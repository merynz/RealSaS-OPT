from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

SEMANTIC_VIEW_ORDER = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
RETAINED_VIEW_LABELS = tuple(f"V{i}" for i in range(8))
ALLOWED_TARGET_KEYS = ("family_id", "P_A", "N_A", "XY_A", "V_A", "direct_obs_A")


def _resolve_view(pose_a_dir: Path, i: int) -> Path:
    candidates = [pose_a_dir / f"{i:02d}_{RETAINED_VIEW_LABELS[i]}.png", pose_a_dir / f"{i:02d}_{SEMANTIC_VIEW_ORDER[i]}.png", pose_a_dir / f"V{i}.png"]
    hits = [p for p in candidates if p.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(f"view {i} resolution failed: hits={hits}, candidates={candidates}")
    return hits[0]


def load_native_rgba(pose_a_dir: str | Path, native_size: int = 1024) -> torch.Tensor:
    pose_a_dir = Path(pose_a_dir)
    views: List[torch.Tensor] = []
    for i in range(8):
        p = _resolve_view(pose_a_dir, i)
        im = Image.open(p)
        if im.size != (native_size, native_size):
            raise ValueError(f"native authority must be {native_size}x{native_size}; {p} is {im.size}")
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        file_alpha = rgba[..., 3].astype(np.float32) / 255.0
        if np.any(rgba[..., 3] < 255):
            alpha = file_alpha
        else:
            green = (rgba[..., 0] == 0) & (rgba[..., 1] == 255) & (rgba[..., 2] == 0)
            alpha = (~green).astype(np.float32)
        rgb[alpha < 0.5] = 0.0
        views.append(torch.from_numpy(np.concatenate([rgb, alpha[..., None]], axis=-1)).permute(2, 0, 1).contiguous())
    return torch.stack(views, dim=0)


def _convert_xy_to_xy01(xy: np.ndarray, reference_size: int) -> Tuple[np.ndarray, str]:
    if not np.isfinite(xy).all() or xy.shape[-1] != 2:
        raise ValueError("XY_A must be finite [...,2]")
    lo, hi = float(xy.min()), float(xy.max())
    if lo >= -1e-5 and hi <= 1.0001:
        return xy.astype(np.float32), "xy01"
    if lo >= -1e-3 and hi <= reference_size - 1 + 1e-3:
        return ((xy + 0.5) / float(reference_size)).astype(np.float32), f"pixel{reference_size}_center"
    raise ValueError(f"cannot establish XY authority: min={lo}, max={hi}, reference_size={reference_size}")


def load_target_npz(path: str | Path, reference_size: int = 256) -> Dict[str, torch.Tensor | int | str]:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ALLOWED_TARGET_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"target missing required fields {missing}")
        out_np = {k: z[k] for k in ALLOWED_TARGET_KEYS}
    P = out_np["P_A"].astype(np.float32)
    N = out_np["N_A"].astype(np.float32)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    xy01, xy_mode = _convert_xy_to_xy01(out_np["XY_A"].astype(np.float32), reference_size)
    V = out_np["V_A"].astype(bool)
    direct = out_np["direct_obs_A"].astype(bool)
    if P.ndim != 2 or P.shape[-1] != 3 or N.shape != P.shape:
        raise ValueError((P.shape, N.shape))
    if xy01.shape[:2] != (8, P.shape[0]) or V.shape != (8, P.shape[0]):
        raise ValueError((xy01.shape, V.shape, P.shape))
    return {"family_id": int(np.asarray(out_np["family_id"]).item()), "P": torch.from_numpy(P), "N": torch.from_numpy(N), "XY01": torch.from_numpy(xy01), "V": torch.from_numpy(V), "direct_obs": torch.from_numpy(direct), "xy_source_mode": xy_mode}


def read_manifest(path: str | Path, split: str | None = None) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    else:
        obj = json.loads(path.read_text()); rows = obj["rows"] if isinstance(obj, dict) and "rows" in obj else obj
    required = {"family_id", "split", "pose_a_dir", "target_npz"}
    for row in rows:
        if not required.issubset(row): raise KeyError(f"manifest row missing {sorted(required-set(row))}: {row}")
    if split is not None: rows = [r for r in rows if str(r["split"]).upper() == split.upper()]
    if not rows: raise ValueError(f"manifest produced no rows for split={split}")
    ids = [int(r["family_id"]) for r in rows]
    if len(ids) != len(set(ids)): raise ValueError("manifest contains duplicate family_id within selected split")
    return rows


class OrthoFamilyDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, native_size: int = 1024, xy_reference_size: int = 256):
        self.rows = read_manifest(manifest_path, split); self.native_size = int(native_size); self.xy_reference_size = int(xy_reference_size)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index: int):
        row = self.rows[index]
        images = load_native_rgba(row["pose_a_dir"], self.native_size)
        target = load_target_npz(row["target_npz"], self.xy_reference_size)
        if int(row["family_id"]) != target["family_id"]: raise ValueError((row["family_id"], target["family_id"]))
        return {"images": images, "target": target, "row": row}


def collate_one_family(batch):
    if len(batch) != 1: raise ValueError("MapAnything Ortho V1 intentionally uses one 8-view family per microbatch")
    item = batch[0]; t = item["target"]
    return {"images": item["images"].unsqueeze(0), "target": {"family_id": torch.tensor([t["family_id"]], dtype=torch.long), "P": t["P"].unsqueeze(0), "N": t["N"].unsqueeze(0), "XY01": t["XY01"].unsqueeze(0), "V": t["V"].unsqueeze(0), "direct_obs": t["direct_obs"].unsqueeze(0), "xy_source_mode": [t["xy_source_mode"]]}, "row": [item["row"]]}
