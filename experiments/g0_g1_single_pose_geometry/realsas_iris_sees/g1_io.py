from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch


SEMANTIC_VIEW_ORDER = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
RETAINED_VIEW_LABELS = tuple(f"V{i}" for i in range(8))
G1_TARGET_KEYS = ("P_A", "N_A", "XY_A", "V_A", "direct_obs_A")
G1_FORBIDDEN_GPU_TARGET_KEYS = (
    "P_B", "N_B", "XY_B", "V_B", "scene_flow", "persistent_obs",
    "R_neighbor_idx", "R_rel_A", "R_rel_B", "D_local_J",
    "D_surface_action", "D_local_residual_rms", "D_source_projector",
    "D_source_singular_values", "D_intrinsic_rank",
    "D_tangent_anisotropy_ratio", "D_ambient_thickness_ratio",
    "carrier_id_TRAINING_ONLY",
)


def _resolve_pose_a_png(pose_a_dir: Path, i: int) -> Path:
    candidates = (
        pose_a_dir / f"{i:02d}_{RETAINED_VIEW_LABELS[i]}.png",
        pose_a_dir / f"{i:02d}_{SEMANTIC_VIEW_ORDER[i]}.png",
    )
    hits = [p for p in candidates if p.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"G1 Pose-A view resolution failed view={i} "
            f"candidates={[str(p) for p in candidates]} hits={[str(p) for p in hits]}"
        )
    return hits[0]


def load_g1_pose_a(pose_a_dir: str | Path, image_size: int = 128) -> torch.Tensor:
    """Load exactly one neutral pose with eight ordered views -> [8,4,H,W].

    Raster preprocessing is byte-semantically inherited from canonical N1D IO:
    bilinear resize, exact-green background test, background RGB zeroing, alpha mask.
    """
    pose_a_dir = Path(pose_a_dir)
    views = []
    for i in range(8):
        p = _resolve_pose_a_png(pose_a_dir, i)
        im = Image.open(p).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        fg = ~((a[..., 0] < 1e-6) & (a[..., 1] > 0.999) & (a[..., 2] < 1e-6))
        rgb = a.copy()
        rgb[~fg] = 0.0
        rgba = np.concatenate([rgb, fg[..., None].astype(np.float32)], axis=-1)
        views.append(torch.from_numpy(rgba).permute(2, 0, 1))
    return torch.stack(views)


def torch_g1_target(npz_path: str | Path, device=None) -> Tuple[Dict[str, torch.Tensor], int]:
    """Load only G1-authorized Pose-A observable truth.

    The source NPZ may contain canonical N1D Pose-B/mechanics fields, but they are not
    copied into the returned target dictionary and therefore cannot become accidental
    forward/loss inputs in G1.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in G1_TARGET_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"G1 target missing required observable fields: {missing}")
        family_id = int(z["family_id"]) if "family_id" in z.files else -1
        out: Dict[str, torch.Tensor] = {}
        for k in G1_TARGET_KEYS:
            t = torch.from_numpy(z[k])
            if t.dtype == torch.uint8:
                pass
            elif t.dtype in (torch.int16, torch.int32, torch.int64):
                t = t.long()
            else:
                t = t.float()
            t = t.unsqueeze(0)
            out[k] = t.to(device) if device is not None else t
    leaked = sorted(set(out).intersection(G1_FORBIDDEN_GPU_TARGET_KEYS))
    if leaked:
        raise RuntimeError(f"forbidden G1 truth leaked into GPU target: {leaked}")
    return out, family_id


def load_g1_sample(
    pose_a_dir: str | Path,
    target_npz: str | Path,
    image_size: int = 128,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    images = load_g1_pose_a(pose_a_dir, image_size=image_size).unsqueeze(0)
    if device is not None:
        images = images.to(device)
    target, family_id = torch_g1_target(target_npz, device=device)
    return images, target, family_id
