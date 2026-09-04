from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
from PIL import Image
import torch

from compiler.realsas_compiler_core.substrate.reconstructed_mesh import rigging_surface_from_obj


CHARACTERGEN_UPSTREAM_REPO = "https://github.com/zjp-shadow/CharacterGen.git"
CHARACTERGEN_PINNED_COMMIT = "f329a835dbd5003060a5653eafd83d4d8868b043"
CHARACTERGEN_LICENSE = "Apache-2.0"
HF_REPO_ID = "zjpshadow/CharacterGen"
CANONICAL_REALSAS_VIEW_ORDER = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
# CharacterGen pretrained 3D stage web UI expects [back, front, right, left].
CARDINAL4_REALSAS_INDICES = (4, 0, 2, 6)


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"CHARACTERGEN_GIT_HEAD_UNAVAILABLE:{repo}:{exc}") from exc


def validate_charactergen_checkout(root: Path, *, allow_upstream_drift: bool = False) -> dict:
    root = root.resolve()
    stage = root / "3D_Stage"
    required = (
        root / "LICENSE",
        stage / "configs" / "infer.yaml",
        stage / "lrm" / "systems" / "multiview_lrm.py",
        stage / "material" / "meta.json",
        stage / "models" / "lrm.ckpt",
        stage / "models" / "base",
        stage / "load" / "tets",
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "CHARACTERGEN_CHECKOUT_INCOMPLETE:" + json.dumps(missing) +
            f"; source={CHARACTERGEN_UPSTREAM_REPO}; weights={HF_REPO_ID}"
        )
    license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("CHARACTERGEN_LICENSE_DRIFT")
    head = _git_head(root)
    if head != CHARACTERGEN_PINNED_COMMIT and not allow_upstream_drift:
        raise RuntimeError(f"CHARACTERGEN_UPSTREAM_DRIFT:{head}!={CHARACTERGEN_PINNED_COMMIT}")
    return {
        "upstream_repo": CHARACTERGEN_UPSTREAM_REPO,
        "upstream_commit": head,
        "expected_commit": CHARACTERGEN_PINNED_COMMIT,
        "license": CHARACTERGEN_LICENSE,
        "hf_repo_id": HF_REPO_ID,
        "checkpoint_sha256": _sha256_file(stage / "models" / "lrm.ckpt"),
    }


def _load_view_rgb(path: Path, *, width: int, height: int, alpha_policy: str) -> np.ndarray:
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if alpha_policy == "upstream_ignore":
        rgb = rgba[..., :3]
    elif alpha_policy == "gray_composite":
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = np.rint(rgba[..., :3].astype(np.float32) * a + 127.5 * (1.0 - a)).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown alpha policy:{alpha_policy}")
    # CharacterGen upstream uses cv2.resize. Import cv2 here only after its environment
    # has been provisioned so source-only tests of this file do not require OpenCV.
    import cv2
    return cv2.resize(rgb, (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def _character_gen_cardinal_c2w(stage: Path) -> np.ndarray:
    meta = json.loads((stage / "material" / "meta.json").read_text(encoding="utf-8"))
    mats = np.asarray([x["transform_matrix"] for x in meta["locations"]], dtype=np.float32)
    if mats.shape != (4, 4, 4) or not np.isfinite(mats).all():
        raise RuntimeError("CHARACTERGEN_CARDINAL_CAMERA_CONTRACT_DRIFT")
    return mats


def _look_at_c2w(position: Iterable[float]) -> np.ndarray:
    p = np.asarray(tuple(position), dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("bad camera position")
    n = float(np.linalg.norm(p))
    if n <= 1e-12:
        raise ValueError("camera at origin")
    z = p / n  # CharacterGen c2w third column points from target to camera.
    up_world = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    x = np.cross(z, up_world)
    xn = float(np.linalg.norm(x))
    if xn <= 1e-12:
        raise ValueError("degenerate camera up")
    x /= xn
    y = np.cross(x, z)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x.astype(np.float32)
    m[:3, 1] = y.astype(np.float32)
    m[:3, 2] = z.astype(np.float32)
    m[:3, 3] = p.astype(np.float32)
    return m


def _character_gen_all8_c2w(radius: float = 1.5) -> np.ndarray:
    # RealSaS compass convention: S is front (-Y), E is +X, N is back (+Y), W is -X.
    positions = []
    for view_index in range(8):
        theta = -0.5 * math.pi + view_index * (math.pi / 4.0)
        positions.append((radius * math.cos(theta), radius * math.sin(theta), 0.0))
    mats = np.stack([_look_at_c2w(p) for p in positions], axis=0)
    return mats


def _conditioning_contract(stage: Path, mode: str) -> tuple[tuple[int, ...], np.ndarray, dict]:
    if mode == "cardinal4":
        indices = CARDINAL4_REALSAS_INDICES
        cameras = _character_gen_cardinal_c2w(stage)
        note = "UPSTREAM_PRETRAINED_CONTRACT_BACK_FRONT_RIGHT_LEFT_FROM_REALSAS_N_S_E_W"
        experimental = False
    elif mode == "all8":
        indices = tuple(range(8))
        cameras = _character_gen_all8_c2w()
        note = "EXPERIMENTAL_EIGHT_VIEW_CONDITIONING_USING_CHARACTERGEN_CAMERA_CONVENTION"
        experimental = True
    else:
        raise ValueError(f"unknown conditioning mode:{mode}")
    return indices, cameras, {
        "mode": mode,
        "realsas_indices": indices,
        "realsas_labels": [CANONICAL_REALSAS_VIEW_ORDER[i] for i in indices],
        "camera_contract": note,
        "zero_shot_view_count_extension": experimental,
    }


@contextmanager
def _character_gen_import_path(stage: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(stage)
        sys.path.insert(0, str(stage))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def run_charactergen_geometry(
    *,
    charactergen_root: Path,
    observation_root: Path,
    out_root: Path,
    conditioning_mode: str,
    alpha_policy: str,
    device: str,
) -> tuple[Path, dict]:
    stage = charactergen_root / "3D_Stage"
    indices, c2w_np, cond_meta = _conditioning_contract(stage, conditioning_mode)

    image_paths = tuple(observation_root / f"V{i}" / "RGBA.png" for i in indices)
    missing = [str(x) for x in image_paths if not x.exists()]
    if missing:
        raise RuntimeError("MISSING_REALSAS_OBSERVATION_VIEWS:" + json.dumps(missing))

    with _character_gen_import_path(stage):
        import lrm
        from lrm.utils.config import load_config

        cfg = load_config(str(stage / "configs" / "infer.yaml"), makedirs=False)
        cfg.system.weights = str(stage / "models" / "lrm.ckpt")
        cfg.system.image_tokenizer.pretrained_model_name_or_path = str(stage / "models" / "base")
        cfg.system.renderer.tet_dir = str(stage / "load" / "tets") + os.sep
        # Geometry substrate does not need UV unwrap/back-projection/texture export.
        cfg.system.exporter.fmt = "obj"
        cfg.system.exporter.visual = "vertex"
        cfg.system.exporter.save_uv = False
        cfg.system.exporter.save_texture = False
        cfg.system.exporter.output_path = str(out_root / "charactergen_export_aux")

        system = lrm.find(cfg.system_cls)(cfg.system).to(device)
        system.eval()
        width = int(cfg.data.cond_width)
        height = int(cfg.data.cond_height)
        rgb = np.stack([
            _load_view_rgb(p, width=width, height=height, alpha_policy=alpha_policy)
            for p in image_paths
        ], axis=0)
        rgb_cond = torch.from_numpy(rgb).float()[None].to(device)
        c2w_cond = torch.from_numpy(c2w_np).float()[None].to(device)

        if rgb_cond.shape[1] != c2w_cond.shape[1]:
            raise RuntimeError("CHARACTERGEN_VIEW_CAMERA_COUNT_MISMATCH")
        with torch.no_grad():
            scene_codes = system({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})
            exporter_output = system.exporter(["00"], scene_codes)

        export_root = out_root / "charactergen_mesh"
        export_root.mkdir(parents=True, exist_ok=True)
        system.set_save_dir(str(export_root))
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(system, save_func_name):
                raise RuntimeError(f"CHARACTERGEN_EXPORT_SAVE_UNSUPPORTED:{save_func_name}")
            getattr(system, save_func_name)(out.save_name, **out.params)

    obj = export_root / "model-00.obj"
    if not obj.exists() or obj.stat().st_size <= 0:
        candidates = sorted(export_root.rglob("*.obj"))
        if len(candidates) != 1:
            raise RuntimeError(f"CHARACTERGEN_OBJ_NOT_UNIQUE:{[str(x) for x in candidates]}")
        obj = candidates[0]
    meta = {
        **cond_meta,
        "source_view_files": [str(x) for x in image_paths],
        "source_view_sha256": [_sha256_file(x) for x in image_paths],
        "alpha_policy": alpha_policy,
        "condition_resolution": [width, height],
        "character_gen_obj": str(obj),
        "character_gen_obj_sha256": _sha256_file(obj),
    }
    return obj, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charactergen-root", required=True)
    ap.add_argument("--observation-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--conditioning-mode", choices=("cardinal4", "all8"), default="cardinal4")
    ap.add_argument("--alpha-policy", choices=("upstream_ignore", "gray_composite"), default="gray_composite")
    ap.add_argument("--surface-samples", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-upstream-drift", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    charactergen_root = Path(args.charactergen_root).resolve()
    observation_root = Path(args.observation_root).resolve()

    upstream = validate_charactergen_checkout(
        charactergen_root,
        allow_upstream_drift=bool(args.allow_upstream_drift),
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CHARACTERGEN_BASELINE_REQUIRES_CUDA")

    prereg = {
        "schema": "RealSaS.CharacterGenGeometryBaseline.v1",
        "status": "PREREGISTERED_BEFORE_INFERENCE",
        "upstream": upstream,
        "conditioning_mode": args.conditioning_mode,
        "alpha_policy": args.alpha_policy,
        "surface_samples": int(args.surface_samples),
        "device": args.device,
        "purpose": "Solved external 4-view/full-3D reconstruction baseline feeding RealSaS RiggingSurfaceIR; all8 is an explicitly experimental view-count extension.",
        "not_claimed": [
            "CharacterGen hidden surfaces are observation truth",
            "all8 zero-shot conditioning is pretrained/validated",
            "CharacterGen reconstruction preserves source pixels exactly",
        ],
    }
    _write_json(out_root / "CHARACTERGEN_BASELINE_PREREG_V1.json", prereg)

    obj, inference_meta = run_charactergen_geometry(
        charactergen_root=charactergen_root,
        observation_root=observation_root,
        out_root=out_root,
        conditioning_mode=args.conditioning_mode,
        alpha_policy=args.alpha_policy,
        device=args.device,
    )
    surface = rigging_surface_from_obj(
        obj,
        sample_count=int(args.surface_samples),
        source_ref=f"CHARACTERGEN:{upstream['upstream_commit']}:{inference_meta['character_gen_obj_sha256']}",
        reconstruction_model="CharacterGen.3D_Stage.MultiviewLRM",
        reconstruction_checkpoint=upstream["checkpoint_sha256"],
    )
    torch.save(
        {
            "schema": "RealSaS.CharacterGenRiggingSurfaceBundle.v1",
            "surface": surface,
            "upstream": upstream,
            "inference": inference_meta,
        },
        out_root / "CHARACTERGEN_RIGGING_SURFACE.pt",
    )
    result = {
        "status": "PASS_CHARACTERGEN_3D_TO_RIGGING_SURFACE",
        "upstream": upstream,
        "inference": inference_meta,
        "surface_nodes": len(surface.surface_nodes),
        "surface_relations": len(surface.local_relations),
        "surface_geometry_lineage_hash": surface.geometry_lineage_hash,
        "surface_builder_id": surface.builder_id,
        "surface_source_domain": surface.metadata.get("source_domain"),
        "surface_full_3d_reconstruction_claim": surface.metadata.get("full_3d_reconstruction_claim"),
        "surface_observational_support_authority": surface.metadata.get("observational_support_authority"),
    }
    _write_json(out_root / "CHARACTERGEN_BASELINE_RESULT_V1.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
