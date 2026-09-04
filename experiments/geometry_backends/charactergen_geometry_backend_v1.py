from __future__ import annotations

"""Pinned CharacterGen geometry baseline reduced immediately to RealSaS substrate.

CharacterGen is used only as a solved multi-view geometry backbone. RealSaS does
not adopt its textured/complete mesh as a product contract. The official network
produces triplanes; the official DMTet renderer's `isosurface()` method is used to
obtain transient vertices/faces, and those are immediately reduced to deterministic
RiggingSurfaceIR samples for Geppetto/Arachne.

No raster rendering, UV export, source-mesh teacher truth, or fabricated
observational visibility is part of this path.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Sequence

import numpy as np

CHARACTERGEN_REPOSITORY = "https://github.com/zjp-shadow/CharacterGen.git"
CHARACTERGEN_COMMIT = "f329a835dbd5003060a5653eafd83d4d8868b043"
CHARACTERGEN_HF_REPO = "zjpshadow/CharacterGen"
CHARACTERGEN_HF_REVISION = "5b733f0e90d9fe51a126c8462ea33d49ae3bdabe"
CHARACTERGEN_LRM_SHA256 = "ec0edc6eed553910bdf8a1ceb204d8837d7b823826dca55f673c84046078094b"
DEFAULT_REALSAS_CARDINAL_MAP = ("N", "S", "E", "W")  # Back, Front, Right, Left
SUPPORTED_VIEW_NAMES = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_charactergen_checkout(root: Path, *, allow_unpinned: bool = False) -> str:
    root = root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"CharacterGen checkout missing .git: {root}")
    license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("CharacterGen checkout no longer carries expected Apache-2.0 code license")
    head = _git_head(root)
    if head != CHARACTERGEN_COMMIT and not allow_unpinned:
        raise RuntimeError(f"CharacterGen commit drift: got {head}, expected {CHARACTERGEN_COMMIT}")
    return head


def verify_charactergen_checkpoint(root: Path) -> str:
    checkpoint = root.resolve() / "3D_Stage" / "models" / "lrm.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CharacterGen checkpoint missing: {checkpoint}")
    digest = _sha256_file(checkpoint)
    if digest != CHARACTERGEN_LRM_SHA256:
        raise RuntimeError(f"CharacterGen lrm.ckpt hash drift: got {digest}, expected {CHARACTERGEN_LRM_SHA256}")
    return digest


def bootstrap_charactergen(root: Path) -> None:
    root = root.resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", CHARACTERGEN_REPOSITORY, str(root)])
    subprocess.check_call(["git", "-C", str(root), "fetch", "--all", "--tags"])
    subprocess.check_call(["git", "-C", str(root), "checkout", "--detach", CHARACTERGEN_COMMIT])
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CharacterGen bootstrap requires huggingface_hub") from exc
    snapshot_download(
        repo_id=CHARACTERGEN_HF_REPO,
        revision=CHARACTERGEN_HF_REVISION,
        allow_patterns=["3D_Stage/models/*"],
        local_dir=str(root),
    )
    verify_charactergen_checkpoint(root)


def _find_view_image(view_dir: Path, view_name: str) -> Path:
    candidates = []
    for stem in (view_name, view_name.lower(), f"view_{view_name}", f"view_{view_name.lower()}"):
        for ext in (".png", ".webp", ".jpg", ".jpeg"):
            p = view_dir / f"{stem}{ext}"
            if p.is_file():
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"missing RealSaS view {view_name} under {view_dir}")
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous RealSaS view {view_name}: {[str(x) for x in candidates]}")
    return candidates[0]


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    from PIL import Image
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb = Image.fromarray(rgba[..., :3], mode="RGB").resize(
        (int(width), int(height)), Image.Resampling.BILINEAR
    )
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _charactergen_inputs(view_dir: Path, cardinal_map: Sequence[str], width: int, height: int):
    if len(cardinal_map) != 4:
        raise ValueError("cardinal_map must contain exactly four RealSaS view names")
    bad = [v for v in cardinal_map if v not in SUPPORTED_VIEW_NAMES]
    if bad:
        raise ValueError(f"unknown RealSaS view names: {bad}")
    paths = tuple(_find_view_image(view_dir, v) for v in cardinal_map)
    rgb = np.stack([_load_rgb(p, width, height) for p in paths], axis=0)[None]
    return paths, rgb


def _install_isosurface_only_nvdiffrast_stub() -> None:
    """Satisfy CharacterGen imports while forbidding accidental raster execution.

    CharacterGen constructs rasterizer contexts during system configuration even
    when only `renderer.isosurface()` is needed. The isosurface method itself uses
    SDF + marching tetrahedra and never calls nvdiffrast. This stub therefore keeps
    the published network/isosurface code intact while making any accidental
    raster/export call fail loudly instead of requiring a CUDA compiler toolchain.
    """
    if "nvdiffrast.torch" in sys.modules:
        return

    class _NoRasterContext:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _forbidden(*args, **kwargs):
        raise RuntimeError("NVDIFFRAST_FORBIDDEN_IN_ISOSURFACE_ONLY_BASELINE")

    parent = types.ModuleType("nvdiffrast")
    torch_mod = types.ModuleType("nvdiffrast.torch")
    torch_mod.RasterizeGLContext = _NoRasterContext
    torch_mod.RasterizeCudaContext = _NoRasterContext
    torch_mod.rasterize = _forbidden
    torch_mod.antialias = _forbidden
    torch_mod.interpolate = _forbidden
    parent.torch = torch_mod
    sys.modules["nvdiffrast"] = parent
    sys.modules["nvdiffrast.torch"] = torch_mod


def _load_charactergen_system(root: Path, device: str):
    stage = root / "3D_Stage"
    sys.path.insert(0, str(stage))
    _install_isosurface_only_nvdiffrast_stub()
    try:
        import torch
        import lrm
        from lrm.utils.config import load_config
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CharacterGen 3D-stage dependencies are not importable") from exc
    with _pushd(stage):
        cfg = load_config("configs/infer.yaml", makedirs=False)
        system = lrm.find(cfg.system_cls)(cfg.system).to(device)
        system.eval()
    return torch, system, cfg


def _load_c2w(stage: Path) -> np.ndarray:
    meta = json.loads((stage / "material" / "meta.json").read_text(encoding="utf-8"))
    mats = np.stack(
        [np.asarray(rec["transform_matrix"], dtype=np.float32) for rec in meta["locations"]],
        axis=0,
    )
    if mats.shape != (4, 4, 4):
        raise RuntimeError(f"CharacterGen camera contract drift: {mats.shape}")
    return mats[None]


def _align_vertices_like_released_webui(vertices: np.ndarray) -> np.ndarray:
    """Apply CharacterGen webui's -90deg X then 180deg Y alignment."""
    p = np.asarray(vertices, dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    ry = np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return ((p @ rx.T) @ ry.T).astype(np.float32)


def _extract_isosurface(system, scene_codes):
    if scene_codes.ndim != 5 or scene_codes.shape[0] != 1:
        raise RuntimeError(f"unexpected CharacterGen scene-code shape: {tuple(scene_codes.shape)}")
    mesh = system.renderer.isosurface(scene_codes[0])
    vertices = mesh.v_pos.detach().float().cpu().numpy()
    faces = mesh.t_pos_idx.detach().long().cpu().numpy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("CharacterGen DMTet isosurface contract drift")
    if len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError("CharacterGen DMTet produced degenerate surface")
    return _align_vertices_like_released_webui(vertices), faces.astype(np.int64, copy=False)


def run_backend(
    *,
    realsas_root: Path,
    charactergen_root: Path,
    view_dir: Path,
    output_dir: Path,
    source_asset_id: str,
    cardinal_map: Sequence[str] = DEFAULT_REALSAS_CARDINAL_MAP,
    sample_count: int = 2048,
    device: str = "cuda",
    allow_unpinned: bool = False,
    retain_debug_mesh: bool = False,
) -> dict:
    head = verify_charactergen_checkout(charactergen_root, allow_unpinned=allow_unpinned)
    checkpoint_sha256 = verify_charactergen_checkpoint(charactergen_root)
    stage = charactergen_root / "3D_Stage"
    torch, system, cfg = _load_charactergen_system(charactergen_root, device)
    paths, rgb_np = _charactergen_inputs(view_dir, cardinal_map, cfg.data.cond_width, cfg.data.cond_height)
    c2w_np = _load_c2w(stage)
    rgb = torch.from_numpy(rgb_np).float().to(device)
    c2w = torch.from_numpy(c2w_np).float().to(device)

    with torch.no_grad(), _pushd(stage):
        scene_codes = system({"rgb_cond": rgb, "c2w_cond": c2w})
        vertices, faces = _extract_isosurface(system, scene_codes)

    # Release the large external model before compiler-side CPU substrate work.
    del scene_codes, system, rgb, c2w
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_npz = None
    if retain_debug_mesh:
        debug_npz = output_dir / "transient_charactergen_isosurface_debug.npz"
        np.savez_compressed(debug_npz, vertices=vertices, faces=faces)

    sys.path.insert(0, str(realsas_root.resolve()))
    from compiler.realsas_compiler_core.substrate.complete_mesh import (
        rigging_surface_from_complete_triangle_mesh,
    )

    surface = rigging_surface_from_complete_triangle_mesh(
        vertices,
        faces,
        provenance_ref=f"CHARACTERGEN|{head}|{source_asset_id}",
        sample_count=int(sample_count),
        backend_id="CHARACTERGEN_DMTET_ISOSURFACE_PINNED_V1",
        source_asset_id=source_asset_id,
        extra_metadata={
            "external_repository": CHARACTERGEN_REPOSITORY,
            "external_commit": head,
            "external_code_license": "Apache-2.0",
            "external_model_repo": CHARACTERGEN_HF_REPO,
            "external_model_revision": CHARACTERGEN_HF_REVISION,
            "external_lrm_sha256": checkpoint_sha256,
            "input_realSaS_cardinal_order_for_back_front_right_left": list(cardinal_map),
            "external_complete_mesh_persisted": bool(retain_debug_mesh),
            "external_rasterizer_executed": False,
            "external_geometry_extraction": "OFFICIAL_RENDERER_ISOSURFACE_ONLY",
        },
    )
    surface_path = output_dir / "rigging_surface.json"
    surface_path.write_text(json.dumps(surface.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "schema": "RealSaS.CharacterGenGeometryBackendReport.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "CHARACTERGEN_DMTET_ISOSURFACE_PINNED_V1",
        "charactergen_commit": head,
        "charactergen_code_license": "Apache-2.0",
        "charactergen_model_license": "Apache-2.0",
        "charactergen_hf_revision": CHARACTERGEN_HF_REVISION,
        "charactergen_lrm_sha256": checkpoint_sha256,
        "source_asset_id": source_asset_id,
        "input_view_dir": str(view_dir.resolve()),
        "input_mapping_back_front_right_left": {
            "back": cardinal_map[0], "front": cardinal_map[1], "right": cardinal_map[2], "left": cardinal_map[3]
        },
        "input_files": [str(x.resolve()) for x in paths],
        "transient_isosurface_vertices": int(len(vertices)),
        "transient_isosurface_faces": int(len(faces)),
        "rigging_surface_samples": int(len(surface.surface_nodes)),
        "rigging_surface_lineage_hash": surface.geometry_lineage_hash,
        "debug_mesh_retained": bool(retain_debug_mesh),
        "debug_mesh_npz": str(debug_npz) if debug_npz else None,
        "surface_json": str(surface_path),
        "rasterizer_executed": False,
        "scientific_status": "BASELINE_GEOMETRY_ONLY__NOT_YET_1FIT",
        "important_boundary": "CharacterGen DMTet surface is transient model prediction; only RiggingSurfaceIR persists, and generated hidden surface is not labeled observation truth.",
    }
    report_path = output_dir / "CHARACTERGEN_GEOMETRY_REPORT_V1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _git_value(repo: Path, *args: str, fallback: str = "UNKNOWN") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip() or fallback
    except Exception:
        return fallback


def append_experiment_ledger(realsas_root: Path, report: dict) -> None:
    ledger = realsas_root.resolve() / "experiments" / "EXPERIMENT_LEDGER.jsonl"
    if not ledger.is_file():
        raise FileNotFoundError(f"append-only experiment ledger missing: {ledger}")
    ts = str(report["timestamp_utc"])
    record = {
        "record_type": "experiment_record",
        "record_id": "CHARACTERGEN_GEOMETRY_" + str(report["rigging_surface_lineage_hash"])[:16],
        "date_utc": ts[:10],
        "trigger_message_utc": None,
        "branch": _git_value(realsas_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head_before": _git_value(realsas_root, "rev-parse", "HEAD"),
        "kind": "CHARACTERGEN_GEOMETRY_BASELINE",
        "question": "Can published CharacterGen four-view reconstruction provide mechanically sufficient geometry for the same RiggingSurfaceIR consumed by Geppetto/Arachne?",
        "inputs": {
            "source_asset_id": report["source_asset_id"],
            "input_files": report["input_files"],
            "mapping_back_front_right_left": report["input_mapping_back_front_right_left"],
            "charactergen_code_commit": report["charactergen_commit"],
            "charactergen_hf_revision": report["charactergen_hf_revision"],
            "charactergen_lrm_sha256": report["charactergen_lrm_sha256"],
        },
        "procedure": [
            "Run pinned official CharacterGen MultiviewLRM on four RealSaS cardinal RGB views using its published camera contract.",
            "Use the official TriplaneDMTetRenderer.isosurface method; forbid nvdiffrast rendering/export in this baseline.",
            "Apply the released webui coordinate alignment deterministically.",
            "Immediately reduce transient DMTet geometry to deterministic area/Halton RiggingSurfaceIR samples with geometric normals.",
            "Do not fabricate observational support_views for model-completed geometry.",
        ],
        "evidence": {
            "transient_isosurface_vertices": report["transient_isosurface_vertices"],
            "transient_isosurface_faces": report["transient_isosurface_faces"],
            "rigging_surface_samples": report["rigging_surface_samples"],
            "rigging_surface_lineage_hash": report["rigging_surface_lineage_hash"],
            "rasterizer_executed": report["rasterizer_executed"],
            "debug_mesh_retained": report["debug_mesh_retained"],
            "surface_json": report["surface_json"],
        },
        "result": {
            "status": report["scientific_status"],
            "geometry_backend_executed": True,
            "downstream_1fit_executed": False,
            "boundary": report["important_boundary"],
        },
        "decision": [
            "Use this solved-literature geometry arm as a baseline against IRIS-Q.",
            "Proceed to same-family Geppetto/Arachne downstream gate; do not reopen bespoke upstream reconstruction unless a measured product requirement fails.",
        ],
        "artifact_paths": [
            report["surface_json"],
            str(Path(report["surface_json"]).with_name("CHARACTERGEN_GEOMETRY_REPORT_V1.json")),
        ] + ([report["debug_mesh_npz"]] if report["debug_mesh_npz"] else []),
    }
    with ledger.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--realsas-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--charactergen-root", type=Path, required=True)
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-asset-id", required=True)
    p.add_argument("--sample-count", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cardinal-map", nargs=4, metavar=("BACK", "FRONT", "RIGHT", "LEFT"), default=DEFAULT_REALSAS_CARDINAL_MAP)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--allow-unpinned", action="store_true")
    p.add_argument("--retain-debug-mesh", action="store_true")
    args = p.parse_args()
    if args.bootstrap:
        bootstrap_charactergen(args.charactergen_root)
    report = run_backend(
        realsas_root=args.realsas_root,
        charactergen_root=args.charactergen_root,
        view_dir=args.view_dir,
        output_dir=args.output_dir,
        source_asset_id=args.source_asset_id,
        cardinal_map=tuple(args.cardinal_map),
        sample_count=args.sample_count,
        device=args.device,
        allow_unpinned=args.allow_unpinned,
        retain_debug_mesh=args.retain_debug_mesh,
    )
    append_experiment_ledger(args.realsas_root, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
