from __future__ import annotations

"""Pinned CharacterGen 3D-stage baseline for RealSaS geometry qualification.

CharacterGen may internally/export transient complete geometry, but RealSaS does
not adopt a complete mesh as its product contract. The only persistent scientific
output of this adapter is the mechanically sufficient RiggingSurfaceIR consumed by
Geppetto/Arachne. Debug mesh retention is opt-in.
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
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
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
    # Alpha is not admitted as learner geometry evidence; match released CharacterGen UI.
    rgb = Image.fromarray(rgba[..., :3], mode="RGB").resize((int(width), int(height)), Image.Resampling.BILINEAR)
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


def _load_charactergen_system(root: Path, device: str):
    stage = root / "3D_Stage"
    sys.path.insert(0, str(stage))
    try:
        import torch
        import lrm
        from lrm.utils.config import load_config
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CharacterGen 3D-stage dependencies are not importable") from exc
    with _pushd(stage):
        cfg = load_config("configs/infer.yaml", makedirs=False)
        if not (stage / "models" / "lrm.ckpt").is_file():
            raise FileNotFoundError("CharacterGen lrm.ckpt missing; run --bootstrap once")
        system = lrm.find(cfg.system_cls)(cfg.system).to(device)
        system.eval()
    return torch, system, cfg


def _load_c2w(stage: Path) -> np.ndarray:
    meta = json.loads((stage / "material" / "meta.json").read_text(encoding="utf-8"))
    mats = np.stack([np.asarray(rec["transform_matrix"], dtype=np.float32) for rec in meta["locations"]], axis=0)
    if mats.shape != (4, 4, 4):
        raise RuntimeError(f"CharacterGen camera contract drift: {mats.shape}")
    return mats[None]


def _save_exporter_outputs(system, exporter_outputs, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    system.set_save_dir(str(save_dir))
    for out in exporter_outputs:
        getattr(system, f"save_{out.save_type}")(out.save_name, **out.params)


def _load_and_align_exported_mesh(save_dir: Path):
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CharacterGen geometry bridge requires trimesh") from exc
    obj = save_dir / "model-00.obj"
    if not obj.is_file():
        candidates = sorted(save_dir.glob("*.obj"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"CharacterGen exporter produced no unique OBJ under {save_dir}")
        obj = candidates[0]
    mesh = trimesh.load(str(obj), force="mesh", process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise RuntimeError("CharacterGen OBJ did not load as a triangle mesh")
    # Match released 3D UI coordinate cleanup; omit optional smoothing/back-projection.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90.0), [-1.0, 0.0, 0.0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180.0), [0.0, 1.0, 0.0]))
    aligned = save_dir / "realsas_aligned_character.obj"
    mesh.export(str(aligned), file_type="obj")
    return mesh, aligned


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
        exporter_outputs = system.exporter(["00"], scene_codes)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="realsas-charactergen-") if not retain_debug_mesh else None
    work_dir = output_dir / "debug_mesh" if retain_debug_mesh else Path(tmp_ctx.name)
    try:
        _save_exporter_outputs(system, exporter_outputs, work_dir)
        mesh, aligned_obj = _load_and_align_exported_mesh(work_dir)
        sys.path.insert(0, str(realsas_root.resolve()))
        from compiler.realsas_compiler_core.substrate.complete_mesh import rigging_surface_from_complete_triangle_mesh
        surface = rigging_surface_from_complete_triangle_mesh(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int64),
            provenance_ref=f"CHARACTERGEN|{head}|{source_asset_id}",
            sample_count=int(sample_count),
            backend_id="CHARACTERGEN_3D_STAGE_PINNED_V1",
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
            },
        )
        surface_path = output_dir / "rigging_surface.json"
        surface_path.write_text(json.dumps(surface.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        report = {
            "schema": "RealSaS.CharacterGenGeometryBackendReport.v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": "CHARACTERGEN_3D_STAGE_PINNED_V1",
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
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "rigging_surface_samples": int(len(surface.surface_nodes)),
            "rigging_surface_lineage_hash": surface.geometry_lineage_hash,
            "debug_mesh_retained": bool(retain_debug_mesh),
            "aligned_obj": str(aligned_obj) if retain_debug_mesh else None,
            "surface_json": str(surface_path),
            "scientific_status": "BASELINE_GEOMETRY_ONLY__NOT_YET_1FIT",
            "important_boundary": "CharacterGen complete geometry is transient model prediction; only RiggingSurfaceIR persists, and generated hidden surface is not labeled observation truth.",
        }
        (output_dir / "CHARACTERGEN_GEOMETRY_REPORT_V1.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _git_value(repo: Path, *args: str, fallback: str = "UNKNOWN") -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip() or fallback
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
            "Run pinned official CharacterGen 3D-stage MultiviewLRM on RealSaS cardinal views using its published camera contract.",
            "Materialize predicted triangle geometry only transiently unless debug retention is explicitly requested.",
            "Apply released UI coordinate alignment deterministically.",
            "Immediately reduce geometry to deterministic area/Halton RiggingSurfaceIR samples with geometric normals.",
            "Do not fabricate observational support_views for model-completed geometry.",
        ],
        "evidence": {
            "mesh_vertices": report["mesh_vertices"],
            "mesh_faces": report["mesh_faces"],
            "rigging_surface_samples": report["rigging_surface_samples"],
            "rigging_surface_lineage_hash": report["rigging_surface_lineage_hash"],
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
            "Proceed to same-family Geppetto/Arachne downstream gate; do not reopen bespoke upstream architecture unless a measured product requirement fails.",
        ],
        "artifact_paths": [
            report["surface_json"],
            str(Path(report["surface_json"]).with_name("CHARACTERGEN_GEOMETRY_REPORT_V1.json")),
        ] + ([report["aligned_obj"]] if report["aligned_obj"] else []),
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
