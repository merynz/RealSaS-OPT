#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed A100 admission preflight for SURF-A1-FROZEN-H."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.flexicubes_v1 import FLEXICUBES_COMMIT, FLEXICUBES_REPOSITORY, PinnedFlexiCubesV1
from models.iris.vnext.surface_rescue_raster_v1 import binary_precision_recall_v1, project_iris_to_pytorch3d_ndc_v1
from models.iris.vnext.surface_rescue_training_v1 import extract_rescued_surface_v1, module_state_sha256_v1
from models.iris.vnext.surface_rescue_v1 import FrozenHSurfaceRescueV1
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head, load_exact_v8_model, load_json, load_subject_source, sha256_file
from tools.iris_vnext.surf_a1_io_v1 import camera_tensors_v1, load_camera_bundles_v1, load_raster_truth_v1, load_surf_a1_binding_v1, load_teacher_rest_iris_v1, require_git_identity_v1

PYTORCH3D_REPOSITORY = "https://github.com/facebookresearch/pytorch3d.git"
PYTORCH3D_COMMIT = "978cd99221b9e0a6a568f1d427854d73363265cf"
SCHEMA = "RealSaS.SurfA1A100Preflight.v1"


def _hard_coverage(vertices_iris: torch.Tensor, faces: torch.Tensor, *, camera: dict, image_size: int) -> torch.Tensor:
    try:
        import pytorch3d
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
    except Exception as exc:
        raise RuntimeError("SURF_A1_PREFLIGHT_PYTORCH3D_IMPORT_FAILED") from exc
    ndc, _ = project_iris_to_pytorch3d_ndc_v1(
        vertices_iris,
        common_center_world=camera["common_center_world"],
        common_half_extent=camera["common_half_extent"],
        camera_origin_world=camera["origin_world"],
        camera_right_world=camera["right_world"],
        camera_screen_up_world=camera["screen_up_world"],
        camera_forward_world=camera["forward_world"],
        camera_half_extent=camera["half_extent"],
    )
    pix, _, _, _ = rasterize_meshes(
        Meshes(verts=[ndc], faces=[faces.long()]),
        image_size=int(image_size), blur_radius=0.0, faces_per_pixel=1,
        perspective_correct=False, clip_barycentric_coords=False,
        cull_backfaces=False, cull_to_frustum=False,
    )
    return pix[0, ..., 0] >= 0


def _verify_pytorch3d_import_root(root: Path) -> str:
    import pytorch3d
    module_path = Path(pytorch3d.__file__).resolve()
    source = root.resolve()
    if not module_path.is_relative_to(source):
        raise RuntimeError(f"SURF_A1_PREFLIGHT_PYTORCH3D_IMPORT_SOURCE_DRIFT:{module_path}:{source}")
    return str(module_path)


def _gpu_gate() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("SURF_A1_PREFLIGHT_CUDA_REQUIRED")
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    gib = float(props.total_memory) / float(1024 ** 3)
    if "A100" not in name.upper():
        raise RuntimeError("SURF_A1_PREFLIGHT_A100_REQUIRED:" + name)
    if gib < 39.0:
        raise RuntimeError(f"SURF_A1_PREFLIGHT_GPU_MEMORY_TOO_SMALL:{gib:.3f}")
    return {"name": name, "total_memory_gib": gib, "torch": torch.__version__, "cuda": torch.version.cuda}


def preflight(args: argparse.Namespace) -> dict:
    repo = Path(args.repo_root).resolve()
    expected_source = str(args.expected_source_sha).lower()
    if git_head(repo) != expected_source:
        raise RuntimeError("SURF_A1_PREFLIGHT_SOURCE_SHA_DRIFT")
    prereg = load_json(Path(args.preregistration).resolve())
    if prereg.get("schema") != "RealSaS.SurfA1FrozenHPreregistration.v1":
        raise ValueError("SURF_A1_PREFLIGHT_PREREG_SCHEMA_INVALID")
    binding = load_surf_a1_binding_v1(Path(args.binding))
    gpu = _gpu_gate()
    require_git_identity_v1(Path(args.flexicubes_root), repository=FLEXICUBES_REPOSITORY, commit=FLEXICUBES_COMMIT, code="SURF_A1_PREFLIGHT_FLEXICUBES")
    require_git_identity_v1(Path(args.pytorch3d_root), repository=PYTORCH3D_REPOSITORY, commit=PYTORCH3D_COMMIT, code="SURF_A1_PREFLIGHT_PYTORCH3D")
    pytorch3d_module = _verify_pytorch3d_import_root(Path(args.pytorch3d_root))

    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("SURF_A1_PREFLIGHT_PREPARED_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("SURF_A1_PREFLIGHT_SUBJECT_ORDER_DRIFT")
    if sha256_file(manifest_path) != binding["prepared"]["manifest"]["sha256"]:
        raise ValueError("SURF_A1_PREFLIGHT_MANIFEST_SHA_DRIFT")
    if sha256_file(Path(args.stage01_bundle)) != binding["stage01"]["sha256"]:
        raise ValueError("SURF_A1_PREFLIGHT_STAGE01_SHA_DRIFT")
    teacher_batch_path = Path(args.teacher_batch_receipt).resolve()
    if sha256_file(teacher_batch_path) != binding["authored_teachers"]["batch_receipt"]["sha256"]:
        raise ValueError("SURF_A1_PREFLIGHT_TEACHER_BATCH_SHA_DRIFT")
    raster_batch = load_json(Path(args.raster_truth_root).resolve() / "SURF_A1_RASTER_TRUTH_BATCH_RECEIPT.json")
    if raster_batch.get("status") != "MATERIALIZED_SURF_A1_RASTER_TRUTH__TRAINING_ONLY" or raster_batch.get("optimizer_steps_executed") != 0:
        raise ValueError("SURF_A1_PREFLIGHT_RASTER_TRUTH_STATUS_INVALID")

    device = torch.device("cuda")
    v8 = load_exact_v8_model(Path(args.v8_checkpoint), device)
    frozen_hash_before = module_state_sha256_v1(v8.scene_core)
    decoder = FrozenHSurfaceRescueV1().to(device=device)
    decoder.eval()
    extractor = PinnedFlexiCubesV1(Path(args.flexicubes_root), device=device, weight_scale=v8.config.flexicubes_weight_scale)
    cameras = load_camera_bundles_v1(Path(args.stage01_bundle), binding)
    teacher_binding = {row["subject_id"]: row for row in binding["authored_teachers"]["subjects"]}

    native_root = Path(args.native_v8_root).resolve()
    zero_step_rows = []
    raster_rows = []
    for subject_index, row in enumerate(rows):
        subject_id = str(row["subject_id"])
        print(f"[SURF-A1 PREFLIGHT] {subject_index+1}/8 {subject_id}: zero-step R24 parity", flush=True)
        shard = Path(args.prepared_root).resolve() / str(row["shard_file"])
        source = load_subject_source(shard, str(row["shard_sha256"]), device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            planes = v8.encode_scene(source["image_tokens"], source["token_xy"], source["camera_features"]).float().detach()
        with torch.no_grad():
            surface = extract_rescued_surface_v1(v8, decoder, planes, extractor, resolution=24, cube_chunk=int(args.cube_chunk), training=False)
        reference_path = native_root / "subjects" / subject_id / "V8_NATIVE_FLEXICUBES_R24.npz"
        with np.load(reference_path, allow_pickle=False) as reference:
            reference_vertices = np.asarray(reference["vertices_normalized"], dtype=np.float32)
            reference_faces = np.asarray(reference["faces"], dtype=np.int64)
        actual_vertices = surface["vertices_iris"].detach().cpu().float().numpy()
        actual_faces = surface["faces"].detach().cpu().numpy().astype(np.int64)
        if actual_vertices.shape != reference_vertices.shape or actual_faces.shape != reference_faces.shape:
            raise RuntimeError("SURF_A1_PREFLIGHT_NATIVE_R24_SHAPE_DRIFT:" + subject_id)
        face_exact = bool(np.array_equal(actual_faces, reference_faces))
        max_vertex = float(np.max(np.abs(actual_vertices - reference_vertices))) if actual_vertices.size else 0.0
        if not face_exact or max_vertex > 2.0e-6:
            raise RuntimeError(f"SURF_A1_PREFLIGHT_NATIVE_R24_PARITY_FAIL:{subject_id}:{face_exact}:{max_vertex}")
        zero_step_rows.append({"subject_id": subject_id, "faces_exact": face_exact, "max_abs_vertex_difference_iris": max_vertex})

        print(f"[SURF-A1 PREFLIGHT] {subject_id}: PyTorch3D hard-raster parity 8/8", flush=True)
        truth = load_raster_truth_v1(Path(args.raster_truth_root), subject_id, device)
        tbind = teacher_binding[subject_id]
        teacher_path = Path(args.teacher_dir).resolve() / str(tbind["file"])
        if sha256_file(teacher_path) != tbind["sha256"]:
            raise ValueError("SURF_A1_PREFLIGHT_TEACHER_SHA_DRIFT:" + subject_id)
        teacher_vertices, teacher_faces = load_teacher_rest_iris_v1(teacher_path, cameras[subject_id])
        tv = torch.from_numpy(teacher_vertices).to(device=device, dtype=torch.float32)
        tf = torch.from_numpy(teacher_faces).to(device=device, dtype=torch.long)
        for view_index in range(8):
            camera = camera_tensors_v1(cameras[subject_id], view_index, device)
            predicted = _hard_coverage(tv, tf, camera=camera, image_size=256)
            metrics = binary_precision_recall_v1(predicted, truth["teacher_center_coverage"][view_index])
            if metrics["precision"] < 0.98 or metrics["recall"] < 0.98:
                raise RuntimeError(f"SURF_A1_PREFLIGHT_RASTER_PARITY_FAIL:{subject_id}:V{view_index}:{metrics}")
            raster_rows.append({"subject_id": subject_id, "view_index": view_index, **metrics})
        del source, planes, surface, tv, tf, truth
        torch.cuda.empty_cache()

    frozen_hash_after = module_state_sha256_v1(v8.scene_core)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("SURF_A1_PREFLIGHT_V8_SCENE_CORE_MUTATED")
    result = {
        "schema": SCHEMA,
        "status": "PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED",
        "execution_source_sha": expected_source,
        "gpu": gpu,
        "pytorch3d": {"repository": PYTORCH3D_REPOSITORY, "commit": PYTORCH3D_COMMIT, "module": pytorch3d_module},
        "flexicubes": extractor.identity,
        "v8_scene_core_sha256": frozen_hash_before,
        "zero_step_native_r24": zero_step_rows,
        "hard_raster_parity": {"minimum_precision": min(r["precision"] for r in raster_rows), "minimum_recall": min(r["recall"] for r in raster_rows), "views": len(raster_rows)},
        "optimizer_steps_executed": 0,
        "teacher_truth_in_inference_forward": False,
        "training_authorized": True,
        "scientific_pass_claimed": False,
        "next_action": "RUN_SURF_A1_2400_STEP_FROZEN_H_TRAINING",
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SURF_A1_PREFLIGHT_PASS", json.dumps({"min_precision": result["hard_raster_parity"]["minimum_precision"], "min_recall": result["hard_raster_parity"]["minimum_recall"], "scene_core": frozen_hash_before[:12]}, sort_keys=True), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json")
    p.add_argument("--preregistration", default="canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--teacher-batch-receipt", required=True)
    p.add_argument("--raster-truth-root", required=True)
    p.add_argument("--native-v8-root", required=True)
    p.add_argument("--flexicubes-root", required=True)
    p.add_argument("--pytorch3d-root", required=True)
    p.add_argument("--cube-chunk", type=int, default=2048)
    p.add_argument("--output", required=True)
    preflight(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
