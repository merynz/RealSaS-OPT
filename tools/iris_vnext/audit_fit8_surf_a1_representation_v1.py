#!/usr/bin/env python3
from __future__ import annotations

"""Independent preregistered representation-rescue court for SURF-A1."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1, vertex_normals_v1
from models.iris.vnext.surface_rescue_training_v1 import extract_rescued_surface_v1, module_state_sha256_v1
from models.iris.vnext.surface_rescue_v1 import FrozenHSurfaceRescueV1
from tools.corpus.iris_observation_admission_v1 import project_mask
from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import measure_surface_against_shard
from tools.iris_vnext.audit_fit8_direct_surface_silhouette_v1 import _mask_metrics, sha256_bytes
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head, load_exact_v8_model, load_json, load_subject_source, sha256_file
from tools.iris_vnext.surf_a1_io_v1 import load_camera_bundles_v1, load_surf_a1_binding_v1

SCHEMA = "RealSaS.SurfA1RepresentationRescueCourt.v1"


def _save_surface(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = vertex_normals_v1(vertices, faces)
    np.savez_compressed(
        path,
        vertices_normalized=vertices.detach().cpu().float().numpy(),
        faces=faces.detach().cpu().to(dtype=torch.int32).numpy(),
        vertex_normals=normals.detach().cpu().float().numpy(),
    )


def audit(args: argparse.Namespace) -> dict:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    if git_head(repo) != source_sha:
        raise RuntimeError("SURF_A1_COURT_SOURCE_SHA_DRIFT")
    prereg = load_json(Path(args.preregistration).resolve())
    if prereg.get("schema") != "RealSaS.SurfA1FrozenHPreregistration.v1":
        raise ValueError("SURF_A1_COURT_PREREG_SCHEMA_INVALID")
    thresholds = prereg["a1_representation_rescue_court"]
    binding = load_surf_a1_binding_v1(Path(args.binding))
    training = load_json(Path(args.training_result).resolve())
    if training.get("status") != "COMPLETED_SURF_A1_2400_STEPS__AWAITING_INDEPENDENT_COURT":
        raise RuntimeError("SURF_A1_COURT_TRAINING_NOT_COMPLETE")
    checkpoint_path = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint_path) != training.get("final_checkpoint_sha256"):
        raise ValueError("SURF_A1_COURT_CHECKPOINT_SHA_DRIFT")

    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    rows = list(manifest.get("subjects") or ())
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4" or tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("SURF_A1_COURT_MANIFEST_INVALID")
    if sha256_file(manifest_path) != binding["prepared"]["manifest"]["sha256"]:
        raise ValueError("SURF_A1_COURT_MANIFEST_SHA_DRIFT")
    if sha256_file(Path(args.stage01_bundle)) != binding["stage01"]["sha256"]:
        raise ValueError("SURF_A1_COURT_STAGE01_SHA_DRIFT")

    device = torch.device("cuda")
    v8 = load_exact_v8_model(Path(args.v8_checkpoint), device)
    scene_hash = module_state_sha256_v1(v8.scene_core)
    if scene_hash != training.get("v8_scene_core_sha256"):
        raise RuntimeError("SURF_A1_COURT_V8_SCENE_HASH_DRIFT")
    extractor = PinnedFlexiCubesV1(Path(args.flexicubes_root), device=device, weight_scale=v8.config.flexicubes_weight_scale)
    decoder = FrozenHSurfaceRescueV1().to(device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "RealSaS.SurfA1FrozenHCheckpoint.v1" or int(checkpoint.get("completed_steps", -1)) != 2400:
        raise ValueError("SURF_A1_COURT_CHECKPOINT_SCHEMA_OR_STEP_DRIFT")
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    decoder.eval()
    cameras = load_camera_bundles_v1(Path(args.stage01_bundle), binding)
    legacy = {row["subject_id"]: row["legacy_subject_id"] for row in binding["authored_teachers"]["subjects"]}
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subject_results = []
    with zipfile.ZipFile(Path(args.stage01_bundle).resolve(), "r") as archive:
        names = archive.namelist()
        for subject_index, row in enumerate(rows):
            subject_id = str(row["subject_id"])
            print(f"[SURF-A1 COURT] {subject_index+1}/8 {subject_id}: materialize R48 + independent courts", flush=True)
            shard_path = Path(args.prepared_root).resolve() / str(row["shard_file"])
            source = load_subject_source(shard_path, str(row["shard_sha256"]), device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                planes = v8.encode_scene(source["image_tokens"], source["token_xy"], source["camera_features"]).float().detach()
            with torch.no_grad():
                surface = extract_rescued_surface_v1(v8, decoder, planes, extractor, resolution=48, cube_chunk=int(args.cube_chunk), training=False)
            subject_root = output_root / "subjects" / subject_id
            candidate_path = subject_root / "SURF_A1_R48.npz"
            _save_surface(candidate_path, surface["vertices_iris"], surface["faces"])
            topology_teacher = (row.get("metadata") or {}).get("topology_teacher") or {}
            geometry = measure_surface_against_shard(
                subject_id=subject_id,
                stage12_path=candidate_path,
                prepared_shard_path=shard_path,
                teacher_component_count=int(topology_teacher["component_count"]) if topology_teacher.get("component_count") is not None else None,
                seed=26101950 + subject_index,
            )

            camera_bundle = cameras[subject_id]
            vertices = surface["vertices_iris"].detach().cpu().double().numpy()
            faces = surface["faces"].detach().cpu().numpy().astype(np.int64)
            center = np.asarray(camera_bundle["common_center_xyz"], dtype=np.float64)
            half = float(camera_bundle["common_half_extent"])
            vertices_world = center[None] + half * vertices
            suffix = f"observations/{legacy[subject_id]}/camera_bundle.json"
            prefixes = [name[: -len("camera_bundle.json")] for name in names if name.endswith(suffix)]
            if len(prefixes) != 1:
                raise ValueError("SURF_A1_COURT_STAGE01_PREFIX_DRIFT:" + subject_id)
            prefix = prefixes[0]
            view_rows = []
            for camera in camera_bundle["cameras"]:
                view_index = int(camera["view_index"])
                resolution = int(camera["resolution"])
                mask_bytes = archive.read(prefix + f"V{view_index}.mask.bin")
                truth = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(resolution, resolution) > 0
                predicted = project_mask(vertices_world, faces, camera)
                view_rows.append({"view_index": view_index, "mask_sha256": sha256_bytes(mask_bytes), **_mask_metrics(predicted, truth)})
            silhouette = {
                "mean_iou": float(np.mean([v["iou"] for v in view_rows])),
                "minimum_iou": float(min(v["iou"] for v in view_rows)),
                "maximum_mask_gap_px": float(max(v["symmetric_maximum_mask_gap_px"] for v in view_rows)),
                "views": view_rows,
            }
            subject_results.append({
                "subject_id": subject_id,
                "candidate_file": str(candidate_path.relative_to(output_root)),
                "candidate_sha256": sha256_file(candidate_path),
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
                "geometry": geometry,
                "silhouette": silhouette,
            })
            del source, planes, surface

    if module_state_sha256_v1(v8.scene_core) != scene_hash:
        raise RuntimeError("SURF_A1_COURT_V8_SCENE_MUTATED")
    mean_chamfer = float(np.mean([r["geometry"]["surface_distance"]["symmetric_chamfer_distance"] for r in subject_results]))
    max_teacher_p95 = float(max(r["geometry"]["surface_distance"]["teacher_to_candidate"]["p95"] for r in subject_results))
    mean_subject_iou = float(np.mean([r["silhouette"]["mean_iou"] for r in subject_results]))
    min_subject_iou = float(min(r["silhouette"]["mean_iou"] for r in subject_results))
    min_view_iou = float(min(v["iou"] for r in subject_results for v in r["silhouette"]["views"]))
    max_gap = float(max(r["silhouette"]["maximum_mask_gap_px"] for r in subject_results))
    min_normal = float(min(r["geometry"]["normal_alignment"]["absolute_dot_mean"] for r in subject_results))
    max_nonmanifold = int(max(r["geometry"]["candidate_topology"]["nonmanifold_edge_count"] for r in subject_results))
    max_degenerate = int(max(r["geometry"]["candidate_topology"]["degenerate_face_count"] for r in subject_results))
    gates = {
        "mean_symmetric_chamfer": mean_chamfer <= float(thresholds["maximum_mean_symmetric_chamfer_distance_iris"]),
        "per_subject_teacher_to_candidate_p95": max_teacher_p95 <= float(thresholds["maximum_per_subject_teacher_to_candidate_p95_distance_iris"]),
        "global_mean_subject_mask_iou": mean_subject_iou >= float(thresholds["minimum_global_mean_subject_mask_iou"]),
        "minimum_subject_mean_mask_iou": min_subject_iou >= float(thresholds["minimum_subject_mean_mask_iou"]),
        "minimum_view_mask_iou": min_view_iou >= float(thresholds["minimum_view_mask_iou"]),
        "maximum_global_mask_gap_px": max_gap <= float(thresholds["maximum_global_mask_gap_px"]),
        "minimum_subject_mean_absolute_normal_alignment": min_normal >= float(thresholds["minimum_mean_absolute_normal_alignment"]),
        "nonmanifold_edges": max_nonmanifold <= int(thresholds["maximum_nonmanifold_edges_per_subject"]),
        "zero_area_faces": max_degenerate <= int(thresholds["maximum_zero_area_faces_per_subject"]),
    }
    passed = all(gates.values()) and len(subject_results) == 8
    result = {
        "schema": SCHEMA,
        "status": "SCIENTIFIC_PASS_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE" if passed else "SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE",
        "execution_source_sha": source_sha,
        "training_checkpoint_sha256": sha256_file(checkpoint_path),
        "v8_scene_core_sha256": scene_hash,
        "optimizer_steps_executed": 0,
        "court_resolution": 48,
        "subjects": subject_results,
        "global": {
            "mean_symmetric_chamfer_distance_iris": mean_chamfer,
            "maximum_subject_teacher_to_candidate_p95_distance_iris": max_teacher_p95,
            "mean_subject_mask_iou": mean_subject_iou,
            "minimum_subject_mask_iou": min_subject_iou,
            "minimum_view_mask_iou": min_view_iou,
            "maximum_mask_gap_px": max_gap,
            "minimum_subject_mean_absolute_normal_alignment": min_normal,
            "maximum_nonmanifold_edges_per_subject": max_nonmanifold,
            "maximum_degenerate_faces_per_subject": max_degenerate,
        },
        "gates": gates,
        "component_count_role": "MEASUREMENT_ONLY_FOR_A1",
        "teacher_truth_in_inference_forward": False,
        "tessa_invoked": False,
        "compiler_qualified": False,
        "scientific_pass_claimed": passed,
        "next_action": "DESIGN_SURF_A2_HIGH_RESOLUTION_DIRECT_A_ARM" if passed else "OPEN_DEDICATED_H_REPRESENTATION_INVESTIGATION__DO_NOT_TRAIN_TESSA",
    }
    output_path = output_root / "SURF_A1_REPRESENTATION_RESCUE_COURT.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SURF_A1_COURT", result["status"], json.dumps(result["global"], sort_keys=True), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json")
    p.add_argument("--preregistration", default="canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json")
    p.add_argument("--training-result", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--flexicubes-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cube-chunk", type=int, default=2048)
    audit(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
