#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.iris.vnext.explicit_geometry_carrier_v1 import continuous_xyz_features_v1
from tools.iris_vnext.h_geo_p1_geometry_v1 import (
    SUBJECT_IDS,
    ExactTriangleBVHV1,
    recover_prepared_normalization_v1,
    triangle_geometry_v1,
)
from tools.iris_vnext.h_geo_p1_runtime_v1 import create_aux_head_v1, load_dataset_split_v1
from tools.iris_vnext.h_geo_p2_runtime_v1 import verify_p1_dataset_v1
from tools.iris_vnext.h_geo_p3_efreq_v1 import (
    EPS_SCALES_IRIS,
    angular_change_degrees_v1,
    cameras_from_features_v1,
    deterministic_unit_direction_v1,
    face_vertex_adjacency_v1,
    nonadjacent_surface_stats_v1,
    radial_highband_fraction_v1,
    select_tail_and_matched_control_v1,
    triangle_view_stats_v1,
    whitened_rms_delta_v1,
)
from tools.iris_vnext.h_geo_p3_runtime_v1 import create_explicit_geometry_carrier_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import (
    load_exact_v8_model,
    load_json,
    load_subject_source,
    require_a100_runtime_v1,
    require_exact_clean_source_v1,
    sha256_file,
)

SCHEMA = "RealSaS.HGeoP3EndpointFrequencyAudit.v1"


def _median(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.median(vals)) if vals else float("nan")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _load_binding(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3EFreqInputBinding.v1":
        raise ValueError("H_GEO_P3_EFREQ_BINDING_SCHEMA_INVALID")
    return value


def _load_prereg(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3EFreqPreregistration.v1":
        raise ValueError("H_GEO_P3_EFREQ_PREREG_SCHEMA_INVALID")
    return value


def _query_h(model, planes: torch.Tensor, xyz: torch.Tensor, *, p3: bool, chunk: int = 4096) -> torch.Tensor:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(xyz), int(chunk)):
            q = xyz[start : start + int(chunk)][None]
            if p3:
                rows.append(model.query_geometry(planes, q)["H_geo"].squeeze(0).float().cpu())
            else:
                rows.append(model.scene_core.query(planes, q)["H"].squeeze(0).float().cpu())
    return torch.cat(rows, dim=0)


def _teacher_geometry(subject: str, teacher_path: Path, shard_path: Path):
    teacher = np.load(teacher_path, allow_pickle=False)
    if not {"rest_vertices_source_world", "faces"}.issubset(set(teacher.files)):
        raise ValueError("H_GEO_P3_EFREQ_TEACHER_FIELDS_MISSING:" + subject)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    surface = shard.get("teacher_surface_points_iris")
    if not isinstance(surface, torch.Tensor):
        raise ValueError("H_GEO_P3_EFREQ_PREPARED_SURFACE_REPLAY_MISSING:" + subject)
    faces = np.asarray(teacher["faces"], dtype=np.int64)
    norm = recover_prepared_normalization_v1(
        subject_id=subject,
        rest_vertices_source_world=np.asarray(teacher["rest_vertices_source_world"], dtype=np.float64),
        faces=faces,
        stored_surface_points_iris=surface.numpy(),
    )
    vertices = norm.apply(np.asarray(teacher["rest_vertices_source_world"], dtype=np.float64))
    tri, _, normals = triangle_geometry_v1(vertices, faces)
    return faces, tri, normals, ExactTriangleBVHV1(vertices, faces), face_vertex_adjacency_v1(faces), norm.maximum_replay_error_iris


def _plane_internal_stats(p3, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    with torch.inference_mode():
        base = p3.geometry_scene(memory).float()
        refined = p3.highres_refiner(base).float()
        b, three, c, h, w = base.shape
        bilinear = F.interpolate(base.reshape(b * three, c, h, w), scale_factor=2.0, mode="bilinear", align_corners=False).reshape_as(refined)
        residual = refined - bilinear
        base_rms = float(torch.sqrt(torch.mean(bilinear.square())).item())
        residual_rms = float(torch.sqrt(torch.mean(residual.square())).item())
        stats = {
            "bilinear_base_rms": base_rms,
            "refiner_residual_rms": residual_rms,
            "refiner_residual_to_base_rms_ratio": residual_rms / max(base_rms, 1e-12),
            "refiner_residual_highband_power_fraction": radial_highband_fraction_v1(residual),
        }
    return base, refined, stats


def audit(args) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    require_exact_clean_source_v1(repo, source_sha)
    binding_path = Path(args.binding).resolve()
    prereg_path = Path(args.preregistration).resolve()
    binding = _load_binding(binding_path)
    prereg = _load_prereg(prereg_path)
    gpu = require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"]))
    device = torch.device("cuda")

    p3_checkpoint = Path(args.p3_checkpoint).resolve()
    p2_checkpoint = Path(args.p2_checkpoint).resolve()
    v8_checkpoint = Path(args.v8_checkpoint).resolve()
    p3_training = Path(args.p3_training_result).resolve()
    p3_court = Path(args.p3_court).resolve()
    dataset_receipt = Path(args.dataset_receipt).resolve()
    teacher_receipt = Path(args.teacher_batch_receipt).resolve()
    for path, expected, code in (
        (p3_checkpoint, binding["p3"]["checkpoint"]["sha256"], "P3_CHECKPOINT"),
        (p2_checkpoint, binding["p2_comparison"]["checkpoint"]["sha256"], "P2_CHECKPOINT"),
        (v8_checkpoint, binding["v8"]["checkpoint"]["sha256"], "V8_CHECKPOINT"),
        (p3_training, binding["p3"]["training_result"]["sha256"], "P3_TRAINING_RESULT"),
        (p3_court, binding["p3"]["independent_court"]["sha256"], "P3_COURT"),
        (dataset_receipt, binding["exact_triangle_dataset"]["receipt"]["sha256"], "DATASET_RECEIPT"),
        (teacher_receipt, binding["teachers"]["batch_receipt"]["sha256"], "TEACHER_BATCH_RECEIPT"),
    ):
        if sha256_file(path) != str(expected):
            raise RuntimeError("H_GEO_P3_EFREQ_AUTHORITY_DRIFT:" + code)
    p3_result = load_json(p3_court)
    if p3_result.get("status") != "SCIENTIFIC_FAIL_H_GEO_P3_EXPLICIT_HIRES_CARRIER":
        raise RuntimeError("H_GEO_P3_EFREQ_TRIGGER_RESULT_INVALID")

    prepared_manifest = Path(args.prepared_manifest).resolve()
    if sha256_file(prepared_manifest) != binding["prepared"]["manifest"]["sha256"]:
        raise RuntimeError("H_GEO_P3_EFREQ_PREPARED_MANIFEST_DRIFT")
    manifest = load_json(prepared_manifest)
    manifest_rows = {str(r["subject_id"]): r for r in manifest["subjects"]}
    expected_prepared = {str(r["subject_id"]): r for r in binding["prepared"]["subjects"]}
    teacher_rows = {str(r["subject_id"]): r for r in binding["teachers"]["subjects"]}
    _, dataset_rows = verify_p1_dataset_v1(dataset_root=Path(args.dataset_root), dataset_receipt=dataset_receipt, binding=binding)

    v8 = load_exact_v8_model(v8_checkpoint, device)
    p2 = load_exact_v8_model(v8_checkpoint, device)
    p2_payload = torch.load(p2_checkpoint, map_location="cpu", weights_only=True)
    if p2_payload.get("schema") != "RealSaS.HGeoP2Checkpoint.v1":
        raise RuntimeError("H_GEO_P3_EFREQ_P2_SCHEMA_INVALID")
    p2.scene_core.scene.load_state_dict(p2_payload["scene_state"], strict=True)
    p2.scene_core.field.load_state_dict(p2_payload["field_state"], strict=True)
    p2.eval()

    p3 = create_explicit_geometry_carrier_v1(v8, device)
    aux = create_aux_head_v1(device)
    p3_payload = torch.load(p3_checkpoint, map_location="cpu", weights_only=True)
    if p3_payload.get("schema") != "RealSaS.HGeoP3Checkpoint.v1" or int(p3_payload.get("completed_steps", -1)) != 4800:
        raise RuntimeError("H_GEO_P3_EFREQ_P3_SCHEMA_INVALID")
    p3.load_state_dict(p3_payload["carrier_state"], strict=True)
    aux.load_state_dict(p3_payload["aux_state"], strict=True)
    p3.eval(); aux.eval(); v8.eval()
    for model in (v8, p2, p3, aux):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    all_rows: list[dict[str, Any]] = []
    subject_summaries: list[dict[str, Any]] = []
    plane_summaries: list[dict[str, Any]] = []
    maximum_replay_error = 0.0

    for subject_index, subject in enumerate(SUBJECT_IDS):
        prow = manifest_rows[subject]; expected = expected_prepared[subject]
        if prow["shard_file"] != expected["file"] or prow["shard_sha256"] != expected["sha256"]:
            raise RuntimeError("H_GEO_P3_EFREQ_PREPARED_BINDING_DRIFT:" + subject)
        shard_path = (Path(args.prepared_root) / expected["file"]).resolve()
        source = load_subject_source(shard_path, expected["sha256"], device)
        dataset_path = (Path(args.dataset_root) / dataset_rows[subject]["payload_file"]).resolve()
        arrays = load_dataset_split_v1(dataset_path, "holdout")
        if "closest_face_index" not in arrays:
            with np.load(dataset_path, allow_pickle=False) as raw:
                arrays["closest_face_index"] = torch.from_numpy(np.asarray(raw["holdout_closest_face_index"], dtype=np.int64))

        teacher_path = (Path(args.teacher_root) / teacher_rows[subject]["file"]).resolve()
        if sha256_file(teacher_path) != teacher_rows[subject]["sha256"]:
            raise RuntimeError("H_GEO_P3_EFREQ_TEACHER_SHA_DRIFT:" + subject)
        _, triangles, face_normals, bvh, adjacency, replay_error = _teacher_geometry(subject, teacher_path, shard_path)
        maximum_replay_error = max(maximum_replay_error, float(replay_error))
        cameras = cameras_from_features_v1(source["camera_features"])

        with torch.inference_mode():
            memory = v8.scene_core.view_bank(source["image_tokens"], source["token_xy"], source["camera_features"]).float()
            p2_planes = p2.scene_core.scene(memory).float()
            _, p3_planes, plane_stats = _plane_internal_stats(p3, memory)
        plane_stats["subject_id"] = subject
        plane_summaries.append(plane_stats)

        xyz_cpu = arrays["xyz_iris"].float()
        xyz = xyz_cpu.to(device)
        p3_h = _query_h(p3, p3_planes, xyz, p3=True, chunk=args.feature_chunk)
        p2_h = _query_h(p2, p2_planes, xyz, p3=False, chunk=args.feature_chunk)
        p3_scale = p3_h.std(dim=0, unbiased=False).clamp_min(1e-6)
        p2_scale = p2_h.std(dim=0, unbiased=False).clamp_min(1e-6)
        with torch.inference_mode():
            pred_chunks = []
            for start in range(0, len(p3_h), int(args.feature_chunk)):
                pred_chunks.append(aux(p3_h[start:start+int(args.feature_chunk)].to(device)))
            pred = {k: torch.cat([c[k].detach().cpu() for c in pred_chunks], dim=0) for k in pred_chunks[0]}
        endpoint = xyz_cpu + pred["offset_to_surface_iris"].float()
        endpoint_error = torch.linalg.vector_norm(endpoint - arrays["closest_surface_iris"].float(), dim=-1).numpy()
        tail_idx, control_idx = select_tail_and_matched_control_v1(endpoint_error, arrays["distance_iris"].numpy(), maximum_per_group=int(prereg["query_selection"]["tail_maximum_queries_per_subject"]))

        subject_rows = []
        for group, indices in (("tail", tail_idx), ("control", control_idx)):
            if not len(indices):
                continue
            idx_tensor = torch.from_numpy(indices).long()
            q0_cpu = xyz_cpu[idx_tensor]
            h3_0 = p3_h[idx_tensor]
            h2_0 = p2_h[idx_tensor]
            with torch.inference_mode():
                pos_res = p3.position_projection(continuous_xyz_features_v1(q0_cpu.to(device)[None])).squeeze(0).cpu()
            pos_rms = torch.sqrt(torch.mean(pos_res.square(), dim=-1)).numpy()

            group_rows = []
            adjacency_sets = {int(face): set(int(x) for x in adjacency[int(face)]) for face in np.unique(arrays["closest_face_index"][idx_tensor].numpy())}
            for local_i, query_index in enumerate(indices.tolist()):
                face = int(arrays["closest_face_index"][query_index].item())
                closest = arrays["closest_surface_iris"][query_index].numpy().astype(np.float64)
                offset0 = arrays["offset_to_surface_iris"][query_index].numpy().astype(np.float64)
                row = {
                    "subject_id": subject,
                    "group": group,
                    "query_index": int(query_index),
                    "distance_iris": float(arrays["distance_iris"][query_index].item()),
                    "aux_endpoint_error_iris": float(endpoint_error[query_index]),
                    "closest_face_index": face,
                    "position_residual_rms": float(pos_rms[local_i]),
                }
                row.update(nonadjacent_surface_stats_v1(closest, face, triangles, adjacency))
                row.update(triangle_view_stats_v1(triangles[face], face_normals[face], cameras))
                direction = deterministic_unit_direction_v1(subject_index, query_index)
                for eps in EPS_SCALES_IRIS:
                    tag = f"eps_{eps:.8f}"
                    flips = []; nonadj_flips = []; angles = []; target_variations = []; closest_motion = []; perturb_points = []
                    for sign in (-1.0, 1.0):
                        qp = np.clip(q0_cpu[local_i].numpy().astype(np.float64) + sign * eps * direction, -1.0, 1.0)
                        cp, fi, _ = bvh.nearest_one(qp)
                        op = cp - qp
                        flips.append(float(fi != face))
                        nonadj_flips.append(float(fi not in adjacency_sets[face]))
                        angles.append(angular_change_degrees_v1(offset0, op))
                        target_variations.append(float(np.linalg.norm(op - offset0) / eps))
                        closest_motion.append(float(np.linalg.norm(cp - closest) / eps))
                        perturb_points.append(qp)
                    row[f"face_flip_rate_{tag}"] = float(np.mean(flips))
                    row[f"nonadjacent_face_flip_rate_{tag}"] = float(np.mean(nonadj_flips))
                    row[f"offset_direction_change_deg_{tag}"] = float(np.mean(angles))
                    row[f"target_offset_variation_per_eps_{tag}"] = float(np.mean(target_variations))
                    row[f"closest_point_motion_per_eps_{tag}"] = float(np.mean(closest_motion))
                    row[f"_perturb_{tag}"] = perturb_points
                group_rows.append(row)

            for eps in EPS_SCALES_IRIS:
                tag = f"eps_{eps:.8f}"
                plus = torch.tensor(np.stack([r[f"_perturb_{tag}"][1] for r in group_rows]), dtype=torch.float32, device=device)
                minus = torch.tensor(np.stack([r[f"_perturb_{tag}"][0] for r in group_rows]), dtype=torch.float32, device=device)
                h3p = _query_h(p3, p3_planes, plus, p3=True, chunk=args.feature_chunk)
                h3m = _query_h(p3, p3_planes, minus, p3=True, chunk=args.feature_chunk)
                h2p = _query_h(p2, p2_planes, plus, p3=False, chunk=args.feature_chunk)
                h2m = _query_h(p2, p2_planes, minus, p3=False, chunk=args.feature_chunk)
                d3 = 0.5 * (whitened_rms_delta_v1(h3p, h3_0, p3_scale) + whitened_rms_delta_v1(h3m, h3_0, p3_scale))
                d2 = 0.5 * (whitened_rms_delta_v1(h2p, h2_0, p2_scale) + whitened_rms_delta_v1(h2m, h2_0, p2_scale))
                for i, row in enumerate(group_rows):
                    row[f"p3_whitened_h_delta_{tag}"] = float(d3[i].item())
                    row[f"p2_whitened_h_delta_{tag}"] = float(d2[i].item())
                    row.pop(f"_perturb_{tag}", None)
            subject_rows.extend(group_rows)
            all_rows.extend(group_rows)

        summary = {"subject_id": subject, "tail_count": int(sum(r["group"] == "tail" for r in subject_rows)), "control_count": int(sum(r["group"] == "control" for r in subject_rows))}
        for group in ("tail", "control"):
            rows = [r for r in subject_rows if r["group"] == group]
            summary[group] = {
                "median_aux_endpoint_error_iris": _median(rows, "aux_endpoint_error_iris"),
                "median_nonadjacent_surface_gap_iris": _median(rows, "nonadjacent_distance_iris"),
                "median_informative_view_count": _median(rows, "informative_view_count"),
                "median_maximum_projected_triangle_area_px2": _median(rows, "maximum_projected_triangle_area_px2"),
                "median_position_residual_rms": _median(rows, "position_residual_rms"),
            }
            for eps in EPS_SCALES_IRIS:
                tag = f"eps_{eps:.8f}"
                summary[group][f"nonadjacent_face_flip_rate_{tag}"] = _mean(rows, f"nonadjacent_face_flip_rate_{tag}")
                summary[group][f"median_offset_direction_change_deg_{tag}"] = _median(rows, f"offset_direction_change_deg_{tag}")
                summary[group][f"median_target_offset_variation_per_eps_{tag}"] = _median(rows, f"target_offset_variation_per_eps_{tag}")
                summary[group][f"median_p3_whitened_h_delta_{tag}"] = _median(rows, f"p3_whitened_h_delta_{tag}")
                summary[group][f"median_p2_whitened_h_delta_{tag}"] = _median(rows, f"p2_whitened_h_delta_{tag}")
        subject_summaries.append(summary)
        print("H_GEO_P3_EFREQ_SUBJECT", subject, json.dumps(summary, sort_keys=True), flush=True)

    pooled = {}
    for group in ("tail", "control"):
        rows = [r for r in all_rows if r["group"] == group]
        pooled[group] = {
            "count": len(rows),
            "median_aux_endpoint_error_iris": _median(rows, "aux_endpoint_error_iris"),
            "median_nonadjacent_surface_gap_iris": _median(rows, "nonadjacent_distance_iris"),
            "median_informative_view_count": _median(rows, "informative_view_count"),
            "median_maximum_projected_triangle_area_px2": _median(rows, "maximum_projected_triangle_area_px2"),
            "median_position_residual_rms": _median(rows, "position_residual_rms"),
        }
        for eps in EPS_SCALES_IRIS:
            tag = f"eps_{eps:.8f}"
            pooled[group][f"nonadjacent_face_flip_rate_{tag}"] = _mean(rows, f"nonadjacent_face_flip_rate_{tag}")
            pooled[group][f"median_offset_direction_change_deg_{tag}"] = _median(rows, f"offset_direction_change_deg_{tag}")
            pooled[group][f"median_target_offset_variation_per_eps_{tag}"] = _median(rows, f"target_offset_variation_per_eps_{tag}")
            pooled[group][f"median_p3_whitened_h_delta_{tag}"] = _median(rows, f"p3_whitened_h_delta_{tag}")
            pooled[group][f"median_p2_whitened_h_delta_{tag}"] = _median(rows, f"p2_whitened_h_delta_{tag}")

    tag = f"eps_{1.0/128.0:.8f}"
    tail = pooled["tail"]; control = pooled["control"]
    ambiguity = bool(
        tail["median_nonadjacent_surface_gap_iris"] <= 0.5 * control["median_nonadjacent_surface_gap_iris"]
        and tail[f"nonadjacent_face_flip_rate_{tag}"] > control[f"nonadjacent_face_flip_rate_{tag}"]
        and tail[f"nonadjacent_face_flip_rate_{tag}"] >= 1.5 * max(control[f"nonadjacent_face_flip_rate_{tag}"], 1e-6)
    )
    view = bool(
        tail["median_maximum_projected_triangle_area_px2"] <= 0.5 * control["median_maximum_projected_triangle_area_px2"]
        or tail["median_informative_view_count"] <= control["median_informative_view_count"] - 1.0
    )
    carrier = bool(
        tail[f"median_target_offset_variation_per_eps_{tag}"] >= 1.5 * control[f"median_target_offset_variation_per_eps_{tag}"]
        and tail[f"median_p3_whitened_h_delta_{tag}"] <= 1.10 * tail[f"median_p2_whitened_h_delta_{tag}"]
    )
    signals = {"target_ambiguity": ambiguity, "view_evidence": view, "carrier_bandwidth": carrier}
    count = sum(bool(v) for v in signals.values())
    if count >= 2:
        classification = "MIXED"
    elif ambiguity:
        classification = "TARGET_AMBIGUITY_DOMINANT"
    elif view:
        classification = "VIEW_EVIDENCE_DOMINANT"
    elif carrier:
        classification = "CARRIER_BANDWIDTH_DOMINANT"
    else:
        classification = "UNRESOLVED"
    next_key = {
        "TARGET_AMBIGUITY_DOMINANT": "TARGET_AMBIGUITY_DOMINANT_next",
        "VIEW_EVIDENCE_DOMINANT": "VIEW_EVIDENCE_DOMINANT_next",
        "CARRIER_BANDWIDTH_DOMINANT": "CARRIER_BANDWIDTH_DOMINANT_next",
        "MIXED": "MIXED_next",
        "UNRESOLVED": "UNRESOLVED_next",
    }[classification]

    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_H_GEO_P3_ENDPOINT_FREQUENCY_AUDIT",
        "execution_source_sha": source_sha,
        "binding_sha256": sha256_file(binding_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "gpu": gpu,
        "optimizer_steps_executed": 0,
        "probe_optimizer_steps_executed": 0,
        "teacher_truth_in_inference_forward": False,
        "tessa_invoked": False,
        "p3_checkpoint_sha256": sha256_file(p3_checkpoint),
        "p2_checkpoint_sha256": sha256_file(p2_checkpoint),
        "dataset_receipt_sha256": sha256_file(dataset_receipt),
        "teacher_batch_receipt_sha256": sha256_file(teacher_receipt),
        "maximum_teacher_normalization_replay_error_iris": maximum_replay_error,
        "pooled": pooled,
        "plane_internal": plane_summaries,
        "subjects": subject_summaries,
        "signals": signals,
        "classification": classification,
        "next_action": prereg["interpretation"][next_key],
        "scientific_pass_claimed": False,
        "surf_a2_authorized": False,
        "tessa_authorized": False,
        "limitations": [
            "P3_TRAINING_AUX_USED_ONLY_TO_RANK_ENDPOINT_TAIL_NOT_TO_REQUALIFY_P3",
            "VIEW_EVIDENCE_METRICS_ARE_ORTHOGRAPHIC_GEOMETRIC_COVERAGE_PROXIES_NOT_TRUE_OCCLUSION_VISIBILITY",
            "NO_NEW_MODEL_OR_PROBE_WAS_TRAINED"
        ]
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = Path(args.query_csv).resolve(); csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(all_rows)
    print("H_GEO_P3_EFREQ_RESULT", classification, result["next_action"], flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/H_GEO_P3_EFREQ_INPUT_BINDING_V1.json")
    p.add_argument("--preregistration", default="canonical/H_GEO_P3_EFREQ_PREREGISTRATION_V1.json")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-receipt", required=True)
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--teacher-batch-receipt", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--p2-checkpoint", required=True)
    p.add_argument("--p3-checkpoint", required=True)
    p.add_argument("--p3-training-result", required=True)
    p.add_argument("--p3-court", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--query-csv", required=True)
    p.add_argument("--feature-chunk", type=int, default=4096)
    audit(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
