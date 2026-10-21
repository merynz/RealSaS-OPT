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
from models.iris.vnext.scene_core_v1 import LocalEvidenceFieldV1
from tools.iris_vnext.h_geo_p1_geometry_v1 import (
    SUBJECT_IDS,
    recover_prepared_normalization_v1,
    triangle_geometry_v1,
)
from tools.iris_vnext.h_geo_p1_runtime_v1 import load_dataset_split_v1
from tools.iris_vnext.h_geo_p2_runtime_v1 import verify_p1_dataset_v1
from tools.iris_vnext.h_geo_p3_efreq_v1 import cameras_from_features_v1, face_vertex_adjacency_v1
from tools.iris_vnext.h_geo_p3_runtime_v1 import create_explicit_geometry_carrier_v1
from tools.iris_vnext.h_geo_p3_scale_xfer_v1 import (
    SCALE_XFER_EPS_IRIS,
    _sample_projected_tokens_v1,
    classify_scale_transfer_v1,
    deterministic_unit_direction_v2,
    nearest_nonadjacent_candidates_v1,
    summarize_stage_delta_v1,
)
from tools.iris_vnext.h_rep_probe_authority_v1 import (
    load_exact_v8_model,
    load_json,
    load_subject_source,
    require_a100_runtime_v1,
    require_exact_clean_source_v1,
    sha256_file,
)

SCHEMA = "RealSaS.HGeoP3ScaleTransferAudit.v1"


def _median(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.median(vals)) if vals else float("nan")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _load_binding(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3ScaleTransferInputBinding.v1":
        raise ValueError("H_GEO_P3_SCALE_XFER_BINDING_SCHEMA_INVALID")
    return value


def _load_prereg(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3ScaleTransferPreregistration.v1":
        raise ValueError("H_GEO_P3_SCALE_XFER_PREREG_SCHEMA_INVALID")
    return value


def _read_efreq_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    out = {s: [] for s in SUBJECT_IDS}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subject = str(row.get("subject_id", ""))
            group = str(row.get("group", ""))
            if subject not in out or group not in {"tail", "control"}:
                raise ValueError("H_GEO_P3_SCALE_XFER_QUERY_ROW_INVALID")
            out[subject].append(row)
    for subject in SUBJECT_IDS:
        if not out[subject]:
            raise ValueError("H_GEO_P3_SCALE_XFER_QUERY_SUBJECT_EMPTY:" + subject)
        seen = set()
        for row in out[subject]:
            key = (row["group"], int(row["query_index"]))
            if key in seen:
                raise ValueError("H_GEO_P3_SCALE_XFER_QUERY_DUPLICATE:" + subject)
            seen.add(key)
    return out


def _teacher_geometry(subject: str, teacher_path: Path, shard_path: Path):
    teacher = np.load(teacher_path, allow_pickle=False)
    if not {"rest_vertices_source_world", "faces"}.issubset(set(teacher.files)):
        raise ValueError("H_GEO_P3_SCALE_XFER_TEACHER_FIELDS_MISSING:" + subject)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    surface = shard.get("teacher_surface_points_iris")
    if not isinstance(surface, torch.Tensor):
        raise ValueError("H_GEO_P3_SCALE_XFER_PREPARED_SURFACE_REPLAY_MISSING:" + subject)
    faces = np.asarray(teacher["faces"], dtype=np.int64)
    norm = recover_prepared_normalization_v1(
        subject_id=subject,
        rest_vertices_source_world=np.asarray(teacher["rest_vertices_source_world"], dtype=np.float64),
        faces=faces,
        stored_surface_points_iris=surface.numpy(),
    )
    vertices = norm.apply(np.asarray(teacher["rest_vertices_source_world"], dtype=np.float64))
    triangles, _, normals = triangle_geometry_v1(vertices, faces)
    return faces, triangles, normals, face_vertex_adjacency_v1(faces), float(norm.maximum_replay_error_iris)


def _sample_stages(p2, p3, p2_planes, base64, bilinear128, refined128, points: torch.Tensor) -> dict[str, torch.Tensor]:
    q = points[None]
    with torch.inference_mode():
        p2_h = p2.scene_core.query(p2_planes, q)["H"].squeeze(0).float().cpu()
        s64 = LocalEvidenceFieldV1.sample_triplanes(base64, q).squeeze(0).float().cpu()
        sb = LocalEvidenceFieldV1.sample_triplanes(bilinear128, q).squeeze(0).float().cpu()
        sr = LocalEvidenceFieldV1.sample_triplanes(refined128, q).squeeze(0).float().cpu()
        field_h = p3.geometry_field(refined128, q)["H"].squeeze(0).float().cpu()
        pos = p3.position_projection(continuous_xyz_features_v1(q)).squeeze(0).float().cpu()
        h_geo = field_h + pos
    return {
        "P2_H_BASELINE": p2_h,
        "P3_BASE64_SAMPLED_PLANES": s64,
        "P3_BILINEAR128_SAMPLED_PLANES": sb,
        "P3_REFINED128_SAMPLED_PLANES": sr,
        "P3_GEOMETRY_FIELD_H": field_h,
        "P3_POSITION_RESIDUAL": pos,
        "P3_H_GEO": h_geo,
    }


def _token_pair_rows(image_tokens: torch.Tensor, cameras, nearest: np.ndarray, competitor: np.ndarray) -> list[dict[str, float]]:
    n = len(nearest)
    points = np.concatenate((nearest, competitor), axis=0)
    feat, coords, in_frame, cell_id = _sample_projected_tokens_v1(image_tokens, cameras, points)
    rows = []
    for i in range(n):
        valid = in_frame[:, i] & in_frame[:, n + i]
        count = int(valid.sum())
        if count == 0:
            rows.append({
                "both_in_frame_view_count": 0.0,
                "distinct_token_cell_view_count": 0.0,
                "same_token_cell_fraction": float("nan"),
                "median_token_cosine_distance": float("nan"),
                "maximum_token_cosine_distance": float("nan"),
                "median_projected_separation_token_cells": float("nan"),
            })
            continue
        f0 = feat[i, valid]
        f1 = feat[n + i, valid]
        cosine_distance = (1.0 - F.cosine_similarity(f0, f1, dim=-1).clamp(-1.0, 1.0)).numpy()
        same = cell_id[valid, i] == cell_id[valid, n + i]
        sep = np.linalg.norm((coords[valid, i] - coords[valid, n + i]) * (37.0 / 2.0), axis=-1)
        rows.append({
            "both_in_frame_view_count": float(count),
            "distinct_token_cell_view_count": float(np.sum(~same)),
            "same_token_cell_fraction": float(np.mean(same)),
            "median_token_cosine_distance": float(np.median(cosine_distance)),
            "maximum_token_cosine_distance": float(np.max(cosine_distance)),
            "median_projected_separation_token_cells": float(np.median(sep)),
        })
    return rows


def _group_summary(rows: list[dict[str, Any]], stage_order: list[str]) -> dict[str, Any]:
    out = {
        "count": len(rows),
        "median_aux_endpoint_error_iris": _median(rows, "aux_endpoint_error_iris"),
        "median_competing_surface_margin_iris": _median(rows, "competing_surface_margin_iris"),
        "median_competing_normal_angle_deg": _median(rows, "competing_normal_angle_deg"),
        "fraction_second_sheet_within_1_over_256": _mean(rows, "second_sheet_within_1_over_256"),
        "fraction_second_sheet_within_1_over_128": _mean(rows, "second_sheet_within_1_over_128"),
        "fraction_second_sheet_within_1_over_64": _mean(rows, "second_sheet_within_1_over_64"),
        "median_both_in_frame_view_count": _median(rows, "both_in_frame_view_count"),
        "median_distinct_token_cell_view_count": _median(rows, "distinct_token_cell_view_count"),
        "median_same_token_cell_fraction": _median(rows, "same_token_cell_fraction"),
        "median_median_token_cosine_distance": _median(rows, "median_token_cosine_distance"),
        "median_maximum_token_cosine_distance": _median(rows, "maximum_token_cosine_distance"),
        "median_projected_separation_token_cells": _median(rows, "median_projected_separation_token_cells"),
    }
    for eps in SCALE_XFER_EPS_IRIS:
        tag = f"eps_{eps:.8f}"
        for stage in stage_order:
            out[f"median_stage_delta_{stage}_{tag}"] = _median(rows, f"stage_delta_{stage}_{tag}")
    return out


def audit(args) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    require_exact_clean_source_v1(repo, source_sha)
    binding_path = Path(args.binding).resolve()
    prereg_path = Path(args.preregistration).resolve()
    binding = _load_binding(binding_path)
    prereg = _load_prereg(prereg_path)
    parent = load_json(repo / binding["parent_input_binding"])
    if parent.get("schema") != "RealSaS.HGeoP3EFreqInputBinding.v1":
        raise RuntimeError("H_GEO_P3_SCALE_XFER_PARENT_BINDING_INVALID")
    gpu = require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"]))
    device = torch.device("cuda")

    efreq_audit = Path(args.efreq_audit).resolve()
    efreq_csv = Path(args.efreq_query_csv).resolve()
    efreq_completion = Path(args.efreq_completion).resolve()
    for path, spec, code in (
        (efreq_audit, binding["efreq_outputs"]["audit"], "EFREQ_AUDIT"),
        (efreq_csv, binding["efreq_outputs"]["query_csv"], "EFREQ_QUERY_CSV"),
        (efreq_completion, binding["efreq_outputs"]["completion"], "EFREQ_COMPLETION"),
    ):
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError("H_GEO_P3_SCALE_XFER_AUTHORITY_DRIFT:" + code)
    if load_json(efreq_audit).get("classification") != "UNRESOLVED" or load_json(efreq_completion).get("classification") != "UNRESOLVED":
        raise RuntimeError("H_GEO_P3_SCALE_XFER_TRIGGER_NOT_UNRESOLVED")

    p3_checkpoint = Path(args.p3_checkpoint).resolve()
    p2_checkpoint = Path(args.p2_checkpoint).resolve()
    v8_checkpoint = Path(args.v8_checkpoint).resolve()
    dataset_receipt = Path(args.dataset_receipt).resolve()
    teacher_receipt = Path(args.teacher_batch_receipt).resolve()
    for path, expected, code in (
        (p3_checkpoint, parent["p3"]["checkpoint"]["sha256"], "P3_CHECKPOINT"),
        (p2_checkpoint, parent["p2_comparison"]["checkpoint"]["sha256"], "P2_CHECKPOINT"),
        (v8_checkpoint, parent["v8"]["checkpoint"]["sha256"], "V8_CHECKPOINT"),
        (dataset_receipt, parent["exact_triangle_dataset"]["receipt"]["sha256"], "DATASET_RECEIPT"),
        (teacher_receipt, parent["teachers"]["batch_receipt"]["sha256"], "TEACHER_RECEIPT"),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError("H_GEO_P3_SCALE_XFER_AUTHORITY_DRIFT:" + code)

    prepared_manifest = Path(args.prepared_manifest).resolve()
    if sha256_file(prepared_manifest) != parent["prepared"]["manifest"]["sha256"]:
        raise RuntimeError("H_GEO_P3_SCALE_XFER_PREPARED_MANIFEST_DRIFT")
    manifest = load_json(prepared_manifest)
    manifest_rows = {str(r["subject_id"]): r for r in manifest["subjects"]}
    expected_prepared = {str(r["subject_id"]): r for r in parent["prepared"]["subjects"]}
    teacher_rows = {str(r["subject_id"]): r for r in parent["teachers"]["subjects"]}
    _, dataset_rows = verify_p1_dataset_v1(dataset_root=Path(args.dataset_root), dataset_receipt=dataset_receipt, binding=parent)
    query_rows = _read_efreq_rows(efreq_csv)

    v8 = load_exact_v8_model(v8_checkpoint, device)
    p2 = load_exact_v8_model(v8_checkpoint, device)
    p2_payload = torch.load(p2_checkpoint, map_location="cpu", weights_only=True)
    if p2_payload.get("schema") != "RealSaS.HGeoP2Checkpoint.v1":
        raise RuntimeError("H_GEO_P3_SCALE_XFER_P2_SCHEMA_INVALID")
    p2.scene_core.scene.load_state_dict(p2_payload["scene_state"], strict=True)
    p2.scene_core.field.load_state_dict(p2_payload["field_state"], strict=True)
    p2.eval()

    p3 = create_explicit_geometry_carrier_v1(v8, device)
    p3_payload = torch.load(p3_checkpoint, map_location="cpu", weights_only=True)
    if p3_payload.get("schema") != "RealSaS.HGeoP3Checkpoint.v1" or int(p3_payload.get("completed_steps", -1)) != 4800:
        raise RuntimeError("H_GEO_P3_SCALE_XFER_P3_SCHEMA_INVALID")
    p3.load_state_dict(p3_payload["carrier_state"], strict=True)
    p3.eval(); v8.eval()
    for model in (v8, p2, p3):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    stage_order = list(prereg["stage_order"])
    all_rows: list[dict[str, Any]] = []
    subject_summaries = []
    maximum_replay_error = 0.0

    for subject_index, subject in enumerate(SUBJECT_IDS):
        prow = manifest_rows[subject]; expected = expected_prepared[subject]
        if prow["shard_file"] != expected["file"] or prow["shard_sha256"] != expected["sha256"]:
            raise RuntimeError("H_GEO_P3_SCALE_XFER_PREPARED_BINDING_DRIFT:" + subject)
        shard_path = (Path(args.prepared_root) / expected["file"]).resolve()
        source = load_subject_source(shard_path, expected["sha256"], device)
        dataset_path = (Path(args.dataset_root) / dataset_rows[subject]["payload_file"]).resolve()
        arrays = load_dataset_split_v1(dataset_path, "holdout")
        if "closest_face_index" not in arrays:
            with np.load(dataset_path, allow_pickle=False) as raw:
                arrays["closest_face_index"] = torch.from_numpy(np.asarray(raw["holdout_closest_face_index"], dtype=np.int64))

        teacher_path = (Path(args.teacher_root) / teacher_rows[subject]["file"]).resolve()
        if sha256_file(teacher_path) != teacher_rows[subject]["sha256"]:
            raise RuntimeError("H_GEO_P3_SCALE_XFER_TEACHER_SHA_DRIFT:" + subject)
        _, triangles, face_normals, adjacency, replay_error = _teacher_geometry(subject, teacher_path, shard_path)
        maximum_replay_error = max(maximum_replay_error, replay_error)
        cameras = cameras_from_features_v1(source["camera_features"])

        rows_in = query_rows[subject]
        max_rows = int(prereg["query_authority"]["maximum_rows_per_group_per_subject"])
        if any(sum(1 for r in rows_in if r["group"] == g) > max_rows for g in ("tail", "control")):
            raise RuntimeError("H_GEO_P3_SCALE_XFER_QUERY_COUNT_EXCEEDS_PREREG:" + subject)
        indices = np.asarray([int(r["query_index"]) for r in rows_in], dtype=np.int64)
        q0_cpu = arrays["xyz_iris"][torch.from_numpy(indices)].float()
        q0 = q0_cpu.to(device)
        nearest = arrays["closest_surface_iris"][torch.from_numpy(indices)].numpy().astype(np.float64)
        distances = arrays["distance_iris"][torch.from_numpy(indices)].numpy().astype(np.float64)
        faces = arrays["closest_face_index"][torch.from_numpy(indices)].numpy().astype(np.int64)

        competitor_points = []
        competitor_meta = []
        for local_i, query_index in enumerate(indices.tolist()):
            candidates = nearest_nonadjacent_candidates_v1(
                q0_cpu[local_i].numpy(), distances[local_i], int(faces[local_i]), triangles, face_normals, adjacency,
                k=int(prereg["target_multisurface_analysis"]["nonadjacent_k"]),
            )
            if not candidates:
                candidates = [{"face_index": -1, "point_iris": nearest[local_i], "distance_iris": float("inf"), "margin_over_nearest_iris": float("inf"), "normal_angle_up_to_sign_deg": float("nan")}]
            competitor_points.append(np.asarray(candidates[0]["point_iris"], dtype=np.float64))
            competitor_meta.append(candidates)

        token_stats = _token_pair_rows(source["image_tokens"], cameras, nearest, np.stack(competitor_points, axis=0))

        with torch.inference_mode():
            memory = v8.scene_core.view_bank(source["image_tokens"], source["token_xy"], source["camera_features"]).float()
            p2_planes = p2.scene_core.scene(memory).float()
            base64 = p3.geometry_scene(memory).float()
            refined128 = p3.highres_refiner(base64).float()
            b, three, c, h, w = base64.shape
            bilinear128 = F.interpolate(base64.reshape(b * three, c, h, w), scale_factor=2.0, mode="bilinear", align_corners=False).reshape_as(refined128)

        center_stages = _sample_stages(p2, p3, p2_planes, base64, bilinear128, refined128, q0)
        stage_scale = {k: v.std(dim=0, unbiased=False).clamp_min(1e-6) for k, v in center_stages.items()}

        subject_rows: list[dict[str, Any]] = []
        for i, src_row in enumerate(rows_in):
            candidates = competitor_meta[i]
            margin = float(candidates[0]["margin_over_nearest_iris"])
            row: dict[str, Any] = {
                "subject_id": subject,
                "group": src_row["group"],
                "query_index": int(src_row["query_index"]),
                "distance_iris": float(distances[i]),
                "aux_endpoint_error_iris": float(src_row["aux_endpoint_error_iris"]),
                "competing_face_index": int(candidates[0]["face_index"]),
                "competing_surface_margin_iris": margin,
                "competing_normal_angle_deg": float(candidates[0]["normal_angle_up_to_sign_deg"]),
                "second_sheet_within_1_over_256": float(margin <= 1.0 / 256.0),
                "second_sheet_within_1_over_128": float(margin <= 1.0 / 128.0),
                "second_sheet_within_1_over_64": float(margin <= 1.0 / 64.0),
                "candidate_count_within_1_over_128": float(sum(float(x["margin_over_nearest_iris"]) <= 1.0 / 128.0 for x in candidates)),
            }
            row.update(token_stats[i])
            subject_rows.append(row)

        directions = torch.tensor(np.stack([deterministic_unit_direction_v2(subject_index, int(q)) for q in indices]), dtype=torch.float32, device=device)
        for eps in SCALE_XFER_EPS_IRIS:
            tag = f"eps_{eps:.8f}"
            plus = torch.clamp(q0 + float(eps) * directions, -1.0, 1.0)
            minus = torch.clamp(q0 - float(eps) * directions, -1.0, 1.0)
            plus_stages = _sample_stages(p2, p3, p2_planes, base64, bilinear128, refined128, plus)
            minus_stages = _sample_stages(p2, p3, p2_planes, base64, bilinear128, refined128, minus)
            for stage in stage_order:
                delta = summarize_stage_delta_v1(plus_stages[stage], minus_stages[stage], center_stages[stage], stage_scale[stage]).numpy()
                for i, row in enumerate(subject_rows):
                    row[f"stage_delta_{stage}_{tag}"] = float(delta[i])

        all_rows.extend(subject_rows)
        summary = {"subject_id": subject}
        for group in ("tail", "control"):
            summary[group] = _group_summary([r for r in subject_rows if r["group"] == group], stage_order)
        subject_summaries.append(summary)
        print("H_GEO_P3_SCALE_XFER_SUBJECT", subject, json.dumps(summary, sort_keys=True), flush=True)

    pooled = {}
    for group in ("tail", "control"):
        pooled[group] = _group_summary([r for r in all_rows if r["group"] == group], stage_order)
    signals, classification, next_action = classify_scale_transfer_v1(pooled, prereg)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_H_GEO_P3_SCALE_TRANSFER_AUDIT",
        "execution_source_sha": source_sha,
        "binding_sha256": sha256_file(binding_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "efreq_audit_sha256": sha256_file(efreq_audit),
        "efreq_query_csv_sha256": sha256_file(efreq_csv),
        "gpu": gpu,
        "optimizer_steps_executed": 0,
        "probe_optimizer_steps_executed": 0,
        "checkpoint_written": False,
        "teacher_truth_in_inference_forward": False,
        "tessa_invoked": False,
        "maximum_teacher_normalization_replay_error_iris": maximum_replay_error,
        "stage_order": stage_order,
        "spatial_scales_iris": list(SCALE_XFER_EPS_IRIS),
        "pooled": pooled,
        "subjects": subject_summaries,
        "signals": signals,
        "classification": classification,
        "next_action": next_action,
        "scientific_pass_claimed": False,
        "p4_training_authorized": False,
        "surf_a2_authorized": False,
        "tessa_authorized": False,
        "limitations": [
            "FROZEN_DINO_TOKEN_FEATURE_CONTRAST_IS_A_SPATIAL_EVIDENCE_PROXY_NOT_TRUE_OCCLUSION_VISIBILITY",
            "EFREQ_TAIL_CONTROL_MEMBERSHIP_REUSED_BYTE_EXACT_WITHOUT_RESELECTION",
            "NO_NEW_MODEL_OR_PROBE_WAS_TRAINED",
        ],
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = Path(args.query_csv).resolve(); csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_rows)
    print("H_GEO_P3_SCALE_XFER_RESULT", classification, next_action, flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/H_GEO_P3_SCALE_XFER_INPUT_BINDING_V1.json")
    p.add_argument("--preregistration", default="canonical/H_GEO_P3_SCALE_XFER_PREREGISTRATION_V1.json")
    p.add_argument("--efreq-audit", required=True)
    p.add_argument("--efreq-query-csv", required=True)
    p.add_argument("--efreq-completion", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-receipt", required=True)
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--teacher-batch-receipt", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--p2-checkpoint", required=True)
    p.add_argument("--p3-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--query-csv", required=True)
    audit(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
