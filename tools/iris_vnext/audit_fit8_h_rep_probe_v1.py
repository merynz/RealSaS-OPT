#!/usr/bin/env python3
from __future__ import annotations

"""Independent held-out H-REP-P0 representation court."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.h_rep_probe_v1 import (
    PROBE_ARMS_V1,
    HRepresentationProbeV1,
    build_probe_input_v1,
    h_rep_probe_metrics_v1,
)
from tools.iris_vnext.h_rep_probe_authority_v1 import (
    load_json,
    require_a100_runtime_v1,
    require_exact_clean_source_v1,
    sha256_file,
)
from tools.iris_vnext.h_rep_probe_runtime_v1 import (
    cache_frozen_v8_probe_features_v1,
    load_h_rep_binding_v1,
)

SCHEMA = "RealSaS.HRepP0HeldoutCourt.v1"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _max(rows: list[dict[str, float]], key: str) -> float:
    return float(max(row[key] for row in rows))


def _min(rows: list[dict[str, float]], key: str) -> float:
    return float(min(row[key] for row in rows))


def _require_finite_tree(value: Any, prefix: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_tree(child, prefix + "." + str(key))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError("H_REP_P0_COURT_NONFINITE_METRIC:" + prefix)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    require_exact_clean_source_v1(repo, source_sha)

    binding_path = Path(args.binding).resolve()
    prereg_path = Path(args.preregistration).resolve()
    dataset_receipt = Path(args.dataset_receipt).resolve()
    training_result_path = Path(args.training_result).resolve()
    binding = load_h_rep_binding_v1(binding_path)
    prereg = load_json(prereg_path)
    if prereg.get("schema") != "RealSaS.HRepP0Preregistration.v1":
        raise ValueError("H_REP_P0_COURT_PREREG_INVALID")
    runtime = prereg.get("runtime_admission") or {}
    if runtime.get("accelerator") != "NVIDIA_A100_CLASS":
        raise ValueError("H_REP_P0_COURT_ACCELERATOR_PREREG_DRIFT")
    minimum_memory_gib = float(runtime.get("minimum_cuda_memory_gib", -1))
    if minimum_memory_gib != 39.0:
        raise ValueError("H_REP_P0_COURT_MEMORY_PREREG_DRIFT")
    gpu = require_a100_runtime_v1(minimum_memory_gib)

    binding_sha = sha256_file(binding_path)
    prereg_sha = sha256_file(prereg_path)
    dataset_sha = sha256_file(dataset_receipt)
    training = load_json(training_result_path)
    if (
        training.get("status")
        != "COMPLETED_H_REP_P0_EQUAL_CAPACITY_PROBES__AWAITING_INDEPENDENT_COURT"
    ):
        raise RuntimeError("H_REP_P0_COURT_TRAINING_NOT_COMPLETE")
    expected_training_authority = {
        "execution_source_sha": source_sha,
        "binding_sha256": binding_sha,
        "preregistration_sha256": prereg_sha,
        "dataset_receipt_sha256": dataset_sha,
    }
    for key, value in expected_training_authority.items():
        if training.get(key) != value:
            raise RuntimeError("H_REP_P0_COURT_TRAINING_AUTHORITY_DRIFT:" + key)
    if training.get("teacher_truth_in_probe_input") is not False:
        raise RuntimeError("H_REP_P0_COURT_TRAINING_INPUT_CONTAMINATION")
    if training.get("mesh_extraction_invoked") is not False or training.get("tessa_invoked") is not False:
        raise RuntimeError("H_REP_P0_COURT_TRAINING_SCOPE_CONTAMINATION")

    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != training.get("checkpoint_sha256"):
        raise ValueError("H_REP_P0_COURT_CHECKPOINT_SHA_DRIFT")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "RealSaS.HRepP0Checkpoint.v1"
        or int(payload.get("completed_steps", -1)) != 1200
    ):
        raise ValueError("H_REP_P0_COURT_CHECKPOINT_INVALID")
    for key, value in expected_training_authority.items():
        if payload.get(key) != value:
            raise RuntimeError("H_REP_P0_COURT_CHECKPOINT_AUTHORITY_DRIFT:" + key)
    if payload.get("v8_scene_core_sha256") != training.get("v8_scene_core_sha256"):
        raise RuntimeError("H_REP_P0_COURT_CHECKPOINT_SCENE_HASH_DRIFT")

    parameter_counts = training.get("parameter_counts") or {}
    if set(parameter_counts) != set(PROBE_ARMS_V1) or len(set(parameter_counts.values())) != 1:
        raise RuntimeError("H_REP_P0_COURT_PARAMETER_COUNT_DRIFT")
    if payload.get("parameter_counts") != parameter_counts:
        raise RuntimeError("H_REP_P0_COURT_CHECKPOINT_PARAMETER_COUNT_DRIFT")

    device = torch.device("cuda")
    cache, scene_hash = cache_frozen_v8_probe_features_v1(
        v8_checkpoint=Path(args.v8_checkpoint),
        prepared_root=Path(args.prepared_root),
        prepared_manifest=Path(args.prepared_manifest),
        dataset_root=Path(args.dataset_root),
        dataset_receipt=dataset_receipt,
        binding=binding,
        binding_sha256=binding_sha,
        split="holdout",
        device=device,
        query_chunk=int(args.query_chunk),
    )
    if scene_hash != training.get("v8_scene_core_sha256"):
        raise RuntimeError("H_REP_P0_COURT_SCENE_HASH_DRIFT")

    models = {arm: HRepresentationProbeV1(arm).to(device).eval() for arm in PROBE_ARMS_V1}
    for arm in PROBE_ARMS_V1:
        if models[arm].trainable_parameter_count() != int(parameter_counts[arm]):
            raise RuntimeError("H_REP_P0_COURT_MODEL_PARAMETER_COUNT_DRIFT:" + arm)
        models[arm].load_state_dict(payload["models"][arm], strict=True)

    subject_results = []
    with torch.no_grad():
        for subject_id in SUBJECT_IDS:
            item = cache[subject_id]
            n = len(item["xyz_iris"])
            arm_metrics = {}
            for arm in PROBE_ARMS_V1:
                chunks = []
                for start in range(0, n, int(args.eval_chunk)):
                    sl = slice(start, min(n, start + int(args.eval_chunk)))
                    h = item["H"][sl].to(device)
                    xyz = item["xyz_iris"][sl].to(device)
                    field = item["raw_field"][sl].to(device)
                    shuffled = (
                        item["H"][item["shuffle_index"][sl]].to(device)
                        if arm == "SHUFFLED_H_XYZ"
                        else None
                    )
                    x = build_probe_input_v1(
                        arm,
                        h=h,
                        xyz_iris=xyz,
                        raw_field=field,
                        shuffled_h=shuffled,
                    )
                    chunks.append(models[arm](x))
                prediction = {
                    key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                    for key in chunks[0]
                }
                arm_metrics[arm] = h_rep_probe_metrics_v1(
                    prediction,
                    xyz_iris=item["xyz_iris"].to(device),
                    nearest_reference_iris=item["nearest_reference_iris"].to(device),
                    distance_iris=item["distance_iris"].to(device),
                    normal_unit=item["normal_unit"].to(device),
                    normal_valid=item["normal_valid"].to(device),
                    near_label=item["near_label"].to(device),
                )
                _require_finite_tree(arm_metrics[arm], subject_id + "." + arm)
            subject_results.append({"subject_id": subject_id, "arms": arm_metrics})
            print(
                "H_REP_P0_COURT_SUBJECT",
                subject_id,
                json.dumps(arm_metrics, sort_keys=True),
                flush=True,
            )

    aggregate = {}
    for arm in PROBE_ARMS_V1:
        rows = [subject["arms"][arm] for subject in subject_results]
        aggregate[arm] = {
            "mean_distance_mae_iris": _mean(rows, "distance_mae_iris"),
            "maximum_subject_distance_mae_iris": _max(rows, "distance_mae_iris"),
            "mean_distance_abs_error_p95_iris": _mean(rows, "distance_abs_error_p95_iris"),
            "maximum_subject_endpoint_error_p95_iris": _max(rows, "endpoint_error_p95_iris"),
            "minimum_subject_normal_absolute_dot_mean": _min(rows, "normal_absolute_dot_mean"),
            "minimum_subject_near_balanced_accuracy": _min(rows, "near_balanced_accuracy"),
            "mean_near_balanced_accuracy": _mean(rows, "near_balanced_accuracy"),
        }
    _require_finite_tree(aggregate, "aggregate")

    thresholds = prereg["heldout_court"]
    h = aggregate["H_XYZ"]
    raw = aggregate["RAW_FIELD_XYZ"]
    xyz = aggregate["XYZ_ONLY"]
    shuffled = aggregate["SHUFFLED_H_XYZ"]

    def reduction(value: float, baseline: float) -> float:
        return 1.0 - value / max(baseline, 1.0e-12)

    h_absolute = {
        "mean_distance_mae": h["mean_distance_mae_iris"] <= float(thresholds["maximum_h_mean_distance_mae_iris"]),
        "max_subject_endpoint_p95": h["maximum_subject_endpoint_error_p95_iris"] <= float(thresholds["maximum_h_subject_endpoint_p95_iris"]),
        "min_subject_normal": h["minimum_subject_normal_absolute_dot_mean"] >= float(thresholds["minimum_h_subject_normal_alignment"]),
        "min_subject_near_ba": h["minimum_subject_near_balanced_accuracy"] >= float(thresholds["minimum_h_subject_near_balanced_accuracy"]),
    }
    h_alignment = {
        "distance_reduction_vs_xyz": reduction(h["mean_distance_mae_iris"], xyz["mean_distance_mae_iris"]) >= float(thresholds["minimum_h_distance_mae_reduction_vs_xyz"]),
        "distance_reduction_vs_shuffled": reduction(h["mean_distance_mae_iris"], shuffled["mean_distance_mae_iris"]) >= float(thresholds["minimum_h_distance_mae_reduction_vs_shuffled"]),
        "near_ba_gain_vs_xyz": h["mean_near_balanced_accuracy"] - xyz["mean_near_balanced_accuracy"] >= float(thresholds["minimum_h_near_ba_gain_vs_xyz"]),
        "near_ba_gain_vs_shuffled": h["mean_near_balanced_accuracy"] - shuffled["mean_near_balanced_accuracy"] >= float(thresholds["minimum_h_near_ba_gain_vs_shuffled"]),
    }
    raw_absolute = {
        "mean_distance_mae": raw["mean_distance_mae_iris"] <= float(thresholds["maximum_raw_mean_distance_mae_iris"]),
        "max_subject_endpoint_p95": raw["maximum_subject_endpoint_error_p95_iris"] <= float(thresholds["maximum_raw_subject_endpoint_p95_iris"]),
        "min_subject_normal": raw["minimum_subject_normal_absolute_dot_mean"] >= float(thresholds["minimum_raw_subject_normal_alignment"]),
        "min_subject_near_ba": raw["minimum_subject_near_balanced_accuracy"] >= float(thresholds["minimum_raw_subject_near_balanced_accuracy"]),
        "distance_reduction_vs_xyz": reduction(raw["mean_distance_mae_iris"], xyz["mean_distance_mae_iris"]) >= float(thresholds["minimum_raw_distance_mae_reduction_vs_xyz"]),
        "near_ba_gain_vs_xyz": raw["mean_near_balanced_accuracy"] - xyz["mean_near_balanced_accuracy"] >= float(thresholds["minimum_raw_near_ba_gain_vs_xyz"]),
    }

    h_pass = all(h_absolute.values()) and all(h_alignment.values())
    raw_pass = all(raw_absolute.values())
    carrier_decodable = h_pass or raw_pass
    status = (
        "SCIENTIFIC_PASS_H_REP_P0_LOCAL_GEOMETRY_DECODABLE"
        if carrier_decodable
        else "SCIENTIFIC_FAIL_H_REP_P0_LOCAL_GEOMETRY_DECODABILITY"
    )
    next_action = (
        "DESIGN_EXTRACTION_TRANSITION_ALTERNATIVE__DO_NOT_RETRAIN_H_YET__DO_NOT_TRAIN_TESSA"
        if carrier_decodable
        else "PREREGISTER_CARRIER_REPRESENTATION_RETRAINING_EXPERIMENT__DO_NOT_TRAIN_TESSA"
    )
    derived = {
        "h_distance_mae_reduction_vs_xyz": reduction(h["mean_distance_mae_iris"], xyz["mean_distance_mae_iris"]),
        "h_distance_mae_reduction_vs_shuffled": reduction(h["mean_distance_mae_iris"], shuffled["mean_distance_mae_iris"]),
        "h_near_ba_gain_vs_xyz": h["mean_near_balanced_accuracy"] - xyz["mean_near_balanced_accuracy"],
        "h_near_ba_gain_vs_shuffled": h["mean_near_balanced_accuracy"] - shuffled["mean_near_balanced_accuracy"],
        "raw_distance_mae_reduction_vs_xyz": reduction(raw["mean_distance_mae_iris"], xyz["mean_distance_mae_iris"]),
        "raw_near_ba_gain_vs_xyz": raw["mean_near_balanced_accuracy"] - xyz["mean_near_balanced_accuracy"],
    }
    _require_finite_tree(derived, "derived")

    result = {
        "schema": SCHEMA,
        "status": status,
        "execution_source_sha": source_sha,
        "binding_sha256": binding_sha,
        "preregistration_sha256": prereg_sha,
        "dataset_receipt_sha256": dataset_sha,
        "training_result_sha256": sha256_file(training_result_path),
        "training_checkpoint_sha256": checkpoint_sha,
        "v8_scene_core_sha256": scene_hash,
        "gpu": gpu,
        "parameter_counts": parameter_counts,
        "subjects": subject_results,
        "aggregate": aggregate,
        "gates": {
            "h_absolute": h_absolute,
            "h_local_alignment": h_alignment,
            "raw_field_absolute_and_xyz_control": raw_absolute,
        },
        "derived_comparisons": derived,
        "h_local_geometry_decodable": h_pass,
        "raw_field_local_geometry_decodable": raw_pass,
        "carrier_local_geometry_decodable": carrier_decodable,
        "point_cloud_proxy_only": True,
        "teacher_truth_in_probe_input": False,
        "mesh_extraction_invoked": False,
        "tessa_invoked": False,
        "optimizer_steps_executed_by_court": 0,
        "scientific_pass_claimed": carrier_decodable,
        "next_action": next_action,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("H_REP_P0_COURT", status, json.dumps(aggregate, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--binding", default="canonical/H_REP_P0_INPUT_BINDING_V1.json")
    parser.add_argument("--preregistration", default="canonical/H_REP_P0_PREREGISTRATION_V1.json")
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-receipt", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--v8-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-chunk", type=int, default=8192)
    parser.add_argument("--eval-chunk", type=int, default=4096)
    audit(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
