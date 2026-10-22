#!/usr/bin/env python3
from __future__ import annotations

"""FIT8 corpus/evidence autopsy for the sealed Structural-Evidence A/B experiment.

Post-hoc mechanism diagnostic only:
1. Audit raw direct-surface presentation-query opportunity/compression statistics.
2. Audit the materialized fixed-budget B-arm presentation memory.
3. Compare subject-level descriptors to the already sealed B-vs-A teacher-forced effect.
4. Rebuild contract-equivalent presentation memories for selected subjects with
   alternative deterministic sample seeds and evaluate them teacher-forced with
   the frozen B checkpoint.

No optimizer/backward step is executed. The sealed directional result is not modified.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from tools.iris_vnext import build_fit8_direct_presentation_memory_v1 as memory_builder
from tools.iris_vnext.build_fit8_vnext_shards_v1 import _seed
from tools.iris_vnext.evaluate_structural_evidence_ab_directional_screen_v1 import _load_arm_model
from tools.iris_vnext.evaluate_structural_evidence_ab_fit8_teacher_census_v1 import _teacher_census
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v1 import _conditioning
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_B,
    load_json,
    validate_preregistration,
)
from models.tessa.v2.experiment_v2 import sha256_file

SCHEMA = "RealSaS.StructuralEvidenceFIT8CorpusEvidenceAutopsyResult.v1"
RECEIPT_SCHEMA = "RealSaS.StructuralEvidenceFIT8CorpusEvidenceAutopsyReceipt.v1"
PREREG_SCHEMA = "RealSaS.StructuralEvidenceFIT8CorpusEvidenceAutopsy.v1"
QUERY_SCHEMA = "RealSaS.DirectIRISPresentationQueriesManifest.v1"
MEMORY_SCHEMA = "RealSaS.DirectIRISPresentationMemoryManifest.v1"
SUBJECT_MEMORY_SCHEMA = "RealSaS.DirectIRISPresentationMemorySubject.v1"
EPS = 1.0e-12


def _phase(message: str) -> None:
    print("FIT8_AUTOPSY " + message, flush=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) <= EPS:
        return 0.0
    return float(np.std(arr, ddof=0) / abs(mean))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _corr(x: list[float], y: list[float]) -> dict[str, float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) != len(ya) or len(xa) < 2:
        raise ValueError("FIT8_AUTOPSY_CORRELATION_INPUT_INVALID")

    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
        if denom <= EPS:
            return 0.0
        return float(np.sum(ac * bc) / denom)

    return {
        "pearson": pearson(xa, ya),
        "spearman": pearson(_rankdata(xa), _rankdata(ya)),
        "n": int(len(xa)),
    }


def _relative_improvement(a: float, b: float) -> float:
    return (float(a) - float(b)) / max(abs(float(a)), EPS)


def _feature_index(feature_names: tuple[str, ...] | list[str], name: str) -> int:
    try:
        return list(feature_names).index(name)
    except ValueError as exc:
        raise ValueError("FIT8_AUTOPSY_FEATURE_MISSING:" + name) from exc


def _memory_payload(path: Path, expected_sha: str, subject_id: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise ValueError("FIT8_AUTOPSY_MEMORY_SHA_DRIFT:" + subject_id)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SUBJECT_MEMORY_SCHEMA or payload.get("subject_id") != subject_id:
        raise ValueError("FIT8_AUTOPSY_MEMORY_PAYLOAD_INVALID:" + subject_id)
    if tuple(payload.get("feature_names") or ()) != tuple(PRESENTATION_FEATURE_NAMES_V1):
        raise ValueError("FIT8_AUTOPSY_MEMORY_FEATURE_DRIFT:" + subject_id)
    if payload.get("teacher_geometry_in_conditioning") is not False:
        raise ValueError("FIT8_AUTOPSY_TEACHER_GEOMETRY_LEAK:" + subject_id)
    if payload.get("teacher_topology_in_conditioning") is not False:
        raise ValueError("FIT8_AUTOPSY_TEACHER_TOPOLOGY_LEAK:" + subject_id)
    if payload.get("materialized_v2_truth_in_conditioning") is not False:
        raise ValueError("FIT8_AUTOPSY_V2_TRUTH_LEAK:" + subject_id)
    return payload


def _view_memory_rows(
    payload: dict[str, Any],
    view_counts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values = payload["values"].detach().cpu().float().numpy()
    valid_mask = payload["valid_mask"].detach().cpu().numpy().astype(bool)
    kind_id = payload["kind_id"].detach().cpu().numpy().astype(np.int64)
    feature_names = tuple(payload["feature_names"])
    if len(values) != len(valid_mask) or len(values) != len(kind_id):
        raise ValueError("FIT8_AUTOPSY_MEMORY_LENGTH_DRIFT")
    cursor = 0
    rows: list[dict[str, Any]] = []
    feature_valids = (
        "front_valid",
        "global_visibility_valid",
        "source_rgba_valid",
        "alpha_boundary_valid",
        "appearance_boundary_valid",
        "source_support_valid",
        "occlusion_valid",
        "deformation_coherence_valid",
        "attachment_persistence_valid",
    )
    for view_row in view_counts:
        counts = [
            ("surface", int(view_row.get("surface", 0))),
            ("pair", int(view_row.get("pair", 0))),
            ("ray_hit", int(view_row.get("ray_hit", 0))),
        ]
        start = cursor
        cursor += sum(c for _, c in counts)
        segment = values[start:cursor]
        segment_valid = valid_mask[start:cursor]
        out: dict[str, Any] = {
            "view_index": int(view_row["view_index"]),
            "memory_token_count": int(len(segment)),
            "top_level_valid_fraction": float(np.mean(segment_valid)) if len(segment) else 0.0,
            "surface_token_count": counts[0][1],
            "pair_token_count": counts[1][1],
            "ray_hit_token_count": counts[2][1],
        }
        for name in feature_valids:
            idx = _feature_index(feature_names, name)
            out[name + "_fraction"] = float(np.mean(segment[:, idx] > 0.5)) if len(segment) else 0.0
        for name in ("alpha_boundary", "appearance_boundary", "source_support", "global_visibility"):
            idx = _feature_index(feature_names, name)
            out[name + "_mean"] = float(np.mean(segment[:, idx])) if len(segment) else 0.0
            out[name + "_positive_fraction"] = (
                float(np.mean(segment[:, idx] > 0.5)) if len(segment) else 0.0
            )
        rows.append(out)
    if cursor != len(values):
        raise ValueError(f"FIT8_AUTOPSY_MEMORY_VIEW_SEGMENT_DRIFT:{cursor}!={len(values)}")
    return rows


def _phase1(
    *,
    query_manifest: dict[str, Any],
    memory_manifest: dict[str, Any],
    memory_root: Path,
    census: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    query_subjects = {str(row["subject_id"]): row for row in query_manifest["subjects"]}
    memory_subjects = {str(row["subject_id"]): row for row in memory_manifest["subjects"]}
    census_votes = census["census_decision"]["subject_votes"]

    per_view: list[dict[str, Any]] = []
    subject_summary: dict[str, dict[str, Any]] = {}

    for subject_id in SUBJECT_IDS:
        qrow = query_subjects[subject_id]
        mrow = memory_subjects[subject_id]
        payload = _memory_payload(
            memory_root / str(mrow["file"]),
            str(mrow["sha256"]),
            subject_id,
        )
        memory_views = {int(r["view_index"]): r for r in _view_memory_rows(payload, mrow["view_counts"])}
        local_rows: list[dict[str, Any]] = []

        for view in qrow["views"]:
            vi = int(view["view_index"])
            mv = memory_views[vi]
            fg = int(view["sampled_source_foreground_rays"])
            hit_rays = int(view["direct_surface_hit_rays"])
            multi = int(view["multi_hit_rays"])
            hits = int(view["total_direct_surface_hits"])
            pairs = int(view["oriented_pair_rows"])
            pair_tokens = int(mv["pair_token_count"])
            ray_tokens = int(mv["ray_hit_token_count"])
            surface_tokens = int(mv["surface_token_count"])
            row = {
                "subject_id": subject_id,
                "view_index": vi,
                "b_relative_total_objective_improvement_vs_a": float(
                    census_votes[subject_id]["b_relative_total_objective_improvement_vs_a"]
                ),
                "sampled_source_foreground_rays": fg,
                "direct_surface_hit_rays": hit_rays,
                "multi_hit_rays": multi,
                "total_direct_surface_hits": hits,
                "oriented_pair_rows": pairs,
                "pair_sampling_fraction": pair_tokens / max(pairs, 1),
                "ray_hit_sampling_fraction": ray_tokens / max(hits, 1),
                "surface_sampling_fraction_proxy": surface_tokens / max(hit_rays, 1),
                "hit_ray_fraction": hit_rays / max(fg, 1),
                "mean_hits_per_hit_ray": hits / max(hit_rays, 1),
                "mean_oriented_pairs_per_multi_hit_ray": pairs / max(multi, 1),
            }
            row.update(mv)
            per_view.append(row)
            local_rows.append(row)

        numeric_metrics = [
            "sampled_source_foreground_rays",
            "direct_surface_hit_rays",
            "multi_hit_rays",
            "total_direct_surface_hits",
            "oriented_pair_rows",
            "pair_sampling_fraction",
            "ray_hit_sampling_fraction",
            "surface_sampling_fraction_proxy",
            "hit_ray_fraction",
            "mean_hits_per_hit_ray",
            "mean_oriented_pairs_per_multi_hit_ray",
            "front_valid_fraction",
            "global_visibility_valid_fraction",
            "source_rgba_valid_fraction",
            "alpha_boundary_valid_fraction",
            "appearance_boundary_valid_fraction",
            "source_support_valid_fraction",
            "occlusion_valid_fraction",
            "deformation_coherence_valid_fraction",
            "attachment_persistence_valid_fraction",
            "alpha_boundary_positive_fraction",
            "appearance_boundary_positive_fraction",
            "global_visibility_positive_fraction",
        ]
        summary: dict[str, Any] = {
            "subject_id": subject_id,
            "b_relative_total_objective_improvement_vs_a": float(
                census_votes[subject_id]["b_relative_total_objective_improvement_vs_a"]
            ),
            "census_vote": str(census_votes[subject_id]["vote"]),
            "memory_token_count": int(mrow["token_count"]),
        }
        for metric in numeric_metrics:
            values = [float(r[metric]) for r in local_rows]
            summary[metric + "_mean"] = _mean(values)
            summary[metric + "_cv"] = _cv(values)
        subject_summary[subject_id] = summary

    effect = [
        float(subject_summary[s]["b_relative_total_objective_improvement_vs_a"])
        for s in SUBJECT_IDS
    ]
    correlation_metrics = [
        "oriented_pair_rows_mean",
        "pair_sampling_fraction_mean",
        "ray_hit_sampling_fraction_mean",
        "surface_sampling_fraction_proxy_mean",
        "hit_ray_fraction_mean",
        "mean_hits_per_hit_ray_mean",
        "mean_oriented_pairs_per_multi_hit_ray_mean",
        "front_valid_fraction_mean",
        "global_visibility_valid_fraction_mean",
        "source_rgba_valid_fraction_mean",
        "alpha_boundary_valid_fraction_mean",
        "appearance_boundary_valid_fraction_mean",
        "source_support_valid_fraction_mean",
        "alpha_boundary_positive_fraction_mean",
        "appearance_boundary_positive_fraction_mean",
        "global_visibility_positive_fraction_mean",
    ]
    correlations = {}
    for metric in correlation_metrics:
        correlations[metric] = _corr(
            [float(subject_summary[s][metric]) for s in SUBJECT_IDS],
            effect,
        )
    return per_view, subject_summary, {
        "small_n_policy": "DESCRIPTIVE_ONLY__NO_SIGNIFICANCE_CLAIM",
        "correlations_vs_b_relative_total_objective_improvement": correlations,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_query_view(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha:
        raise ValueError("FIT8_AUTOPSY_QUERY_VIEW_SHA_DRIFT:" + path.name)
    with np.load(path, allow_pickle=False) as npz:
        return {name: np.asarray(npz[name]) for name in npz.files}


def _payload_from_queries(
    *,
    subject_row: dict[str, Any],
    query_root: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    subject_id = str(subject_row["subject_id"])
    epsilon = float(subject_row["direct_surface_depth_tie_epsilon"])
    parts = []
    old_choice = memory_builder._choice

    if seed is not None:
        def alternative_choice(count: int, limit: int, *, key: str) -> np.ndarray:
            if count <= 0 or limit <= 0:
                return np.zeros(0, dtype=np.int64)
            rng = np.random.default_rng(_seed(f"fit8-autopsy-v1:{seed}:{key}"))
            return rng.permutation(count)[: min(count, limit)].astype(np.int64)
        memory_builder._choice = alternative_choice
    try:
        for view_row in subject_row["views"]:
            vi = int(view_row["view_index"])
            data = _load_query_view(query_root / str(view_row["file"]), str(view_row["sha256"]))
            for part in (
                memory_builder._surface_part(
                    data, subject_id=subject_id, view_index=vi, epsilon=epsilon
                ),
                memory_builder._pair_part(
                    data, subject_id=subject_id, view_index=vi, epsilon=epsilon
                ),
                memory_builder._ray_part(
                    data, subject_id=subject_id, view_index=vi, epsilon=epsilon
                ),
            ):
                if part is not None:
                    parts.append(part)
    finally:
        memory_builder._choice = old_choice

    evidence = memory_builder._concat(parts)
    if int(evidence.values.shape[1]) > int(memory_builder.MAX_PRESENTATION_TOKENS_PER_SUBJECT):
        raise AssertionError("FIT8_AUTOPSY_RESAMPLE_TOKEN_BUDGET_DRIFT")
    return {
        "schema": SUBJECT_MEMORY_SCHEMA,
        "subject_id": subject_id,
        "values": evidence.values[0].contiguous(),
        "valid_mask": evidence.valid_mask[0].contiguous(),
        "kind_id": evidence.kind_id[0].contiguous(),
        "feature_names": PRESENTATION_FEATURE_NAMES_V1,
        "source_query_subject_binding": {
            "direct_surface_sha256": subject_row["direct_surface_sha256"],
            "direct_surface_candidate_binding_hash": subject_row[
                "direct_surface_candidate_binding_hash"
            ],
            "camera_bundle_sha256": subject_row["camera_bundle_sha256"],
        },
        "teacher_geometry_in_conditioning": False,
        "teacher_topology_in_conditioning": False,
        "materialized_v2_truth_in_conditioning": False,
    }


def _assert_reproduction(original: dict[str, Any], rebuilt: dict[str, Any], subject_id: str) -> None:
    scalar_fields = (
        "schema",
        "subject_id",
        "feature_names",
        "source_query_subject_binding",
        "teacher_geometry_in_conditioning",
        "teacher_topology_in_conditioning",
        "materialized_v2_truth_in_conditioning",
    )
    for field in scalar_fields:
        if original[field] != rebuilt[field]:
            raise RuntimeError(f"FIT8_AUTOPSY_ORIGINAL_REPRODUCTION_FIELD_DRIFT:{subject_id}:{field}")
    for field in ("values", "valid_mask", "kind_id"):
        if not torch.equal(original[field].detach().cpu(), rebuilt[field].detach().cpu()):
            raise RuntimeError(f"FIT8_AUTOPSY_ORIGINAL_REPRODUCTION_TENSOR_DRIFT:{subject_id}:{field}")


def _classify_stability(*, original_gain: float, gains: list[float]) -> str:
    positive = [g > 0.0 for g in gains]
    negative = [g < 0.0 for g in gains]
    if original_gain > 0.0 and all(positive):
        return "STABLE_POSITIVE"
    if original_gain < 0.0 and all(negative):
        return "STABLE_NEGATIVE"
    return "SAMPLING_SENSITIVE"


def _phase2(
    *,
    prereg: dict[str, Any],
    query_manifest: dict[str, Any],
    query_root: Path,
    memory_manifest: dict[str, Any],
    memory_root: Path,
    corpus_manifest: dict[str, Any],
    corpus_root: Path,
    training_root: Path,
    parent_prereg_path: Path,
    census: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    query_subjects = {str(row["subject_id"]): row for row in query_manifest["subjects"]}
    memory_subjects = {str(row["subject_id"]): row for row in memory_manifest["subjects"]}
    corpus_subjects = {str(row["subject_id"]): row for row in corpus_manifest["subjects"]}
    selected_subjects = tuple(prereg["phase_2_sampling_stability"]["primary_subjects"])
    seeds = tuple(int(s) for s in prereg["phase_2_sampling_stability"]["seeds"])

    reproduction = {}
    for subject_id in selected_subjects:
        mrow = memory_subjects[subject_id]
        original = _memory_payload(memory_root / str(mrow["file"]), str(mrow["sha256"]), subject_id)
        rebuilt = _payload_from_queries(
            subject_row=query_subjects[subject_id],
            query_root=query_root,
            seed=None,
        )
        _assert_reproduction(original, rebuilt, subject_id)
        reproduction[subject_id] = True
        _phase(f"subject={subject_id} stage=original_memory_reproduction_pass")

    parent = load_json(parent_prereg_path)
    validate_preregistration(parent)
    parent_sha = sha256_file(parent_prereg_path)
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    memory_manifest_path = memory_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    corpus_sha = sha256_file(corpus_path)
    memory_sha = sha256_file(memory_manifest_path)

    bridge, receipt, checkpoint_path = _load_arm_model(
        arm_id=ARM_B,
        training_root=training_root,
        device=device,
        training_source_sha=str(prereg["frozen_inputs"]["training_execution_source_sha"]),
        parent_prereg_sha=parent_sha,
        corpus_sha=corpus_sha,
        presentation_sha=memory_sha,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_b = str(prereg["frozen_inputs"]["arm_b_checkpoint_sha256"])
    if checkpoint_sha != expected_b:
        raise RuntimeError("FIT8_AUTOPSY_B_CHECKPOINT_DRIFT")

    subjects = {
        s: load_subject(
            corpus_root / str(corpus_subjects[s]["file"]),
            str(corpus_subjects[s]["sha256"]),
        )
        for s in selected_subjects
    }

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for subject_id in selected_subjects:
        a_obj = float(census["census_decision"]["subject_votes"][subject_id]["a_total_objective"])
        original_b_obj = float(census["census_decision"]["subject_votes"][subject_id]["b_total_objective"])
        original_gain = _relative_improvement(a_obj, original_b_obj)
        seed_gains: list[float] = []
        seed_objectives: list[float] = []

        for seed in seeds:
            payload = _payload_from_queries(
                subject_row=query_subjects[subject_id],
                query_root=query_root,
                seed=seed,
            )
            memory = _conditioning(
                bridge,
                arm_id=ARM_B,
                subject=subjects[subject_id],
                presentation_payload=payload,
                device=device,
            )
            detail = _teacher_census(
                bridge.structuralizer,
                memory=memory,
                subject=subjects[subject_id],
                device=device,
                heartbeat_prefix=f"autopsy subject={subject_id} seed={seed}",
            )
            objective = float(detail["total_objective"])
            gain = _relative_improvement(a_obj, objective)
            seed_objectives.append(objective)
            seed_gains.append(gain)
            rows.append(
                {
                    "subject_id": subject_id,
                    "seed": seed,
                    "a_total_objective": a_obj,
                    "original_b_total_objective": original_b_obj,
                    "resampled_b_total_objective": objective,
                    "original_b_gain_vs_a": original_gain,
                    "resampled_b_gain_vs_a": gain,
                    "relative_change_resampled_b_vs_original_b": _relative_improvement(
                        original_b_obj, objective
                    ),
                    "sign_vs_a": "B_BETTER" if gain > 0.0 else "A_BETTER" if gain < 0.0 else "TIE",
                }
            )
            _phase(
                f"subject={subject_id} seed={seed} stage=resample_teacher_pass "
                f"objective={objective:.9f} gain_vs_a={gain:.6f}"
            )
            del memory, payload
            gc.collect()
            torch.cuda.empty_cache()

        summary[subject_id] = {
            "original_b_gain_vs_a": original_gain,
            "original_b_total_objective": original_b_obj,
            "a_total_objective": a_obj,
            "resampled_b_total_objective_mean": _mean(seed_objectives),
            "resampled_b_total_objective_std": float(np.std(seed_objectives, ddof=0)),
            "resampled_b_total_objective_min": float(np.min(seed_objectives)),
            "resampled_b_total_objective_max": float(np.max(seed_objectives)),
            "resampled_b_gain_vs_a_mean": _mean(seed_gains),
            "resampled_b_gain_vs_a_min": float(np.min(seed_gains)),
            "resampled_b_gain_vs_a_max": float(np.max(seed_gains)),
            "sign_stability": _classify_stability(
                original_gain=original_gain,
                gains=seed_gains,
            ),
            "all_original_builder_reproduced": True,
        }

    del bridge
    gc.collect()
    torch.cuda.empty_cache()
    return rows, summary, {
        "original_memory_reproduction": reproduction,
        "b_checkpoint_sha256": checkpoint_sha,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_sha = str(args.expected_evaluator_sha).strip().lower()
    if git_head(repo_root) != expected_sha:
        raise RuntimeError("FIT8_AUTOPSY_CHECKOUT_SHA_DRIFT")

    prereg_path = Path(args.autopsy_preregistration).resolve()
    prereg = load_json(prereg_path)
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("FIT8_AUTOPSY_PREREG_SCHEMA_INVALID")
    if prereg.get("status") != "LOCKED_POST_HOC_MECHANISM_DIAGNOSTIC__NOT_QUALIFICATION":
        raise ValueError("FIT8_AUTOPSY_PREREG_STATUS_INVALID")
    if tuple(prereg.get("subjects") or ()) != tuple(SUBJECT_IDS):
        raise ValueError("FIT8_AUTOPSY_SUBJECT_ORDER_DRIFT")
    if prereg["explicit_limits"]["retraining"] is not False:
        raise ValueError("FIT8_AUTOPSY_RETRAINING_ILLEGAL")
    if int(prereg["explicit_limits"]["optimizer_steps"]) != 0:
        raise ValueError("FIT8_AUTOPSY_OPTIMIZER_BUDGET_ILLEGAL")

    query_root = Path(args.query_root).resolve()
    query_manifest_path = query_root / "DIRECT_IRIS_PRESENTATION_QUERIES_V1_MANIFEST.json"
    query_manifest = load_json(query_manifest_path)
    if query_manifest.get("schema") != QUERY_SCHEMA:
        raise ValueError("FIT8_AUTOPSY_QUERY_MANIFEST_INVALID")
    if sha256_file(query_manifest_path) != str(prereg["frozen_inputs"]["query_manifest_sha256"]):
        raise ValueError("FIT8_AUTOPSY_QUERY_MANIFEST_SHA_DRIFT")
    if int(query_manifest.get("optimizer_steps_executed", -1)) != 0:
        raise ValueError("FIT8_AUTOPSY_QUERY_OPTIMIZER_DRIFT")

    memory_root = Path(args.presentation_root).resolve()
    memory_manifest_path = memory_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    memory_manifest = load_json(memory_manifest_path)
    if memory_manifest.get("schema") != MEMORY_SCHEMA:
        raise ValueError("FIT8_AUTOPSY_MEMORY_MANIFEST_INVALID")
    if sha256_file(memory_manifest_path) != str(
        prereg["frozen_inputs"]["presentation_memory_manifest_sha256"]
    ):
        raise ValueError("FIT8_AUTOPSY_MEMORY_MANIFEST_SHA_DRIFT")

    census_path = Path(args.census_result).resolve()
    if sha256_file(census_path) != str(args.expected_census_result_sha256):
        raise ValueError("FIT8_AUTOPSY_CENSUS_RESULT_SHA_DRIFT")
    census = load_json(census_path)
    if census.get("status") != "SELECT_B" or census.get("selected_arm") != ARM_B:
        raise ValueError("FIT8_AUTOPSY_PARENT_CENSUS_NOT_SELECT_B")
    if int(census.get("optimizer_steps_executed", -1)) != 0:
        raise ValueError("FIT8_AUTOPSY_PARENT_CENSUS_OPTIMIZER_DRIFT")
    if census.get("parameters_updated") is not False:
        raise ValueError("FIT8_AUTOPSY_PARENT_CENSUS_PARAMETER_DRIFT")

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("FIT8_AUTOPSY_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    _phase("stage=phase1_begin")
    per_view, subject_summary, correlations = _phase1(
        query_manifest=query_manifest,
        memory_manifest=memory_manifest,
        memory_root=memory_root,
        census=census,
    )
    _write_csv(output_root / "FIT8_CORPUS_EVIDENCE_AUTOPSY_SUBJECT_VIEW.csv", per_view)
    _atomic_json(output_root / "PHASE1_SUBJECT_SUMMARY.json", subject_summary)
    _atomic_json(output_root / "PHASE1_CORRELATIONS.json", correlations)
    _phase("stage=phase1_pass")

    if not torch.cuda.is_available():
        raise RuntimeError("FIT8_AUTOPSY_PHASE2_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("FIT8_AUTOPSY_PHASE2_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    corpus_root = Path(args.corpus_root).resolve()
    corpus_manifest_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus_manifest = load_json(corpus_manifest_path)

    _phase("stage=phase2_begin")
    resample_rows, stability, phase2_provenance = _phase2(
        prereg=prereg,
        query_manifest=query_manifest,
        query_root=query_root,
        memory_manifest=memory_manifest,
        memory_root=memory_root,
        corpus_manifest=corpus_manifest,
        corpus_root=corpus_root,
        training_root=Path(args.training_root).resolve(),
        parent_prereg_path=Path(args.parent_preregistration).resolve(),
        census=census,
        device=device,
    )
    _write_csv(output_root / "FIT8_CORPUS_EVIDENCE_AUTOPSY_RESAMPLING.csv", resample_rows)
    _atomic_json(output_root / "PHASE2_SAMPLING_STABILITY.json", stability)
    _phase("stage=phase2_pass")

    classifications = {s: stability[s]["sign_stability"] for s in stability}
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_POST_HOC_DIAGNOSTIC__NO_QUALIFICATION",
        "experiment_id": "STRUCTURAL-EVIDENCE-FIT8-CORPUS-EVIDENCE-AUTOPSY-V1",
        "execution_source_sha": expected_sha,
        "training_execution_source_sha": prereg["frozen_inputs"]["training_execution_source_sha"],
        "parent_census_result_sha256": sha256_file(census_path),
        "query_manifest_sha256": sha256_file(query_manifest_path),
        "presentation_memory_manifest_sha256": sha256_file(memory_manifest_path),
        "phase_1": {
            "subject_summary": subject_summary,
            "correlations": correlations,
        },
        "phase_2": {
            "sampling_stability": stability,
            "classifications": classifications,
            "provenance": phase2_provenance,
        },
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "parameters_updated": False,
        "parent_directional_result_mutated": False,
        "qualification_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "shared_scene_hypothesis_proven": False,
        "causal_attribution_claimed": False,
        "interpretation_scope": "POST_HOC_CORPUS_EXTRACTOR_SAMPLER_MECHANISM_DIAGNOSTIC_ONLY",
    }
    result_path = output_root / "FIT8_CORPUS_EVIDENCE_AUTOPSY_RESULT.json"
    _atomic_json(result_path, result)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "SEALED_ZERO_STEP_AUTOPSY_RESULT",
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "qualification_pass_claimed": False,
    }
    _atomic_json(output_root / "FIT8_CORPUS_EVIDENCE_AUTOPSY_RECEIPT.json", receipt)
    _phase("stage=complete classifications=" + json.dumps(classifications, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-evaluator-sha", required=True)
    parser.add_argument("--autopsy-preregistration", required=True)
    parser.add_argument("--parent-preregistration", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--presentation-root", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--census-result", required=True)
    parser.add_argument("--expected-census-result-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
