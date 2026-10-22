#!/usr/bin/env python3
from __future__ import annotations

"""Zero-training causal-utilization ablations on the sealed B presentation checkpoint."""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from models.tessa.v2.experiment_v2 import sha256_file
from tools.iris_vnext.evaluate_structural_evidence_ab_fit8_teacher_census_v1 import (
    _teacher_census,
)
from tools.iris_vnext.evaluate_structural_evidence_ab_directional_screen_v1 import (
    _load_arm_model,
)
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v1 import _conditioning
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_B,
    _load_presentation_subject,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceFrozenBCausalUtilizationResult.v1"
PREREG_SCHEMA = "RealSaS.StructuralEvidenceFrozenBCausalUtilizationAblation.v1"
VARIANTS = (
    "B_CORRECT",
    "B_VALUE_ZERO_KEEP_VALID",
    "B_SEMANTICS_ABSTAIN",
    "B_SEMANTICS_SHUFFLED_WITHIN_KIND",
    "B_MINUS_FRONT",
    "B_MINUS_VISIBILITY",
    "B_MINUS_BOUNDARY",
    "B_MINUS_RGBA_SUPPORT",
)
EPS = 1.0e-12


def _phase(message: str) -> None:
    print("FROZEN_B_UTILIZATION " + message, flush=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (float(baseline) - float(challenger)) / max(abs(float(baseline)), EPS)


def _feature_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(PRESENTATION_FEATURE_NAMES_V1)}


IDX = _feature_index()
SEMANTIC_VALUE_NAMES = (
    "source_r", "source_g", "source_b", "source_a",
    "alpha_boundary", "appearance_boundary", "front_probability",
    "occlusion_probability", "global_visibility", "deformation_coherence",
    "attachment_persistence", "source_support",
)
SEMANTIC_VALID_NAMES = (
    "source_rgba_valid", "alpha_boundary_valid", "appearance_boundary_valid",
    "front_valid", "occlusion_valid", "global_visibility_valid",
    "deformation_coherence_valid", "attachment_persistence_valid", "source_support_valid",
)
SEMANTIC_ALL_NAMES = SEMANTIC_VALUE_NAMES + SEMANTIC_VALID_NAMES


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["values"] = payload["values"].detach().cpu().clone()
    out["valid_mask"] = payload["valid_mask"].detach().cpu().clone()
    out["kind_id"] = payload["kind_id"].detach().cpu().clone()
    return out


def _zero_fields(values: torch.Tensor, names: tuple[str, ...]) -> None:
    for name in names:
        values[:, IDX[name]] = 0.0


def _shuffle_semantics_within_kind(
    payload: dict[str, Any], *, subject_id: str, seed: int
) -> dict[str, Any]:
    out = _clone_payload(payload)
    original = out["values"].clone()
    values = out["values"]
    kind = out["kind_id"]
    semantic_idx = torch.tensor([IDX[n] for n in SEMANTIC_ALL_NAMES], dtype=torch.long)
    changed = False
    for kind_value in sorted(set(int(x) for x in kind.tolist())):
        positions = torch.nonzero(kind == kind_value, as_tuple=False).reshape(-1)
        if int(positions.numel()) <= 1:
            continue
        digest = hashlib.sha256(f"frozen-b-util-v1:{seed}:{subject_id}:{kind_value}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(local_seed)
        perm_np = rng.permutation(int(positions.numel()))
        if np.all(perm_np == np.arange(len(perm_np))):
            perm_np = np.roll(perm_np, 1)
        perm = torch.from_numpy(perm_np.astype(np.int64))
        src_positions = positions[perm]
        values[positions[:, None], semantic_idx[None, :]] = original[
            src_positions[:, None], semantic_idx[None, :]
        ]
        if not torch.equal(
            values[positions[:, None], semantic_idx[None, :]],
            original[positions[:, None], semantic_idx[None, :]],
        ):
            changed = True
    semantic_bearing = original[:, semantic_idx].abs().sum().item() > 0.0
    if semantic_bearing and not changed:
        raise RuntimeError("FROZEN_B_UTILIZATION_SHUFFLE_DID_NOT_CHANGE_SEMANTICS:" + subject_id)
    return out


def _intervene(payload: dict[str, Any], *, variant: str, subject_id: str, shuffle_seed: int) -> dict[str, Any]:
    if variant == "B_CORRECT":
        return _clone_payload(payload)
    if variant == "B_SEMANTICS_SHUFFLED_WITHIN_KIND":
        return _shuffle_semantics_within_kind(payload, subject_id=subject_id, seed=shuffle_seed)

    out = _clone_payload(payload)
    values = out["values"]
    if variant == "B_VALUE_ZERO_KEEP_VALID":
        _zero_fields(values, SEMANTIC_VALUE_NAMES)
    elif variant == "B_SEMANTICS_ABSTAIN":
        _zero_fields(values, SEMANTIC_ALL_NAMES)
    elif variant == "B_MINUS_FRONT":
        _zero_fields(values, ("front_probability", "front_valid"))
    elif variant == "B_MINUS_VISIBILITY":
        _zero_fields(values, ("global_visibility", "global_visibility_valid"))
    elif variant == "B_MINUS_BOUNDARY":
        _zero_fields(values, (
            "alpha_boundary", "alpha_boundary_valid",
            "appearance_boundary", "appearance_boundary_valid",
        ))
    elif variant == "B_MINUS_RGBA_SUPPORT":
        _zero_fields(values, (
            "source_r", "source_g", "source_b", "source_a", "source_rgba_valid",
            "source_support", "source_support_valid",
        ))
    else:
        raise ValueError("FROZEN_B_UTILIZATION_UNKNOWN_VARIANT:" + variant)
    return out


def _assert_nonsemantic_invariance(original: dict[str, Any], changed: dict[str, Any], variant: str) -> None:
    if not torch.equal(original["valid_mask"], changed["valid_mask"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_TOP_MASK_DRIFT:" + variant)
    if not torch.equal(original["kind_id"], changed["kind_id"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_KIND_DRIFT:" + variant)
    semantic = {IDX[n] for n in SEMANTIC_ALL_NAMES}
    keep = [i for i in range(len(PRESENTATION_FEATURE_NAMES_V1)) if i not in semantic]
    if not torch.equal(original["values"][:, keep], changed["values"][:, keep]):
        raise RuntimeError("FROZEN_B_UTILIZATION_NONSEMANTIC_DRIFT:" + variant)


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    rows = list(corpus.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("FROZEN_B_UTILIZATION_CORPUS_ORDER_DRIFT")

    presentation_root = Path(args.presentation_root).resolve()
    pm_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    pm = load_json(pm_path)
    prows = list(pm.get("subjects") or ())
    if tuple(row.get("subject_id") for row in prows) != SUBJECT_IDS:
        raise ValueError("FROZEN_B_UTILIZATION_PRESENTATION_ORDER_DRIFT")
    presentations = [
        _load_presentation_subject(
            presentation_root / str(row["file"]), str(row["sha256"]), str(row["subject_id"])
        )
        for row in prows
    ]
    return corpus, pm, rows, presentations


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_sha = str(args.expected_evaluator_sha).strip().lower()
    if git_head(repo_root) != expected_sha:
        raise RuntimeError("FROZEN_B_UTILIZATION_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("FROZEN_B_UTILIZATION_CUDA_REQUIRED")
    if "A100" not in str(torch.cuda.get_device_properties(0).name).upper():
        raise RuntimeError("FROZEN_B_UTILIZATION_A100_REQUIRED")
    device = torch.device("cuda")

    prereg = load_json(Path(args.utilization_preregistration).resolve())
    if prereg.get("schema") != PREREG_SCHEMA or not str(prereg.get("status", "")).startswith("LOCKED_POST_HOC"):
        raise ValueError("FROZEN_B_UTILIZATION_PREREG_INVALID")
    if tuple(prereg["subjects"]) != SUBJECT_IDS:
        raise ValueError("FROZEN_B_UTILIZATION_SUBJECT_DRIFT")
    if int(prereg["evaluation"]["optimizer_steps"]) != 0:
        raise ValueError("FROZEN_B_UTILIZATION_OPTIMIZER_BUDGET_DRIFT")
    shuffle_seed = int(prereg["shuffle_seed"])
    tolerance = float(prereg["evaluation"]["correct_b_parent_reproduction_absolute_tolerance"])

    parent_prereg_path = Path(args.parent_preregistration).resolve()
    parent = load_json(parent_prereg_path)
    validate_preregistration(parent)
    parent_prereg_sha = sha256_file(parent_prereg_path)

    census_path = Path(args.parent_census_result).resolve()
    if sha256_file(census_path) != str(prereg["parent_census_result_sha256"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_PARENT_CENSUS_SHA_DRIFT")
    census = load_json(census_path)
    if census.get("status") != "SELECT_B" or census.get("selected_arm") != ARM_B:
        raise RuntimeError("FROZEN_B_UTILIZATION_PARENT_CENSUS_NOT_SELECT_B")

    autopsy_path = Path(args.parent_autopsy_result).resolve()
    if sha256_file(autopsy_path) != str(prereg["parent_autopsy_result_sha256"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_PARENT_AUTOPSY_SHA_DRIFT")

    corpus, pm, rows, presentations = _load_inputs(args)
    corpus_sha = sha256_file(Path(args.corpus_root).resolve() / "TESSA_V2_CORPUS_MANIFEST.json")
    presentation_sha = sha256_file(Path(args.presentation_root).resolve() / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json")
    if presentation_sha != str(prereg["frozen_inputs"]["presentation_memory_manifest_sha256"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_PRESENTATION_MANIFEST_SHA_DRIFT")

    bridge, receipt, checkpoint_path = _load_arm_model(
        arm_id=ARM_B,
        training_root=Path(args.training_root).resolve(),
        device=device,
        training_source_sha=str(prereg["frozen_inputs"]["training_execution_source_sha"]),
        parent_prereg_sha=parent_prereg_sha,
        corpus_sha=corpus_sha,
        presentation_sha=presentation_sha,
    )
    if sha256_file(checkpoint_path) != str(prereg["frozen_inputs"]["arm_b_checkpoint_sha256"]):
        raise RuntimeError("FROZEN_B_UTILIZATION_B_CHECKPOINT_SHA_DRIFT")

    subjects = [
        load_subject(Path(args.corpus_root).resolve() / str(row["file"]), str(row["sha256"]))
        for row in rows
    ]

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("FROZEN_B_UTILIZATION_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    per_subject: dict[str, Any] = {}
    variant_macro: dict[str, list[float]] = {v: [] for v in VARIANTS}

    for subject_id, subject, original in zip(SUBJECT_IDS, subjects, presentations, strict=True):
        parent_vote = census["census_decision"]["subject_votes"][subject_id]
        a_obj = float(parent_vote["a_total_objective"])
        parent_b_obj = float(parent_vote["b_total_objective"])
        correct_gain = _relative_improvement(a_obj, parent_b_obj)
        subject_rows: list[dict[str, Any]] = []

        for variant in VARIANTS:
            payload = _intervene(original, variant=variant, subject_id=subject_id, shuffle_seed=shuffle_seed)
            _assert_nonsemantic_invariance(original, payload, variant)
            memory = _conditioning(
                bridge, arm_id=ARM_B, subject=subject,
                presentation_payload=payload, device=device,
            )
            detail = _teacher_census(
                bridge.structuralizer, memory=memory, subject=subject,
                device=device, heartbeat_prefix=f"frozen_b_util subject={subject_id} variant={variant}",
            )
            obj = float(detail["total_objective"])
            if variant == "B_CORRECT" and abs(obj - parent_b_obj) > tolerance:
                raise RuntimeError(
                    f"FROZEN_B_UTILIZATION_PARENT_B_REPRODUCTION_DRIFT:{subject_id}:"
                    f"observed={obj}:parent={parent_b_obj}:tol={tolerance}"
                )
            gain_vs_a = _relative_improvement(a_obj, obj)
            degradation_vs_correct_b = (obj - parent_b_obj) / max(abs(parent_b_obj), EPS)
            retained = None
            if abs(correct_gain) > EPS:
                retained = gain_vs_a / correct_gain
            row = {
                "subject_id": subject_id,
                "variant": variant,
                "a_total_objective": a_obj,
                "parent_b_total_objective": parent_b_obj,
                "variant_total_objective": obj,
                "correct_b_gain_vs_a": correct_gain,
                "variant_gain_vs_a": gain_vs_a,
                "relative_degradation_vs_correct_b": degradation_vs_correct_b,
                "retained_correct_b_gain_ratio": retained,
                "parent_vote": parent_vote["vote"],
            }
            result_rows.append(row)
            subject_rows.append(row)
            variant_macro[variant].append(gain_vs_a)
            _phase(
                f"subject={subject_id} variant={variant} objective={obj:.9f} "
                f"gain_vs_a={gain_vs_a:+.6f} degradation_vs_B={degradation_vs_correct_b:+.6f}"
            )
            del payload, memory
            gc.collect(); torch.cuda.empty_cache()

        per_subject[subject_id] = {
            "parent_vote": parent_vote["vote"],
            "correct_b_gain_vs_a": correct_gain,
            "variants": {row["variant"]: row for row in subject_rows},
        }

    macro = {
        variant: {
            "mean_subject_relative_gain_vs_a": float(np.mean(gains)),
            "median_subject_relative_gain_vs_a": float(np.median(gains)),
            "positive_subject_count": int(sum(g > 0.0 for g in gains)),
            "negative_subject_count": int(sum(g < 0.0 for g in gains)),
        }
        for variant, gains in variant_macro.items()
    }

    b_winners = [
        s for s in SUBJECT_IDS
        if census["census_decision"]["subject_votes"][s]["vote"] == ARM_B
    ]
    correct_positive = [float(per_subject[s]["correct_b_gain_vs_a"]) for s in b_winners]
    utilization = {}
    for variant in ("B_VALUE_ZERO_KEEP_VALID", "B_SEMANTICS_ABSTAIN", "B_SEMANTICS_SHUFFLED_WITHIN_KIND"):
        retained_ratios = []
        erased_half = 0
        for s, cg in zip(b_winners, correct_positive, strict=True):
            vg = float(per_subject[s]["variants"][variant]["variant_gain_vs_a"])
            ratio = vg / max(abs(cg), EPS)
            retained_ratios.append(ratio)
            if cg > 0.0 and vg <= 0.5 * cg:
                erased_half += 1
        utilization[variant] = {
            "b_winner_subjects": b_winners,
            "mean_retained_positive_gain_ratio": float(np.mean(retained_ratios)),
            "median_retained_positive_gain_ratio": float(np.median(retained_ratios)),
            "subjects_losing_at_least_half_correct_b_gain": erased_half,
        }

    channel_variants = ("B_MINUS_FRONT", "B_MINUS_VISIBILITY", "B_MINUS_BOUNDARY", "B_MINUS_RGBA_SUPPORT")
    channel_effects = {}
    for variant in channel_variants:
        losses = []
        rescues = []
        for s in SUBJECT_IDS:
            correct = float(per_subject[s]["variants"]["B_CORRECT"]["variant_total_objective"])
            ablated = float(per_subject[s]["variants"][variant]["variant_total_objective"])
            losses.append((ablated - correct) / max(abs(correct), EPS))
            if per_subject[s]["correct_b_gain_vs_a"] < 0.0:
                rescues.append(
                    float(per_subject[s]["variants"][variant]["variant_gain_vs_a"])
                    - float(per_subject[s]["correct_b_gain_vs_a"])
                )
        channel_effects[variant] = {
            "mean_relative_objective_degradation_vs_correct_b": float(np.mean(losses)),
            "median_relative_objective_degradation_vs_correct_b": float(np.median(losses)),
            "mean_negative_transfer_rescue_delta_gain_vs_a": float(np.mean(rescues)) if rescues else None,
        }

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_B_CAUSAL_UTILIZATION_DIAGNOSTIC__NO_QUALIFICATION",
        "execution_source_sha": expected_sha,
        "parent_census_result_sha256": sha256_file(census_path),
        "parent_autopsy_result_sha256": sha256_file(autopsy_path),
        "arm_b_checkpoint_sha256": sha256_file(checkpoint_path),
        "variants": list(VARIANTS),
        "subject_results": per_subject,
        "macro": macro,
        "semantic_utilization_summary": utilization,
        "single_channel_effects": channel_effects,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "parameters_updated": False,
        "qualification_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "world_causal_attribution_claimed": False,
        "parent_directional_result_mutated": False,
    }
    result_path = output_root / "FROZEN_B_CAUSAL_UTILIZATION_RESULT.json"
    _atomic_json(result_path, result)

    import csv
    csv_path = output_root / "FROZEN_B_CAUSAL_UTILIZATION_ROWS.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader(); writer.writerows(result_rows)

    receipt_out = {
        "schema": "RealSaS.StructuralEvidenceFrozenBCausalUtilizationReceipt.v1",
        "status": "SEALED_ZERO_STEP_FROZEN_B_UTILIZATION_RESULT",
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "qualification_pass_claimed": False,
    }
    _atomic_json(output_root / "FROZEN_B_CAUSAL_UTILIZATION_RECEIPT.json", receipt_out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-evaluator-sha", required=True)
    parser.add_argument("--utilization-preregistration", required=True)
    parser.add_argument("--parent-preregistration", required=True)
    parser.add_argument("--parent-census-result", required=True)
    parser.add_argument("--parent-autopsy-result", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--presentation-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({
        "status": result["status"],
        "semantic_utilization_summary": result["semantic_utilization_summary"],
        "single_channel_effects": result["single_channel_effects"],
        "optimizer_steps_executed": result["optimizer_steps_executed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
