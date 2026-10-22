#!/usr/bin/env python3
from __future__ import annotations

"""Two-subject bounded directional screen for sealed Structural-Evidence A/B checkpoints.

This is deliberately NOT the parent FIT8 qualification court.  It answers only
which already-trained arm is the better research direction to continue.  It
combines a cheap deterministic teacher-forced probe with an exact bounded greedy
free-running probe on George + Bat.  Bounded-horizon exhaustion is not an EOS
failure; typed structural-invalid EOS/dead-end states are recorded and do not
crash the process.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.generation_v2 import TESSAGenerationPolicyV2, TESSAGrammarStateV2
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from models.tessa.v2.windowed_incremental_generation_v2 import assert_windowed_incremental_reference_parity_v2
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v1 import _conditioning
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v4_optimized import (
    NO_ALLOWED_STOP,
    generate_failclosed_windowed,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    CHECKPOINT_SCHEMA,
    _load_presentation_subject,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceABDirectionalScreenResult.v1"
SCREEN_SCHEMA = "RealSaS.StructuralEvidenceABDirectionalScreen.v1"
SUBJECTS = ("George", "Bat")
MAX_TF_WINDOWS = 8
EPS = 1.0e-12


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def _phase(message: str) -> None:
    print("DIRECTIONAL_SCREEN " + message, flush=True)


def _even_window_indices(count: int, maximum: int = MAX_TF_WINDOWS) -> list[int]:
    if count <= 0:
        raise ValueError("DIRECTIONAL_SCREEN_NO_TOPOLOGY_WINDOWS")
    if count <= maximum:
        return list(range(count))
    raw = np.linspace(0, count - 1, num=maximum)
    out = sorted({int(round(x)) for x in raw.tolist()})
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != count - 1:
        out.append(count - 1)
    return out[:maximum]


def _weighted_mean(sum_value: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(sum_value) / float(count)


@torch.no_grad()
def _teacher_probe(model, *, memory: torch.Tensor, subject: dict[str, Any], device: torch.device) -> dict[str, Any]:
    inputs_bank = subject["topology_input_ids"]
    labels_bank = subject["topology_labels"]
    offsets_bank = subject["topology_window_offsets"]
    indices = _even_window_indices(int(inputs_bank.shape[0]))
    weights = TESSAObjectiveWeightsV2()

    valid_sum = coord_sum = structural_sum = 0
    total_weighted = token_ce_weighted = 0.0
    coordinate_metric_weighted = coordinate_abs_weighted = 0.0
    special_weighted = 0.0
    rows: list[dict[str, Any]] = []

    for index in indices:
        input_ids = inputs_bank[index : index + 1].to(device=device, dtype=torch.long)
        labels = labels_bank[index : index + 1].to(device=device, dtype=torch.long)
        offsets = offsets_bank[index : index + 1].to(device=device, dtype=torch.long)
        output = model.decode_tokens(
            memory=memory,
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=offsets,
            objective_weights=weights,
        )
        objective = output.objective
        if objective is None:
            raise AssertionError("DIRECTIONAL_SCREEN_OBJECTIVE_MISSING")
        valid = int(objective.valid_token_count)
        coord = int(objective.coordinate_token_count)
        structural = int(objective.structural_token_count)
        total = float(objective.total.detach().float().cpu())
        token_ce = float(objective.token_cross_entropy.detach().float().cpu())
        coord_metric = float(objective.coordinate_metric.detach().float().cpu())
        coord_abs = float(objective.coordinate_expected_abs_error_bins.detach().float().cpu())
        special = float(objective.structural_special_token.detach().float().cpu())

        total_weighted += total * valid
        token_ce_weighted += token_ce * valid
        coordinate_metric_weighted += coord_metric * coord
        coordinate_abs_weighted += coord_abs * coord
        special_weighted += special * structural
        valid_sum += valid
        coord_sum += coord
        structural_sum += structural
        rows.append({
            "window_index": int(index),
            "valid_token_count": valid,
            "coordinate_token_count": coord,
            "structural_token_count": structural,
            "total_objective": total,
            "token_cross_entropy": token_ce,
            "coordinate_metric": coord_metric,
            "coordinate_expected_abs_error_bins": coord_abs,
            "structural_special_token_ce": special,
        })
        del input_ids, labels, offsets, output

    return {
        "window_count_total": int(inputs_bank.shape[0]),
        "window_indices_evaluated": indices,
        "evaluated_valid_tokens": valid_sum,
        "evaluated_coordinate_tokens": coord_sum,
        "evaluated_structural_tokens": structural_sum,
        "total_objective": _weighted_mean(total_weighted, valid_sum),
        "token_cross_entropy": _weighted_mean(token_ce_weighted, valid_sum),
        "coordinate_metric": _weighted_mean(coordinate_metric_weighted, coord_sum),
        "coordinate_expected_abs_error_bins": _weighted_mean(coordinate_abs_weighted, coord_sum),
        "structural_special_token_ce": _weighted_mean(special_weighted, structural_sum),
        "windows": rows,
    }


def _free_run_class(candidate) -> tuple[str, bool]:
    topo = candidate.topology_court.to_dict()
    if candidate.stop_reason == NO_ALLOWED_STOP:
        return "GRAMMAR_DEAD_END__NO_ALLOWED_TOKEN", True
    if candidate.completed and candidate.stop_reason == "EOS":
        if topo.get("grammar_valid") is not True or topo.get("decode_failure"):
            return "EOS_CODEC_INVALID", True
        if topo.get("finite_geometry") is not True or topo.get("within_scale_policy") is not True:
            return "EOS_TOPOLOGY_COURT_INVALID", True
        return "EOS_VALID", False
    if candidate.stop_reason == "MAXIMUM_NEW_TOKENS_REACHED":
        return "BOUNDED_HORIZON_REACHED__NOT_EOS_FAILURE", False
    return "UNEXPECTED_STOP:" + str(candidate.stop_reason), True


@torch.no_grad()
def _free_run_probe(model, *, memory: torch.Tensor, maximum_new_tokens: int,
                    seed: int, heartbeat_prefix: str) -> tuple[dict[str, Any], np.ndarray]:
    policy = TESSAGenerationPolicyV2(
        maximum_new_tokens=int(maximum_new_tokens),
        sampling=False,
        temperature=1.0,
        top_k=0,
    )
    candidate = generate_failclosed_windowed(
        model,
        memory=memory,
        policy=policy,
        seed=int(seed),
        heartbeat_prefix=heartbeat_prefix,
    )
    status, hard_invalid = _free_run_class(candidate)
    grammar = TESSAGrammarStateV2.from_prefix(candidate.token_ids, config=model.config)
    generated = max(1, len(candidate.token_ids) - 1)
    faces = int(grammar.face_count)
    result = {
        "status_class": status,
        "hard_invalid": bool(hard_invalid),
        "completed_eos": bool(candidate.completed and candidate.stop_reason == "EOS"),
        "stop_reason": candidate.stop_reason,
        "generated_token_count": generated,
        "total_token_count_including_bos": len(candidate.token_ids),
        "face_count": faces,
        "component_count": int(grammar.component_count),
        "restart_count": int(grammar.restart_count),
        "grammar_phase": grammar.phase.value,
        "tokens_per_face": float(generated) / float(faces) if faces > 0 else None,
        "mean_selected_token_log_probability": float(candidate.token_log_probability_sum) / float(generated),
        "topology_court": candidate.topology_court.to_dict() if candidate.completed else None,
        "bounded_horizon_is_not_eos_failure": candidate.stop_reason == "MAXIMUM_NEW_TOKENS_REACHED",
    }
    return result, np.asarray(candidate.token_ids, dtype=np.int64)


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (float(baseline) - float(challenger)) / max(abs(float(baseline)), EPS)


def _subject_vote(a: dict[str, Any], b: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    af = a["free_running"]
    bf = b["free_running"]
    if bool(af["hard_invalid"]) != bool(bf["hard_invalid"]):
        winner = ARM_B if af["hard_invalid"] else ARM_A
        return {"vote": winner, "reason": "HARD_INVALID_PRECEDENCE"}

    at = a["teacher_forced"]
    bt = b["teacher_forced"]
    threshold = rule["teacher_forced_subject_vote_if_validity_tied"]
    min_gain = float(threshold["minimum_relative_total_objective_improvement"])
    max_coord_reg = float(threshold["maximum_relative_regression_on_coordinate_abs_error_bins"])
    max_special_reg = float(threshold["maximum_relative_regression_on_structural_special_token_ce"])

    b_gain = _relative_improvement(at["total_objective"], bt["total_objective"])
    b_coord_reg = -_relative_improvement(at["coordinate_expected_abs_error_bins"], bt["coordinate_expected_abs_error_bins"])
    b_special_reg = -_relative_improvement(at["structural_special_token_ce"], bt["structural_special_token_ce"])
    a_gain = _relative_improvement(bt["total_objective"], at["total_objective"])
    a_coord_reg = -_relative_improvement(bt["coordinate_expected_abs_error_bins"], at["coordinate_expected_abs_error_bins"])
    a_special_reg = -_relative_improvement(bt["structural_special_token_ce"], at["structural_special_token_ce"])

    if b_gain >= min_gain and b_coord_reg <= max_coord_reg and b_special_reg <= max_special_reg:
        vote = ARM_B
    elif a_gain >= min_gain and a_coord_reg <= max_coord_reg and a_special_reg <= max_special_reg:
        vote = ARM_A
    else:
        vote = "TIE"
    return {
        "vote": vote,
        "reason": "TEACHER_FORCED_DIRECTION_WITH_FREE_RUN_VALIDITY_TIED",
        "b_relative_total_objective_improvement_vs_a": b_gain,
        "b_coordinate_abs_error_relative_regression_vs_a": b_coord_reg,
        "b_structural_special_ce_relative_regression_vs_a": b_special_reg,
        "a_relative_total_objective_improvement_vs_b": a_gain,
    }


def _macro_total(rows: dict[str, dict[str, dict[str, Any]]], arm: str) -> float:
    return float(np.mean([rows[s][arm]["teacher_forced"]["total_objective"] for s in SUBJECTS]))


def _decision(rows: dict[str, dict[str, dict[str, Any]]], screen: dict[str, Any], horizon: int) -> dict[str, Any]:
    rule = screen["directional_decision_rule"]
    votes = {s: _subject_vote(rows[s][ARM_A], rows[s][ARM_B], rule) for s in SUBJECTS}
    values = [votes[s]["vote"] for s in SUBJECTS]
    if values[0] == values[1] and values[0] in ARMS:
        selected = values[0]
        reason = "UNANIMOUS_TWO_SUBJECT_VOTE"
    else:
        selected = None
        reason = "NO_UNANIMOUS_VOTE"
        non_ties = [v for v in values if v in ARMS]
        if len(non_ties) == 1:
            candidate = non_ties[0]
            other = ARM_B if candidate == ARM_A else ARM_A
            candidate_macro = _macro_total(rows, candidate)
            other_macro = _macro_total(rows, other)
            macro_gain = _relative_improvement(other_macro, candidate_macro)
            candidate_invalid = sum(bool(rows[s][candidate]["free_running"]["hard_invalid"]) for s in SUBJECTS)
            other_invalid = sum(bool(rows[s][other]["free_running"]["hard_invalid"]) for s in SUBJECTS)
            if macro_gain >= 0.01 and candidate_invalid <= other_invalid:
                selected = candidate
                reason = "ONE_VOTE_ONE_TIE_WITH_MACRO_GAIN_AND_NO_HARD_INVALID_REGRESSION"

    if selected == ARM_A:
        status = "SELECT_A"
    elif selected == ARM_B:
        status = "SELECT_B"
    else:
        primary = int(screen["measurements"]["bounded_free_running_probe"]["primary_horizon_new_tokens"])
        extension = int(screen["measurements"]["bounded_free_running_probe"]["extension_horizon_new_tokens"])
        status = "INCONCLUSIVE__RUN_32K_EXTENSION" if int(horizon) == primary else "INCONCLUSIVE"
        if int(horizon) not in (primary, extension):
            status = "INCONCLUSIVE__NONPREREG_HORIZON"

    return {
        "status": status,
        "selected_arm": selected,
        "reason": reason,
        "subject_votes": votes,
        "a_macro_total_objective": _macro_total(rows, ARM_A),
        "b_macro_total_objective": _macro_total(rows, ARM_B),
        "b_macro_relative_total_objective_improvement_vs_a": _relative_improvement(
            _macro_total(rows, ARM_A), _macro_total(rows, ARM_B)
        ),
        "a_hard_invalid_count": sum(bool(rows[s][ARM_A]["free_running"]["hard_invalid"]) for s in SUBJECTS),
        "b_hard_invalid_count": sum(bool(rows[s][ARM_B]["free_running"]["hard_invalid"]) for s in SUBJECTS),
    }


def _load_arm_model(*, arm_id: str, training_root: Path, device: torch.device,
                    training_source_sha: str, parent_prereg_sha: str,
                    corpus_sha: str, presentation_sha: str):
    arm_root = training_root / arm_id
    receipt_path = arm_root / "TRAINING_RECEIPT.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "RealSaS.StructuralEvidenceABArmTrainingReceipt.v1" or receipt.get("arm_id") != arm_id:
        raise ValueError("DIRECTIONAL_SCREEN_TRAINING_RECEIPT_INVALID:" + arm_id)
    for key, expected in {
        "execution_source_sha": training_source_sha,
        "preregistration_sha256": parent_prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
    }.items():
        if receipt.get(key) != expected:
            raise ValueError("DIRECTIONAL_SCREEN_LINEAGE_DRIFT:" + arm_id + ":" + key)
    checkpoint_path = arm_root / str(receipt["final_checkpoint"])
    if sha256_file(checkpoint_path) != receipt.get("final_checkpoint_sha256"):
        raise ValueError("DIRECTIONAL_SCREEN_CHECKPOINT_SHA_DRIFT:" + arm_id)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("arm_id") != arm_id:
        raise ValueError("DIRECTIONAL_SCREEN_CHECKPOINT_SCHEMA_INVALID:" + arm_id)
    if checkpoint.get("execution_source_sha") != training_source_sha:
        raise ValueError("DIRECTIONAL_SCREEN_CHECKPOINT_SOURCE_DRIFT:" + arm_id)
    bridge = IRISVNextStructuralizationBridgeV4().to(device=device, dtype=torch.float32)
    bridge.load_state_dict(checkpoint["model"], strict=True)
    bridge.eval()
    del checkpoint
    gc.collect()
    return bridge, receipt, checkpoint_path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_sha = str(args.expected_evaluator_sha).strip().lower()
    training_source = str(args.training_source_sha).strip().lower()
    if git_head(repo_root) != expected_sha:
        raise RuntimeError("DIRECTIONAL_SCREEN_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("DIRECTIONAL_SCREEN_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("DIRECTIONAL_SCREEN_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    screen_path = Path(args.screen_preregistration).resolve()
    screen = load_json(screen_path)
    if screen.get("schema") != SCREEN_SCHEMA or screen.get("status") != "LOCKED_DIRECTIONAL_SELECTION_SCREEN__NOT_QUALIFICATION":
        raise ValueError("DIRECTIONAL_SCREEN_PREREG_INVALID")
    if tuple(screen["subject_selection"]["primary_subjects"]) != SUBJECTS:
        raise ValueError("DIRECTIONAL_SCREEN_SUBJECT_DRIFT")
    horizon = int(args.maximum_new_tokens)
    allowed_horizons = {
        int(screen["measurements"]["bounded_free_running_probe"]["primary_horizon_new_tokens"]),
        int(screen["measurements"]["bounded_free_running_probe"]["extension_horizon_new_tokens"]),
    }
    if horizon not in allowed_horizons:
        raise ValueError("DIRECTIONAL_SCREEN_HORIZON_NOT_PREREGISTERED")

    parent_path = Path(args.parent_preregistration).resolve()
    parent = load_json(parent_path)
    protocol = validate_preregistration(parent)
    parent_sha = sha256_file(parent_path)

    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    corpus_sha = sha256_file(corpus_path)
    corpus_rows = {str(row["subject_id"]): row for row in corpus.get("subjects") or ()}
    if any(s not in corpus_rows for s in SUBJECTS):
        raise ValueError("DIRECTIONAL_SCREEN_CORPUS_SUBJECT_MISSING")

    presentation_root = Path(args.presentation_root).resolve()
    pmanifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    pmanifest = load_json(pmanifest_path)
    presentation_sha = sha256_file(pmanifest_path)
    prows = {str(row["subject_id"]): row for row in pmanifest.get("subjects") or ()}
    presentations = {
        s: _load_presentation_subject(
            presentation_root / str(prows[s]["file"]), str(prows[s]["sha256"]), s
        )
        for s in SUBJECTS
    }
    subjects = {
        s: load_subject(corpus_root / str(corpus_rows[s]["file"]), str(corpus_rows[s]["sha256"]))
        for s in SUBJECTS
    }

    _phase("stage=reference_parity_begin")
    parity = assert_windowed_incremental_reference_parity_v2(device=device)
    _phase("stage=reference_parity_pass " + json.dumps(parity, sort_keys=True))

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("DIRECTIONAL_SCREEN_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SUBJECTS}
    checkpoint_shas: dict[str, str] = {}
    for arm_index, arm in enumerate(ARMS):
        _phase(f"arm={arm} stage=model_load_begin")
        bridge, receipt, checkpoint_path = _load_arm_model(
            arm_id=arm,
            training_root=Path(args.training_root).resolve(),
            device=device,
            training_source_sha=training_source,
            parent_prereg_sha=parent_sha,
            corpus_sha=corpus_sha,
            presentation_sha=presentation_sha,
        )
        checkpoint_shas[arm] = sha256_file(checkpoint_path)
        _phase(f"arm={arm} stage=model_load_pass")

        for subject_index, subject_id in enumerate(SUBJECTS):
            _phase(f"arm={arm} subject={subject_id} stage=conditioning_begin")
            memory = _conditioning(
                bridge,
                arm_id=arm,
                subject=subjects[subject_id],
                presentation_payload=presentations[subject_id],
                device=device,
            )
            _phase(f"arm={arm} subject={subject_id} stage=teacher_probe_begin")
            teacher = _teacher_probe(
                bridge.structuralizer,
                memory=memory,
                subject=subjects[subject_id],
                device=device,
            )
            _phase(
                f"arm={arm} subject={subject_id} stage=teacher_probe_pass "
                f"total={teacher['total_objective']:.6f} ce={teacher['token_cross_entropy']:.6f}"
            )
            _phase(f"arm={arm} subject={subject_id} stage=free_run_begin horizon={horizon}")
            free, tokens = _free_run_probe(
                bridge.structuralizer,
                memory=memory,
                maximum_new_tokens=horizon,
                seed=int(protocol["seed_set"][0]) + (0 if subject_id == "George" else 1),
                heartbeat_prefix=f"arm={arm} subject={subject_id} screen_horizon={horizon}",
            )
            subject_root = output_root / arm / subject_id
            subject_root.mkdir(parents=True, exist_ok=True)
            token_path = subject_root / f"FREE_RUNNING_{horizon}.tokens.npy"
            np.save(token_path, tokens)
            rows[subject_id][arm] = {
                "teacher_forced": teacher,
                "free_running": free,
                "tokens_file": token_path.relative_to(output_root).as_posix(),
                "tokens_sha256": sha256_file(token_path),
            }
            _atomic_json(subject_root / f"SCREEN_{horizon}.json", rows[subject_id][arm])
            _phase(
                f"arm={arm} subject={subject_id} stage=free_run_done "
                f"status={free['status_class']} tokens={free['generated_token_count']} "
                f"faces={free['face_count']} components={free['component_count']}"
            )
            del memory, tokens
            gc.collect(); torch.cuda.empty_cache()

        del bridge
        gc.collect(); torch.cuda.empty_cache()

    decision = _decision(rows, screen, horizon)
    result = {
        "schema": SCHEMA,
        "status": decision["status"],
        "experiment_id": screen["experiment_id"],
        "evaluator_source_sha": expected_sha,
        "training_execution_source_sha": training_source,
        "screen_preregistration_sha256": sha256_file(screen_path),
        "parent_preregistration_sha256": parent_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "checkpoint_sha256": checkpoint_shas,
        "subjects": list(SUBJECTS),
        "maximum_new_tokens": horizon,
        "reference_parity": parity,
        "measurements": rows,
        "decision": decision,
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "qualification_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "canonical_m_authorized": False,
        "product_promotion_authorized": False,
        "next_action": (
            "CONTINUE_A_DIRECTION" if decision["status"] == "SELECT_A" else
            "CONTINUE_B_DIRECTION" if decision["status"] == "SELECT_B" else
            "RUN_PREREGISTERED_32K_EXTENSION" if decision["status"] == "INCONCLUSIVE__RUN_32K_EXTENSION" else
            "DIRECTIONAL_SCREEN_INCONCLUSIVE"
        ),
    }
    _atomic_json(output_root / "DIRECTIONAL_SCREEN_RESULT.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-evaluator-sha", required=True)
    p.add_argument("--training-source-sha", required=True)
    p.add_argument("--screen-preregistration", required=True)
    p.add_argument("--parent-preregistration", required=True)
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--presentation-root", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--maximum-new-tokens", required=True, type=int)
    result = evaluate(p.parse_args())
    return 0 if result["status"] in {"SELECT_A", "SELECT_B", "INCONCLUSIVE__RUN_32K_EXTENSION", "INCONCLUSIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
