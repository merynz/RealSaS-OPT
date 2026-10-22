#!/usr/bin/env python3
from __future__ import annotations

"""Full-FIT8 teacher-forced directional census for sealed Structural-Evidence A/B checkpoints.

This is NOT the parent qualification court. It evaluates every topology window for
all eight FIT8 subjects using the sealed final A/B checkpoints. Total objective is
the only per-subject voting metric. If the census cannot select an arm under the
locked rule, one bounded exact free-run on Horse per arm may break the tie only
through a structural-validity separation.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from models.tessa.v2.windowed_incremental_generation_v2 import (
    assert_windowed_incremental_reference_parity_v2,
)
from tools.iris_vnext.evaluate_structural_evidence_ab_directional_screen_v1 import (
    _free_run_probe,
    _load_arm_model,
)
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v1 import _conditioning
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    _load_presentation_subject,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceABFit8TeacherCensusResult.v1"
PREREG_SCHEMA = "RealSaS.StructuralEvidenceABFit8TeacherCensus.v1"
EPS = 1.0e-12


def _phase(message: str) -> None:
    print("FIT8_CENSUS " + message, flush=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (float(baseline) - float(challenger)) / max(abs(float(baseline)), EPS)


@torch.no_grad()
def _teacher_census(
    model,
    *,
    memory: torch.Tensor,
    subject: dict[str, Any],
    device: torch.device,
    heartbeat_prefix: str,
) -> dict[str, Any]:
    inputs_bank = subject["topology_input_ids"]
    labels_bank = subject["topology_labels"]
    offsets_bank = subject["topology_window_offsets"]
    count = int(inputs_bank.shape[0])
    if count <= 0:
        raise ValueError("FIT8_CENSUS_NO_TOPOLOGY_WINDOWS")

    weights = TESSAObjectiveWeightsV2()
    valid_sum = coord_sum = structural_sum = 0
    token_ce_sum = coord_metric_sum = coord_abs_sum = special_sum = 0.0
    window_rows: list[dict[str, Any]] = []

    for index in range(count):
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
            raise AssertionError("FIT8_CENSUS_OBJECTIVE_MISSING")
        valid = int(objective.valid_token_count)
        coord = int(objective.coordinate_token_count)
        structural = int(objective.structural_token_count)
        token_ce = float(objective.token_cross_entropy.detach().float().cpu())
        coord_metric = float(objective.coordinate_metric.detach().float().cpu())
        coord_abs = float(objective.coordinate_expected_abs_error_bins.detach().float().cpu())
        special = float(objective.structural_special_token.detach().float().cpu())
        total_window = float(objective.total.detach().float().cpu())

        valid_sum += valid
        coord_sum += coord
        structural_sum += structural
        token_ce_sum += token_ce * valid
        coord_metric_sum += coord_metric * coord
        coord_abs_sum += coord_abs * coord
        special_sum += special * structural
        window_rows.append(
            {
                "window_index": index,
                "valid_token_count": valid,
                "coordinate_token_count": coord,
                "structural_token_count": structural,
                "total_objective": total_window,
                "token_cross_entropy": token_ce,
                "coordinate_metric": coord_metric,
                "coordinate_expected_abs_error_bins": coord_abs,
                "structural_special_token_ce": special,
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == count:
            _phase(
                f"{heartbeat_prefix} stage=teacher_window_progress "
                f"windows={index + 1}/{count}"
            )
        del input_ids, labels, offsets, output

    if valid_sum <= 0:
        raise RuntimeError("FIT8_CENSUS_EMPTY_VALID_TOKEN_SET")
    token_ce_global = token_ce_sum / float(valid_sum)
    coord_metric_global = coord_metric_sum / float(coord_sum) if coord_sum else 0.0
    coord_abs_global = coord_abs_sum / float(coord_sum) if coord_sum else 0.0
    special_global = special_sum / float(structural_sum) if structural_sum else 0.0
    total_global = (
        float(weights.token_cross_entropy) * token_ce_global
        + float(weights.coordinate_metric) * coord_metric_global
        + float(weights.structural_special_token) * special_global
    )

    return {
        "window_count_total": count,
        "window_count_evaluated": count,
        "evaluated_valid_tokens": valid_sum,
        "evaluated_coordinate_tokens": coord_sum,
        "evaluated_structural_tokens": structural_sum,
        "total_objective": total_global,
        "token_cross_entropy": token_ce_global,
        "coordinate_metric": coord_metric_global,
        "coordinate_expected_abs_error_bins": coord_abs_global,
        "structural_special_token_ce": special_global,
        "all_windows_evaluated": True,
        "windows": window_rows,
    }


def _subject_votes(
    rows: dict[str, dict[str, dict[str, Any]]],
    *,
    margin: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for subject_id in SUBJECT_IDS:
        a = float(rows[subject_id][ARM_A]["total_objective"])
        b = float(rows[subject_id][ARM_B]["total_objective"])
        b_gain = _relative_improvement(a, b)
        a_gain = _relative_improvement(b, a)
        if b_gain >= margin:
            vote = ARM_B
        elif a_gain >= margin:
            vote = ARM_A
        else:
            vote = "TIE"
        out[subject_id] = {
            "vote": vote,
            "a_total_objective": a,
            "b_total_objective": b,
            "b_relative_total_objective_improvement_vs_a": b_gain,
            "a_relative_total_objective_improvement_vs_b": a_gain,
        }
    return out


def _macro(rows: dict[str, dict[str, dict[str, Any]]], arm: str) -> float:
    return float(np.mean([float(rows[s][arm]["total_objective"]) for s in SUBJECT_IDS]))


def _census_decision(
    rows: dict[str, dict[str, dict[str, Any]]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    margin = float(prereg["teacher_forced_census"]["subject_vote_relative_margin"])
    votes = _subject_votes(rows, margin=margin)
    counts = {
        ARM_A: sum(v["vote"] == ARM_A for v in votes.values()),
        ARM_B: sum(v["vote"] == ARM_B for v in votes.values()),
        "TIE": sum(v["vote"] == "TIE" for v in votes.values()),
    }
    a_macro = _macro(rows, ARM_A)
    b_macro = _macro(rows, ARM_B)
    b_macro_gain = _relative_improvement(a_macro, b_macro)
    a_macro_gain = _relative_improvement(b_macro, a_macro)

    rule = prereg["census_selection_rule"]
    min_wins = int(rule["minimum_subject_wins"])
    min_macro = float(rule["minimum_macro_relative_total_objective_improvement"])
    selected = None
    if counts[ARM_B] >= min_wins and b_macro_gain >= min_macro:
        selected = ARM_B
    elif counts[ARM_A] >= min_wins and a_macro_gain >= min_macro:
        selected = ARM_A

    plurality = None
    if counts[ARM_A] > counts[ARM_B]:
        plurality = ARM_A
    elif counts[ARM_B] > counts[ARM_A]:
        plurality = ARM_B

    return {
        "selected_arm": selected,
        "status": (
            "SELECT_A" if selected == ARM_A
            else "SELECT_B" if selected == ARM_B
            else "NEEDS_HORSE_TIEBREAK"
        ),
        "subject_votes": votes,
        "vote_counts": counts,
        "plurality_arm": plurality,
        "a_macro_total_objective": a_macro,
        "b_macro_total_objective": b_macro,
        "b_macro_relative_total_objective_improvement_vs_a": b_macro_gain,
        "a_macro_relative_total_objective_improvement_vs_b": a_macro_gain,
    }


def _horse_validity_winner(
    a: dict[str, Any],
    b: dict[str, Any],
) -> tuple[str | None, str]:
    a_invalid = bool(a["hard_invalid"])
    b_invalid = bool(b["hard_invalid"])
    if a_invalid and not b_invalid:
        return ARM_B, "A_HARD_INVALID__B_NON_HARD_INVALID"
    if b_invalid and not a_invalid:
        return ARM_A, "B_HARD_INVALID__A_NON_HARD_INVALID"
    if a_invalid and b_invalid:
        return None, "BOTH_HARD_INVALID"
    return None, "BOTH_NON_HARD_INVALID__NO_STRUCTURAL_VALIDITY_SEPARATION"


def _finalize_with_horse(
    census: dict[str, Any],
    horse: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    winner, horse_reason = _horse_validity_winner(horse[ARM_A], horse[ARM_B])
    if winner is None:
        return {
            "status": "INCONCLUSIVE",
            "selected_arm": None,
            "reason": horse_reason,
            "horse_validity_winner": None,
        }

    plurality = census["plurality_arm"]
    if plurality is not None and plurality != winner:
        return {
            "status": "INCONCLUSIVE",
            "selected_arm": None,
            "reason": "HORSE_WINNER_OPPOSES_CENSUS_PLURALITY",
            "horse_validity_winner": winner,
        }

    opposite = ARM_B if winner == ARM_A else ARM_A
    threshold = float(
        prereg["horse_tiebreak"][
            "final_selection_forbidden_if_opposite_arm_has_macro_improvement_at_least"
        ]
    )
    opposite_macro_gain = (
        census["b_macro_relative_total_objective_improvement_vs_a"]
        if opposite == ARM_B
        else census["a_macro_relative_total_objective_improvement_vs_b"]
    )
    if float(opposite_macro_gain) >= threshold:
        return {
            "status": "INCONCLUSIVE",
            "selected_arm": None,
            "reason": "HORSE_WINNER_CONTRADICTED_BY_OPPOSITE_MACRO_ADVANTAGE",
            "horse_validity_winner": winner,
        }

    return {
        "status": "SELECT_A" if winner == ARM_A else "SELECT_B",
        "selected_arm": winner,
        "reason": "HORSE_STRUCTURAL_VALIDITY_TIEBREAK_CONSISTENT_WITH_CENSUS",
        "horse_validity_winner": winner,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_sha = str(args.expected_evaluator_sha).strip().lower()
    training_source = str(args.training_source_sha).strip().lower()
    if git_head(repo_root) != expected_sha:
        raise RuntimeError("FIT8_CENSUS_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("FIT8_CENSUS_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("FIT8_CENSUS_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.census_preregistration).resolve()
    prereg = load_json(prereg_path)
    if (
        prereg.get("schema") != PREREG_SCHEMA
        or prereg.get("status") != "LOCKED_DIRECTIONAL_CENSUS__NOT_QUALIFICATION"
        or tuple(prereg.get("subjects") or ()) != tuple(SUBJECT_IDS)
    ):
        raise ValueError("FIT8_CENSUS_PREREG_INVALID")
    if prereg.get("training_execution_source_sha") != training_source:
        raise ValueError("FIT8_CENSUS_TRAINING_SOURCE_DRIFT")

    parent_path = Path(args.parent_preregistration).resolve()
    parent = load_json(parent_path)
    protocol = validate_preregistration(parent)
    parent_sha = sha256_file(parent_path)

    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    corpus_sha = sha256_file(corpus_path)
    corpus_rows = {str(row["subject_id"]): row for row in corpus.get("subjects") or ()}
    if tuple(corpus_rows) != tuple(SUBJECT_IDS):
        raise ValueError("FIT8_CENSUS_CORPUS_SUBJECT_ORDER_DRIFT")

    presentation_root = Path(args.presentation_root).resolve()
    pmanifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    pmanifest = load_json(pmanifest_path)
    presentation_sha = sha256_file(pmanifest_path)
    prows = {str(row["subject_id"]): row for row in pmanifest.get("subjects") or ()}
    if any(s not in prows for s in SUBJECT_IDS):
        raise ValueError("FIT8_CENSUS_PRESENTATION_SUBJECT_MISSING")

    subjects = {
        s: load_subject(
            corpus_root / str(corpus_rows[s]["file"]),
            str(corpus_rows[s]["sha256"]),
        )
        for s in SUBJECT_IDS
    }
    presentations = {
        s: _load_presentation_subject(
            presentation_root / str(prows[s]["file"]),
            str(prows[s]["sha256"]),
            s,
        )
        for s in SUBJECT_IDS
    }

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("FIT8_CENSUS_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SUBJECT_IDS}
    checkpoint_shas: dict[str, str] = {}

    for arm in ARMS:
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
        checkpoint_sha = sha256_file(checkpoint_path)
        expected_checkpoint = str(prereg["sealed_checkpoint_sha256"][arm])
        if checkpoint_sha != expected_checkpoint:
            raise RuntimeError("FIT8_CENSUS_SEALED_CHECKPOINT_DRIFT:" + arm)
        checkpoint_shas[arm] = checkpoint_sha
        _phase(f"arm={arm} stage=model_load_pass")

        for subject_id in SUBJECT_IDS:
            _phase(f"arm={arm} subject={subject_id} stage=conditioning_begin")
            memory = _conditioning(
                bridge,
                arm_id=arm,
                subject=subjects[subject_id],
                presentation_payload=presentations[subject_id],
                device=device,
            )
            _phase(f"arm={arm} subject={subject_id} stage=teacher_census_begin")
            detail = _teacher_census(
                bridge.structuralizer,
                memory=memory,
                subject=subjects[subject_id],
                device=device,
                heartbeat_prefix=f"arm={arm} subject={subject_id}",
            )
            summary = {k: v for k, v in detail.items() if k != "windows"}
            rows[subject_id][arm] = summary
            subject_root = output_root / "teacher_forced" / arm / subject_id
            _atomic_json(subject_root / "CENSUS_DETAIL.json", detail)
            _phase(
                f"arm={arm} subject={subject_id} stage=teacher_census_pass "
                f"windows={summary['window_count_evaluated']} "
                f"total={summary['total_objective']:.6f} "
                f"coord_abs={summary['coordinate_expected_abs_error_bins']:.6f}"
            )
            del memory
            gc.collect()
            torch.cuda.empty_cache()

        del bridge
        gc.collect()
        torch.cuda.empty_cache()

    census = _census_decision(rows, prereg)
    _phase(
        "stage=census_decision "
        + json.dumps(
            {
                "status": census["status"],
                "vote_counts": census["vote_counts"],
                "plurality_arm": census["plurality_arm"],
                "a_macro": census["a_macro_total_objective"],
                "b_macro": census["b_macro_total_objective"],
            },
            sort_keys=True,
        )
    )

    horse_result = None
    final = {
        "status": census["status"],
        "selected_arm": census["selected_arm"],
        "reason": "DIRECT_FIT8_TEACHER_CENSUS_SELECTION",
        "horse_validity_winner": None,
    }

    if census["selected_arm"] is None:
        _phase("stage=horse_tiebreak_reference_parity_begin")
        parity = assert_windowed_incremental_reference_parity_v2(device=device)
        _phase("stage=horse_tiebreak_reference_parity_pass " + json.dumps(parity, sort_keys=True))
        horse_id = str(prereg["horse_tiebreak"]["subject"])
        horizon = int(prereg["horse_tiebreak"]["maximum_new_tokens"])
        horse_result = {"reference_parity": parity, "maximum_new_tokens": horizon, "arms": {}}
        for arm_index, arm in enumerate(ARMS):
            _phase(f"arm={arm} subject={horse_id} stage=horse_model_load_begin")
            bridge, receipt, checkpoint_path = _load_arm_model(
                arm_id=arm,
                training_root=Path(args.training_root).resolve(),
                device=device,
                training_source_sha=training_source,
                parent_prereg_sha=parent_sha,
                corpus_sha=corpus_sha,
                presentation_sha=presentation_sha,
            )
            memory = _conditioning(
                bridge,
                arm_id=arm,
                subject=subjects[horse_id],
                presentation_payload=presentations[horse_id],
                device=device,
            )
            _phase(f"arm={arm} subject={horse_id} stage=horse_free_run_begin horizon={horizon}")
            free, tokens = _free_run_probe(
                bridge.structuralizer,
                memory=memory,
                maximum_new_tokens=horizon,
                seed=int(protocol["seed_set"][0]) + 3,
                heartbeat_prefix=f"arm={arm} subject={horse_id} census_tiebreak",
            )
            token_path = output_root / "horse_tiebreak" / arm / f"FREE_RUNNING_{horizon}.tokens.npy"
            token_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(token_path, tokens)
            horse_result["arms"][arm] = {
                "free_running": free,
                "tokens_file": token_path.relative_to(output_root).as_posix(),
                "tokens_sha256": sha256_file(token_path),
            }
            _atomic_json(
                token_path.parent / "HORSE_TIEBREAK.json",
                horse_result["arms"][arm],
            )
            _phase(
                f"arm={arm} subject={horse_id} stage=horse_free_run_done "
                f"status={free['status_class']} hard_invalid={free['hard_invalid']} "
                f"tokens={free['generated_token_count']}"
            )
            del memory, tokens, bridge
            gc.collect()
            torch.cuda.empty_cache()

        final = _finalize_with_horse(
            census,
            {
                ARM_A: horse_result["arms"][ARM_A]["free_running"],
                ARM_B: horse_result["arms"][ARM_B]["free_running"],
            },
            prereg,
        )

    result = {
        "schema": SCHEMA,
        "experiment_id": prereg["experiment_id"],
        "status": final["status"],
        "selected_arm": final["selected_arm"],
        "reason": final["reason"],
        "evaluator_source_sha": expected_sha,
        "training_execution_source_sha": training_source,
        "census_preregistration_sha256": sha256_file(prereg_path),
        "parent_preregistration_sha256": parent_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "checkpoint_sha256": checkpoint_shas,
        "subjects": list(SUBJECT_IDS),
        "teacher_forced": rows,
        "census_decision": census,
        "horse_tiebreak": horse_result,
        "horse_validity_winner": final.get("horse_validity_winner"),
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "qualification_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "canonical_m_authorized": False,
        "product_promotion_authorized": False,
        "next_action": (
            "CONTINUE_A_DIRECTION"
            if final["status"] == "SELECT_A"
            else "CONTINUE_B_DIRECTION"
            if final["status"] == "SELECT_B"
            else "A_B_DIRECTION_REMAINS_MIXED__REVISIT_EVIDENCE_DESIGN"
        ),
    }
    _atomic_json(output_root / "FIT8_TEACHER_CENSUS_RESULT.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-evaluator-sha", required=True)
    p.add_argument("--training-source-sha", required=True)
    p.add_argument("--census-preregistration", required=True)
    p.add_argument("--parent-preregistration", required=True)
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--presentation-root", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--output-root", required=True)
    result = evaluate(p.parse_args())
    return 0 if result["status"] in {"SELECT_A", "SELECT_B", "INCONCLUSIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
