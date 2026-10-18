from __future__ import annotations

"""Evaluate the preregistered V9 scientific court after the fixed schedule.

The court consumes the zero-step preflight, the single final held-out evaluation,
and the final run-state.  Thresholds come only from the locked preregistration;
this tool does not invent or relax criteria after seeing results.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "RealSaS.IRISVNextFocusedRelationScientificCourt.v1"
STATIC_FAMILIES = ("same_drawable_affinity", "cut_probability", "attachment_affinity")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("VNEXT_V9_COURT_JSON_OBJECT_REQUIRED")
    return value


def _direction_pass(value: float, direction: str, threshold: float) -> bool:
    if direction == "MIN":
        return value >= threshold
    if direction == "MAX":
        return value <= threshold
    raise ValueError("VNEXT_V9_COURT_DIRECTION_INVALID")


def _global_metric(heldout: dict[str, Any], family: str, metric: str) -> float:
    value = (((heldout.get("families") or {}).get(family) or {}).get(metric))
    if value is None:
        raise ValueError(f"VNEXT_V9_COURT_GLOBAL_METRIC_MISSING:{family}:{metric}")
    return float(value)


def _minimum_per_view_metric(heldout: dict[str, Any], family: str, metric: str) -> float:
    views = heldout.get("per_view_aggregate") or {}
    values: list[float] = []
    for view_index in range(8):
        value = (((views.get(str(view_index)) or {}).get(family) or {}).get(metric))
        if value is None:
            raise ValueError(f"VNEXT_V9_COURT_PER_VIEW_METRIC_MISSING:{family}:{metric}:V{view_index}")
        values.append(float(value))
    return min(values)


def audit_court(
    *,
    preregistration_path: Path,
    zero_step_preflight_path: Path,
    final_heldout_path: Path,
    run_state_path: Path,
) -> dict[str, Any]:
    prereg = _load(preregistration_path)
    baseline = _load(zero_step_preflight_path)
    final = _load(final_heldout_path)
    run_state = _load(run_state_path)
    blockers: list[str] = []
    gate_results: list[dict[str, Any]] = []

    if prereg.get("schema") != "RealSaS.IRISVNextFocusedRelationPreregistration.v1":
        blockers.append("PREREGISTRATION_SCHEMA_INVALID")
    if prereg.get("thresholds_locked_before_training") is not True:
        blockers.append("THRESHOLDS_NOT_LOCKED")
    if prereg.get("scientific_fit_started") is not False:
        blockers.append("PREREGISTRATION_ALREADY_CLAIMED_START")

    source = str((prereg.get("exact_artifact_bindings") or {}).get("source_commit_sha") or "")
    schedule = prereg.get("schedule") or {}
    maximum_steps = int(schedule.get("maximum_optimizer_steps", -1))
    if maximum_steps <= 0:
        blockers.append("MAXIMUM_STEPS_INVALID")

    if baseline.get("status") != "PASS__ZERO_STEP_ONLY__TRAINING_NOT_STARTED":
        blockers.append("ZERO_STEP_PREFLIGHT_NOT_PASS")
    if baseline.get("source_commit_sha") != source:
        blockers.append("ZERO_STEP_SOURCE_DRIFT")
    if int(baseline.get("training_steps_executed", -1)) != 0:
        blockers.append("ZERO_STEP_PREFLIGHT_NOT_ZERO")
    baseline_held = baseline.get("heldout") or {}
    if int(baseline_held.get("global_step", -1)) != 0:
        blockers.append("ZERO_STEP_HELDOUT_NOT_ZERO")

    if final.get("schema") != "RealSaS.IRISVNextFocusedRelationHeldOutMetrics.v2":
        blockers.append("FINAL_HELDOUT_SCHEMA_INVALID")
    if int(final.get("global_step", -1)) != maximum_steps:
        blockers.append("FINAL_HELDOUT_STEP_DRIFT")
    if int(final.get("training_steps_executed", -1)) != maximum_steps:
        blockers.append("FINAL_HELDOUT_TRAINING_STEP_DRIFT")
    if final.get("supervision_policy") != prereg.get("supervision_policy"):
        blockers.append("FINAL_SUPERVISION_POLICY_DRIFT")

    if run_state.get("schema") != "RealSaS.IRISVNextFocusedRelationRunState.v2":
        blockers.append("RUN_STATE_SCHEMA_INVALID")
    if run_state.get("status") != "SCHEDULE_STOP_REACHED":
        blockers.append("RUN_STATE_NOT_COMPLETE")
    if run_state.get("source_commit_sha") != source:
        blockers.append("RUN_STATE_SOURCE_DRIFT")
    if int(run_state.get("training_steps_executed", -1)) != maximum_steps:
        blockers.append("RUN_STATE_STEP_DRIFT")
    if run_state.get("binding_sha256") != baseline.get("binding_sha256"):
        blockers.append("RUN_BINDING_DRIFT_FROM_ZERO_STEP")
    if run_state.get("supervision_policy") != prereg.get("supervision_policy"):
        blockers.append("RUN_SUPERVISION_POLICY_DRIFT")

    for gate in prereg.get("held_out_gates") or []:
        family = str(gate["family"])
        metric = str(gate["metric"])
        scope = str(gate.get("scope") or "GLOBAL")
        direction = str(gate["direction"])
        threshold = float(gate["threshold"])
        try:
            if scope == "GLOBAL":
                value = _global_metric(final, family, metric)
            elif scope == "MIN_PER_VIEW_AGGREGATE":
                value = _minimum_per_view_metric(final, family, metric)
            else:
                raise ValueError("VNEXT_V9_COURT_SCOPE_INVALID:" + scope)
            passed = _direction_pass(value, direction, threshold)
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append("HELDOUT_GATE_EVALUATION_ERROR:" + str(exc))
            continue
        row = {
            "kind": "HELD_OUT",
            "family": family,
            "metric": metric,
            "scope": scope,
            "direction": direction,
            "threshold": threshold,
            "value": value,
            "pass": passed,
        }
        gate_results.append(row)
        if not passed:
            blockers.append(f"HELDOUT_GATE_FAILED:{family}:{metric}:{scope}")

    for gate in prereg.get("non_regression_gates") or []:
        family = str(gate["family"])
        metric = str(gate["metric"])
        direction = str(gate["direction"])
        threshold = float(gate["threshold"])
        try:
            if metric == "balanced_accuracy_delta_from_step0":
                value = _global_metric(final, family, "balanced_accuracy") - _global_metric(
                    baseline_held, family, "balanced_accuracy"
                )
            elif metric == "balanced_bce_delta_from_step0":
                value = _global_metric(final, family, "balanced_bce_mean") - _global_metric(
                    baseline_held, family, "balanced_bce_mean"
                )
            elif family == "FROZEN_PARAMETER_DOMAINS" and metric == "all_hashes_exact":
                frozen = final.get("frozen_non_regression") or {}
                value = 1.0 if frozen and all(bool(x) for x in frozen.values()) else 0.0
            elif family == "STATIC_RELATION_ABSTENTION" and metric == "valid_target_count":
                value = float(
                    sum(int((((final.get("families") or {}).get(name) or {}).get("valid") or 0)) for name in STATIC_FAMILIES)
                )
            elif family == "occlusion_probability" and metric == "binary_court_eligible":
                value = 1.0 if bool((((final.get("families") or {}).get(family) or {}).get("court_eligible"))) else 0.0
            else:
                raise ValueError(f"VNEXT_V9_COURT_NONREG_METRIC_UNKNOWN:{family}:{metric}")
            passed = _direction_pass(value, direction, threshold)
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append("NONREG_GATE_EVALUATION_ERROR:" + str(exc))
            continue
        row = {
            "kind": "NON_REGRESSION",
            "family": family,
            "metric": metric,
            "direction": direction,
            "threshold": threshold,
            "value": value,
            "pass": passed,
        }
        gate_results.append(row)
        if not passed:
            blockers.append(f"NONREG_GATE_FAILED:{family}:{metric}")

    status = "SCIENTIFIC_PASS" if not blockers else "SCIENTIFIC_FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "source_commit_sha": source,
        "preregistration_sha256": _sha(preregistration_path),
        "zero_step_preflight_sha256": _sha(zero_step_preflight_path),
        "final_heldout_sha256": _sha(final_heldout_path),
        "run_state_sha256": _sha(run_state_path),
        "maximum_optimizer_steps": maximum_steps,
        "gate_results": gate_results,
        "blockers": blockers,
        "scientific_fit_started": True,
        "scientific_fit_pass_claimed": not blockers,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", type=Path, required=True)
    p.add_argument("--zero-step-preflight", type=Path, required=True)
    p.add_argument("--final-heldout", type=Path, required=True)
    p.add_argument("--run-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    report = audit_court(
        preregistration_path=args.preregistration.resolve(),
        zero_step_preflight_path=args.zero_step_preflight.resolve(),
        final_heldout_path=args.final_heldout.resolve(),
        run_state_path=args.run_state.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_V9_SCIENTIFIC_COURT",
        f"status={report['status']}",
        f"blockers={len(report['blockers'])}",
    )
    if args.require_pass and report["status"] != "SCIENTIFIC_PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())