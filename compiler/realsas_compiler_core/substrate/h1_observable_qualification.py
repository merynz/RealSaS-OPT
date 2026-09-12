from __future__ import annotations

"""Qualification contract for scene-first H1 signed-surface outputs.

The historical H1 V2 FIT gate mixed two different questions:

1. Does the product-input-derived surface explain the exact observed 8-view raster evidence?
2. Does the learned signed field reconstruct the entire FIT teacher as a globally signed,
   watertight 3D object?

Those questions are equivalent only when a closed/watertight sign contract has been
qualified. They are not equivalent for open or mixed character surfaces (cloth sheets,
rigid accessories, shells, cards, etc.). This module preserves full-teacher and global
signed metrics as diagnostics while making observable product evidence the hard product
surface gate. No teacher geometry is admitted at product inference.
"""

from dataclasses import dataclass, asdict
import math
from typing import Mapping, Any


OPEN_OR_MIXED_SURFACE = "OPEN_OR_MIXED_SURFACE"
QUALIFIED_CLOSED_WATERTIGHT = "QUALIFIED_CLOSED_WATERTIGHT"
_ALLOWED_SURFACE_CLASSES = {OPEN_OR_MIXED_SURFACE, QUALIFIED_CLOSED_WATERTIGHT}
_EXPECTED_PRODUCT_INPUTS = ("8x1024_RGBA", "8xORTHOGRAPHIC_CAMERA")


@dataclass(frozen=True)
class H1ObservableQualificationThresholds:
    min_alpha_recall: float = 0.95
    min_alpha_precision: float = 0.95
    min_alpha_iou: float = 0.92
    observable_first_hit_p95_norm_max: float = 0.05
    closed_zero_bracket_rate_min: float = 0.90
    closed_full_teacher_p95_norm_max: float = 0.05


class H1QualificationError(RuntimeError):
    pass


def _finite_number(value: Any) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise H1QualificationError("H1_QUALIFICATION_NONFINITE_METRIC")
    return v


def _require_alpha_report(alpha_report: Mapping[str, Any], t: H1ObservableQualificationThresholds) -> tuple[bool, dict]:
    mins = dict(alpha_report.get("minimums") or {})
    if not {"recall", "precision", "iou"}.issubset(mins):
        raise H1QualificationError("H1_ALPHA_REPORT_INCOMPLETE")
    recall = _finite_number(mins["recall"])
    precision = _finite_number(mins["precision"])
    iou = _finite_number(mins["iou"])
    checks = {
        "min_recall": recall >= t.min_alpha_recall,
        "min_precision": precision >= t.min_alpha_precision,
        "min_iou": iou >= t.min_alpha_iou,
    }
    return all(checks.values()), {
        "minimums": {"recall": recall, "precision": precision, "iou": iou},
        "thresholds": {
            "min_recall": t.min_alpha_recall,
            "min_precision": t.min_alpha_precision,
            "min_iou": t.min_alpha_iou,
        },
        "checks": checks,
    }


def _require_observable_strata(strata: Mapping[str, Any], t: H1ObservableQualificationThresholds) -> tuple[bool, dict]:
    rows = []
    for name in sorted(strata):
        row = dict(strata[name] or {})
        n = int(row.get("n", 0) or 0)
        p95 = row.get("p95_norm")
        if n <= 0:
            rows.append({"name": name, "n": n, "p95_norm": None, "applicable": False, "pass": True})
            continue
        if p95 is None:
            raise H1QualificationError(f"H1_OBSERVABLE_STRATUM_MISSING_P95:{name}")
        p95f = _finite_number(p95)
        rows.append({
            "name": name,
            "n": n,
            "p95_norm": p95f,
            "applicable": True,
            "pass": p95f <= t.observable_first_hit_p95_norm_max,
        })
    applicable = [r for r in rows if r["applicable"]]
    if not applicable:
        raise H1QualificationError("H1_NO_OBSERVABLE_FIRST_HIT_STRATA")
    return all(r["pass"] for r in applicable), {
        "p95_norm_max": t.observable_first_hit_p95_norm_max,
        "strata": rows,
    }


def qualify_h1_observable_surface_v1(
    metrics: Mapping[str, Any],
    *,
    product_inference_inputs: tuple[str, ...] | list[str],
    teacher_mesh_used_at_inference: bool,
    surface_sign_contract: str,
    thresholds: H1ObservableQualificationThresholds | None = None,
) -> dict:
    """Return a fail-closed qualification report for one H1 surface candidate.

    For OPEN_OR_MIXED_SURFACE, full-teacher bidirectional distances and global
    +/- zero-bracket are retained as FIT diagnostics only. They cannot veto an
    otherwise qualified observed product surface because a global inside/outside
    sign is not well-defined for arbitrary open sheets/shells.

    For QUALIFIED_CLOSED_WATERTIGHT, the historical global sign/full-teacher gates
    remain hard gates in addition to observation gates.
    """
    t = thresholds or H1ObservableQualificationThresholds()
    if surface_sign_contract not in _ALLOWED_SURFACE_CLASSES:
        raise H1QualificationError(f"H1_UNKNOWN_SURFACE_SIGN_CONTRACT:{surface_sign_contract}")
    if tuple(product_inference_inputs) != _EXPECTED_PRODUCT_INPUTS:
        raise H1QualificationError("H1_PRODUCT_INPUT_CONTRACT_DRIFT")
    if bool(teacher_mesh_used_at_inference):
        raise H1QualificationError("H1_TEACHER_GEOMETRY_AT_PRODUCT_INFERENCE_FORBIDDEN")

    alpha_report = metrics.get("input_alpha_coverage")
    if not isinstance(alpha_report, Mapping):
        raise H1QualificationError("H1_INPUT_ALPHA_COVERAGE_MISSING")
    alpha_pass, alpha_detail = _require_alpha_report(alpha_report, t)

    strata = metrics.get("observable_support_strata")
    if not isinstance(strata, Mapping):
        raise H1QualificationError("H1_OBSERVABLE_SUPPORT_STRATA_MISSING")
    observable_pass, observable_detail = _require_observable_strata(strata, t)

    diagnostic = {
        "zero_bracket_rate": _finite_number(metrics.get("zero_bracket_rate")),
        "truth_to_pred_p95_norm": _finite_number(metrics.get("truth_to_pred_p95_norm")),
        "pred_to_truth_p95_norm": _finite_number(metrics.get("pred_to_truth_p95_norm")),
    }

    global_sign_gate_applicable = surface_sign_contract == QUALIFIED_CLOSED_WATERTIGHT
    if global_sign_gate_applicable:
        historical_checks = {
            "zero_bracket_rate": diagnostic["zero_bracket_rate"] >= t.closed_zero_bracket_rate_min,
            "truth_to_pred_p95_norm": diagnostic["truth_to_pred_p95_norm"] <= t.closed_full_teacher_p95_norm_max,
            "pred_to_truth_p95_norm": diagnostic["pred_to_truth_p95_norm"] <= t.closed_full_teacher_p95_norm_max,
        }
        historical_pass = all(historical_checks.values())
        historical_role = "HARD_GATE__CLOSED_WATERTIGHT_SIGN_CONTRACT_QUALIFIED"
    else:
        historical_checks = {}
        historical_pass = True
        historical_role = "DIAGNOSTIC_ONLY__GLOBAL_SIGN_OR_FULL_HIDDEN_TEACHER_NOT_PRODUCT_AUTHORITY"

    hard_pass = bool(alpha_pass and observable_pass and historical_pass)
    return {
        "schema": "RealSaS.H1ObservableSurfaceQualification.v1",
        "status": "PASS_H1_OBSERVABLE_PRODUCT_SURFACE" if hard_pass else "FAIL_H1_OBSERVABLE_PRODUCT_SURFACE",
        "pass": hard_pass,
        "surface_sign_contract": surface_sign_contract,
        "product_inference_inputs": list(_EXPECTED_PRODUCT_INPUTS),
        "teacher_mesh_used_at_inference": False,
        "hard_gates": {
            "input_alpha_coverage": {"pass": alpha_pass, **alpha_detail},
            "observable_first_hit_geometry": {"pass": observable_pass, **observable_detail},
            "closed_watertight_global_sign": {
                "applicable": global_sign_gate_applicable,
                "pass": historical_pass,
                "checks": historical_checks,
                "thresholds": {
                    "zero_bracket_rate_min": t.closed_zero_bracket_rate_min,
                    "full_teacher_p95_norm_max": t.closed_full_teacher_p95_norm_max,
                },
            },
        },
        "historical_full_teacher_and_sign_metrics": {
            **diagnostic,
            "qualification_role": historical_role,
            "preserved_as_scientific_history": True,
        },
        "authority_boundary": {
            "teacher_geometry_allowed_at_fit_evaluation": True,
            "teacher_geometry_allowed_at_product_inference": False,
            "full_hidden_teacher_completeness_hard_gate": global_sign_gate_applicable,
            "product_observation_authority": "EXACT_8_RGBA_ALPHA_PLUS_EXACT_8_CAMERAS",
        },
        "thresholds": asdict(t),
        "canonical_promotion_authorized": False,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
    }


__all__ = [
    "OPEN_OR_MIXED_SURFACE",
    "QUALIFIED_CLOSED_WATERTIGHT",
    "H1ObservableQualificationThresholds",
    "H1QualificationError",
    "qualify_h1_observable_surface_v1",
]
