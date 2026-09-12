import pytest

from compiler.realsas_compiler_core.substrate.h1_observable_qualification import (
    OPEN_OR_MIXED_SURFACE,
    QUALIFIED_CLOSED_WATERTIGHT,
    H1QualificationError,
    qualify_h1_observable_surface_v1,
)


def _metrics():
    return {
        "zero_bracket_rate": 0.798828125,
        "truth_to_pred_p95_norm": 0.13903570709611163,
        "pred_to_truth_p95_norm": 0.07577677401856454,
        "observable_support_strata": {
            "support_1_2": {"n": 10744, "p95_norm": 0.040687738730924204},
            "support_3_4": {"n": 572, "p95_norm": 0.024123951564970204},
            "support_5_8": {"n": 2, "p95_norm": 0.008030098852479932},
        },
        "input_alpha_coverage": {
            "minimums": {
                "recall": 0.9511473445488728,
                "precision": 0.9806321480449604,
                "iou": 0.9356283394389125,
            }
        },
    }


def _qual(metrics=None, sign=OPEN_OR_MIXED_SURFACE):
    return qualify_h1_observable_surface_v1(
        metrics or _metrics(),
        product_inference_inputs=("8x1024_RGBA", "8xORTHOGRAPHIC_CAMERA"),
        teacher_mesh_used_at_inference=False,
        surface_sign_contract=sign,
    )


def test_mage_1024_open_surface_requalifies_observable_product_surface_without_erasing_historical_fail_metrics():
    report = _qual()
    assert report["pass"] is True
    assert report["status"] == "PASS_H1_OBSERVABLE_PRODUCT_SURFACE"
    assert report["hard_gates"]["input_alpha_coverage"]["pass"] is True
    assert report["hard_gates"]["observable_first_hit_geometry"]["pass"] is True
    assert report["hard_gates"]["closed_watertight_global_sign"]["applicable"] is False
    diag = report["historical_full_teacher_and_sign_metrics"]
    assert diag["zero_bracket_rate"] == pytest.approx(0.798828125)
    assert diag["truth_to_pred_p95_norm"] == pytest.approx(0.13903570709611163)
    assert diag["preserved_as_scientific_history"] is True
    assert report["canonical_promotion_authorized"] is False
    assert report["product_pass_claimed"] is False


def test_same_metrics_still_fail_when_closed_watertight_sign_contract_is_explicitly_claimed():
    report = _qual(sign=QUALIFIED_CLOSED_WATERTIGHT)
    assert report["pass"] is False
    gate = report["hard_gates"]["closed_watertight_global_sign"]
    assert gate["applicable"] is True
    assert gate["pass"] is False


def test_alpha_coverage_remains_hard_gate():
    metrics = _metrics()
    metrics["input_alpha_coverage"] = {"minimums": {"recall": 0.90, "precision": 0.99, "iou": 0.90}}
    report = _qual(metrics)
    assert report["pass"] is False
    assert report["hard_gates"]["input_alpha_coverage"]["pass"] is False


def test_observable_first_hit_geometry_remains_hard_gate():
    metrics = _metrics()
    metrics["observable_support_strata"]["support_1_2"]["p95_norm"] = 0.051
    report = _qual(metrics)
    assert report["pass"] is False
    assert report["hard_gates"]["observable_first_hit_geometry"]["pass"] is False


def test_teacher_geometry_at_product_inference_fails_closed():
    with pytest.raises(H1QualificationError, match="TEACHER_GEOMETRY"):
        qualify_h1_observable_surface_v1(
            _metrics(),
            product_inference_inputs=("8x1024_RGBA", "8xORTHOGRAPHIC_CAMERA"),
            teacher_mesh_used_at_inference=True,
            surface_sign_contract=OPEN_OR_MIXED_SURFACE,
        )


def test_input_contract_drift_fails_closed():
    with pytest.raises(H1QualificationError, match="PRODUCT_INPUT_CONTRACT_DRIFT"):
        qualify_h1_observable_surface_v1(
            _metrics(),
            product_inference_inputs=("teacher_mesh",),
            teacher_mesh_used_at_inference=False,
            surface_sign_contract=OPEN_OR_MIXED_SURFACE,
        )
