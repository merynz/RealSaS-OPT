from __future__ import annotations

import json

from experiments.geppetto_arachne_r6_20260901.oracle_substrate_coverage_stress_v1 import (
    build_stressed_u1_surface,
)
from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import build_u0_surface
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import REQUIRED_STABLE, WITNESSES
from experiments.geppetto_arachne_r6_20260901.test_r6_oracle_substrate_arachne_one_family_v1 import (
    EXPECTED_SHIPPING_ARACHNE_HASH,
    EXPECTED_SHIPPING_CODEC_HASH,
    _run_arm,
)


WITNESS = next(w for w in WITNESSES if w.name == "branch_blend_4")


def _passed(result: dict) -> bool:
    if result.get("status") != "PASS":
        return False
    a0 = result.get("a0") or {}
    a1 = result.get("a1") or {}
    return (
        int(a0.get("stable_passes", 0)) >= REQUIRED_STABLE
        and int(a1.get("stable_passes", 0)) >= REQUIRED_STABLE
        and result.get("codec_config_hash") == EXPECTED_SHIPPING_CODEC_HASH
        and a1.get("config_hash") == EXPECTED_SHIPPING_ARACHNE_HASH
        and a1.get("codec_config_hash") == EXPECTED_SHIPPING_CODEC_HASH
    )


def test_r6_arachne_observable_surface_coverage_stress_measurement() -> None:
    _, samples, cameras, visibility = build_u0_surface(WITNESS)
    records = []
    for stress_name in ("CORE_75", "CORE_60"):
        surface, evidence, telemetry = build_stressed_u1_surface(
            WITNESS,
            samples,
            cameras,
            visibility,
            stress_name=stress_name,
        )
        assert evidence.metadata["coverage_stress_applied_before_gsa"] is True
        assert evidence.metadata["hidden_surface_completion"] is False
        assert evidence.metadata["source_mesh_consumer_input"] is False
        assert surface.builder_id == "RealSaS.GeometricSubstrateAssembler.current"

        result = _run_arm(
            WITNESS,
            f"U1_OBSERVATION_ORACLE_GSA_{stress_name}",
            surface,
            telemetry,
        )
        record = {"stress": stress_name, "telemetry": telemetry, "result": result, "passed": _passed(result)}
        records.append(record)
        print("R6_A_COVERAGE_STRESS_RESULT=" + json.dumps(record, sort_keys=True))

    by_name = {r["stress"]: r for r in records}
    core75_pass = bool(by_name["CORE_75"]["passed"])
    core60_pass = bool(by_name["CORE_60"]["passed"])
    if not core75_pass:
        verdict = "ARACHNE_COVERAGE_SENSITIVE_CORE75_FAIL"
    elif not core60_pass:
        verdict = "ARACHNE_SUBSTANTIAL_TOLERANCE_BUT_BOUNDARY_BETWEEN_CORE75_AND_CORE60"
    else:
        verdict = "STRONG_SYNTHETIC_GSA_ARACHNE_COVERAGE_TOLERANCE"
    summary = {
        "core75_pass": core75_pass,
        "core60_pass": core60_pass,
        "verdict": verdict,
        "records": records,
    }
    print("R6_A_COVERAGE_STRESS_SUMMARY=" + json.dumps(summary, sort_keys=True))

    r75 = float(by_name["CORE_75"]["telemetry"]["retained_fraction_vs_u0"])
    r60 = float(by_name["CORE_60"]["telemetry"]["retained_fraction_vs_u0"])
    assert 0.70 <= r75 <= 0.80, summary
    assert 0.55 <= r60 <= 0.65, summary
    assert r60 < r75 < 0.90, summary
