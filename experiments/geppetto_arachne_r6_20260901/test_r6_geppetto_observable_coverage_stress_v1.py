from __future__ import annotations

import json

import numpy as np
import torch

from experiments.geppetto_arachne_r6_20260901.oracle_substrate_coverage_stress_v1 import (
    build_stressed_u1_surface,
)
from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import build_u0_surface
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import WITNESSES
from experiments.geppetto_arachne_r6_20260901.test_r6_oracle_substrate_geppetto_one_family_v1 import (
    REQUIRED_STABLE,
    _fit_arm,
)


WITNESS = next(w for w in WITNESSES if w.name == "branch_blend_4")


def _passed(result: dict) -> bool:
    return result["pass_step"] is not None and int(result["stable_passes"]) >= REQUIRED_STABLE


def test_r6_geppetto_observable_surface_coverage_stress_measurement() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(WITNESS.seed)
    np.random.seed(WITNESS.seed)

    u0, samples, cameras, visibility = build_u0_surface(WITNESS)
    u0_result = _fit_arm("U0_REFERENCE_FULL_SURFACE", u0, WITNESS, WITNESS.seed)
    print("R6_G_COVERAGE_STRESS_U0=" + json.dumps(u0_result, sort_keys=True))

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
        assert all("reference_full_surface" not in n.metadata for n in surface.surface_nodes)
        assert all(
            n.derived_normal is None
            or n.metadata.get("derived_normal_operator_id") == "RealSaS.DTB-ND1.RobustLocalPlane.v1"
            for n in surface.surface_nodes
        )

        result = _fit_arm(
            f"U1_OBSERVATION_ORACLE_GSA_{stress_name}",
            surface,
            WITNESS,
            WITNESS.seed,
        )
        record = {"stress": stress_name, "telemetry": telemetry, "result": result, "passed": _passed(result)}
        records.append(record)
        print("R6_G_COVERAGE_STRESS_RESULT=" + json.dumps(record, sort_keys=True))

    by_name = {r["stress"]: r for r in records}
    u0_pass = _passed(u0_result)
    core75_pass = bool(by_name["CORE_75"]["passed"])
    core60_pass = bool(by_name["CORE_60"]["passed"])
    if not u0_pass:
        verdict = "APPARATUS_INVALID_U0_FAIL"
    elif not core75_pass:
        verdict = "COVERAGE_SENSITIVE_CORE75_FAIL"
    elif not core60_pass:
        verdict = "SUBSTANTIAL_TOLERANCE_BUT_BOUNDARY_BETWEEN_CORE75_AND_CORE60"
    else:
        verdict = "STRONG_SYNTHETIC_GSA_GEPPETTO_COVERAGE_TOLERANCE"

    summary = {
        "u0_pass": u0_pass,
        "core75_pass": core75_pass,
        "core60_pass": core60_pass,
        "verdict": verdict,
        "records": records,
    }
    print("R6_G_COVERAGE_STRESS_SUMMARY=" + json.dumps(summary, sort_keys=True))

    # Integrity/contrast assertions only. Consumer PASS/FAIL is the measured outcome,
    # classified above per the preregistration rather than hidden by test control flow.
    assert u0_pass, summary
    r75 = float(by_name["CORE_75"]["telemetry"]["retained_fraction_vs_u0"])
    r60 = float(by_name["CORE_60"]["telemetry"]["retained_fraction_vs_u0"])
    assert 0.70 <= r75 <= 0.80, summary
    assert 0.55 <= r60 <= 0.65, summary
    assert r60 < r75 < 0.90, summary
