from __future__ import annotations

import json

import pytest

from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import (
    build_u0_surface,
    build_u1_surface,
    substrate_telemetry,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
)
from experiments.geppetto_arachne_r6_20260901.test_r6_oracle_substrate_arachne_one_family_v1 import (
    EXPECTED_SHIPPING_ARACHNE_HASH,
    EXPECTED_SHIPPING_CODEC_HASH,
    _run_arm,
)


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_r6_arachne_heterogeneous_u1_observation_oracle(witness: Witness):
    u0, samples, cameras, visibility = build_u0_surface(witness)
    u1, _ = build_u1_surface(witness, samples, cameras, visibility)
    telemetry = substrate_telemetry(u0, u1, visibility)

    result = _run_arm(witness, "U1_OBSERVATION_ORACLE_SUBSTRATE", u1, telemetry)
    print("R6_A1_HET_U1_RESULT=" + json.dumps(result, sort_keys=True))

    assert result["status"] == "PASS", result
    assert result["codec_config_hash"] == EXPECTED_SHIPPING_CODEC_HASH, result
    assert result["a0"]["stable_passes"] >= REQUIRED_STABLE, result
    assert result["a1"]["config_hash"] == EXPECTED_SHIPPING_ARACHNE_HASH, result
    assert result["a1"]["codec_config_hash"] == EXPECTED_SHIPPING_CODEC_HASH, result
    assert result["a1"]["stable_passes"] >= REQUIRED_STABLE, result
