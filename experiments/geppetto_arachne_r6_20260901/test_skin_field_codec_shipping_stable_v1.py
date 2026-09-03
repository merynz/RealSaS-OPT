from __future__ import annotations

import json

import pytest

from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
)
from experiments.geppetto_arachne_r6_20260901.test_skin_field_codec_shipping_cooling_diagnostic_v1 import (
    _run_shipping_lane,
)


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_shipping_codec_stable_cosine_protocol_through_compiler_and_lbs(witness: Witness):
    result = _run_shipping_lane(witness, cosine=True, lane="SHIPPING_A0_GENERIC_COSINE_V1")
    print("SKIN_FIELD_CODEC_SHIPPING_STABLE_V1=" + json.dumps(result, sort_keys=True))

    assert result["codec_config_hash"] == "24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715"
    assert result["scheduler"] == "COSINE_TO_ZERO"
    assert result["pass_step"] is not None, result
    assert result["stable_passes"] >= REQUIRED_STABLE, result
    assert result["final"]["compiler_total_correction_l1"] <= 1e-5, result
