import json
from pathlib import Path


def test_observed_snake_failure_is_preoptimizer_and_not_scientific_fail():
    row = json.loads(
        Path("canonical/VNEXT_DIRECT_SURFACE_EXACT_CONTAINMENT_FAILURE_20261021.json")
        .read_text(encoding="utf-8")
    )
    assert row["status"] == "OBSERVED_PRE_OPTIMIZER_LEGACY_PROXY_FAILURE__NOT_SCIENTIFIC_FAIL"
    assert row["execution_source_sha"] == "39476c024b09fa61bc5bc3e0404123204afeed50"
    assert row["subject_id"] == "Snake"
    assert row["stage"] == "DECODE_T512_T1024_STARTED"
    assert row["worker_return_code"] == 3
    assert row["worker_signal"] is None
    assert row["exception_type"] == "SparseSurfaceTruncationError"
    assert row["exception_message"] == (
        "SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:62735467 boundary crossing tetrahedra"
    )
    assert row["optimizer_steps_executed"] == 0
    assert row["backward_steps_executed"] == 0
    assert row["scientific_ab_court_executed"] is False
    assert row["scientific_pass_or_fail_observed"] is False
    assert row["required_correction"]["forbid_blind_proxy_disable"] is True
