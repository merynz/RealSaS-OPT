from __future__ import annotations

import json
from pathlib import Path

from tools.corpus import run_truth139_v32_finalize_remaining_full139_v3 as runner


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_response_svd_rescue_must_bind_selected_response_chain(tmp_path: Path) -> None:
    court = tmp_path / "court.json"
    _write(court, {"status": "PASS"})

    chain = tmp_path / "chain.json"
    chain_value = {
        "bindings": {"response_court_identity_sha256": "c" * 64},
        "outputs": {"response_court": {"path": str(court), "sha256": runner.impl.sha256(court)}},
    }
    _write(chain, chain_value)

    response = tmp_path / "response.json"
    response_value = {
        "code_commit_sha": "a" * 40,
        "subject_output_bindings": {
            runner.RESCUE_SUBJECT: {
                "chain_path": str(chain),
                "chain_sha256": runner.impl.sha256(chain),
            }
        },
    }
    _write(response, response_value)

    rescue = tmp_path / "rescue.json"
    rescue_value = {
        "status": runner.RESCUE_STATUS,
        "subject_id": runner.RESCUE_SUBJECT,
        "base_repository_commit": "a" * 40,
        "patched_validator_sha256": runner.RESCUE_VALIDATOR_SHA256,
        "diagnosis": {
            "producer_truth_changed": False,
            "teacher_truth_changed": False,
            "tolerance_relaxed": False,
            "new_independent_formulation": "REST_TANGENT_SVD_GENERALIZED_PRINCIPAL_STRETCH",
        },
        "court": {
            "sha256": runner.impl.sha256(court),
            "court_identity_sha256": "c" * 64,
            "maximum_nonrigid_abs_error": 1.0e-12,
        },
    }
    _write(rescue, rescue_value)

    validated = runner._validate_response_rescue(rescue, response)
    assert validated["status"] == runner.RESCUE_STATUS

    rescue_value["diagnosis"]["tolerance_relaxed"] = True
    _write(rescue, rescue_value)
    try:
        runner._validate_response_rescue(rescue, response)
    except ValueError as exc:
        assert str(exc) == "V32_FINAL_RESPONSE_RESCUE_TOLERANCE_RELAXED"
    else:
        raise AssertionError("relaxed rescue tolerance must fail closed")
