import json
from pathlib import Path

import pytest

from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256
from tools.truth_foundry.build_fit8_presentation_truth_bundle_v3 import bind_logical_attempts_v3


def _output(tmp_path: Path, attempt: str, schema: str, payload: dict):
    path = tmp_path / attempt / ("summary.json" if "Candidate" in schema else "exclusion.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "schema": schema,
        "authority_class": "TEST",
    }


def _observation(tmp_path: Path):
    attempts = [
        {"attempt_id": "a-snake", "subject_id": "uuid-snake", "role": "COHORT", "state": "COMPLETED"},
        {"attempt_id": "a-bat", "subject_id": "uuid-bat", "role": "COHORT", "state": "COMPLETED"},
        {"attempt_id": "a-knight", "subject_id": "uuid-knight", "role": "REGRESSION", "state": "COMPLETED"},
    ]
    exports = [
        {"attempt_id": "a-snake", "subject_id": "uuid-snake", "role": "COHORT", "outputs": [_output(tmp_path, "a-snake", "RealSaS.PresentationResponseTruthCandidate.v1", {"subject_id": "fit8_snake_v1"})]},
        {"attempt_id": "a-bat", "subject_id": "uuid-bat", "role": "COHORT", "outputs": [_output(tmp_path, "a-bat", "RealSaS.PresentationResponseTruthCandidate.v1", {"subject_id": "fit8_bat_v1"})]},
        {"attempt_id": "a-knight", "subject_id": "uuid-knight", "role": "REGRESSION", "outputs": [_output(tmp_path, "a-knight", "RealSaS.PresentationTruthRegressionExclusion.v1", {"subject_id": "SUBJECT2_KNIGHT", "new_truth_produced": False, "training_authorized": False})]},
    ]
    return {"attempts": attempts, "exports": exports}


def test_uuid_transport_identity_is_separate_from_logical_scientific_key(tmp_path):
    cohort, regression = bind_logical_attempts_v3(_observation(tmp_path), ("fit8_snake_v1", "fit8_bat_v1"), ("knight",))
    assert set(cohort) == {"fit8_snake_v1", "fit8_bat_v1"}
    assert cohort["fit8_snake_v1"]["platform_subject_id"] == "uuid-snake"
    assert cohort["fit8_snake_v1"]["attempt_id"] == "a-snake"
    assert regression["knight"]["platform_subject_id"] == "uuid-knight"
    assert regression["knight"]["producer_subject_id"] == "SUBJECT2_KNIGHT"
    assert regression["knight"]["logical_subject_id"] == "knight"


def test_duplicate_logical_cohort_key_is_rejected(tmp_path):
    observation = _observation(tmp_path)
    duplicate_path = Path(observation["exports"][1]["outputs"][0]["path"])
    duplicate_path.write_text(json.dumps({"subject_id": "fit8_snake_v1"}) + "\n")
    observation["exports"][1]["outputs"][0]["sha256"] = sha256(duplicate_path)
    observation["exports"][1]["outputs"][0]["size_bytes"] = duplicate_path.stat().st_size
    with pytest.raises(ValueError, match="FIT8_PRESENTATION_V3_LOGICAL_SUBJECT_DUPLICATE"):
        bind_logical_attempts_v3(observation, ("fit8_snake_v1", "fit8_bat_v1"), ("knight",))


def test_regression_exclusion_flags_are_fail_closed(tmp_path):
    observation = _observation(tmp_path)
    ref = observation["exports"][2]["outputs"][0]
    path = Path(ref["path"])
    path.write_text(json.dumps({"subject_id": "SUBJECT2_KNIGHT", "new_truth_produced": True, "training_authorized": False}) + "\n")
    ref["sha256"] = sha256(path)
    ref["size_bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="FIT8_PRESENTATION_BUNDLE_REGRESSION_EXCLUSION_DRIFT"):
        bind_logical_attempts_v3(observation, ("fit8_snake_v1", "fit8_bat_v1"), ("knight",))
