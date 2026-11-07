"""Synthetic preflight for the real 370-subject semantic source-group sweep."""
import hashlib
import json
from pathlib import Path
import pytest

from tools.corpus.run_corpus370_semantic_source_group_sweep_v1 import preflight


ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / "canonical/REALSAS_SEMANTIC_PRIOR_POLICY_V1.json").read_text())
HINTS = json.loads((ROOT / "canonical/REALSAS_SEMANTIC_SOURCE_HINT_POLICY_V1.json").read_text())
SHA = "a" * 64


def _fixture():
    subjects = [{"subject_id": f"subject_{i:03d}", "source_sha256": SHA}
                for i in range(370)]
    v5 = {
        "schema": "RealSaS.Corpus370PreparedObservationPackets.v5",
        "subject_count": 370, "prepared_subject_count": 352,
        "failure_count": 18, "training_authorized": False,
        "subjects": [dict(x) for x in subjects[:352]],
        "failures": [dict(x) for x in subjects[352:]],
    }
    roster = {
        "schema": "RealSaS.Corpus370ExactSourceRoster.v1",
        "subject_count": 370, "training_authorized": False,
        "subjects": [dict(x) for x in subjects],
    }
    rest = {
        "schema": "RealSaS.Corpus370V5SourceRestIndependentCourtSweep.v1",
        "observation_qualified_count": 352, "observation_failed_count": 18,
        "training_authorized": False,
        "subjects": {row["subject_id"]: {
            "status": "PASS_SOURCE_REST_INDEPENDENT_COURT",
            "training_authorized": False,
        } for row in subjects[:352]},
    }
    return v5, roster, rest


def test_all_original_370_rows_partition_and_prior_version():
    source, qualified, excluded = preflight(*_fixture(), POLICY, HINTS)
    assert (len(source), len(qualified), len(excluded)) == (370, 352, 18)
    assert not set(qualified) & set(excluded)
    assert POLICY["concepts"]["HAIR_LIKE"]["prior"]["SECONDARY_MOTION_ELIGIBLE"]
    assert POLICY["concepts"]["UNKNOWN"]["prior"] == {}


def test_missing_observation_subject_cannot_vanish():
    v5, roster, rest = _fixture()
    v5["failures"].pop()
    with pytest.raises(ValueError, match="V5_COHORT|UNIVERSE_PARTITION"):
        preflight(v5, roster, rest, POLICY, HINTS)


def test_forged_student_admission_is_rejected():
    v5, roster, rest = _fixture()
    v5["training_authorized"] = True
    with pytest.raises(ValueError, match="V5_COHORT"):
        preflight(v5, roster, rest, POLICY, HINTS)
    v5, roster, rest = _fixture()
    rest["training_authorized"] = True
    with pytest.raises(ValueError, match="REST_INDEX"):
        preflight(v5, roster, rest, POLICY, HINTS)


def test_source_sha_drift_and_missing_rest_are_rejected():
    v5, roster, rest = _fixture()
    v5["subjects"][0]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SOURCE_SHA_MATCH"):
        preflight(v5, roster, rest, POLICY, HINTS)
    v5, roster, rest = _fixture()
    rest["subjects"].pop("subject_000")
    with pytest.raises(ValueError, match="REST_SUBJECT_SET"):
        preflight(v5, roster, rest, POLICY, HINTS)
