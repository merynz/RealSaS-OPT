from __future__ import annotations

import pytest

from tools.corpus.requalify_corpus370_v5_source_camera_families_v1 import claim


def test_pass_requires_actual_pass_court_and_evidence():
    result=claim("SOURCE_IMPORT","PASS",[
        {"schema":"ProvenSource","sha256":"a"*64,"path":"/test/reference"}
    ],{"schema":"RealSaS.Corpus370V5SourceImportIndependentCourt.v1",
       "sha256":"b"*64,"path":"/test/court","status":"PASS"})
    assert result["status"]=="PASS"
    assert result["reason_codes"]==[]
    assert result["metadata"]["source_version"]=="V5_352_PLUS_18"


def test_forged_pass_status_fails():
    with pytest.raises(ValueError,match="V5_SOURCE_CAMERA_COURT_STATUS"):
        claim("CAMERA_OBSERVATION","PASS",[{"path":"/reference"}],
              {"status":"FAIL","path":"/court"})


def test_false_source_family_name_fails():
    with pytest.raises(ValueError,match="V5_SOURCE_CAMERA_FAMILY"):
        claim("IRIS_GLOBAL_SOLID_SDF","PASS",[{"path":"/reference"}],
              {"status":"PASS","path":"/court"})


def test_fail_without_reason_fails():
    with pytest.raises(ValueError,match="V5_SOURCE_CAMERA_CLAIM_EVIDENCE"):
        claim("CAMERA_OBSERVATION","FAIL",[{"path":"/reference"}],
              {"status":"FAIL","path":"/court"})


def test_fake_camera_masked_claim_is_rejected():
    with pytest.raises(ValueError,match="V5_SOURCE_CAMERA_STATUS"):
        claim("CAMERA_OBSERVATION","MASKED",[{"path":"/reference"}],
              {"status":"MASKED","path":"/court"})


def test_cannot_mint_pass_without_independent_proof():
    with pytest.raises(ValueError,match="V5_SOURCE_CAMERA_CLAIM_EVIDENCE"):
        claim("SOURCE_IMPORT","PASS",[],{"status":"PASS","path":"/court"})
