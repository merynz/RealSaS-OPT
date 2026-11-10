from __future__ import annotations

import pytest

from tools.corpus.build_corpus370_v5_geometry_family_evidence_v1 import claim


def row(surface="QUALIFIED_CORE_CANDIDATE",normal="QUALIFIED_CORE_CANDIDATE",face_count=10,valid_count=10):
    return {"family_candidate_outcomes":{"SURFACE_GEOMETRY":surface,"ORIENTED_NORMALS":normal},
            "counts":{"face_count":face_count,"valid_face_count":valid_count,"query_count":8192}}


M={"path":"/evidence/manifest","sha256":"a"*64,"schema":"RealSaS.ComponentGeometryAuthority.v1"}
C={"path":"/evidence/court","sha256":"b"*64,"schema":"RealSaS.ComponentGeometryAuthorityCourt.v1"}


def test_source_surface_udf_admitted_as_candidate_only():
    r=claim("SURFACE_GEOMETRY_UDF",row(),M,C)
    assert r["status"]=="PASS"
    assert r["metadata"]["global_solid_sign_proven"] is False
    assert r["metadata"]["tessa_mechanical_surface_teacher_proven"] is False
    assert r["metadata"]["training_authorized"] is False


def test_partial_valid_source_faces_need_mask():
    r=claim("SURFACE_NORMALS",row(normal="QUALIFIED_MASKED_CANDIDATE",valid_count=9),M,C)
    assert r["status"]=="PASS"
    assert r["metadata"]["row_level_validity_mask_required"] is True
    assert r["metadata"]["qualified_row_count"]==9


def test_component_sign_cannot_be_promoted_to_global_sdf():
    with pytest.raises(ValueError,match="V5_GEOMETRY_EVIDENCE_FAMILY"):
        claim("GLOBAL_SOLID_SDF",row(),M,C)


def test_zero_valid_rows_never_admitted():
    with pytest.raises(ValueError,match="V5_GEOMETRY_EVIDENCE_NO_VALID_ROW"):
        claim("SURFACE_NORMALS",row(normal="QUALIFIED_MASKED_CANDIDATE",valid_count=0),M,C)


def test_unavailable_target_never_admitted():
    with pytest.raises(ValueError,match="V5_GEOMETRY_EVIDENCE_UNQUALIFIED_FAMILY"):
        claim("SURFACE_NORMALS",row(normal="UNAVAILABLE_CANDIDATE"),M,C)
