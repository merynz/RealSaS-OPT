import numpy as np
import pytest

from tools.corpus.fit8_static_ray_census_v1 import StaticRayCensusPolicyV1
from tools.corpus.fit8_static_ray_census_v2 import build_static_ray_census_v2


def _tetra():
    vertices = np.asarray(
        [[0.0, 0.0, 0.8], [-0.8, -0.6, -0.6], [0.8, -0.6, -0.6], [0.0, 0.8, -0.6]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    material = np.asarray([1, 1, 2, 2], dtype=np.int32)
    owner = np.asarray([7, 7, 9, 9], dtype=np.int32)
    return vertices, faces, material, owner


def test_diagnostic_owner_labels_are_not_promoted_to_authority():
    vertices, faces, material, owner = _tetra()
    summary, _ = build_static_ray_census_v2(
        subject_key="UNIT_DIAGNOSTIC_OWNER",
        vertices=vertices,
        faces=faces,
        face_material_id=material,
        face_owner_id=owner,
        policy=StaticRayCensusPolicyV1(resolution=64, provenance="owner authority guard"),
        owner_authority_state="DIAGNOSTIC",
    )
    claims = summary["claims"]
    assert summary["schema"] == "RealSaS.FIT8StaticRayCensus.v2"
    assert summary["parent_static_ray_schema"] == "RealSaS.FIT8StaticRayCensus.v1"
    assert claims["mechanical_owner_labels_present"] is True
    assert claims["mechanical_owner_authority_state"] == "DIAGNOSTIC"
    assert claims["mechanical_owner_authority"] is False
    assert claims["mechanical_owner_diagnostic_only"] is True
    assert claims["owner_authority_inferred_from_label_presence"] is False


def test_unspecified_owner_labels_remain_unqualified():
    vertices, faces, material, owner = _tetra()
    summary, _ = build_static_ray_census_v2(
        subject_key="UNIT_UNSPECIFIED_OWNER",
        vertices=vertices,
        faces=faces,
        face_material_id=material,
        face_owner_id=owner,
        policy=StaticRayCensusPolicyV1(resolution=64, provenance="owner authority guard"),
    )
    claims = summary["claims"]
    assert claims["mechanical_owner_authority_state"] == "UNSPECIFIED"
    assert claims["mechanical_owner_authority"] is False
    assert claims["mechanical_owner_diagnostic_only"] is False


def test_qualified_owner_authority_requires_labels():
    vertices, faces, material, _ = _tetra()
    with pytest.raises(ValueError, match="OWNER_AUTHORITY_WITHOUT_LABELS"):
        build_static_ray_census_v2(
            subject_key="UNIT_NO_OWNER",
            vertices=vertices,
            faces=faces,
            face_material_id=material,
            face_owner_id=None,
            policy=StaticRayCensusPolicyV1(resolution=64, provenance="owner authority guard"),
            owner_authority_state="QUALIFIED",
        )


def test_invalid_owner_authority_state_fails_closed():
    vertices, faces, material, owner = _tetra()
    with pytest.raises(ValueError, match="OWNER_AUTHORITY_STATE_INVALID"):
        build_static_ray_census_v2(
            subject_key="UNIT_BAD_OWNER_STATE",
            vertices=vertices,
            faces=faces,
            face_material_id=material,
            face_owner_id=owner,
            policy=StaticRayCensusPolicyV1(resolution=64, provenance="owner authority guard"),
            owner_authority_state="SOURCE_EXACT",
        )
