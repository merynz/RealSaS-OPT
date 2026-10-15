import pytest

from tools.corpus.fit8_fbx_static_carrier_v1 import (
    MATERIAL_SCOPE,
    OWNER_SCOPE,
    PROBE_SCHEMA,
    qualify_ufbx_probe_payload_v1,
)


SOURCE_SHA = "1" * 64
PROBE_SHA = "2" * 64
RECEIPT_SHA = "3" * 64


def _payload():
    return {
        "schema": PROBE_SCHEMA,
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "source_polygons": [[0, 1, 2, 3]],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "triangle_polygon_id": [0, 0],
        "face_material_id": [7, 7],
        "face_owner_id": [11, 11],
        "vertex_count": 4,
        "polygon_count": 1,
        "material_count": 1,
        "bone_count": 1,
        "triangulation_proposal": "UFBX_TRIANGULATE_FACE_REQUIRES_REALSAS_QUALIFICATION",
        "material_scope": MATERIAL_SCOPE,
        "owner_scope": OWNER_SCOPE,
        "native_motion_names": ["Idle"],
    }


def test_valid_quad_triangulation_is_admitted_and_labeled():
    manifest, arrays = qualify_ufbx_probe_payload_v1(
        _payload(),
        source_sha256=SOURCE_SHA,
        probe_executable_sha256=PROBE_SHA,
    )
    assert manifest["claims"]["triangulation_topology_qualified"] is True
    assert manifest["claims"]["material_id_source_exact"] is True
    assert manifest["claims"]["mechanical_owner_diagnostic_only"] is True
    assert manifest["claims"]["semantic_owner_authority"] is False
    assert manifest["claims"]["ufbx_revision_verified"] is False
    assert manifest["vertex_count"] == 4
    assert manifest["face_count"] == 2
    assert manifest["component_count"] == 1
    assert arrays["face_material_id"].tolist() == [7, 7]
    assert arrays["face_owner_id"].tolist() == [11, 11]
    assert arrays["face_component_id"].tolist() == [0, 0]


def test_bad_polygon_triangulation_fails_closed():
    payload = _payload()
    payload["faces"] = [[0, 1, 2], [0, 1, 3]]
    with pytest.raises(ValueError, match="TRIANGULATION_NOT_QUALIFIED"):
        qualify_ufbx_probe_payload_v1(
            payload,
            source_sha256=SOURCE_SHA,
            probe_executable_sha256=PROBE_SHA,
        )


def test_revision_verification_requires_build_receipt():
    with pytest.raises(ValueError, match="REVISION_VERIFIED_WITHOUT_BUILD_RECEIPT"):
        qualify_ufbx_probe_payload_v1(
            _payload(),
            source_sha256=SOURCE_SHA,
            probe_executable_sha256=PROBE_SHA,
            build_receipt_verifies_expected_ufbx_revision=True,
        )


def test_revision_verification_can_be_bound_to_receipt_sha():
    manifest, _ = qualify_ufbx_probe_payload_v1(
        _payload(),
        source_sha256=SOURCE_SHA,
        probe_executable_sha256=PROBE_SHA,
        build_receipt_sha256=RECEIPT_SHA,
        build_receipt_verifies_expected_ufbx_revision=True,
    )
    assert manifest["build_receipt_sha256"] == RECEIPT_SHA
    assert manifest["claims"]["ufbx_revision_verified"] is True
