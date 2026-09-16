from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.continuity_contact_patch_v4 import (
    CONTACT_AUTHORITY,
    CONTACT_PATCH_SEMANTICS,
    FOREGROUND_ALPHA_AUTHORITY,
    FOREGROUND_CARRIER_ROLE,
    UNDERLAY_AUTHORITY,
    ContactPatchPolicyV4,
    assert_contact_patch_measurement,
    build_contact_patch_set,
    qualify_contact_patch,
    validate_contact_patch,
)
from compiler.realsas_compiler_core.types import QualificationError


def _patch(**overrides):
    kw = dict(
        view_index=0,
        component_id="BOOK_FOREGROUND",
        canonical_parent_joint_id="J:right",
        source_contact_pair_count=64,
        selected_contact_pair_count=20,
        rest_body_sample_count=48,
        local_body_face_count=24,
        selected_contact_min_socket_distance_px=3.5,
        source_owner_partition_sha256="a" * 64,
        foreground_alpha_authority_sha256="b" * 64,
        component_lineage_hash="c" * 64,
        contact_component_sha256="d" * 64,
        rest_body_sample_set_sha256="e" * 64,
        local_body_face_set_sha256="f" * 64,
    )
    kw.update(overrides)
    return qualify_contact_patch(**kw)


def test_contact_patch_semantics_are_source_contact_not_whole_boundary_or_socket_bridge():
    p = _patch()
    assert p.metadata["measurement_semantics"] == CONTACT_PATCH_SEMANTICS
    assert p.metadata["contact_authority"] == CONTACT_AUTHORITY
    assert p.metadata["underlay_authority"] == UNDERLAY_AUTHORITY
    assert p.metadata["foreground_alpha_authority"] == FOREGROUND_ALPHA_AUTHORITY
    assert p.metadata["foreground_carrier_role"] == FOREGROUND_CARRIER_ROLE
    assert p.metadata["owner_boundary_wholesale_promoted_to_mechanical_seam"] is False
    assert p.metadata["canonical_socket_promoted_to_raster_contact"] is False
    assert p.metadata["fixed_endpoint_pairing_used"] is False
    assert p.metadata["rest_background_required_zero"] is False


def test_contact_patch_forbids_carrier_quad_occupancy_and_fixed_endpoint_pairing():
    with pytest.raises(QualificationError, match="CARRIER_QUAD_OCCUPANCY_FORBIDDEN"):
        _patch(foreground_carrier_mesh_used_as_occupancy=True)
    with pytest.raises(QualificationError, match="FIXED_ENDPOINT_PAIRING_FORBIDDEN"):
        _patch(fixed_endpoint_pairing_used=True)


def test_contact_patch_requires_real_selected_source_contact_and_body_samples():
    with pytest.raises(QualificationError, match="CONTACT_PAIR_COUNTS_INVALID"):
        _patch(selected_contact_pair_count=0)
    with pytest.raises(QualificationError, match="REST_BODY_SAMPLE_COUNT_TOO_LOW"):
        _patch(rest_body_sample_count=7)


def test_contact_patch_hash_and_semantic_drift_fail_closed():
    p = _patch()
    validate_contact_patch(p)
    with pytest.raises(QualificationError, match="WHOLE_OWNER_BOUNDARY_FORBIDDEN"):
        validate_contact_patch(replace(p, metadata={**p.metadata, "owner_boundary_wholesale_promoted_to_mechanical_seam": True}))
    with pytest.raises(QualificationError, match="SOCKET_AS_RASTER_CONTACT_FORBIDDEN"):
        validate_contact_patch(replace(p, metadata={**p.metadata, "canonical_socket_promoted_to_raster_contact": True}))


def test_contact_patch_policy_cannot_relax_two_percent_gate():
    ContactPatchPolicyV4().validate()
    with pytest.raises(QualificationError, match="THRESHOLD_RELAXATION_FORBIDDEN"):
        ContactPatchPolicyV4(max_new_background_fraction=0.021).validate()


def test_contact_patch_measurement_and_component_set():
    policy = ContactPatchPolicyV4()
    s = build_contact_patch_set(view_index=0, patches=(_patch(),), policy=policy)
    assert assert_contact_patch_measurement(
        patch_set=s, newly_exposed_sample_count=1, evaluated_rest_body_sample_count=100, policy=policy
    ) == 0.01
    with pytest.raises(QualificationError, match="DYNAMIC_NEW_BACKGROUND"):
        assert_contact_patch_measurement(
            patch_set=s, newly_exposed_sample_count=3, evaluated_rest_body_sample_count=100, policy=policy
        )
