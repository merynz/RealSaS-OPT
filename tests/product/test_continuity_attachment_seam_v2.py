from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.continuity_attachment_seam_v2 import (
    ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS,
    AttachmentSeamPolicyV2,
    assert_attachment_seam_measurement,
    qualify_attachment_seam_continuity_measurement,
    qualify_attachment_seam_sample_set,
)
from compiler.realsas_compiler_core.continuity_underlay import QualifiedContinuityUnderlayIR
from compiler.realsas_compiler_core.types import QualificationError


def _underlay():
    return QualifiedContinuityUnderlayIR(
        view_index=0,
        substrate_component_id="BODY",
        substrate_component_state_hash="component",
        mesh_lineage_hash="mesh",
        mesh_skin_lineage_hash="skin",
        appearance_lineage_hash="appearance",
        external_render_support_qualification_hash="external",
        materialization_manifest_sha256="materialization",
        direct_binding_manifest_sha256="binding",
        component_assembly_hash="assembly",
        partition_authority_sha256="partition",
        deformable_component_ids=("BODY",),
        foreground_component_ids=("BOOK",),
        partition_membership_hash="partition-membership",
        qualification_hash="underlay",
        metadata={},
    )


def _sample_set(**overrides):
    args = dict(
        view_index=0,
        underlay=_underlay(),
        component_assembly_hash="assembly",
        mechanical_candidate_count=100,
        parent_supported_candidate_count=80,
        rest_closed_sample_count=76,
        parent_support_evidence_sha256="parent-support",
        rest_closure_evidence_sha256="rest-closure",
        sample_set_sha256="sample-set",
    )
    args.update(overrides)
    return qualify_attachment_seam_sample_set(**args)


def test_rest_open_pairs_are_excluded_not_baseline_subtracted():
    sample_set = _sample_set()
    value = qualify_attachment_seam_continuity_measurement(
        sample_set=sample_set,
        underlay=_underlay(),
        composed_raster_sha256="raster",
        newly_exposed_pixel_count=1,
        evaluated_closed_seam_pixel_count=100,
        foreground_alpha_authority_sha256="alpha",
    )
    assert value.passed is True
    assert value.metadata["measurement_semantics"] == ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS
    assert value.metadata["rest_open_pairs_are_not_seams"] is True
    assert value.metadata["rest_baseline_exposure_is_subtracted"] is False
    assert_attachment_seam_measurement(value, sample_set=sample_set, underlay=_underlay())


def test_dynamic_threshold_is_not_relaxed():
    sample_set = _sample_set()
    value = qualify_attachment_seam_continuity_measurement(
        sample_set=sample_set,
        underlay=_underlay(),
        composed_raster_sha256="raster",
        newly_exposed_pixel_count=3,
        evaluated_closed_seam_pixel_count=100,
        foreground_alpha_authority_sha256="alpha",
    )
    assert value.passed is False
    assert value.max_allowed_exposed_seam_fraction == pytest.approx(0.02)


def test_fail_closed_when_too_few_parent_supported_samples():
    with pytest.raises(QualificationError, match="PARENT_SUPPORTED_SAMPLE_COUNT_TOO_LOW"):
        _sample_set(parent_supported_candidate_count=12, rest_closed_sample_count=12)


def test_fail_closed_when_rest_closed_fraction_is_too_low():
    with pytest.raises(QualificationError, match="REST_CLOSED_FRACTION_TOO_LOW"):
        _sample_set(parent_supported_candidate_count=80, rest_closed_sample_count=60)


def test_carrier_quad_cannot_be_used_as_occupancy():
    sample_set = _sample_set()
    with pytest.raises(QualificationError, match="CARRIER_MESH_OCCUPANCY_FORBIDDEN"):
        qualify_attachment_seam_continuity_measurement(
            sample_set=sample_set,
            underlay=_underlay(),
            composed_raster_sha256="raster",
            newly_exposed_pixel_count=0,
            evaluated_closed_seam_pixel_count=100,
            foreground_alpha_authority_sha256="alpha",
            foreground_carrier_mesh_used_as_occupancy=True,
        )


def test_policy_parent_weight_is_fixed_majority():
    policy = AttachmentSeamPolicyV2()
    assert policy.min_body_parent_weight == pytest.approx(0.5)
    assert policy.max_new_exposed_fraction == pytest.approx(0.02)
