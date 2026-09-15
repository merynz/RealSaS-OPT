from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.continuity_interface_evidence import (
    PAIRED_INTERFACE_MEASUREMENT_SEMANTICS,
    assert_paired_interface_measurement,
    qualify_paired_interface_continuity_measurement,
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


def test_paired_interface_measurement_binds_semantics_and_rejects_legacy_same_pixel_meaning():
    value = qualify_paired_interface_continuity_measurement(
        view_index=0,
        underlay=_underlay(),
        boundary_sample_set_sha256="boundary",
        composed_raster_sha256="raster",
        exposed_seam_pixel_count=1,
        evaluated_boundary_pixel_count=100,
        max_allowed_exposed_seam_fraction=0.02,
        rest_calibration_exposed_fraction=0.0,
        endpoint_binding_coverage_fraction=0.97,
    )
    assert value.passed is True
    assert value.metadata["measurement_semantics"] == PAIRED_INTERFACE_MEASUREMENT_SEMANTICS
    assert value.metadata["foreground_pixel_requires_body_underlay_at_same_pixel"] is False
    assert_paired_interface_measurement(value)


def test_paired_interface_measurement_fails_closed_on_bad_rest_calibration():
    with pytest.raises(QualificationError, match="REST_CALIBRATION_FAILED"):
        qualify_paired_interface_continuity_measurement(
            view_index=0,
            underlay=_underlay(),
            boundary_sample_set_sha256="boundary",
            composed_raster_sha256="raster",
            exposed_seam_pixel_count=0,
            evaluated_boundary_pixel_count=100,
            max_allowed_exposed_seam_fraction=0.02,
            rest_calibration_exposed_fraction=0.20,
            endpoint_binding_coverage_fraction=0.97,
        )


def test_paired_interface_measurement_fails_closed_on_low_endpoint_coverage():
    with pytest.raises(QualificationError, match="BINDING_COVERAGE_FAILED"):
        qualify_paired_interface_continuity_measurement(
            view_index=0,
            underlay=_underlay(),
            boundary_sample_set_sha256="boundary",
            composed_raster_sha256="raster",
            exposed_seam_pixel_count=0,
            evaluated_boundary_pixel_count=100,
            max_allowed_exposed_seam_fraction=0.02,
            rest_calibration_exposed_fraction=0.0,
            endpoint_binding_coverage_fraction=0.50,
        )
