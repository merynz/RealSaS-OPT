from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.continuity_interface_evidence import (
    FOREGROUND_CARRIER_ROLE,
    FOREGROUND_OCCUPANCY_AUTHORITY,
    PAIRED_INTERFACE_MEASUREMENT_SEMANTICS,
    REQUIRED_FOREGROUND_ALPHA_THRESHOLD,
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


def _qualify(**overrides):
    args = dict(
        view_index=0,
        underlay=_underlay(),
        boundary_sample_set_sha256="boundary",
        composed_raster_sha256="raster",
        exposed_seam_pixel_count=1,
        evaluated_boundary_pixel_count=100,
        max_allowed_exposed_seam_fraction=0.02,
        rest_calibration_exposed_fraction=0.0,
        endpoint_binding_coverage_fraction=0.97,
        foreground_alpha_authority_sha256="atlas-panel-alpha",
    )
    args.update(overrides)
    return qualify_paired_interface_continuity_measurement(**args)


def test_paired_interface_measurement_binds_semantics_and_owner_alpha_authority():
    value = _qualify()
    assert value.passed is True
    assert value.metadata["measurement_semantics"] == PAIRED_INTERFACE_MEASUREMENT_SEMANTICS
    assert value.metadata["foreground_pixel_requires_body_underlay_at_same_pixel"] is False
    assert value.metadata["foreground_occupancy_authority"] == FOREGROUND_OCCUPANCY_AUTHORITY
    assert value.metadata["foreground_carrier_role"] == FOREGROUND_CARRIER_ROLE
    assert value.metadata["foreground_alpha_threshold"] == REQUIRED_FOREGROUND_ALPHA_THRESHOLD
    assert value.metadata["foreground_carrier_mesh_used_as_occupancy"] is False
    assert_paired_interface_measurement(value)


def test_paired_interface_measurement_fails_closed_on_bad_rest_calibration():
    with pytest.raises(QualificationError, match="REST_CALIBRATION_FAILED"):
        _qualify(rest_calibration_exposed_fraction=0.20)


def test_paired_interface_measurement_fails_closed_on_low_endpoint_coverage():
    with pytest.raises(QualificationError, match="BINDING_COVERAGE_FAILED"):
        _qualify(endpoint_binding_coverage_fraction=0.50)


def test_paired_interface_measurement_requires_exact_owner_alpha_threshold():
    with pytest.raises(QualificationError, match="ALPHA_THRESHOLD_DRIFT"):
        _qualify(foreground_alpha_threshold=1)


def test_paired_interface_measurement_forbids_carrier_quad_as_occupancy():
    with pytest.raises(QualificationError, match="CARRIER_MESH_OCCUPANCY_FORBIDDEN"):
        _qualify(foreground_carrier_mesh_used_as_occupancy=True)


def test_paired_interface_measurement_requires_alpha_authority_hash():
    with pytest.raises(QualificationError, match="ALPHA_AUTHORITY_REQUIRED"):
        _qualify(foreground_alpha_authority_sha256="")
