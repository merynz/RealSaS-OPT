from __future__ import annotations

from typing import Any, Mapping

from .continuity_underlay import (
    ContinuityRasterMeasurementIR,
    QualifiedContinuityUnderlayIR,
    qualify_continuity_raster_measurement,
)
from .types import QualificationError


PAIRED_INTERFACE_MEASUREMENT_SEMANTICS = "DEFORMED_PAIRED_INTERFACE_BACKGROUND_CRACK_V1"
EXACT_ENDPOINT_BINDING = "EXACT_TRIANGLE_BARYCENTRIC_FROM_REST_RASTER"
FOREGROUND_OCCUPANCY_AUTHORITY = "EXACT_SOURCE_OWNER_MASK_ATLAS_ALPHA_GE_8"
FOREGROUND_CARRIER_ROLE = "RIGID_TRANSFORM_COORDINATE_CARRIER_ONLY"
REQUIRED_FOREGROUND_ALPHA_THRESHOLD = 8


def qualify_paired_interface_continuity_measurement(
    *,
    view_index: int,
    underlay: QualifiedContinuityUnderlayIR,
    boundary_sample_set_sha256: str,
    composed_raster_sha256: str,
    exposed_seam_pixel_count: int,
    evaluated_boundary_pixel_count: int,
    max_allowed_exposed_seam_fraction: float,
    rest_calibration_exposed_fraction: float,
    foreground_alpha_authority_sha256: str,
    foreground_alpha_threshold: int = REQUIRED_FOREGROUND_ALPHA_THRESHOLD,
    foreground_carrier_mesh_used_as_occupancy: bool = False,
    max_allowed_rest_calibration_fraction: float = 0.005,
    endpoint_binding_coverage_fraction: float,
    min_endpoint_binding_coverage_fraction: float = 0.90,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuityRasterMeasurementIR:
    rest_fraction = float(rest_calibration_exposed_fraction)
    rest_limit = float(max_allowed_rest_calibration_fraction)
    coverage = float(endpoint_binding_coverage_fraction)
    min_coverage = float(min_endpoint_binding_coverage_fraction)
    alpha_authority = str(foreground_alpha_authority_sha256 or "")
    alpha_threshold = int(foreground_alpha_threshold)

    if not (0.0 <= rest_fraction <= 1.0 and 0.0 <= rest_limit <= 1.0):
        raise QualificationError("PAIRED_INTERFACE_INVALID_REST_CALIBRATION")
    if rest_fraction > rest_limit:
        raise QualificationError(
            f"PAIRED_INTERFACE_REST_CALIBRATION_FAILED:{rest_fraction}:{rest_limit}"
        )
    if not (0.0 <= coverage <= 1.0 and 0.0 < min_coverage <= 1.0):
        raise QualificationError("PAIRED_INTERFACE_INVALID_BINDING_COVERAGE")
    if coverage < min_coverage:
        raise QualificationError(
            f"PAIRED_INTERFACE_BINDING_COVERAGE_FAILED:{coverage}:{min_coverage}"
        )
    if not alpha_authority:
        raise QualificationError("PAIRED_INTERFACE_FOREGROUND_ALPHA_AUTHORITY_REQUIRED")
    if alpha_threshold != REQUIRED_FOREGROUND_ALPHA_THRESHOLD:
        raise QualificationError(
            f"PAIRED_INTERFACE_FOREGROUND_ALPHA_THRESHOLD_DRIFT:{alpha_threshold}:"
            f"{REQUIRED_FOREGROUND_ALPHA_THRESHOLD}"
        )
    if bool(foreground_carrier_mesh_used_as_occupancy):
        raise QualificationError("PAIRED_INTERFACE_CARRIER_MESH_OCCUPANCY_FORBIDDEN")

    supplied = dict(metadata or {})
    forbidden = {
        "measurement_semantics",
        "endpoint_binding_authority",
        "rest_calibration_exposed_fraction",
        "max_allowed_rest_calibration_fraction",
        "endpoint_binding_coverage_fraction",
        "min_endpoint_binding_coverage_fraction",
        "foreground_occupancy_authority",
        "foreground_alpha_authority_sha256",
        "foreground_alpha_threshold",
        "foreground_carrier_role",
        "foreground_carrier_mesh_used_as_occupancy",
    }
    if forbidden.intersection(supplied):
        raise QualificationError("PAIRED_INTERFACE_RESERVED_METADATA_OVERRIDE")

    return qualify_continuity_raster_measurement(
        view_index=int(view_index),
        underlay=underlay,
        boundary_sample_set_sha256=boundary_sample_set_sha256,
        composed_raster_sha256=composed_raster_sha256,
        exposed_seam_pixel_count=int(exposed_seam_pixel_count),
        evaluated_boundary_pixel_count=int(evaluated_boundary_pixel_count),
        max_allowed_exposed_seam_fraction=float(max_allowed_exposed_seam_fraction),
        metadata={
            "measurement_semantics": PAIRED_INTERFACE_MEASUREMENT_SEMANTICS,
            "endpoint_binding_authority": EXACT_ENDPOINT_BINDING,
            "rest_calibration_exposed_fraction": rest_fraction,
            "max_allowed_rest_calibration_fraction": rest_limit,
            "endpoint_binding_coverage_fraction": coverage,
            "min_endpoint_binding_coverage_fraction": min_coverage,
            "foreground_occupancy_authority": FOREGROUND_OCCUPANCY_AUTHORITY,
            "foreground_alpha_authority_sha256": alpha_authority,
            "foreground_alpha_threshold": alpha_threshold,
            "foreground_carrier_role": FOREGROUND_CARRIER_ROLE,
            "foreground_carrier_mesh_used_as_occupancy": False,
            "foreground_pixel_requires_body_underlay_at_same_pixel": False,
            **supplied,
        },
    )


def assert_paired_interface_measurement(value: ContinuityRasterMeasurementIR) -> None:
    if value.metadata.get("measurement_semantics") != PAIRED_INTERFACE_MEASUREMENT_SEMANTICS:
        raise QualificationError("PAIRED_INTERFACE_MEASUREMENT_SEMANTICS_DRIFT")
    if value.metadata.get("endpoint_binding_authority") != EXACT_ENDPOINT_BINDING:
        raise QualificationError("PAIRED_INTERFACE_ENDPOINT_BINDING_DRIFT")
    if bool(value.metadata.get("foreground_pixel_requires_body_underlay_at_same_pixel", True)):
        raise QualificationError("PAIRED_INTERFACE_LEGACY_FOREGROUND_UNDERLAY_TEST_FORBIDDEN")
    rest = float(value.metadata.get("rest_calibration_exposed_fraction", 1.0))
    rest_limit = float(value.metadata.get("max_allowed_rest_calibration_fraction", -1.0))
    if rest_limit < 0.0 or rest > rest_limit:
        raise QualificationError("PAIRED_INTERFACE_REST_CALIBRATION_DRIFT")
    coverage = float(value.metadata.get("endpoint_binding_coverage_fraction", -1.0))
    minimum = float(value.metadata.get("min_endpoint_binding_coverage_fraction", 2.0))
    if coverage < minimum:
        raise QualificationError("PAIRED_INTERFACE_BINDING_COVERAGE_DRIFT")
    if value.metadata.get("foreground_occupancy_authority") != FOREGROUND_OCCUPANCY_AUTHORITY:
        raise QualificationError("PAIRED_INTERFACE_FOREGROUND_OCCUPANCY_AUTHORITY_DRIFT")
    if not str(value.metadata.get("foreground_alpha_authority_sha256") or ""):
        raise QualificationError("PAIRED_INTERFACE_FOREGROUND_ALPHA_AUTHORITY_MISSING")
    if int(value.metadata.get("foreground_alpha_threshold", -1)) != REQUIRED_FOREGROUND_ALPHA_THRESHOLD:
        raise QualificationError("PAIRED_INTERFACE_FOREGROUND_ALPHA_THRESHOLD_DRIFT")
    if value.metadata.get("foreground_carrier_role") != FOREGROUND_CARRIER_ROLE:
        raise QualificationError("PAIRED_INTERFACE_FOREGROUND_CARRIER_ROLE_DRIFT")
    if bool(value.metadata.get("foreground_carrier_mesh_used_as_occupancy", True)):
        raise QualificationError("PAIRED_INTERFACE_CARRIER_MESH_OCCUPANCY_DRIFT")


__all__ = [
    "PAIRED_INTERFACE_MEASUREMENT_SEMANTICS",
    "EXACT_ENDPOINT_BINDING",
    "FOREGROUND_OCCUPANCY_AUTHORITY",
    "FOREGROUND_CARRIER_ROLE",
    "REQUIRED_FOREGROUND_ALPHA_THRESHOLD",
    "qualify_paired_interface_continuity_measurement",
    "assert_paired_interface_measurement",
]
