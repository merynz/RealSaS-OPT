from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.geometry_substrate_v2 import (
    GeometrySubstrateViewIR,
    build_geometry_substrate_qualification,
    geometry_substrate_evidence_from_dict,
    geometry_substrate_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError


def _failed_value():
    rows = tuple(
        GeometrySubstrateViewIR(
            view_index=i,
            silhouette_recall=0.998,
            silhouette_precision=0.98,
            largest_coherent_hole_fraction=0.00036,
            interior_uncovered_fraction=0.00028,
            source_foreground_pixel_count=1000,
            predicted_foreground_pixel_count=1018,
            component_recall=0.998,
            silhouette_edge_p95_px=3.0,
            passed=False,
            metadata={"fixture": True},
        )
        for i in range(8)
    )
    return build_geometry_substrate_qualification(
        zero_surface_binding_hash="z" * 64,
        observation_set_binding_hash="o" * 64,
        camera_set_binding_hash="c" * 64,
        normalization_binding_hash="n" * 64,
        policy={
            "min_recall": 0.999,
            "min_precision": 0.999,
            "max_largest_coherent_hole_fraction": 0.00025,
            "max_interior_uncovered_fraction": 0.0005,
            "min_component_recall": 0.999,
            "component_min_foreground_fraction": 0.0,
            "max_silhouette_edge_p95_px": 0.5,
        },
        views=rows,
        metadata={"product_authority_claimed": False},
    )


def test_failed_geometry_can_be_read_as_measurement_evidence_only():
    value = _failed_value()
    decoded = geometry_substrate_evidence_from_dict(value.to_dict())
    assert decoded.substrate_hash == value.substrate_hash
    assert decoded.qualification_report["every_view_passed"] is False
    assert decoded.qualification_report["status"] == "FAIL_GEOMETRY_SUBSTRATE"


def test_failed_geometry_remains_rejected_as_qualified_substrate():
    value = _failed_value()
    with pytest.raises(QualificationError, match="GEOMETRY_SUBSTRATE_HAS_FAILED_VIEW"):
        geometry_substrate_from_dict(value.to_dict())


def test_evidence_decoder_still_rejects_hash_drift():
    payload = _failed_value().to_dict()
    payload["substrate_hash"] = "0" * 64
    with pytest.raises(QualificationError, match="GEOMETRY_SUBSTRATE_HASH_MISMATCH"):
        geometry_substrate_evidence_from_dict(payload)
