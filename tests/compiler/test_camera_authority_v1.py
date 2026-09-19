from __future__ import annotations

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.camera_authority_v1 import (
    build_qualified_camera_set,camera_projection_binding_hash,validate_qualified_camera_set,
)
from compiler.realsas_compiler_core.types import QualificationError


def _rows():
    rows=[]
    for i in range(8):
        rows.append({
            "schema_version":"RealSaS.FullSurfaceCameraProjection.v3",
            "view_id":f"V{i}",
            "view_index":i,
            "origin":[0.0,0.0,-2.0-float(i)],
            "right":[1.0,0.0,0.0],
            "screen_up":[0.0,1.0,0.0],
            "forward":[0.0,0.0,1.0],
            "half_extent":2.0,
            "resolution":64,
        })
    return rows


def test_camera_set_binds_exact_dense_eight_view_parameters():
    value=build_qualified_camera_set(_rows(),source_bundle_sha256="a"*64)
    assert len(value.cameras)==8
    assert value.camera_binding_hashes==tuple(camera_projection_binding_hash(c) for c in value.cameras)
    assert len(value.camera_set_hash)==64
    validate_qualified_camera_set(value)


def test_camera_set_rejects_frame_scale_drift():
    rows=_rows()
    rows[7]={**rows[7],"half_extent":2.5}
    with pytest.raises(QualificationError,match="COMMON_FRAME"):
        build_qualified_camera_set(rows,source_bundle_sha256="a"*64)


def test_camera_set_hash_changes_when_camera_parameter_changes():
    a=build_qualified_camera_set(_rows(),source_bundle_sha256="a"*64)
    rows=_rows()
    rows[7]={**rows[7],"origin":[0.0,0.0,-99.0]}
    b=build_qualified_camera_set(rows,source_bundle_sha256="a"*64)
    assert a.camera_set_hash!=b.camera_set_hash
    assert a.camera_binding_hashes[7]!=b.camera_binding_hashes[7]
