from __future__ import annotations

"""Depth ownership is dynamic geometry, not source visibility or color."""

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.pose_depth_raster_v1 import (
    rasterize_pose_depth_v1,
)
from compiler.realsas_compiler_core.pose_depth_visibility_v1 import (
    inspect_pose_depth_pixel_v1,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_controllable_puppet_runtime_v1 import (
    qualified_synthetic_bundle,
    _compiled_graphics_consumed_from_same_synthetic_canonical_mesh,
)

FAR = ((0., 0., 3.), (3., 0., 3.), (0., 3., 3.))
NEAR = ((0., 0., 1.), (3., 0., 1.), (0., 3., 1.))


def _raster(faces, labels, *, width=4, height=4, **kwargs):
    return rasterize_pose_depth_v1(
        view_id="V0", graphics_hash="graph-hash", projected_faces=faces,
        face_fragment_ids=labels, width=width, height=height, **kwargs,
    )


def test_depth_front_is_geometry_selected_not_triangle_array_order():
    first = _raster((FAR, NEAR), ("back", "front"))
    swapped = _raster((NEAR, FAR), ("front", "back"))
    assert first.front_face_by_pixel[0][0] == 1
    assert swapped.front_face_by_pixel[0][0] == 0
    assert first.front_fragment_by_pixel[0][0] == "front"
    assert swapped.front_fragment_by_pixel[0][0] == "front"
    assert first.nearest_depth_by_pixel[0][0] == 1.0
    assert first.unique_front_pixel_count > 0
    assert first.ambiguous_front_pixel_count == 0
    assert first.empty_pixel_count > 0
    oracle = inspect_pose_depth_pixel_v1(
        view_id="V0", graphics_hash="graph-hash",
        projected_faces=(FAR, NEAR), face_fragment_ids=("back", "front"),
        screen_x=.5, screen_y=.5,
    )
    assert oracle.geometrically_front_fragment_ids == ("front",)
    assert first.pose_projection_hash == oracle.pose_projection_hash
    assert first.raster_hash == content_sha256({
        k:v for k,v in first.to_dict().items() if k != "raster_hash"
    })


def test_occlusion_order_changes_with_pose_and_ties_remain_unknown():
    a = _raster((FAR, NEAR), ("arm", "body"))
    b = _raster((NEAR, FAR), ("arm", "body"))
    assert a.front_fragment_by_pixel[0][0] == "body"
    assert b.front_fragment_by_pixel[0][0] == "arm"
    tied = _raster((NEAR, NEAR), ("body", "tool"))
    assert tied.front_face_by_pixel[0][0] == -2
    assert tied.front_fragment_by_pixel[0][0] is None
    assert tied.ambiguous_front_pixel_count > 0
    for result in (a,b,tied):
        assert result.source_visibility_qualified is False
        assert result.source_alpha_qualified is False
        assert result.source_hidden_appearance_qualified is False
        assert result.pose_appearance_authority is False
        assert result.product_revision_minted is False


def test_raster_rejects_out_of_budget_inputs_and_bad_vertex_coordinates():
    with pytest.raises(QualificationError, match="FACE_PIXEL_WORK_BUDGET"):
        _raster((FAR, NEAR), ("back", "front"), width=64, height=64,
                max_face_pixel_tests=1)
    with pytest.raises(QualificationError, match="INPUT_DOMAIN_INVALID"):
        _raster((FAR,), ("front",), width=True)
    with pytest.raises(QualificationError, match="VERTEX_DOMAIN_INVALID"):
        invalid = ((0.,0.,1.), (float("nan"),0.,1.), (0.,2.,1.))
        _raster((invalid,), ("bad",))


def test_ik_studio_uses_the_current_deformed_mesh_for_full_depth_raster():
    bundle = qualified_synthetic_bundle()
    with pytest.raises(QualificationError, match="REQUIRES_GRAPH"):
        ControllablePuppetSessionV1(bundle).inspect_graphics_depth_raster(64,64)

    graph = _compiled_graphics_consumed_from_same_synthetic_canonical_mesh(bundle)
    session = ControllablePuppetSessionV1(replace(bundle, graphics=graph))
    rest = session.inspect_graphics_depth_raster(64,64)
    session.drag_world("C_ROOT", (0.5, 0.0, 0.0))
    moved = session.inspect_graphics_depth_raster(64,64)
    assert rest.view_id == moved.view_id == "V0"
    assert rest.graphics_hash == moved.graphics_hash == graph.runtime_graphics_hash
    assert rest.pose_projection_hash != moved.pose_projection_hash
    session.set_view_index(1)
    rotated = session.inspect_graphics_depth_raster(64,64)
    assert rotated.view_id == "V1"
    assert rotated.pose_projection_hash != moved.pose_projection_hash
    assert rotated.source_hidden_appearance_qualified is False
