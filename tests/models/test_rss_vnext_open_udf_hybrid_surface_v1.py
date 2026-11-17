from __future__ import annotations

"""Untrained analytic open sheet only: structural regression, not product proof."""

from dataclasses import replace
import pytest
import torch

from models.rss_vnext.open_udf_surface_v1 import (
    extract_open_udf_surface_v1,
)
from models.rss_vnext.hybrid_signed_open_iris_seed_v1 import (
    join_hybrid_iris_seed_v1,OPEN_KIND,SIGNED_KIND,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,tessa_adaptive_split_edges_v1,
)
from models.rss_vnext.open_udf_surface_v1 import tensor_bytes_sha256_v1


def _scene():
    return torch.ones((1,3,4,4,4),dtype=torch.float32)


def _open_plane(scene,*,shift=None):
    def field(x):
        z=x[:,2] if shift is None else x[:,2]-shift
        return z.abs()
    return extract_open_udf_surface_v1(
        scene_triplanes=scene,
        query_unsigned_distance=field,
        resolution=(12,12,12),
        query_chunk=256,
        max_active_cubes=4096,
    )


def _closed_tetra(scene):
    xyz=torch.tensor([[
        [-.4,-.4,-.4],[.4,-.4,-.4],[0.,.4,-.4],[0.,0.,.4],
    ]],dtype=torch.float32)
    faces=torch.tensor([
        [0,2,1],[0,1,3],[1,2,3],[2,0,3],
    ],dtype=torch.long)
    return build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,iris_surface_faces=faces,
        component_ids_by_vertex=torch.zeros(4,dtype=torch.long),
        scene_sha256=tensor_bytes_sha256_v1(scene),
        original_surface_sha256="a"*64,
    )


def test_open_plane_from_pure_unsigned_distance_has_true_open_boundary():
    scene=_scene()
    result=_open_plane(scene)
    result.validate()
    assert len(result.boundary_edges)>0
    assert result.xyz.shape[1]>15
    assert result.faces.shape[0]>10
    assert float(result.xyz[0,:,2].abs().max())<.045
    assert result.source_authority.startswith("UNQUALIFIED_")
    assert not result.canonical_mesh_authority


def test_open_udf_surface_positions_keep_gradient_to_same_query_head():
    scene=_scene()
    shift=torch.tensor(.035,requires_grad=True)
    result=_open_plane(scene,shift=shift)
    result.xyz[:,:,2].mean().backward()
    assert shift.grad is not None
    assert torch.isfinite(shift.grad)
    assert float(shift.grad.abs())>.001


def test_constant_or_negative_field_fails_closed():
    scene=_scene()
    with pytest.raises(ValueError,match="NO_OPEN_SURFACE_CANDIDATE"):
        extract_open_udf_surface_v1(
            scene_triplanes=scene,
            query_unsigned_distance=lambda x:x[:,0]*0+.15,
            resolution=(12,12,12),query_chunk=256,
        )
    with pytest.raises(ValueError,match="UNSIGNED_FIELD_INVALID"):
        extract_open_udf_surface_v1(
            scene_triplanes=scene,
            query_unsigned_distance=lambda x:x[:,0]*0-.1,
            resolution=(12,12,12),query_chunk=256,
        )


def test_hybrid_one_m0_preserves_closed_and_open_components_before_tessa():
    scene=_scene()
    signed=_closed_tetra(scene)
    open_surface=_open_plane(scene)
    result=join_hybrid_iris_seed_v1(
        signed=signed,open_udf=open_surface,scene_triplanes=scene,
        open_layer_name="cape",open_presentation_roles=("outer_covering",),
    )
    result.validate()
    assert result.seed.original_face_count == (
        len(signed.faces)+len(open_surface.faces)
    )
    assert set(result.component_surface_kinds)=={SIGNED_KIND,OPEN_KIND}
    assert result.seed.vertex_layer_ids[:4].tolist()==[0]*4
    assert bool((result.seed.vertex_layer_ids[4:]==1).all())
    assert bool((result.seed.faces[len(signed.faces):]>=4).all())
    # TESSA refinement remains on ONE globally indexed mesh, with exact
    # open-layer boundaries preserved and no cross-layer constructed edges.
    refined=tessa_adaptive_split_edges_v1(
        result.seed,importance_logits=torch.full(
            (1,result.seed.xyz.shape[1]),9.,
        ),max_added_vertices=4,
    )
    refined.validate()
    assert refined.vertex_layer_ids is not None
    assert bool((refined.vertex_layer_ids[refined.faces]
                 ==refined.vertex_layer_ids[refined.faces[:,0:1]]).all())


def test_two_different_scene_receipts_cannot_silently_form_one_hybrid_mesh():
    scene=_scene()
    signed=_closed_tetra(scene)
    open_surface=_open_plane(scene)
    with pytest.raises(ValueError,match="TWO_SCENES_FORBIDDEN"):
        join_hybrid_iris_seed_v1(
            signed=replace(signed,scene_sha256="f"*64),
            open_udf=open_surface,scene_triplanes=scene,
            open_layer_name="cape",open_presentation_roles=("outer_covering",),
        )
