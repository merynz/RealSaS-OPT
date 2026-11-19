from __future__ import annotations

"""M* indexed surface teacher identity is discrete, source-bound and stable."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.seeded_mstar_teacher_address_v1 import (
    seeded_mstar_teacher_address_sha256_v1 as address,
    require_qualified_mstar_teacher_address_v1 as require,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)


def _source(*, refine: bool = False):
    xyz=torch.tensor([[0.,0.,.5],[-.5,-.5,-.5],[.5,-.5,-.5],[0.,.5,-.5]])
    faces=torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=torch.long)
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz[None],iris_surface_faces=faces,
        component_ids_by_vertex=torch.zeros(4,dtype=torch.long),
        scene_sha256="a"*64,original_surface_sha256="b"*64,
        one_to_four_refine=refine,
    )
    return seed


def test_live_geometry_updates_do_not_invalidate_discrete_mstar_binding():
    seed=_source()
    before=address(seed)
    moved=replace(
        seed,
        xyz=seed.xyz.clone() + torch.tensor([[[0.01,0.,0.]]]),
        scene_sha256="c"*64,
        original_surface_sha256="d"*64,
    )
    assert address(moved)==before
    require(transport_address_sha256=before,active_mstar_address_sha256=address(moved))


def test_refinement_changes_identity_without_changing_original_source():
    original=_source()
    refined=_source(refine=True)
    assert original.original_face_count==refined.original_face_count
    assert original.original_vertex_count==refined.original_vertex_count
    assert address(original)!=address(refined)
    with pytest.raises(ValueError,match="MSTAR_TEACHER_TOPOLOGY_ADDRESS_DRIFT"):
        require(
            transport_address_sha256=address(original),
            active_mstar_address_sha256=address(refined),
        )


def test_parent_support_changes_topology_address_even_for_same_final_faces():
    seed=_source()
    flipped=replace(seed,parent_seed_edges=((1,1),(0,0),(2,2),(3,3)))
    flipped.validate()  # Same indexed F and V: parent provenance alone differs.
    assert torch.equal(seed.faces,flipped.faces)
    assert address(seed)!=address(flipped)


def test_layer_namespace_and_component_partition_are_addressed():
    xyz=torch.tensor([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [2.,0.,0.],[3.,0.,0.],[2.,1.,0.],
    ])
    faces=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.long)
    comp=torch.tensor([0,0,0,1,1,1])
    layers=comp.clone()
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz[None],iris_surface_faces=faces,
        component_ids_by_vertex=comp,
        scene_sha256="a"*64,original_surface_sha256="b"*64,
        original_vertex_layer_ids=layers,
        layer_names=("body","cloth"),
        layer_roles=(("primary_form",),("outer_covering",)),
        layer_namespace_sha256="c"*64,
    )
    assert address(seed)!=address(replace(seed,layer_namespace_sha256="d"*64))
    assert address(seed)!=address(replace(seed,vertex_component_ids=torch.tensor(
        [0,0,0,2,2,2],dtype=torch.long,
    ),face_component_ids=torch.tensor([0,2],dtype=torch.long)))
    assert address(seed)!=address(replace(seed,layer_names=("body","wing")))


def test_missing_or_forged_topology_address_is_never_unknown_success():
    seed=_source()
    for other in ("", "f"*64, "x"*64):
        with pytest.raises(ValueError,match="MSTAR_TEACHER_TOPOLOGY_ADDRESS_DRIFT"):
            require(
                transport_address_sha256=other,
                active_mstar_address_sha256=address(seed),
            )
