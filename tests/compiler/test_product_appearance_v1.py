from __future__ import annotations

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.product_appearance_v1 import (
    build_product_appearance_set,validate_product_appearance_set,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,RiggingSurfaceIR,SurfaceNode,SurfaceSupportBinding,
)


def _fixture():
    nodes=(
        SurfaceNode("s0",(0.0,0.0,0.0),tuple(range(8)),("p0",),("o0",),
                    tuple((v,(10.5+v,10.5)) for v in range(8))),
        SurfaceNode("s1",(1.0,0.0,0.0),tuple(range(8)),("p1",),("o1",),
                    tuple((v,(20.5+v,10.5)) for v in range(8))),
        SurfaceNode("s2",(0.0,1.0,0.0),tuple(range(8)),("p2",),("o2",),
                    tuple((v,(10.5+v,20.5)) for v in range(8))),
    )
    surface=RiggingSurfaceIR(
        nodes,(),"surface-hash",
        metadata={"raster_coordinate_system":"PIXEL_CENTER_XY","resolution":64}
    )
    verts=tuple(
        type("V",(),{
            "canonical_mesh_vertex_id":f"v{i}",
            "support_binding":SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{i}",1.0),)),
        })()
        for i in range(3)
    )
    mesh=type("M",(),{
        "surface_binding_hash":"surface-hash",
        "mesh_lineage_hash":"mesh-hash",
        "vertices":verts,
        "faces":(("v0","v1","v2"),),
    })()
    views=tuple(
        QualifiedObservationViewIR(
            v,64,64,f"obs-{v}",content_sha256({"rgba":v}),content_sha256({"fg":v}),
            f"cam-{v}","PASS",(f"e-{v}",)
        )
        for v in range(8)
    )
    return surface,mesh,build_qualified_observation_set(views)


def test_product_appearance_uses_same_canonical_mesh_and_one_local_donor_per_face():
    surface,mesh,observations=_fixture()
    value=build_product_appearance_set(surface=surface,mesh=mesh,observation_set=observations)
    assert len(value.bindings)==8
    for view,binding in enumerate(value.bindings):
        assert binding.target_view_index==view
        assert binding.mesh_binding_hash=="mesh-hash"
        assert {c.donor_view_index for c in binding.corner_bindings}=={view}
        assert all(c.source_observation_hash==f"obs-{view}" for c in binding.corner_bindings)
        assert binding.metadata["face_uniform_donor_required"] is True
    validate_product_appearance_set(value,surface=surface,mesh=mesh,observation_set=observations)


def test_face_uniform_cross_view_fallback_is_used_when_target_lacks_support():
    surface,mesh,observations=_fixture()
    nodes=list(surface.surface_nodes)
    nodes[2]=replace(
        nodes[2],
        support_views=tuple(range(1,8)),
        raster_bindings=tuple((v,(10.5+v,20.5)) for v in range(1,8)),
    )
    sparse=replace(surface,surface_nodes=tuple(nodes))
    value=build_product_appearance_set(surface=sparse,mesh=mesh,observation_set=observations)
    v0=value.bindings[0]
    assert {c.donor_view_index for c in v0.corner_bindings}=={1}
    assert {c.authority_class for c in v0.corner_bindings}=={"OBSERVED_CROSS_VIEW"}


def test_face_without_common_observed_donor_fails_closed_instead_of_blending():
    surface,mesh,observations=_fixture()
    nodes=list(surface.surface_nodes)
    nodes[0]=replace(nodes[0],support_views=(0,),raster_bindings=((0,(10.5,10.5)),))
    nodes[1]=replace(nodes[1],support_views=(1,),raster_bindings=((1,(21.5,10.5)),))
    bad=replace(surface,surface_nodes=tuple(nodes))
    with pytest.raises(QualificationError,match="NO_OBSERVED_DONOR"):
        build_product_appearance_set(surface=bad,mesh=mesh,observation_set=observations)


def test_source_raster_identity_changes_appearance_set_even_when_coordinates_do_not():
    surface,mesh,observations=_fixture()
    first=build_product_appearance_set(surface=surface,mesh=mesh,observation_set=observations)
    rows=list(observations.views)
    rows[0]=replace(rows[0],source_raster_sha256="f"*64)
    changed=build_qualified_observation_set(tuple(rows))
    second=build_product_appearance_set(surface=surface,mesh=mesh,observation_set=changed)
    assert first.appearance_set_hash!=second.appearance_set_hash
    assert first.bindings[0].atlas_payload_hash!=second.bindings[0].atlas_payload_hash
