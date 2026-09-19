from __future__ import annotations

from types import SimpleNamespace
import pytest

from compiler.realsas_compiler_core.presentation_structure_v1 import build_presentation_structure
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,ComponentRegionIR,build_component_carrier_policy,
    build_mechanical_partition,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,QualifiedSkeletonIR,QualifiedSkinIR,QualifiedSkinRow,
    QualificationError,RiggingSurfaceIR,SurfaceNode,
)


def _fixture(*,rigid=True,carrier="MESH"):
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("s0",(0,0,0),(0,),("p",),("o",)),
            SurfaceNode("s1",(1,0,0),(0,),("p",),("o",)),
            SurfaceNode("s2",(0,1,0),(0,),("p",),("o",)),
            SurfaceNode("s3",(2,0,0),(0,),("p",),("o",)),
        ),(),"surface-hash"
    )
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("root",(0,0,0),None,("s0","s1"),"p0"),
            QualifiedJoint("j1",(2,0,0),"root",("s2","s3"),"p1"),
        ),"root",{"status":"PASS"},"skeleton-hash"
    )
    rows=(
        QualifiedSkinRow("s0",(("root",0.6),("j1",0.4)),0,0),
        QualifiedSkinRow("s1",(("root",0.6),("j1",0.4)),0,0),
        QualifiedSkinRow("s2",(("j1",1.0),),0,0),
        QualifiedSkinRow("s3",(("j1",1.0),) if rigid else (("j1",0.8),("root",0.2)),0,0),
    )
    skin=QualifiedSkinIR(rows,"surface-hash","skeleton-hash",{"status":"PASS"},"skin-hash")
    partition=build_mechanical_partition(
        surface=surface,
        components=(
            ComponentRegionIR("body",("s0","s1")),
            ComponentRegionIR("part",("s2","s3")),
        ),
        boundary_constraints=(),
    )
    carrier_policy=build_component_carrier_policy(
        partition=partition,
        decisions=(
            ComponentCarrierDecisionIR("body","MESH",("e",)),
            ComponentCarrierDecisionIR("part",carrier,("e",)),
        ),
    )
    mesh=SimpleNamespace(
        mesh_lineage_hash="mesh-hash",
        vertices=(
            SimpleNamespace(canonical_mesh_vertex_id="b0",component_id="body"),
            SimpleNamespace(canonical_mesh_vertex_id="b1",component_id="body"),
            SimpleNamespace(canonical_mesh_vertex_id="b2",component_id="body"),
            SimpleNamespace(canonical_mesh_vertex_id="p0",component_id="part"),
            SimpleNamespace(canonical_mesh_vertex_id="p1",component_id="part"),
            SimpleNamespace(canonical_mesh_vertex_id="p2",component_id="part"),
        ),
        faces=(("b0","b1","b2"),("p0","p1","p2")),
    )
    state=SimpleNamespace(product_state_hash="state-hash")
    return surface,skeleton,skin,partition,carrier_policy,mesh,state


def test_role_free_structure_uses_root_for_deformable_and_exact_skin_owner_for_rigid():
    args=_fixture(rigid=True)
    value=build_presentation_structure(
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],mesh=args[5],product_state=args[6],
    )
    by_component={slot.metadata["mechanical_component_id"]:slot for slot in value.slots}
    assert by_component["body"].bone_id=="root"
    assert by_component["part"].bone_id=="j1"
    att={row.mechanical_component_ids[0]:row for row in value.attachments}
    assert att["body"].mechanical_class=="DEFORMABLE"
    assert att["part"].mechanical_class=="RIGID"
    assert att["part"].metadata["detachability_authority"]=="UNPROVEN"
    assert value.metadata["categorical_recognition_used"] is False


def test_non_one_hot_component_stays_deformable_without_object_semantics():
    args=_fixture(rigid=False)
    value=build_presentation_structure(
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],mesh=args[5],product_state=args[6],
    )
    att={row.mechanical_component_ids[0]:row for row in value.attachments}
    assert att["part"].mechanical_class=="DEFORMABLE"


def test_planar_carrier_blocks_until_proxy_plane_authority_exists():
    args=_fixture(carrier="PLANAR")
    with pytest.raises(QualificationError,match="PLANAR_PROXY_NOT_IMPLEMENTED"):
        build_presentation_structure(
            surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
            carrier_policy=args[4],mesh=args[5],product_state=args[6],
        )
