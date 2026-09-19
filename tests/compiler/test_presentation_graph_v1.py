from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.presentation_graph_v1 import build_qualified_presentation_bundle
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,ComponentRegionIR,build_component_carrier_policy,
    build_mechanical_partition,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,QualifiedSkeletonIR,QualifiedSkinIR,QualifiedSkinRow,
    RiggingSurfaceIR,SurfaceNode,SurfaceSupportBinding,
)


def _fixture():
    nodes=(
        SurfaceNode("s0",(0,0,0),tuple(range(8)),("p",),("o0",),tuple((v,(10.5,10.5)) for v in range(8))),
        SurfaceNode("s1",(1,0,0),tuple(range(8)),("p",),("o1",),tuple((v,(20.5,10.5)) for v in range(8))),
        SurfaceNode("s2",(0,1,0),tuple(range(8)),("p",),("o2",),tuple((v,(10.5,20.5)) for v in range(8))),
    )
    surface=RiggingSurfaceIR(nodes,(),"surface-hash",metadata={"raster_coordinate_system":"PIXEL_CENTER_XY","resolution":64})
    skeleton=QualifiedSkeletonIR(
        (QualifiedJoint("root",(0,0,0),None,("s0","s1","s2"),"p0"),),
        "root",{"status":"PASS"},"skeleton-hash"
    )
    skin=QualifiedSkinIR(
        tuple(QualifiedSkinRow(sid,(("root",1.0),),0,0) for sid in ("s0","s1","s2")),
        "surface-hash","skeleton-hash",{"status":"PASS"},"skin-hash"
    )
    partition=build_mechanical_partition(
        surface=surface,components=(ComponentRegionIR("c0",("s0","s1","s2")),),boundary_constraints=()
    )
    carrier=build_component_carrier_policy(
        partition=partition,decisions=(ComponentCarrierDecisionIR("c0","MESH",("e",)),)
    )
    vertices=tuple(
        SimpleNamespace(
            canonical_mesh_vertex_id=f"v{i}",
            support_binding=SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{i}",1.0),)),
            component_id="c0",
        )
        for i in range(3)
    )
    mesh=SimpleNamespace(
        surface_binding_hash="surface-hash",mesh_lineage_hash="mesh-hash",
        vertices=vertices,faces=(("v0","v1","v2"),),
        qualification_report={"g5_evidence_hash":"g5-hash"},
    )
    observations=build_qualified_observation_set(tuple(
        QualifiedObservationViewIR(
            i,64,64,f"obs-{i}",content_sha256({"rgba":i}),content_sha256({"fg":i}),
            f"cam-{i}","PASS",(f"e-{i}",)
        )
        for i in range(8)
    ))
    state=SimpleNamespace(
        product_state_hash="state-hash",
        skeleton_lineage_hash="skeleton-hash",
        mesh_lineage_hash="mesh-hash",
        partition_lineage_hash=partition.partition_lineage_hash,
        carrier_policy_lineage_hash=carrier.carrier_policy_lineage_hash,
    )
    envelope=SimpleNamespace(camera_binding_hashes=tuple(f"cam-{i}" for i in range(8)))
    return surface,skeleton,skin,mesh,partition,carrier,envelope,state,observations


def test_bundle_binds_structure_appearance_composition_and_final_graph_exactly():
    args=_fixture()
    structure,appearance,composition,graph=build_qualified_presentation_bundle(
        surface=args[0],skeleton=args[1],skin=args[2],mesh=args[3],
        partition=args[4],carrier_policy=args[5],envelope=args[6],
        product_state=args[7],observation_set=args[8],
    )
    assert graph.presentation_structure_binding_hash==structure.structure_lineage_hash
    assert graph.appearance_set_binding_hash==appearance.appearance_set_hash
    assert graph.composition_set_binding_hash==composition.composition_set_hash
    assert len(graph.view_overlays)==8
    assert graph.qualification_report["single_canonical_mesh"] is True
    assert graph.qualification_report["slot_order_solves_physical_occlusion"] is False
    assert graph.qualification_report["categorical_recognition_used"] is False
    assert graph.qualification_report["completion_used"] is False
    for view,overlay in enumerate(graph.view_overlays):
        assert overlay.appearance_binding_hash==appearance.bindings[view].appearance_lineage_hash
        assert overlay.composition_binding_hash==composition.views[view].composition_binding_hash
