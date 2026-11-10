from __future__ import annotations

"""Consumer-first, synthetic-but-REAL Compiler FK/IK/LBS and eight-view court.

The fixture is intentionally not a model prediction or a product authority.
It tests what a *qualified* source MUST provide to the runtime consumer.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, CanonicalMeshVertexIR, canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2, CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2, CanonicalRigConstraintIRV2,
    CanonicalRigControlIRV2, CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, CanonicalSkinRowIRV2,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetBundleV1, ControllablePuppetSessionV1,
    RuntimeSelectionBindingV1, VIEW_IDS_V1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    build_full_surface_runtime_meshes_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance, RuntimeV3AppearancePatch,
    RuntimeV3PlaybackContract, RuntimeV3Slot,
)
from compiler.realsas_compiler_core.types import (
    QualificationError, SurfaceSupportBinding,
)


def qualified_synthetic_bundle() -> ControllablePuppetBundleV1:
    def sv(i: int) -> SurfaceSupportBinding:
        return SurfaceSupportBinding("EXACT_SURFACE_NODE", ((f"S:{i}", 1.0),), {})
    vertex_positions = ((0., 0., 0.), (.5, 1., 0.), (0., 2., 0.))
    provisional = CanonicalMeshIR(
        vertices=tuple(CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=f"MV:{i}",
            support_binding=sv(i),
            component_id="body",
            P=p,
            source_candidate_vertex_id=f"CV:{i}",
        ) for i, p in enumerate(vertex_positions)),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0","MV:1"),("MV:1","MV:2"),("MV:0","MV:2")),
        source_evidence_binding_hash="1"*64,
        partition_binding_hash="2"*64,
        carrier_policy_binding_hash="3"*64,
        static_qualification_hash="4"*64,
        carrier_evidence_hash="5"*64,
        carrier_topology_hash="6"*64,
        carrier_geometry_hash="7"*64,
        proposal_evidence_refs=("8"*64,),
        source_candidate_lineage_hash="9"*64,
        mesh_lineage_hash="",
        metadata={"authority":"SYNTHETIC_UNIT_SOURCE_ONLY"},
    )
    mesh = replace(
        provisional, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(provisional),
    )
    I = ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    rig = CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("J0",(0.,0.,0.),None,"PJ0"),
            CanonicalDeformJointIRV2("J1",(0.,1.,0.),"J0","PJ1"),
            CanonicalDeformJointIRV2("J2",(0.,2.,0.),"J1","PJ2"),
        ),
        local_frames=tuple(CanonicalJointFrameIRV2(
            jid, I, ("RX","RY","RZ"),
        ) for jid in ("J0","J1","J2")),
        controls=(
            CanonicalRigControlIRV2("C_ROOT","ROOT",("J0",),None,(0.,0.,0.),"PC0"),
            CanonicalRigControlIRV2("C_IK","IK_EFFECTOR",("J2",),"C_ROOT",(0.,2.,0.),"PC1"),
            CanonicalRigControlIRV2("C_POLE","IK_POLE",(),"C_ROOT",(1.,1.,0.),"PC2"),
        ),
        chains=(CanonicalRigChainIRV2(
            "K0",("J0","J1","J2"),"GENERIC_KINEMATIC_CHAIN","PK0",
        ),),
        constraints=(CanonicalRigConstraintIRV2(
            "Q0","IK",("C_IK","C_POLE"),("J2",),"K0",
            {"p0":1.,"p1":1.,"p2":1.}, "PQ0",
        ),),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash="1"*64,
        proposal_evidence_hash="a"*64,
        rig_program_qualification_hash="b"*64,
        deform_structure_hash="c"*64,
        control_program_hash="d"*64,
        rig_lineage_hash="e"*64,
        metadata={"authority":"SYNTHETIC_UNIT_RIG"},
    )
    skin = CanonicalSkinIRV2(
        rows=tuple(CanonicalSkinRowIRV2(
            f"MV:{i}", ((f"J{i}",1.),), f"CV:{i}",
        ) for i in range(3)),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash="f"*64,
        skin_code_proposal_hash="1"*64,
        skin_codec_architecture_id="codec",
        skin_codec_checkpoint_hash="2"*64,
        skin_lineage_hash="3"*64,
        metadata={"authority":"SYNTHETIC_UNIT_SKIN"},
    )
    cameras = {}
    for i, view in enumerate(VIEW_IDS_V1):
        theta = math.pi * i / 4
        si, co = math.sin(theta), math.cos(theta)
        cameras[view] = {
            "view_index":i,
            "origin":(3*si,0.,3*co),
            "right":(co,0.,-si),
            "screen_up":(0.,1.,0.),
            "forward":(-si,0.,-co),
            "half_extent":2.,
            "resolution":128,
        }
    uv = np.array([[0,1],[1,.5],[0,0]],dtype=np.float64)
    surface = build_full_surface_runtime_meshes_v3(
        np.asarray(vertex_positions),
        np.array([[0,1,2]],dtype=np.int64),
        cameras=cameras,
        uv_by_view={view:uv for view in VIEW_IDS_V1},
        slot_id="S_BODY",attachment_id="A_BODY",
    )
    playback = RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("S_BODY","J0",0,"A_BODY"),),
        meshes=surface.meshes,
        appearance_patches=tuple(RuntimeV3AppearancePatch(
            patch_id=f"AP:{view}",mesh_id=f"{view}:A_BODY",
            face_indices=(0,),
            provenance=AppearanceProvenance.UNSEEN,
            donor_view_index=None,atlas_id=None,
        ) for view in VIEW_IDS_V1),
    )
    return ControllablePuppetBundleV1(
        mesh=mesh, rig=rig, skin=skin, playback=playback,
        cameras_by_view=cameras,
        vertex_ids_by_attachment={
            "A_BODY":("MV:0","MV:1","MV:2"),
        },
        selection_bindings=(RuntimeSelectionBindingV1(
            "A_BODY","body","SYNTHETIC_UNIT_SOURCE_FACE","C_IK",
        ),),
        product_revision_id="SYNTHETIC_TEST_REVISION_NOT_PRODUCT",
        product_qualification_receipt="SYNTHETIC_TEST_ONLY",
    )


def test_runtime_eight_views_reproject_one_exact_3d_mesh():
    bundle = qualified_synthetic_bundle()
    session = ControllablePuppetSessionV1(bundle)
    expected_ids = tuple(v.canonical_mesh_vertex_id for v in bundle.mesh.vertices)
    old_mesh_hash = bundle.mesh.mesh_lineage_hash
    positions = []
    for i in range(8):
        session.set_view_index(i)
        frame = session.frame()
        assert frame.view_id == f"V{i}"
        assert frame.product_revision_id == bundle.product_revision_id
        assert frame.authority == "QUALIFIED_SOURCE_GEOMETRY_INTERACTION_CONSUMER"
        assert frame.deformed.canonical_mesh_vertex_ids == expected_ids
        assert frame.deformed.mesh_lineage_hash == old_mesh_hash
        assert len(frame.meshes) == 1
        assert frame.meshes[0].attachment_id == "A_BODY"
        positions.append(tuple((v.x,v.y,v.z) for v in frame.meshes[0].vertices))
    # These are distinct camera projections, not eight duplicated sprites.
    assert positions[0] != positions[1]
    assert positions[1] != positions[2]
    assert session.bundle.mesh.mesh_lineage_hash == old_mesh_hash
    with pytest.raises(QualificationError,match="VIEW_INDEX_OUT_OF_RANGE"):
        session.set_view_index(8)


def test_mouse_screen_drag_executes_real_ik_and_lbs_without_model_inference():
    session = ControllablePuppetSessionV1(qualified_synthetic_bundle())
    initial = session.frame()
    original_vertices = initial.deformed.positions
    # Front camera x=80,y=16 corresponds x=.5,y=1.5,z=0.0.
    session.drag_screen("C_IK",80.,16.)
    moved = session.frame()
    by_joint = {j.canonical_joint_id: np.asarray(j.solved_P) for j in moved.pose.joints}
    assert np.linalg.norm(by_joint["J2"] - np.array([.5,1.5,0.])) < 1e-4
    assert moved.pose.constraint_residuals[0][1] < 1e-4
    assert moved.pose.metadata["learned_inference_performed"] is False
    assert moved.pose.metadata["rig_program_mutated"] is False
    assert moved.deformed.positions != original_vertices
    assert moved.deformed.canonical_mesh_vertex_ids == initial.deformed.canonical_mesh_vertex_ids
    assert abs(np.linalg.norm(by_joint["J1"]-by_joint["J0"])-1.) < 1e-4
    # 45-degree turn changes presentation, not mechanics or vertex ID.
    session.set_view_index(1)
    turned = session.frame()
    assert turned.pose.joints == moved.pose.joints
    assert turned.deformed.positions == moved.deformed.positions
    assert [(v.x,v.y,v.z) for v in turned.meshes[0].vertices] != [
        (v.x,v.y,v.z) for v in moved.meshes[0].vertices
    ]


def test_mouse_picking_resolves_canonical_component_and_requires_alpha():
    session = ControllablePuppetSessionV1(qualified_synthetic_bundle())
    hit = session.pick(69,32,sample_alpha=lambda view,att,u,v:1.)
    assert hit is not None
    assert hit.canonical_component_id == "body"
    assert hit.canonical_face_vertex_ids == ("MV:0","MV:1","MV:2")
    assert hit.source_support_evidence_id == "SYNTHETIC_UNIT_SOURCE_FACE"
    assert hit.control_id == "C_IK"
    # Transparent pixel must not consume user clicks.
    assert session.pick(69,32,sample_alpha=lambda *args:0.) is None
    assert session.selected is None
    with pytest.raises(QualificationError,match="PICK_ALPHA_UNQUALIFIED"):
        session.pick(69,32,sample_alpha=lambda *args:float("nan"))
    with pytest.raises(QualificationError,match="PICK_INPUT_INVALID"):
        session.pick(69,32,sample_alpha=None)


def test_product_addresses_and_eight_camera_requirements_are_fail_closed():
    bundle = qualified_synthetic_bundle()
    with pytest.raises(QualificationError,match="EIGHT_CAMERAS_REQUIRED"):
        ControllablePuppetSessionV1(replace(
            bundle,cameras_by_view={k:v for k,v in bundle.cameras_by_view.items() if k!="V7"},
        ))
    with pytest.raises(QualificationError,match="VIEW_VERTEX_ID_UNKNOWN"):
        ControllablePuppetSessionV1(replace(
            bundle,vertex_ids_by_attachment={"A_BODY":("MV:0","MV:1","FABRICATED")},
        ))
    with pytest.raises(QualificationError,match="RIG_MESH_LINEAGE_DRIFT"):
        ControllablePuppetSessionV1(replace(
            bundle,rig=replace(bundle.rig,mesh_lineage_hash="wrong"),
        ))
    with pytest.raises(QualificationError,match="PICK_CONTROL_UNBOUND"):
        ControllablePuppetSessionV1(replace(
            bundle,selection_bindings=(replace(bundle.selection_bindings[0],control_id="FAKE"),),
        ))


def test_no_automatic_attachment_invention_and_no_parent_space_guessing():
    s = ControllablePuppetSessionV1(qualified_synthetic_bundle())
    with pytest.raises(QualificationError,match="ATTACHMENT_NOT_BOUND_IN_ALL_VIEWS"):
        s.set_attachment("S_BODY","UNSEEN_SWORD")
    s.set_attachment("S_BODY",None)
    assert not s.frame().meshes
    s.set_attachment("S_BODY","A_BODY")
    assert len(s.frame().meshes)==1
    s.set_control(ControlTransformIRV1("C_ROOT",translation=(.5,0.,0.)))
    with pytest.raises(QualificationError,match="PARENT_SPACE_INVERSION_REQUIRED"):
        s.drag_world("C_IK",(.5,1.5,0.))



def test_runtime_snapshot_replays_controls_across_view_changes_without_model():
    session=ControllablePuppetSessionV1(qualified_synthetic_bundle())
    session.drag_world("C_IK",(.5,1.5,0.))
    session.set_view_index(1)
    pose=session.frame()
    checkpoint=session.snapshot()
    assert checkpoint.state_hash==checkpoint.checked_hash()
    assert checkpoint.view_index==1
    session.drag_world("C_IK",(-.25,1.75,0.))
    session.set_view_index(7)
    assert session.frame().deformed.positions != pose.deformed.positions
    session.restore(checkpoint)
    assert session.view_index==1
    assert session.frame().pose.joints==pose.pose.joints
    assert session.frame().deformed.positions==pose.deformed.positions
    assert session.snapshot().state_hash==checkpoint.state_hash


def test_runtime_snapshot_tampering_rejected_and_session_state_is_atomic():
    session=ControllablePuppetSessionV1(qualified_synthetic_bundle())
    session.set_view_index(2)
    baseline=session.snapshot()
    with pytest.raises(QualificationError,match="SNAPSHOT_HASH_DRIFT"):
        session.restore(replace(baseline,view_index=3))
    assert session.view_index==2
    foreign=replace(
        baseline,product_revision_id="OTHER_RECOMPILED_ASSET",state_hash="",
    )
    from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import replace_snapshot_hash
    with pytest.raises(QualificationError,match="SNAPSHOT_PARENT_DRIFT"):
        session.restore(replace_snapshot_hash(foreign))
    assert session.snapshot().state_hash==baseline.state_hash
    forged=replace(
        baseline,
        control_states=(ControlTransformIRV1("UNKNOWN_CONTROL"),),
        state_hash="",
    )
    with pytest.raises(QualificationError,match="SNAPSHOT_CONTROLS_UNKNOWN"):
        session.restore(replace_snapshot_hash(forged))
    forged=replace(baseline,active_attachments=(),state_hash="")
    with pytest.raises(QualificationError,match="SNAPSHOT_SLOT_COVERAGE"):
        session.restore(replace_snapshot_hash(forged))
    assert session.snapshot().state_hash==baseline.state_hash
