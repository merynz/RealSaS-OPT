from __future__ import annotations

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_compile_v1 import (
    CanonicalJointTrackIR, CompiledMotionClipIR, MotionCompileConstraintSetIR,
    MotionContactConstraintIR, MotionKeyframeIR, QualifiedMotionIR,
    compiled_motion_clip_hash, motion_constraint_set_hash, qualified_motion_hash,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v1 import build_qualified_dynamic_motion
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR, MeshQualificationPolicyIR, PresentationAttachmentIR,
    PresentationSlotIR, QualifiedMeshIR, QualifiedMeshVertexIR, QualifiedPresentationGraphIR,
    mesh_qualification_policy_lineage_hash, qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
)
from compiler.realsas_compiler_core.canonical_puppet_state_v1 import CanonicalPuppetStateIR
from compiler.realsas_compiler_core.types import (
    QualifiedJoint, QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualifiedSkeletonIR,
    QualificationError, SurfaceSupportBinding,
)


def _proof_observation_args():
    cameras=tuple(
        CameraProjectionV3(
            f"V{i}",i,(0.0,0.0,-2.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),2.0,32
        )
        for i in range(8)
    )
    masks={i:bytes([1])*(32*32) for i in range(8)}
    return cameras,masks


def _fixture(*,contact=False):
    skeleton=QualifiedSkeletonIR(
        (QualifiedJoint("root",(0.0,0.0,0.0),None,(),"p0"),),
        "root",{"status":"PASS"},"skel",
    )
    verts=(
        QualifiedMeshVertexIR("v0",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s0",1.0),)),"c0",(0.0,0.0,0.0),"v0"),
        QualifiedMeshVertexIR("v1",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s1",1.0),)),"c0",(1.0,0.0,0.0),"v1"),
        QualifiedMeshVertexIR("v2",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s2",1.0),)),"c0",(0.0,1.0,0.0),"v2"),
    )
    policy=MeshQualificationPolicyIR(
        1.0,1.0,7.5,16.0,
        (CarrierCoverageThresholdIR("MESH",0,0,1,1),CarrierCoverageThresholdIR("PLANAR",0,0,1,1)),
        "",0.05,20.0,16.0,
    )
    policy=replace(policy,qualification_policy_lineage_hash=mesh_qualification_policy_lineage_hash(policy))
    mesh=QualifiedMeshIR(
        verts,(("v0","v1","v2"),),(("v0","v1"),("v0","v2"),("v1","v2")),
        "surface","partition","carrier","env",policy.qualification_policy_lineage_hash,
        {"g3_stress_probe_status":"PASS","g3_envelope_binding_hash":"env"},"",
    )
    mesh=replace(mesh,mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    rows=tuple(QualifiedMeshSkinRow(v.canonical_mesh_vertex_id,(("root",1.0),),(("s0",1.0),),0.0,0.0) for v in verts)
    mesh_skin=QualifiedMeshSkinIR(rows,"surface","skel","skin",mesh.mesh_lineage_hash,"TEST",{"status":"PASS"},"")
    raw=mesh_skin.to_dict(); raw.pop("mesh_skin_lineage_hash")
    mesh_skin=replace(mesh_skin,mesh_skin_lineage_hash=content_sha256(raw))
    state=CanonicalPuppetStateIR(
        "surface","skel","skin","partition","carrier","env",policy.qualification_policy_lineage_hash,
        mesh.mesh_lineage_hash,mesh_skin.mesh_skin_lineage_hash,(),"",
    )
    raw=state.to_dict(); raw.pop("product_state_hash")
    state=replace(state,product_state_hash=content_sha256(raw))
    slot=PresentationSlotIR("slot","root",0,"att",("ATTACHMENT",))
    att=PresentationAttachmentIR("att","slot",("c0",),"DEFORMABLE","MESH",mesh.mesh_lineage_hash)
    presentation=QualifiedPresentationGraphIR(
        (slot,),(att,),(),(),"skel",mesh.mesh_lineage_hash,"partition","carrier",state.product_state_hash,
        "structure","appearance","composition",{"status":"PASS"},"",
    )
    presentation=replace(presentation,presentation_lineage_hash=qualified_presentation_lineage_hash(presentation))
    keys=(MotionKeyframeIR(0.0,rotation_deg=0.0),MotionKeyframeIR(1.0,rotation_deg=10.0))
    track=CanonicalJointTrackIR("root","hips",("ROTATION_DEG",),keys)
    clip=CompiledMotionClipIR("run","RUN",1.0,True,"asset","SOURCE_RIG_TRACKS_V1","ARTIST_SOURCE","IN_PLACE",(track,),"")
    clip=replace(clip,clip_lineage_hash=compiled_motion_clip_hash(clip))
    contacts=(MotionContactConstraintIR("plant","run","root",0.0,1.0,"PLANT_2D"),) if contact else ()
    constraints=MotionCompileConstraintSetIR(
        state.product_state_hash,"skel","env",presentation.presentation_lineage_hash,
        (("run","IN_PLACE"),),(("run",(("hips","root"),)),),contacts,
        "STATIC_QUALIFIED_PRESENTATION_ONLY_V1","",
    )
    constraints=replace(constraints,constraint_set_hash=motion_constraint_set_hash(constraints))
    motion=QualifiedMotionIR(
        "seal","sources",state.product_state_hash,"skel","env",presentation.presentation_lineage_hash,
        constraints.constraint_set_hash,(clip,),{"status":"PASS_COMPILED_MECHANICS_ONLY","motion_quality_claimed":False,"dynamic_proof_passed":False},"",
    )
    motion=replace(motion,motion_lineage_hash=qualified_motion_hash(motion))
    return motion,constraints,state,skeleton,mesh,mesh_skin,presentation,policy


def test_dynamic_proof_is_single_canonical_3d_and_professional_artist_evidence():
    motion,constraints,state,skeleton,mesh,mesh_skin,presentation,policy=_fixture(contact=True)
    proof=build_qualified_dynamic_motion(
        motion=motion,constraints=constraints,product_state=state,skeleton=skeleton,
        mesh=mesh,mesh_skin=mesh_skin,presentation=presentation,mesh_policy=policy,
        cameras=_proof_observation_args()[0],source_foreground_masks=_proof_observation_args()[1],
    )
    assert proof.qualification_report["dynamic_proof_passed"] is True
    assert proof.qualification_report["canonical_3d_single_mesh_truth"] is True
    assert proof.qualification_report["professional_motion_clip_count"]==1
    assert proof.clips[0].contact_proofs[0].status=="PASS"
    assert len(proof.clips[0].frames)>=17
    assert set(v for v,_ in proof.clips[0].frames[-1].posed_vertex_xyz)=={"v0","v1","v2"}


def test_contact_plant_rejects_translating_root():
    motion,constraints,state,skeleton,mesh,mesh_skin,presentation,policy=_fixture(contact=True)
    clip=motion.clips[0]
    track=replace(clip.tracks[0],channel_contract=("ROTATION_DEG","TRANSLATION_XY"),keyframes=(
        MotionKeyframeIR(0.0,rotation_deg=0.0,translation_xy=(0.0,0.0)),
        MotionKeyframeIR(1.0,rotation_deg=0.0,translation_xy=(0.1,0.0)),
    ))
    clip=replace(clip,tracks=(track,),clip_lineage_hash="")
    clip=replace(clip,clip_lineage_hash=compiled_motion_clip_hash(clip))
    motion=replace(motion,clips=(clip,),motion_lineage_hash="")
    motion=replace(motion,motion_lineage_hash=qualified_motion_hash(motion))
    with pytest.raises(QualificationError,match="MOTION_DYNAMIC_CONTACT_FAIL"):
        build_qualified_dynamic_motion(
            motion=motion,constraints=constraints,product_state=state,skeleton=skeleton,
            mesh=mesh,mesh_skin=mesh_skin,presentation=presentation,mesh_policy=policy,
            cameras=_proof_observation_args()[0],source_foreground_masks=_proof_observation_args()[1],
        )
