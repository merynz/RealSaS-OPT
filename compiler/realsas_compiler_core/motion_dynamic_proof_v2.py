from __future__ import annotations

"""Stage-35 execution of full-3D quaternion motion on exact canonical M/B.

Input motion is RealSaS.QualifiedMotionIR.v2. Output deliberately reuses the existing
QualifiedDynamicMotionIR frame format because runtime needs exact posed XYZ, not the
authoring quaternion representation.
"""

from dataclasses import replace
import math
from typing import Any

import numpy as np

from .canonical_puppet_state_v1 import canonical_puppet_state_hash
from .joint_frames_v1 import derive_joint_frames_from_skeleton, frame_set_hash
from .mesh.product_coverage_v1 import rasterize_visible_face_pixel_counts
from .motion_compile_v2 import (
    QualifiedMotion3DIR,
    MotionCompileConstraintSetV2IR,
    qualified_motion_v2_hash,
    motion_constraint_set_v2_hash,
)
from .motion_dynamic_proof_v1 import (
    CanonicalDynamicFrameIR,
    DynamicContactProofIR,
    DynamicMotionClipProofIR,
    QualifiedDynamicMotionIR,
    canonical_dynamic_frame_hash,
    dynamic_contact_proof_hash,
    dynamic_motion_clip_proof_hash,
    qualified_dynamic_motion_hash,
    _frame_metrics,
    _mesh_skin_hash,
    _policy_hash,
    _scale,
    _skin_vertices,
    _weights,
)
from .product_authority_v1 import (
    qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
)
from .types import QualificationError


DYNAMIC_EVALUATOR_SEMANTIC_VERSION_V2="RealSaS.CanonicalDynamicMotionEvaluator.QuaternionV3"
_UNIFORM_SAMPLE_COUNT=17
_CONTACT_REL_TOL=1e-5
_MOTION_REL_EPS=1e-8


def _quat_matrix_xyzw(q):
    q=np.asarray(q,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all():
        raise QualificationError("MOTION_V2_DYNAMIC_QUAT_INVALID")
    n=float(np.linalg.norm(q))
    if n<=1e-12:
        raise QualificationError("MOTION_V2_DYNAMIC_QUAT_DEGENERATE")
    x,y,z,w=(q/n).tolist()
    xx=x*x; yy=y*y; zz=z*z
    xy=x*y; xz=x*z; yz=y*z
    wx=w*x; wy=w*y; wz=w*z
    return np.asarray([
        [1-2*(yy+zz),2*(xy-wz),2*(xz+wy)],
        [2*(xy+wz),1-2*(xx+zz),2*(yz-wx)],
        [2*(xz-wy),2*(yz+wx),1-2*(xx+yy)],
    ],dtype=np.float64)


def _slerp(q0,q1,u):
    a=np.asarray(q0,dtype=np.float64)
    b=np.asarray(q1,dtype=np.float64)
    a=a/max(float(np.linalg.norm(a)),1e-12)
    b=b/max(float(np.linalg.norm(b)),1e-12)
    dot=float(np.dot(a,b))
    if dot<0.0:
        b=-b; dot=-dot
    dot=max(-1.0,min(1.0,dot))
    if dot>0.9995:
        q=a+float(u)*(b-a)
        q=q/max(float(np.linalg.norm(q)),1e-12)
        return tuple(map(float,q))
    theta=math.acos(dot)
    s=math.sin(theta)
    q=(math.sin((1-float(u))*theta)/s)*a+(math.sin(float(u)*theta)/s)*b
    q=q/max(float(np.linalg.norm(q)),1e-12)
    return tuple(map(float,q))


def _sample_track(track,time_seconds):
    keys=tuple(track.keyframes)
    if not keys:
        raise QualificationError("MOTION_V2_DYNAMIC_TRACK_EMPTY")
    if time_seconds<=keys[0].time_seconds:
        return (
            keys[0].local_rotation_quat_xyzw,
            keys[0].local_translation_xyz,
            keys[0].local_scale_xyz,
        )
    if time_seconds>=keys[-1].time_seconds:
        return (
            keys[-1].local_rotation_quat_xyzw,
            keys[-1].local_translation_xyz,
            keys[-1].local_scale_xyz,
        )
    for a,b in zip(keys,keys[1:]):
        ta=float(a.time_seconds); tb=float(b.time_seconds)
        if ta<=time_seconds<=tb:
            u=(float(time_seconds)-ta)/(tb-ta)
            q=_slerp(a.local_rotation_quat_xyzw,b.local_rotation_quat_xyzw,u)
            t=tuple(
                (1-u)*float(x)+u*float(y)
                for x,y in zip(a.local_translation_xyz,b.local_translation_xyz)
            )
            s=tuple(
                (1-u)*float(x)+u*float(y)
                for x,y in zip(a.local_scale_xyz,b.local_scale_xyz)
            )
            return q,t,s
    raise QualificationError("MOTION_V2_DYNAMIC_TRACK_SAMPLE_FAIL")


def _mat4(R=None,t=(0.0,0.0,0.0),scale=(1.0,1.0,1.0)):
    out=np.eye(4,dtype=np.float64)
    if R is not None:
        out[:3,:3]=np.asarray(R,dtype=np.float64)@np.diag(np.asarray(scale,dtype=np.float64))
    out[:3,3]=np.asarray(t,dtype=np.float64)
    return out


def _rest_global_transforms(skeleton,frames):
    by={j.canonical_joint_id:j for j in skeleton.joints}
    out={}
    for jid in by:
        frame=frames[jid]
        R=np.asarray(frame.rotation_matrix,dtype=np.float64)
        p=np.asarray(frame.rest_position,dtype=np.float64)
        out[jid]=_mat4(R,p)
    return out


def _topological_order(skeleton):
    by={j.canonical_joint_id:j for j in skeleton.joints}
    if len(by)!=len(skeleton.joints) or skeleton.root_id not in by:
        raise QualificationError("MOTION_V2_DYNAMIC_SKELETON_INVALID")
    order=[]
    visiting=set(); done=set()
    def visit(jid):
        if jid in done:return
        if jid in visiting:raise QualificationError("MOTION_V2_DYNAMIC_SKELETON_CYCLE")
        visiting.add(jid)
        p=by[jid].parent_canonical_id
        if p is not None:
            if p not in by:raise QualificationError("MOTION_V2_DYNAMIC_PARENT_UNKNOWN")
            visit(p)
        visiting.remove(jid); done.add(jid); order.append(jid)
    for jid in sorted(by):
        visit(jid)
    return tuple(order),by


def _joint_pose_v2(*,skeleton,tracks,time_seconds,cameras):
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    rest_global=_rest_global_transforms(skeleton,frames)
    order,by=_topological_order(skeleton)
    posed={}
    for jid in order:
        joint=by[jid]
        parent=joint.parent_canonical_id
        track=tracks.get(jid)
        if track is None:
            q=(0.0,0.0,0.0,1.0); translation=(0.0,0.0,0.0); scale=(1.0,1.0,1.0)
        else:
            q,translation,scale=_sample_track(track,time_seconds)
        D=_mat4(_quat_matrix_xyzw(q),translation,scale)
        if parent is None:
            local_rest=rest_global[jid]
            posed[jid]=local_rest@D
        else:
            local_rest=np.linalg.inv(rest_global[parent])@rest_global[jid]
            posed[jid]=posed[parent]@local_rest@D
    skin={jid:posed[jid]@np.linalg.inv(rest_global[jid]) for jid in order}
    positions={jid:tuple(map(float,posed[jid][:3,3])) for jid in order}
    return skin,positions,frame_set_hash(frames)


def _sample_times(clip,contact_rows):
    duration=float(clip.duration_seconds)
    times={float(x) for x in np.linspace(0.0,duration,_UNIFORM_SAMPLE_COUNT)}
    for track in clip.tracks:
        for key in track.keyframes:
            times.add(float(key.time_seconds))
    for row in contact_rows:
        times.add(float(row.start_time_seconds))
        times.add(float(row.end_time_seconds))
    return tuple(sorted(times))


def _validate_bindings(*,motion,constraints,product_state,skeleton,mesh,mesh_skin,presentation,mesh_policy):
    if motion.motion_lineage_hash!=qualified_motion_v2_hash(motion):
        raise QualificationError("MOTION_V2_DYNAMIC_SOURCE_HASH_DRIFT")
    if constraints.constraint_set_hash!=motion_constraint_set_v2_hash(constraints):
        raise QualificationError("MOTION_V2_DYNAMIC_CONSTRAINT_HASH_DRIFT")
    if product_state.product_state_hash!=canonical_puppet_state_hash(product_state):
        raise QualificationError("MOTION_V2_DYNAMIC_PRODUCT_STATE_HASH_DRIFT")
    if mesh.mesh_lineage_hash!=qualified_mesh_lineage_hash(mesh):
        raise QualificationError("MOTION_V2_DYNAMIC_MESH_HASH_DRIFT")
    if mesh_skin.mesh_skin_lineage_hash!=_mesh_skin_hash(mesh_skin):
        raise QualificationError("MOTION_V2_DYNAMIC_MESH_SKIN_HASH_DRIFT")
    if presentation.presentation_lineage_hash!=qualified_presentation_lineage_hash(presentation):
        raise QualificationError("MOTION_V2_DYNAMIC_PRESENTATION_HASH_DRIFT")
    if mesh_policy.qualification_policy_lineage_hash!=_policy_hash(mesh_policy):
        raise QualificationError("MOTION_V2_DYNAMIC_MESH_POLICY_HASH_DRIFT")
    checks=(
        (motion.product_state_binding_hash,product_state.product_state_hash,"PRODUCT"),
        (motion.skeleton_binding_hash,skeleton.skeleton_lineage_hash,"SKELETON"),
        (motion.presentation_binding_hash,presentation.presentation_lineage_hash,"PRESENTATION"),
        (motion.constraint_set_binding_hash,constraints.constraint_set_hash,"CONSTRAINT"),
        (mesh_skin.mesh_binding_hash,mesh.mesh_lineage_hash,"MESH_SKIN_MESH"),
        (mesh_skin.skeleton_binding_hash,skeleton.skeleton_lineage_hash,"MESH_SKIN_SKELETON"),
        (mesh.qualification_policy_hash,mesh_policy.qualification_policy_lineage_hash,"MESH_POLICY"),
    )
    for actual,wanted,label in checks:
        if actual!=wanted:
            raise QualificationError("MOTION_V2_DYNAMIC_BINDING_DRIFT:"+label)
    if str(mesh.qualification_report.get("g3_stress_probe_status"))!="PASS":
        raise QualificationError("MOTION_V2_DYNAMIC_REQUIRES_STAGE27_G3_PASS")
    components={v.component_id for v in mesh.vertices}
    owners={}
    for attachment in presentation.attachments:
        for component_id in attachment.mechanical_component_ids:
            owners.setdefault(component_id,[]).append(attachment.attachment_id)
    if set(owners)!=components:
        raise QualificationError("MOTION_V2_DYNAMIC_COMPONENT_ATTACHMENT_ACCOUNTING_DRIFT")


def build_qualified_dynamic_motion_v2(
    *,
    motion:QualifiedMotion3DIR,
    constraints:MotionCompileConstraintSetV2IR,
    product_state,
    skeleton,
    mesh,
    mesh_skin,
    presentation,
    mesh_policy,
    cameras,
    source_foreground_masks,
):
    _validate_bindings(
        motion=motion,constraints=constraints,product_state=product_state,
        skeleton=skeleton,mesh=mesh,mesh_skin=mesh_skin,
        presentation=presentation,mesh_policy=mesh_policy,
    )
    cameras=tuple(sorted(tuple(cameras),key=lambda c:int(c.view_index)))
    if len(cameras)!=8 or tuple(int(c.view_index) for c in cameras)!=tuple(range(8)):
        raise QualificationError("MOTION_V2_DYNAMIC_REQUIRES_8_CAMERAS")
    source_foreground_masks={int(k):bytes(v) for k,v in dict(source_foreground_masks).items()}
    if set(source_foreground_masks)!=set(range(8)):
        raise QualificationError("MOTION_V2_DYNAMIC_SOURCE_MASKS_INCOMPLETE")

    vids,jids,W=_weights(mesh,mesh_skin,skeleton)
    rest=np.asarray([v.P for v in mesh.vertices],dtype=np.float64)
    if not np.isfinite(rest).all():
        raise QualificationError("MOTION_V2_DYNAMIC_REST_NONFINITE")

    rest_observed_faces=set()
    for camera in cameras:
        counts=rasterize_visible_face_pixel_counts(
            mesh,camera,positions=rest,
            width=int(camera.resolution),height=int(camera.resolution),
            pixel_mask=source_foreground_masks[int(camera.view_index)],
        )
        rest_observed_faces.update(i for i,count in enumerate(counts) if count>0)
    truly_unseen_faces=set(range(len(mesh.faces)))-rest_observed_faces

    scale=_scale(skeleton)
    contact_tol=_CONTACT_REL_TOL*scale
    motion_eps=_MOTION_REL_EPS*scale
    contacts_by_clip={}
    for row in constraints.contact_constraints:
        contacts_by_clip.setdefault(row.clip_id,[]).append(row)

    clips=[]
    any_nonzero=False
    frame_set_hashes=set()

    for clip in motion.clips:
        tracks={track.canonical_joint_id:track for track in clip.tracks}
        if len(tracks)!=len(clip.tracks):
            raise QualificationError("MOTION_V2_DYNAMIC_DUPLICATE_TRACK")
        contact_rows=tuple(contacts_by_clip.get(clip.clip_id,()))
        times=_sample_times(clip,contact_rows)
        frames=[]
        poses={}
        clip_max=0.0

        for time_seconds in times:
            skin,joint_positions,rest_frame_hash=_joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=time_seconds,
                cameras=cameras,
            )
            frame_set_hashes.add(rest_frame_hash)
            posed=_skin_vertices(rest,W,jids,skin)
            displacement=np.linalg.norm(posed-rest,axis=1)
            max_disp=float(displacement.max(initial=0.0))
            clip_max=max(clip_max,max_disp)
            min_area,max_area,max_condition=_frame_metrics(mesh,rest,posed,mesh_policy)

            frame=CanonicalDynamicFrameIR(
                time_seconds=float(time_seconds),
                joint_world_positions=tuple(
                    (jid,tuple(map(float,joint_positions[jid])))
                    for jid in sorted(joint_positions)
                ),
                posed_vertex_xyz=tuple(
                    (vid,tuple(map(float,posed[i])))
                    for i,vid in enumerate(vids)
                ),
                max_vertex_displacement=max_disp,
                min_triangle_area_ratio=float(min_area),
                max_triangle_area_ratio=float(max_area),
                max_triangle_condition_number=float(max_condition),
                frame_hash="",
                metadata={
                    "canonical_3d_authority":True,
                    "view_projection_performed":False,
                    "motion_input":"LOCAL_QUATERNION_V2",
                    "derived_joint_frame_set_hash":rest_frame_hash,
                },
            )
            frame=replace(frame,frame_hash=canonical_dynamic_frame_hash(frame))

            for camera in cameras:
                counts=rasterize_visible_face_pixel_counts(
                    mesh,camera,positions=posed,
                    width=int(camera.resolution),height=int(camera.resolution),
                )
                exposed=[i for i in truly_unseen_faces if counts[i]>0]
                if exposed:
                    raise QualificationError(
                        "MOTION_V2_DYNAMIC_TRULY_UNSEEN_EXPOSURE:"
                        f"{clip.clip_id}:t={float(time_seconds)}:"
                        f"v={int(camera.view_index)}:faces={len(exposed)}"
                    )
            frames.append(frame)
            poses[float(time_seconds)]=joint_positions

        contact_proofs=[]
        for row in contact_rows:
            anchor=np.asarray(
                poses[float(row.start_time_seconds)][row.canonical_joint_id][:2],
                dtype=np.float64,
            )
            relevant=[
                t for t in times
                if float(row.start_time_seconds)-1e-12<=t<=float(row.end_time_seconds)+1e-12
            ]
            drifts=[
                float(
                    np.linalg.norm(
                        np.asarray(poses[t][row.canonical_joint_id][:2],dtype=np.float64)-anchor
                    )
                )
                for t in relevant
            ]
            max_drift=max(drifts,default=0.0)
            status="PASS" if max_drift<=contact_tol else "FAIL"
            proof=DynamicContactProofIR(
                row.contact_id,row.clip_id,row.canonical_joint_id,
                tuple(map(float,anchor)),max_drift,contact_tol,len(relevant),
                status,"",
                metadata={"mode":row.mode,"anchor_time_seconds":float(row.start_time_seconds)},
            )
            proof=replace(proof,proof_hash=dynamic_contact_proof_hash(proof))
            if status!="PASS":
                raise QualificationError("MOTION_V2_DYNAMIC_CONTACT_FAIL:"+row.contact_id)
            contact_proofs.append(proof)

        nonzero=clip_max>motion_eps
        any_nonzero=any_nonzero or nonzero
        clip_proof=DynamicMotionClipProofIR(
            clip_id=clip.clip_id,
            clip_kind=clip.clip_kind,
            classification=clip.classification,
            duration_seconds=clip.duration_seconds,
            loop=clip.loop,
            frames=tuple(frames),
            contact_proofs=tuple(contact_proofs),
            max_vertex_displacement=clip_max,
            professional_motion_evidence=bool(clip.classification=="ARTIST_SOURCE" and nonzero),
            clip_proof_hash="",
            metadata={
                "canonical_motion_nonzero":nonzero,
                "motion_nonzero_epsilon":motion_eps,
                "frame_count":len(frames),
                "source_rig_is_final_skeleton_authority":False,
                "target_skeleton_authority":skeleton.skeleton_lineage_hash,
                "truly_unseen_source_face_count":int(len(truly_unseen_faces)),
                "truly_unseen_dynamic_exposure_status":"PASS_ZERO_EXPOSED_PIXELS",
            },
        )
        clips.append(
            replace(
                clip_proof,
                clip_proof_hash=dynamic_motion_clip_proof_hash(clip_proof),
            )
        )

    if not clips:
        raise QualificationError("MOTION_V2_DYNAMIC_CLIP_SET_EMPTY")
    if not any_nonzero:
        raise QualificationError("MOTION_V2_DYNAMIC_ALL_CLIPS_STATIC")
    if len(frame_set_hashes)!=1:
        raise QualificationError("MOTION_V2_DYNAMIC_JOINT_FRAME_SET_DRIFT")

    value=QualifiedDynamicMotionIR(
        qualified_motion_binding_hash=motion.motion_lineage_hash,
        constraint_set_binding_hash=constraints.constraint_set_hash,
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        mesh_skin_binding_hash=mesh_skin.mesh_skin_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        mesh_policy_binding_hash=mesh_policy.qualification_policy_lineage_hash,
        evaluator_semantic_version=DYNAMIC_EVALUATOR_SEMANTIC_VERSION_V2,
        clips=tuple(sorted(clips,key=lambda c:c.clip_id)),
        qualification_report={
            "status":"PASS_DYNAMIC_CANONICAL_3D",
            "dynamic_proof_passed":True,
            "full_3d_local_quaternion_motion":True,
            "canonical_3d_single_mesh_truth":True,
            "view_specific_motion_mesh_truth":False,
            "all_contacts_satisfied":True,
            "all_sampled_frames_conditioned":True,
            "truly_unseen_dynamic_exposure_passed":True,
            "truly_unseen_exposed_pixel_budget":0,
            "truly_unseen_source_face_count":int(len(truly_unseen_faces)),
            "any_clip_nonzero":True,
            "professional_motion_clip_count":sum(c.professional_motion_evidence for c in clips),
            "artist_source_clip_count":sum(c.classification=="ARTIST_SOURCE" for c in clips),
            "mechanical_probe_clip_count":0,
            "source_rig_is_final_skeleton_authority":False,
        },
        dynamic_motion_hash="",
        metadata={
            "runtime_export_authorized":True,
            "runtime_must_project_these_exact_canonical_frames":True,
            "uniform_sample_count":_UNIFORM_SAMPLE_COUNT,
            "contact_relative_tolerance":_CONTACT_REL_TOL,
            "evaluator_semantic_version":DYNAMIC_EVALUATOR_SEMANTIC_VERSION_V2,
            "derived_joint_frame_set_hash":next(iter(frame_set_hashes)),
            "truly_unseen_policy":"FAIL_ON_ANY_NEW_Z_VISIBLE_PIXEL_IN_8_RUNTIME_VIEWS",
        },
    )
    return replace(value,dynamic_motion_hash=qualified_dynamic_motion_hash(value))
