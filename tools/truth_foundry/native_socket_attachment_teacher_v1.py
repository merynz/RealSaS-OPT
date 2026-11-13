from __future__ import annotations

"""Source-native held/holder supervision candidate from authored rig + interventions.

This is a SOURCE-ONLY Truth Foundry court, never a two-view student input,
Compiler discovery operator or a trusted training admission registry.

Three separate propositions must NOT be collapsed:
  (1) native authoring declares a specific parent socket and held/holder faces;
  (2) measured discovery and HELD-OUT source motion supports rigid transport
      and the independent source parent-frame follows those surfaces;
  (3) a separately verified source-native/TESSA face bijection transfers the
      two face addresses into one specific teacher topology.

A rig's influence weights, co-moving surfaces under a global action, proximity,
unverified parent frames, or absence of an authored binding never supplies a
positive. A verified NO_FACE target requires an EXHAUSTIVE authored empty
attachment inventory and an independently exercised nonrigid decoy response.
UNKNOWN is MASKED, never converted to NO_FACE.

Output does not self-award admission: source provenance, actual native
constraint inventory, action independence, source rights and held-out
counterfactual identity must be admitted by an external foundry launcher.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np

from models.iris.truth.mechanics_oracle_v1 import (
    SurfacePairQueryV1,measure_relative_transport_v1,
    RelativeTransportPolicyV1,
)
from tools.truth_foundry.native_tessa_face_reindex_v1 import (
    NativeTESSAFaceReindexCandidateV1,
    read_native_tessa_face_reindex_candidate_v1,
)

SCHEMA="RealSaS.TruthFoundry.NativeSocketAttachmentTeacherCandidate.v1"
SOURCE_ONLY="SOURCE_NATIVE_EVIDENCE_CANDIDATE_EXTERNAL_ADMISSION_REQUIRED"
_AUTHORING={
    "EXPLICIT_SOURCE_AUTHORED_RIGID_ATTACHMENT",
    "EXHAUSTIVE_SOURCE_AUTHORED_NO_ATTACHMENT",
    "UNKNOWN",
}
_POS="POSITIVE_CANDIDATE_NOT_YET_ADMITTED"
_NEG="NEGATIVE_NO_FACE_CANDIDATE_NOT_YET_ADMITTED"
_UNK="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"


def _fail(msg:str)->None:
    raise ValueError("RSS_VNEXT_NATIVE_SOCKET_TRUTH_"+msg)


def _sha(x:object)->bool:
    return isinstance(x,str) and type(x) is str and len(x)==64 and all(
        c in "0123456789abcdef" for c in x
    )


def _tensor_digest(a:np.ndarray)->str:
    x=np.ascontiguousarray(a)
    head=json.dumps({"dtype":str(x.dtype),"shape":list(x.shape)},
                    sort_keys=True,separators=(",",":")).encode()
    return sha256(head+b"\x00"+x.tobytes()).hexdigest()


def _record_digest(obj:dict)->str:
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def _homogeneous_frames(values:np.ndarray, *, name:str)->np.ndarray:
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(4,4) or not np.isfinite(a).all() or a.shape[0]<1:
        _fail(name+"_FRAMES_SHAPE_OR_NONFINITE")
    if not np.allclose(a[:,3,:],np.array([0.,0.,0.,1.]),atol=1.e-8,rtol=0):
        _fail(name+"_HOMOGENEOUS_LAST_ROW")
    for rot in a[:,:3,:3]:
        if (
            not np.allclose(rot.T@rot,np.eye(3),atol=1.e-6,rtol=0)
            or not abs(float(np.linalg.det(rot))-1.)<1.e-6
        ):
            _fail(name+"_ROTATION_NOT_SO3")
    return a


def _angle(r1:np.ndarray,r2:np.ndarray)->float:
    cosine=np.clip((np.trace(r1.T@r2)-1.)*.5,-1.,1.)
    return float(np.arccos(cosine))


def _triangle_frame(vertices:np.ndarray,tri:np.ndarray)->np.ndarray|None:
    p=vertices[tri]
    edge=p[1]-p[0]
    n=np.cross(edge,p[2]-p[0])
    ln=np.linalg.norm(edge); nn=np.linalg.norm(n)
    if ln<1.e-9 or nn<1.e-12:
        return None
    x=edge/ln; z=n/nn
    y=np.cross(z,x)
    frame=np.eye(4,dtype=np.float64)
    frame[:3,:3]=np.stack((x,y,z),axis=1)
    frame[:3,3]=p.mean(axis=0)
    return frame


def _max_parent_face_transport(
    *, rest:np.ndarray, faces:np.ndarray, face_index:int,
    posed:np.ndarray, parent_rest:np.ndarray, parent_frames:np.ndarray,
    subject_scale:float,
)->tuple[bool,float,float]:
    ref=_triangle_frame(rest,faces[face_index])
    if ref is None:
        return False,float("inf"),float("inf")
    ref_local=np.linalg.inv(parent_rest)@ref
    worst_angle=0.;worst_offset=0.
    for vertices,frame in zip(posed,parent_frames):
        measured=_triangle_frame(vertices,faces[face_index])
        if measured is None:
            return False,float("inf"),float("inf")
        local=np.linalg.inv(frame)@measured
        worst_angle=max(worst_angle,_angle(ref_local[:3,:3],local[:3,:3]))
        worst_offset=max(
            worst_offset,float(np.linalg.norm(
                ref_local[:3,3]-local[:3,3]
            )/subject_scale),
        )
    return True,worst_angle,worst_offset


def _parent_excitation(
    *, rest:np.ndarray, discovery:np.ndarray, validation:np.ndarray,
    subject_scale:float,
)->tuple[float,float]:
    counts=[]
    for bank in (discovery,validation):
        delta=[
            max(_angle(rest[:3,:3],p[:3,:3]),
                float(np.linalg.norm(p[:3,3]-rest[:3,3])/subject_scale))
            for p in bank
        ]
        counts.append(max(delta))
    return tuple(counts)


@dataclass(frozen=True)
class NativeSocketAttachmentRequestV1:
    axis_anchor_event_index:int
    socket_control_index:int
    authoring_class:Literal[
        "EXPLICIT_SOURCE_AUTHORED_RIGID_ATTACHMENT",
        "EXHAUSTIVE_SOURCE_AUTHORED_NO_ATTACHMENT",
        "UNKNOWN",
    ]
    native_held_face_index:int|None
    native_holder_face_index:int|None
    # SHA of original, independently extracted rig/constraint or exhaustive
    # inventory record, NOT a string minted from the model's prediction.
    authored_native_attachment_record_sha256:str
    parent_control_native_frame_sha256:str
    # External authority marks independent parent and holder relative
    # intervention action diversity. This flag alone NEVER grants admission.
    independently_actuated_parent:bool


@dataclass(frozen=True)
class NativeSocketAttachmentCandidateV1:
    axis_anchor_event_index:int
    socket_control_index:int
    authoring_class:str
    relationship_state:str
    source_held_face_index:int|None
    source_holder_face_index:int|None
    teacher_held_face_index:int|None
    teacher_holder_face_index:int|None
    teacher_held_target:int
    teacher_holder_target:int
    held_target_mask:bool
    holder_target_mask:bool
    relative_transport_status:str
    heldout_confirmed:bool
    max_held_parent_angle_rad:float|None
    max_held_parent_translation_fraction:float|None
    max_holder_parent_angle_rad:float|None
    max_holder_parent_translation_fraction:float|None
    discovery_parent_excitation:float
    validation_parent_excitation:float
    authoring_provenance_sha256:str
    parent_frame_provenance_sha256:str
    source_asset_sha256:str
    source_intervention_bank_sha256:str
    normalized_mesh_digest:str
    tessa_topology_digest:str
    native_tessa_reindex_candidate_sha256:str
    scene_measurement_sha256:str
    authority:str=SOURCE_ONLY
    trainer_admission_qualified:bool=False
    source_contact_qualified:bool=False
    executable_attachment_qualified:bool=False


def build_native_socket_attachment_teacher_candidate_v1(
    *,
    source_vertices_normalized:np.ndarray,
    source_rest_vertices_world:np.ndarray,
    source_faces:np.ndarray,
    discovery_positions_world:np.ndarray,
    validation_positions_world:np.ndarray,
    source_world_from_normalized:np.ndarray,
    native_tessa_reindex:NativeTESSAFaceReindexCandidateV1,
    request:NativeSocketAttachmentRequestV1,
    parent_rest_world:np.ndarray,
    parent_discovery_world:np.ndarray,
    parent_validation_world:np.ndarray,
    source_asset_sha256:str,
    source_intervention_bank_sha256:str,
    subject_scale:float,
    degeneracy_tolerance:float=1.e-12,
    rigid_parent_angle_tolerance_rad:float=1.e-3,
    rigid_parent_translation_fraction:float=1.e-4,
    minimum_excitation:float=5.e-3,
)->NativeSocketAttachmentCandidateV1:
    if type(request) is not NativeSocketAttachmentRequestV1:
        _fail("AUTHORED_REQUEST_TYPE")
    if (
        request.authoring_class not in _AUTHORING
        or any(type(i) is not int or i<0 for i in (
            request.axis_anchor_event_index,request.socket_control_index
        ))
        or not all(_sha(x) for x in (
            request.authored_native_attachment_record_sha256,
            request.parent_control_native_frame_sha256,
            source_asset_sha256,source_intervention_bank_sha256,
        ))
        or type(request.independently_actuated_parent) is not bool
    ):
        _fail("EXTERNAL_SOURCE_OR_AUTHORING_DOMAIN")
    if native_tessa_reindex is None:
        _fail("EXACT_SOURCE_TESSA_REINDEX_REQUIRED")
    # This caller has no ability to mark a structural face mapping as
    # admitted. Its roundtrip validates hash, indices and fixed authority.
    read_native_tessa_face_reindex_candidate_v1(
        __import__("dataclasses").asdict(native_tessa_reindex),
    )
    x=np.asarray(source_vertices_normalized,dtype=np.float64)
    rest=np.asarray(source_rest_vertices_world,dtype=np.float64)
    faces=np.asarray(source_faces)
    discovery=np.asarray(discovery_positions_world,dtype=np.float64)
    validation=np.asarray(validation_positions_world,dtype=np.float64)
    transform=np.asarray(source_world_from_normalized,dtype=np.float64)
    if (
        x.ndim!=2 or x.shape[1:]!=(3,) or not np.isfinite(x).all()
        or rest.shape!=x.shape or not np.isfinite(rest).all()
        or faces.ndim!=2 or faces.shape[1:]!=(3,)
        or faces.dtype.kind not in "iu" or len(faces)!=len(native_tessa_reindex.source_face_to_tessa_face)
        or len(rest)<3 or np.any(faces<0) or np.any(faces>=len(rest))
        or discovery.ndim!=3 or validation.ndim!=3
        or discovery.shape[1:]!=rest.shape or validation.shape[1:]!=rest.shape
        or len(discovery)<2 or len(validation)<1
        or not np.isfinite(discovery).all() or not np.isfinite(validation).all()
        or transform.shape!=(4,4) or not np.isfinite(transform).all()
        or not np.allclose(transform[3,:],[0.,0.,0.,1.],atol=1.e-8)
        or not np.isfinite(subject_scale) or subject_scale<=0
        or not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance<=0
        or min(rigid_parent_angle_tolerance_rad,
               rigid_parent_translation_fraction,minimum_excitation)<=0
    ):
        _fail("SOURCE_GEOMETRY_OR_INTERVENTION_DOMAIN")
    world=np.concatenate((x,np.ones((len(x),1))),axis=1)@transform.T
    if not np.allclose(
        world[:,:3],rest,atol=1.e-6*subject_scale,rtol=0,
    ):
        _fail("NATIVE_TO_TESSA_COORDINATE_FRAME_NOT_PROVEN")
    if not np.isclose(
        abs(np.linalg.det(transform[:3,:3])),0.,atol=1.e-10,
    ) and not np.linalg.cond(transform[:3,:3])>1.e8:
        pass
    else:
        _fail("SOURCE_NORMALIZATION_NONINVERTIBLE")
    # Must match the independently measured canonical normalized mesh exactly.
    from tools.truth_foundry.native_tessa_face_reindex_v1 import (
        _source_arrays_digest,
    )
    source_components=np.asarray(
        native_tessa_reindex.source_face_components,dtype=np.int64,
    )
    canonical=_source_arrays_digest(xyz=x,faces=faces,components=source_components)
    if canonical!=native_tessa_reindex.normalized_source_mesh_canonical_sha256:
        _fail("SOURCE_MESH_REINDEX_BYTES_DRIFT")
    rest_parent=_homogeneous_frames(
        np.asarray(parent_rest_world)[None],name="REST_PARENT",
    )[0]
    discovery_parent=_homogeneous_frames(
        parent_discovery_world,name="DISCOVERY_PARENT",
    )
    validation_parent=_homogeneous_frames(
        parent_validation_world,name="VALIDATION_PARENT",
    )
    if (
        len(discovery_parent)!=len(discovery)
        or len(validation_parent)!=len(validation)
    ):
        _fail("PARENT_POSE_TIMEBASE_MISMATCH")
    disc_exc,val_exc=_parent_excitation(
        rest=rest_parent,discovery=discovery_parent,
        validation=validation_parent,subject_scale=subject_scale,
    )
    # A global-body-only passive dataset is not an independent action.
    # Caller must also supply trusted native source intervention provenance.
    exciting=(
        request.independently_actuated_parent
        and disc_exc>=minimum_excitation
        and val_exc>=minimum_excitation
    )
    held=request.native_held_face_index
    holder=request.native_holder_face_index
    if request.authoring_class=="UNKNOWN" and (held is not None or holder is not None):
        _fail("UNKNOWN_CANNOT_DECLARED_FACE_TRUTH")
    if request.authoring_class!="UNKNOWN":
        if (
            type(held) is not int or type(holder) is not int
            or not 0<=held<len(faces) or not 0<=holder<len(faces)
            or held==holder
        ):
            _fail("NATIVE_AUTHORING_FACE_ADDRESS_INVALID")
    status="ABSTAIN"
    heldout=False
    hp_angle=hp_offset=bp_angle=bp_offset=None
    evidence_digest=_record_digest({
        "rest":_tensor_digest(rest),
        "faces":_tensor_digest(np.asarray(faces,dtype=np.int64)),
        "discovery":_tensor_digest(discovery),
        "validation":_tensor_digest(validation),
        "parent_rest":_tensor_digest(rest_parent),
        "parent_discovery":_tensor_digest(discovery_parent),
        "parent_validation":_tensor_digest(validation_parent),
        "source_asset":source_asset_sha256,
        "source_intervention_bank":source_intervention_bank_sha256,
    })
    relation=_UNK
    if held is not None:
        measurement=measure_relative_transport_v1(
            rest,faces,discovery,validation,
            SurfacePairQueryV1(face_a=held,face_b=holder),
            subject_scale=subject_scale,
            degeneracy_tolerance=degeneracy_tolerance,
            policy=RelativeTransportPolicyV1(),
        )
        status=measurement.final_status
        heldout=measurement.heldout_confirmed
        valid_held,hp_angle,hp_offset=_max_parent_face_transport(
            rest=rest,faces=faces,face_index=held,
            posed=np.concatenate((discovery,validation),axis=0),
            parent_rest=rest_parent,
            parent_frames=np.concatenate((discovery_parent,validation_parent),axis=0),
            subject_scale=subject_scale,
        )
        valid_holder,bp_angle,bp_offset=_max_parent_face_transport(
            rest=rest,faces=faces,face_index=holder,
            posed=np.concatenate((discovery,validation),axis=0),
            parent_rest=rest_parent,
            parent_frames=np.concatenate((discovery_parent,validation_parent),axis=0),
            subject_scale=subject_scale,
        )
        parent_supported=(
            valid_held and valid_holder
            and max(hp_angle,bp_angle)<=rigid_parent_angle_tolerance_rad
            and max(hp_offset,bp_offset)<=rigid_parent_translation_fraction
        )
        if (
            request.authoring_class=="EXPLICIT_SOURCE_AUTHORED_RIGID_ATTACHMENT"
            and exciting and heldout
            and status=="RIGID_RELATIVE_TRANSPORT_SUPPORTED"
            and parent_supported
            and (native_tessa_reindex.source_face_components[held]
                 !=native_tessa_reindex.source_face_components[holder])
        ):
            relation=_POS
        elif (
            request.authoring_class=="EXHAUSTIVE_SOURCE_AUTHORED_NO_ATTACHMENT"
            and exciting and heldout
            and status=="NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
        ):
            relation=_NEG

    nface=len(faces)
    if relation==_POS:
        target_held=native_tessa_reindex.source_face_to_tessa_face[held]
        target_holder=native_tessa_reindex.source_face_to_tessa_face[holder]
        train_held=True;train_holder=True
    elif relation==_NEG:
        target_held=target_holder=nface  # explicit source verified NO_FACE
        train_held=True;train_holder=True
    else:
        target_held=target_holder=-1
        train_held=False;train_holder=False

    return NativeSocketAttachmentCandidateV1(
        axis_anchor_event_index=request.axis_anchor_event_index,
        socket_control_index=request.socket_control_index,
        authoring_class=request.authoring_class,
        relationship_state=relation,
        source_held_face_index=held,source_holder_face_index=holder,
        teacher_held_face_index=(
            None if held is None else
            native_tessa_reindex.source_face_to_tessa_face[held]
        ),
        teacher_holder_face_index=(
            None if holder is None else
            native_tessa_reindex.source_face_to_tessa_face[holder]
        ),
        teacher_held_target=target_held,
        teacher_holder_target=target_holder,
        held_target_mask=train_held,holder_target_mask=train_holder,
        relative_transport_status=status,heldout_confirmed=heldout,
        max_held_parent_angle_rad=hp_angle,
        max_held_parent_translation_fraction=hp_offset,
        max_holder_parent_angle_rad=bp_angle,
        max_holder_parent_translation_fraction=bp_offset,
        discovery_parent_excitation=disc_exc,
        validation_parent_excitation=val_exc,
        authoring_provenance_sha256=request.authored_native_attachment_record_sha256,
        parent_frame_provenance_sha256=request.parent_control_native_frame_sha256,
        source_asset_sha256=source_asset_sha256,
        source_intervention_bank_sha256=source_intervention_bank_sha256,
        normalized_mesh_digest=canonical,
        tessa_topology_digest=native_tessa_reindex.tessa_mesh_topology_digest,
        native_tessa_reindex_candidate_sha256=native_tessa_reindex.candidate_sha256,
        scene_measurement_sha256=evidence_digest,
    )


def compile_native_socket_training_targets_v1(
    *,
    candidates:tuple[NativeSocketAttachmentCandidateV1,...],
    axis_socket_anchor_events:tuple[int,...],
    tessa_face_count:int,
)->tuple[np.ndarray,np.ndarray]:
    """Training-only mask/label proposal, NOT an admission credential.

    The independent launcher must check source credentials and exact AXIS
    source program/teacher mesh before supplying a trusted registry row.
    """
    if (
        type(candidates) is not tuple or type(axis_socket_anchor_events) is not tuple
        or not 1<=len(candidates)<=512 or len(candidates)!=len(axis_socket_anchor_events)
        or type(tessa_face_count) is not int or tessa_face_count<1
        or len(set(axis_socket_anchor_events))!=len(axis_socket_anchor_events)
    ):
        _fail("TEACHER_TARGET_RESOURCE_OR_AXIS_ORDER")
    targets=np.full((len(candidates),2),-1,dtype=np.int64)
    mask=np.zeros((len(candidates),2),dtype=np.bool_)
    for i,(candidate,event) in enumerate(zip(candidates,axis_socket_anchor_events)):
        if (
            type(candidate) is not NativeSocketAttachmentCandidateV1
            or type(event) is not int or candidate.axis_anchor_event_index!=event
            or candidate.authority!=SOURCE_ONLY
            or candidate.trainer_admission_qualified
            or candidate.source_contact_qualified
            or candidate.executable_attachment_qualified
            or candidate.teacher_held_target>=tessa_face_count+1
            or candidate.teacher_holder_target>=tessa_face_count+1
        ):
            _fail("TEACHER_TARGET_SOURCE_OR_AXIS_PROVENANCE_DRIFT")
        if candidate.relationship_state in (_POS,_NEG):
            if not candidate.heldout_confirmed:
                _fail("NO_HELDOUT_TEACHER_AUTHORITY")
            targets[i]=(
                candidate.teacher_held_target,candidate.teacher_holder_target
            )
            mask[i]=True
            if (
                (candidate.relationship_state==_NEG and
                 tuple(targets[i])!=(tessa_face_count,tessa_face_count))
                or (candidate.relationship_state==_POS and
                    (targets[i].min()<0 or targets[i].max()>=tessa_face_count))
            ):
                _fail("TEACHER_TARGET_TRUE_NEGATIVE_OR_FACE_RANGE")
        elif candidate.relationship_state!=_UNK:
            _fail("TEACHER_TARGET_RELATION_INVALID")
    return targets,mask


__all__=[
    "SCHEMA","SOURCE_ONLY","NativeSocketAttachmentRequestV1",
    "NativeSocketAttachmentCandidateV1",
    "build_native_socket_attachment_teacher_candidate_v1",
    "compile_native_socket_training_targets_v1",
]
