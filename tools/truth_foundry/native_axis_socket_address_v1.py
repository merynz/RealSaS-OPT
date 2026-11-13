from __future__ import annotations

"""Source-only AXIS V6 socket/anchor parent-frame contract.

Take the real source-authored socket candidate and the *actual* AXIS source
teacher causal sequence, and check that the declared anchor event belongs to
the declared SOCKET_CONTROL, is parent-local rather than world XYZ, and
quantizes the explicitly authored native anchor coordinates. This prevents
attaching a correct source face pair to the WRONG AXIS anchor/control program.

This court is intentionally NOT a trusted-source registrar. All authoring
sha256 fields denote externally obtained source records, and a separate
launcher must verify their bytes/rights/lineage and sign/accept any training
registry row. No student inference uses native anchor XYZ.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, AXISRigProgramConfigV6, RigTokenKindV6,
    RigGrammarStateV6, AXISRigProgramGrammarV6Product,
    CONTROL_KINDS_V6,
)
from models.axis.rig_program_decode_v6 import _local_vector
from models.rss_vnext.attachment_face_correspondence_v1 import (
    axis_sequence_digest_v1,
)
from tools.truth_foundry.native_socket_attachment_teacher_v1 import (
    NativeSocketAttachmentCandidateV1,SOURCE_ONLY,
)

SCHEMA="RealSaS.TruthFoundry.NativeAXISSocketAnchorAddressCandidate.v1"
AUTHORITY="AXIS_CAUSAL_PARENT_LOCAL_COORDINATE_CONSISTENCY_ONLY"


def _fail(code:str)->None:
    raise ValueError("RSS_VNEXT_NATIVE_AXIS_SOCKET_"+code)


def _sha(value:object)->bool:
    return type(value) is str and len(value)==64 and all(
        c in "0123456789abcdef" for c in value
    )


@dataclass(frozen=True)
class NativeAXISSocketAnchorAddressCandidateV1:
    axis_anchor_event_index:int
    axis_parent_control_event_index:int
    axis_parent_control_ordinal:int
    axis_anchor_coord_bins:tuple[int,int,int]
    decoded_axis_parent_local_xyz:tuple[float,float,float]
    authored_native_parent_local_xyz:tuple[float,float,float]
    max_coord_quantization_residual:float
    source_world_anchor_xyz:tuple[float,float,float]
    axis_teacher_sequence_sha256:str
    source_socket_face_evidence_sha256:str
    source_authored_attachment_record_sha256:str
    native_parent_frame_provenance_sha256:str
    schema:str=SCHEMA
    authority:str=AUTHORITY
    parent_source_independently_admitted:bool=False
    source_contact_qualified:bool=False
    source_attachment_executable:bool=False
    product_revision_qualified:bool=False


def validate_native_axis_socket_anchor_teacher_v1(
    *,
    source_candidate:NativeSocketAttachmentCandidateV1,
    axis_teacher_sequence:AXISSequenceV6,
    axis_config:AXISRigProgramConfigV6,
    tokens_per_joint:int,
    axis_normalization_scale:float,
    authored_native_parent_local_xyz:tuple[float,float,float],
    authored_native_world_anchor_xyz:tuple[float,float,float],
    native_parent_rest_world:np.ndarray,
    authored_attachment_record_sha256:str,
    parent_native_frame_provenance_sha256:str,
)->NativeAXISSocketAnchorAddressCandidateV1:
    if (
        type(source_candidate) is not NativeSocketAttachmentCandidateV1
        or source_candidate.authority != SOURCE_ONLY
        or any((
            source_candidate.trainer_admission_qualified,
            source_candidate.source_contact_qualified,
            source_candidate.executable_attachment_qualified,
        ))
        or source_candidate.authoring_provenance_sha256!=authored_attachment_record_sha256
        or source_candidate.parent_frame_provenance_sha256!=parent_native_frame_provenance_sha256
        or not _sha(authored_attachment_record_sha256)
        or not _sha(parent_native_frame_provenance_sha256)
        or type(axis_teacher_sequence) is not AXISSequenceV6
        or type(axis_config) is not AXISRigProgramConfigV6
        or type(tokens_per_joint) is not int or tokens_per_joint<1
        or not math.isfinite(axis_normalization_scale) or axis_normalization_scale<=0
    ):
        _fail("SOURCE_OR_AXIS_TEACHER_DOMAIN")
    axis_teacher_sequence.validate(axis_config)
    if (
        axis_teacher_sequence.kind_ids.shape[0]!=1
        or not bool(axis_teacher_sequence.attention_mask.all())
        or axis_teacher_sequence.kind_ids.shape[1]<3
        or int(axis_teacher_sequence.kind_ids[0,0])!=int(RigTokenKindV6.BOS)
        or int(axis_teacher_sequence.kind_ids[0,-1])!=int(RigTokenKindV6.EOS)
    ):
        _fail("CAUSAL_PROGRAM_COMPLETE_UNPADDED_REQUIRED")
    sequence=axis_teacher_sequence
    state=RigGrammarStateV6()
    for token in sequence.kind_ids[0].tolist():
        try:
            state=AXISRigProgramGrammarV6Product.advance(
                state,RigTokenKindV6(int(token)),
                tokens_per_joint=tokens_per_joint,
            )
        except (ValueError,TypeError) as exc:
            _fail("CAUSAL_PROGRAM_GRAMMAR_DRIFT")
    if state.section.name!="DONE":
        _fail("CAUSAL_PROGRAM_INCOMPLETE")
    controls=torch.where(
        sequence.kind_ids[0]==int(RigTokenKindV6.CONTROL),
    )[0].tolist()
    idx=source_candidate.axis_anchor_event_index
    ord_=source_candidate.socket_control_index
    if (
        type(idx) is not int or type(ord_) is not int
        or not 0<=idx<sequence.kind_ids.shape[1]
        or not 0<=ord_<len(controls)
        or int(sequence.kind_ids[0,idx])!=int(RigTokenKindV6.ANCHOR)
        or int(sequence.anchor_parent_kind[0,idx])!=1
        or int(sequence.pointer_b[0,idx])!=ord_
    ):
        _fail("ANCHOR_PARENT_CONTROL_POINTER_DRIFT")
    control_event=controls[ord_]
    if (
        int(sequence.aux_type_ids[0,control_event])
        !=CONTROL_KINDS_V6.index("SOCKET_CONTROL")
        or control_event>=idx
    ):
        _fail("SOCKET_KIND_OR_CAUSAL_ORDER_DRIFT")
    bins=tuple(int(i) for i in sequence.coord_indices[0,idx].tolist())
    if any(i<0 or i>=axis_config.coord_bins for i in bins):
        _fail("ANCHOR_COORDINATE_BIN_DOMAIN")
    for vec in (authored_native_parent_local_xyz,authored_native_world_anchor_xyz):
        if (
            type(vec) is not tuple or len(vec)!=3
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in vec)
        ):
            _fail("SOURCE_NATIVE_ANCHOR_COORDINATE_DOMAIN")
    parent=np.asarray(native_parent_rest_world,dtype=np.float64)
    if (
        parent.shape!=(4,4) or not np.isfinite(parent).all()
        or not np.allclose(parent[3,:],[0.,0.,0.,1.],atol=1.e-8,rtol=0)
        or not np.allclose(parent[:3,:3].T@parent[:3,:3],
                           np.eye(3),atol=1.e-6,rtol=0)
        or abs(float(np.linalg.det(parent[:3,:3]))-1.)>1.e-6
    ):
        _fail("NATIVE_REST_PARENT_FRAME_NOT_SE3")
    native=np.asarray(authored_native_parent_local_xyz,dtype=np.float64)
    world=parent[:3,:3]@native+parent[:3,3]
    source_world=np.asarray(authored_native_world_anchor_xyz,dtype=np.float64)
    tolerance=1.e-7*max(1.,axis_normalization_scale)
    if not np.allclose(world,source_world,atol=tolerance,rtol=0):
        _fail("NATIVE_PARENT_LOCAL_VS_WORLD_ANCHOR_DRIFT")

    actual=np.asarray(
        _local_vector(axis_config,bins,scale=axis_normalization_scale),
        dtype=np.float64,
    )
    step=axis_normalization_scale*(
        axis_config.coord_max-axis_config.coord_min
    )/(axis_config.coord_bins-1)
    if not math.isfinite(step) or step<=0:
        _fail("AXIS_ANCHOR_COORDINATE_CODEC_POLICY_INVALID")
    residual=float(np.max(np.abs(actual-native)))
    if residual>step*.50001+1.e-8:
        _fail("AXIS_LOCAL_ANCHOR_NOT_SOURCE_QUANTIZED")
    axis_sha=axis_sequence_digest_v1(sequence)
    source_sha=sha256(json.dumps({
        "source_asset_sha256":source_candidate.source_asset_sha256,
        "source_intervention_bank_sha256":source_candidate.source_intervention_bank_sha256,
        "scene_measurement_sha256":source_candidate.scene_measurement_sha256,
        "native_tessa_reindex_candidate_sha256":
            source_candidate.native_tessa_reindex_candidate_sha256,
        "source_held_face_index":source_candidate.source_held_face_index,
        "source_holder_face_index":source_candidate.source_holder_face_index,
        "authoring_class":source_candidate.authoring_class,
        "relationship_state":source_candidate.relationship_state,
        "axis_anchor_event_index":idx,
        "socket_control_index":ord_,
    },sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return NativeAXISSocketAnchorAddressCandidateV1(
        axis_anchor_event_index=idx,
        axis_parent_control_event_index=control_event,
        axis_parent_control_ordinal=ord_,
        axis_anchor_coord_bins=bins,
        decoded_axis_parent_local_xyz=tuple(float(x) for x in actual),
        authored_native_parent_local_xyz=authored_native_parent_local_xyz,
        max_coord_quantization_residual=residual,
        source_world_anchor_xyz=authored_native_world_anchor_xyz,
        axis_teacher_sequence_sha256=axis_sha,
        source_socket_face_evidence_sha256=source_sha,
        source_authored_attachment_record_sha256=authored_attachment_record_sha256,
        native_parent_frame_provenance_sha256=parent_native_frame_provenance_sha256,
    )


__all__=[
    "SCHEMA","AUTHORITY","NativeAXISSocketAnchorAddressCandidateV1",
    "validate_native_axis_socket_anchor_teacher_v1",
]
