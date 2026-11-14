from __future__ import annotations

"""Trainable AXIS causal socket/anchor -> TESSA exact face-pair correspondence.

A source-teachable MISSING neural producer, not an attachment solver.

Two independent categorical outputs select HELD face and HOLDER face,
respectively, with an explicit last-index NO_FACE class. This is O(A*F),
NOT O(A*F**2); only requested or top-k distinct directed pairs are scored.
The head reads the actual IRIS-backed TESSA mesh and AXIS event hidden states
from one checkpoint. It sees no teacher source mesh/rig/pose in inference.

Source native face labels are ONLY accepted by a separate, externally
registry-qualified loss on a teacher-replayed TESSA face index namespace.
Unknown/unlabelled != NO_FACE; false negatives are never manufactured.
No learned prediction is canonically owned, rendered, or executable.
"""

from dataclasses import dataclass
from typing import Mapping
import hashlib
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.axis.rig_program_model_v6 import AXISSequenceV6, RigTokenKindV6
from .animation_ready_mesh import RSSvNextStructuralMeshCandidate

_AUTHORITY = "UNQUALIFIED_NEURAL_SOCKET_FACE_CORRESPONDENCE_V1"
_SOCKET_KIND = 7


@dataclass
class RSSvNextSocketFaceNeuralOutputV1:
    anchor_event_indices: tuple[int, ...]
    socket_control_indices: tuple[int, ...]
    held_face_logits: Tensor     # [A,F+1] last is real NO_FACE / abstain class
    holder_face_logits: Tensor   # [A,F+1]
    face_component_ids: Tensor   # [F], exact TESSA candidates, not truth
    face_region_probabilities: Tensor  # [F,R+1], UNKNOWN last
    region_relation_logits: Tensor     # [R,R,3], last directed
    mesh_topology_digest: str
    axis_sequence_digest: str
    authority: str = _AUTHORITY
    source_correspondence_qualified: bool = False
    executable_attachment_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextNeuralSocketFaceProposalV1:
    anchor_event_index: int
    socket_control_index: int
    held_face_index: int | None
    holder_face_index: int | None
    same_tessa_component: bool | None
    known_cross_slot_support: float
    predicted_directed_relation: float | None
    held_unknown_probability: float
    holder_unknown_probability: float
    predicted_rank_log_score: float | None
    state: str
    authority: str = _AUTHORITY
    source_contact_qualified: bool = False
    parent_rest_frame_qualified: bool = False
    attachment_qualified: bool = False


def _digest_tensor(value: Tensor) -> str:
    cpu=value.detach().contiguous().cpu()
    header=json.dumps(
        {"dtype":str(cpu.dtype),"shape":list(cpu.shape)},
        sort_keys=True,separators=(",",":"),
    ).encode()
    return hashlib.sha256(header+b"\x00"+cpu.numpy().tobytes()).hexdigest()


def _digest_fields(fields: Mapping[str,str]) -> str:
    return hashlib.sha256(json.dumps(
        dict(fields),sort_keys=True,separators=(",",":"),
    ).encode()).hexdigest()


def mesh_topology_digest_v1(mesh: RSSvNextStructuralMeshCandidate) -> str:
    return _digest_fields({
        "topology_tokens_sha256":hashlib.sha256(json.dumps(
            list(mesh.topology_token_ids),separators=(",",":"),
        ).encode()).hexdigest(),
        "faces_sha256":_digest_tensor(mesh.faces),
        "components_sha256":_digest_tensor(mesh.face_component_ids),
        "vertices_sha256":_digest_tensor(mesh.vertex_component_ids),
    })


def axis_sequence_digest_v1(seq: AXISSequenceV6) -> str:
    return _digest_fields({
        f.name:_digest_tensor(getattr(seq,f.name))
        for f in __import__("dataclasses").fields(seq)
    })


class RSSvNextSocketFaceCorrespondenceHeadV1(nn.Module):
    """Actual trainable face ownership classifier, shared checkpoint expert."""

    def __init__(
        self, *, field_width:int, axis_width:int, region_slots:int,
        hidden_width:int, max_faces:int=32768, max_anchors:int=512,
    ):
        super().__init__()
        if min(field_width,axis_width,region_slots,hidden_width,max_faces,max_anchors)<1:
            raise ValueError("RSS_VNEXT_SOCKET_FACE_HEAD_DIMENSIONS")
        self.field_width=field_width
        self.axis_width=axis_width
        self.region_slots=region_slots
        self.max_faces=max_faces
        self.max_anchors=max_anchors
        self.anchor_position=nn.Sequential(
            nn.Linear(3,hidden_width),nn.GELU(),
        )
        self.anchor_projection=nn.Sequential(
            nn.LayerNorm(axis_width*2+hidden_width),
            nn.Linear(axis_width*2+hidden_width,hidden_width),
            nn.GELU(),nn.Linear(hidden_width,hidden_width),
        )
        face_width=field_width+6+region_slots+1
        self.face_projection=nn.Sequential(
            nn.LayerNorm(face_width),nn.Linear(face_width,hidden_width),
            nn.GELU(),nn.Linear(hidden_width,hidden_width),
        )
        self.held_query=nn.Linear(hidden_width,hidden_width,bias=False)
        self.holder_query=nn.Linear(hidden_width,hidden_width,bias=False)
        self.held_key=nn.Linear(hidden_width,hidden_width,bias=False)
        self.holder_key=nn.Linear(hidden_width,hidden_width,bias=False)
        # NONE is a real learnable output, *not* fallback to vertex zero.
        self.held_none=nn.Linear(hidden_width,1)
        self.holder_none=nn.Linear(hidden_width,1)
        self.inv_scale=hidden_width**-.5

    @staticmethod
    def _socket_anchor_positions(seq:AXISSequenceV6)->tuple[tuple[int,...],tuple[int,...],tuple[int,...]]:
        if seq.kind_ids.shape[0]!=1:
            raise ValueError("RSS_VNEXT_SOCKET_FACE_BATCH_ONE_ONLY")
        kind=seq.kind_ids[0]
        valid=seq.attention_mask[0].bool()
        control_event_positions=torch.where(
            valid&(kind==int(RigTokenKindV6.CONTROL))
        )[0].tolist()
        selected=[]
        parent_control_event=[]
        socket_control_ordinals=[]
        for idx in torch.where(valid&(kind==int(RigTokenKindV6.ANCHOR)))[0].tolist():
            if int(seq.anchor_parent_kind[0,idx])!=1:
                continue
            parent=int(seq.pointer_b[0,idx])
            if not 0<=parent<len(control_event_positions):
                raise ValueError("RSS_VNEXT_SOCKET_FACE_PARENT_CONTROL_POINTER_INVALID")
            ce=control_event_positions[parent]
            if int(seq.aux_type_ids[0,ce]) != _SOCKET_KIND:
                continue
            if not bool(((seq.coord_indices[0,idx]>=0)&
                         (seq.coord_indices[0,idx]<256)).all()):
                raise ValueError("RSS_VNEXT_SOCKET_FACE_ANCHOR_COORD_RANGE")
            selected.append(idx)
            parent_control_event.append(ce)
            socket_control_ordinals.append(parent)
        return tuple(selected),tuple(parent_control_event),tuple(socket_control_ordinals)

    def forward(
        self, *, mesh:RSSvNextStructuralMeshCandidate,
        axis_hidden:Tensor, axis_sequence:AXISSequenceV6,
        region_relation_logits:Tensor,
    )->RSSvNextSocketFaceNeuralOutputV1:
        if type(mesh) is not RSSvNextStructuralMeshCandidate:
            raise ValueError("RSS_VNEXT_SOCKET_FACE_INDEXED_TESSA_MESH_REQUIRED")
        mesh.validate()
        faces=mesh.faces
        nfaces=int(faces.shape[0])
        if (
            nfaces<1 or nfaces>self.max_faces
            or mesh.local_features.shape!=(1,mesh.rest_vertices_iris.shape[1],self.field_width)
            or mesh.semantic_logits.shape!=(1,mesh.rest_vertices_iris.shape[1],self.region_slots+1)
            or axis_hidden.ndim!=3
            or axis_hidden.shape!=(1,axis_sequence.kind_ids.shape[1],self.axis_width)
            or region_relation_logits.shape!=(1,self.region_slots,self.region_slots,3)
            or len({t.device for t in (mesh.local_features,mesh.rest_vertices_iris,
                     mesh.normals,faces,axis_hidden,region_relation_logits)})!=1
            or len({t.dtype for t in (mesh.local_features,mesh.rest_vertices_iris,
                     mesh.normals,axis_hidden,region_relation_logits)})!=1
            or not all(bool(torch.isfinite(t).all()) for t in (
                mesh.local_features,mesh.rest_vertices_iris,mesh.normals,
                axis_hidden,region_relation_logits,mesh.semantic_logits
            ))
        ):
            raise ValueError("RSS_VNEXT_SOCKET_FACE_HEAD_ADDRESS_OR_FIELD_DRIFT")
        indices,parents,controls=self._socket_anchor_positions(axis_sequence)
        if len(indices)>self.max_anchors:
            raise ValueError("RSS_VNEXT_SOCKET_FACE_ANCHOR_BUDGET")
        probs=torch.softmax(mesh.semantic_logits[0],dim=-1)
        feat=mesh.local_features[0,faces].mean(dim=1)
        xyz=mesh.rest_vertices_iris[0,faces].mean(dim=1)
        normal=F.normalize(mesh.normals[0,faces].mean(dim=1),dim=-1,eps=1e-8)
        regional=probs[faces].mean(dim=1)
        fstate=self.face_projection(torch.cat((feat,xyz,normal,regional),dim=-1))
        if indices:
            anchor_state=axis_hidden[0,list(indices)]
            parent_state=axis_hidden[0,list(parents)]
            coords=(axis_sequence.coord_indices[0,list(indices)].to(
                dtype=anchor_state.dtype)/255.)*2.-1.
            astate=self.anchor_projection(torch.cat((
                anchor_state,parent_state,self.anchor_position(coords)
            ),dim=-1))
            held=(self.held_query(astate)@self.held_key(fstate).T)*self.inv_scale
            holder=(self.holder_query(astate)@self.holder_key(fstate).T)*self.inv_scale
            held=torch.cat((held,self.held_none(astate)),dim=-1)
            holder=torch.cat((holder,self.holder_none(astate)),dim=-1)
        else:
            held=fstate.new_zeros((0,nfaces+1))
            holder=fstate.new_zeros((0,nfaces+1))
        return RSSvNextSocketFaceNeuralOutputV1(
            anchor_event_indices=indices,
            socket_control_indices=controls,
            held_face_logits=held,
            holder_face_logits=holder,
            face_component_ids=mesh.face_component_ids,
            face_region_probabilities=regional,
            region_relation_logits=region_relation_logits[0],
            mesh_topology_digest=mesh_topology_digest_v1(mesh),
            axis_sequence_digest=axis_sequence_digest_v1(axis_sequence),
        )


@torch.no_grad()
def propose_socket_face_pairs_v1(
    output:RSSvNextSocketFaceNeuralOutputV1,
    *, max_face_candidates_per_role:int=4,
)->tuple[RSSvNextNeuralSocketFaceProposalV1,...]:
    """Bounded model-side selection; no nearest triangle, no product authority.

    A NONE maximum abstains even if a learned relation strongly prefers a pair.
    Pair direction is explicitly HELD→HOLDER. For all hypotheses, preserve
    the exact candidate TESSA component IDs and UNKNOWN mass.
    """
    if type(output) is not RSSvNextSocketFaceNeuralOutputV1:
        raise ValueError("RSS_VNEXT_SOCKET_FACE_PROPOSAL_TYPE")
    if not 1<=max_face_candidates_per_role<=16:
        raise ValueError("RSS_VNEXT_SOCKET_FACE_TOPK_BUDGET")
    f=int(output.face_component_ids.shape[0])
    a=len(output.anchor_event_indices)
    if (
        output.held_face_logits.shape!=(a,f+1)
        or output.holder_face_logits.shape!=(a,f+1)
        or output.face_region_probabilities.shape[0]!=f
        or not bool(torch.isfinite(output.held_face_logits).all())
        or not bool(torch.isfinite(output.holder_face_logits).all())
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_PROPOSAL_SHAPE_DRIFT")
    predictions=[]
    region=torch.sigmoid(output.region_relation_logits[:,:,2])
    diagonal=torch.eye(region.shape[0],device=region.device,dtype=torch.bool)
    region=region.masked_fill(diagonal,0.)
    for i in range(a):
        h=F.log_softmax(output.held_face_logits[i].float(),dim=-1)
        p=F.log_softmax(output.holder_face_logits[i].float(),dim=-1)
        selected=None
        # NO_FACE is an independently trained class; not an unknown target.
        if int(h.argmax())!=f and int(p.argmax())!=f:
            kh=h[:f].topk(min(f,max_face_candidates_per_role)).indices.tolist()
            kp=p[:f].topk(min(f,max_face_candidates_per_role)).indices.tolist()
            best=-float("inf")
            for held in kh:
                for holder in kp:
                    if held==holder:
                        continue
                    a_probs=output.face_region_probabilities[held,:-1]
                    b_probs=output.face_region_probabilities[holder,:-1]
                    joint=torch.outer(a_probs,b_probs).masked_fill(diagonal,0.)
                    support=float(joint.sum().item())
                    aff=None if support<1e-6 else float((joint*region).sum().item()/support)
                    rank=float((h[held]+p[holder]).item())
                    if aff is not None:
                        rank+=math.log(max(aff,1e-12))
                    if rank>best:
                        best=rank
                        selected=(held,holder,support,aff,rank)
        if selected is None:
            predictions.append(RSSvNextNeuralSocketFaceProposalV1(
                anchor_event_index=output.anchor_event_indices[i],
                socket_control_index=output.socket_control_indices[i],
                held_face_index=None,holder_face_index=None,
                same_tessa_component=None,known_cross_slot_support=0.,
                predicted_directed_relation=None,held_unknown_probability=1.,
                holder_unknown_probability=1.,predicted_rank_log_score=None,
                state="ABSTAIN_MODEL_NO_FACE_OR_PAIR",
            ))
            continue
        held,holder,support,aff,rank=selected
        predictions.append(RSSvNextNeuralSocketFaceProposalV1(
            anchor_event_index=output.anchor_event_indices[i],
            socket_control_index=output.socket_control_indices[i],
            held_face_index=held,holder_face_index=holder,
            same_tessa_component=bool(
                output.face_component_ids[held]==output.face_component_ids[holder]
            ),
            known_cross_slot_support=support,
            predicted_directed_relation=aff,
            held_unknown_probability=float(output.face_region_probabilities[held,-1]),
            holder_unknown_probability=float(output.face_region_probabilities[holder,-1]),
            predicted_rank_log_score=rank,
            state=(
                "UNQUALIFIED_PREDICTED_DIRECTIONAL_FACE_PAIR"
                if aff is not None else
                "UNQUALIFIED_PREDICTED_PAIR_UNKNOWN_REGION"
            ),
        ))
    return tuple(predictions)


@dataclass(frozen=True)
class RSSvNextSourceQualifiedAnchorFaceLossV1:
    loss:Tensor
    eligible_held:int
    eligible_holder:int
    eligible_negative:int
    authority:str="RECEIPTED_SOURCE_NATIVE_TEACHER_ONLY"


def source_qualified_socket_face_loss_v1(
    output:RSSvNextSocketFaceNeuralOutputV1,
    *, source_face_targets:Tensor, source_face_mask:Tensor,
    source_asset_sha256:str, dataset_manifest_sha256:str,
    source_admission_key:str,
    trusted_registry:Mapping[str,Mapping[str,object]],
)->RSSvNextSourceQualifiedAnchorFaceLossV1:
    """Only real source-native, exactly reindexed mesh/rig supervision.

    `F` denotes explicitly source-qualified NO_FACE, never unlabelled;
    unknown labels have mask=False. The registry must be supplied by the
    independent dataset qualification launcher, not minted by this head.
    """
    def sha(value:object)->bool:
        return type(value) is str and len(value)==64 and all(
            c in "0123456789abcdef" for c in value
        )
    if (
        type(output) is not RSSvNextSocketFaceNeuralOutputV1
        or not sha(source_asset_sha256)
        or not sha(dataset_manifest_sha256)
        or not source_admission_key
        or source_admission_key not in trusted_registry
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_SOURCE_ADMISSION_REQUIRED")
    a,fplus=output.held_face_logits.shape
    if (
        a<1 or fplus<2
        or output.holder_face_logits.shape!=(a,fplus)
        or source_face_targets.shape!=(a,2)
        or source_face_mask.shape!=(a,2)
        or source_face_targets.dtype!=torch.long
        or source_face_mask.dtype!=torch.bool
        or source_face_targets.device!=output.held_face_logits.device
        or source_face_mask.device!=source_face_targets.device
        or not bool(source_face_mask.any())
        or not bool(torch.isfinite(output.held_face_logits).all())
        or not bool(torch.isfinite(output.holder_face_logits).all())
        or bool((source_face_targets[source_face_mask]<0).any())
        or bool((source_face_targets[source_face_mask]>=fplus).any())
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_TEACHER_TARGET_DOMAIN")
    r=trusted_registry[source_admission_key]
    if (
        r.get("admission_status") != "QUALIFIED_SOURCE_NATIVE_ATTACHMENT_FACE_REINDEX"
        or r.get("source_semantics") !=
           "NATIVE_RIG_ATTACHMENT_HELD_HOLDER_FACE_OR_VERIFIED_NO_FACE"
        or r.get("source_asset_sha256")!=source_asset_sha256
        or r.get("dataset_manifest_sha256")!=dataset_manifest_sha256
        or r.get("tessa_mesh_topology_sha256")!=output.mesh_topology_digest
        or r.get("axis_teacher_sequence_sha256")!=output.axis_sequence_digest
        or r.get("source_face_targets_sha256")!=_digest_tensor(source_face_targets)
        or r.get("source_face_mask_sha256")!=_digest_tensor(source_face_mask)
        or type(r.get("source_reindex_proven")) is not bool
        or not r.get("source_reindex_proven")
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_SOURCE_RECEIPT_DRIFT")
    losses=[]
    for i,pred in enumerate((output.held_face_logits,output.holder_face_logits)):
        mask=source_face_mask[:,i]
        if bool(mask.any()):
            losses.append(F.cross_entropy(pred[mask].float(),source_face_targets[mask,i]))
    return RSSvNextSourceQualifiedAnchorFaceLossV1(
        loss=torch.stack(losses).mean(),
        eligible_held=int(source_face_mask[:,0].sum()),
        eligible_holder=int(source_face_mask[:,1].sum()),
        eligible_negative=int(((source_face_targets==fplus-1)&source_face_mask).sum()),
    )


__all__=[
    "RSSvNextSocketFaceNeuralOutputV1",
    "RSSvNextNeuralSocketFaceProposalV1",
    "RSSvNextSocketFaceCorrespondenceHeadV1",
    "RSSvNextSourceQualifiedAnchorFaceLossV1",
    "mesh_topology_digest_v1","axis_sequence_digest_v1",
    "propose_socket_face_pairs_v1","source_qualified_socket_face_loss_v1",
]
