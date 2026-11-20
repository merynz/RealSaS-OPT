from __future__ import annotations

"""Exact final-M* OPEN hem pin support: learned candidate vs native source truth.

Socket face (HELD/HOLDER) is NOT a pin-vertex label. This separate learnable
candidate uses one H(x), one indexed M* and the causal AXIS anchor states.
Only SOURCE-NATIVE pin-to-hem plus held-out host intervention witnesses may
teach it. It never grants a material, physical or Compiler authority.
"""
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .animation_ready_mesh import RSSvNextStructuralMeshCandidate
from .attachment_face_correspondence_v1 import (
    RSSvNextSocketFaceCorrespondenceHeadV1,
    mesh_topology_digest_v1, axis_sequence_digest_v1, _digest_tensor,
)
from models.axis.rig_program_model_v6 import AXISSequenceV6

AUTHORITY = "UNQUALIFIED_NEURAL_OPEN_PIN_VERTEX_PROPOSAL_V1"
TEACHER_SCHEMA = "RealSaS.RSSvNext.SourceNativeOpenHemPinVertexTeacher.v1"


def _sha(x: object) -> bool:
    return type(x) is str and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x
    )


def open_hem_boundary_mask_v1(
    mesh: RSSvNextStructuralMeshCandidate,
) -> Tensor:
    """Exact topology test, not a geometry/skin proximity heuristic.

    Computes boundary edge incidence over the ORIGINAL final indexed M*
    faces and respects TESSA component IDs even when sheets touch at XYZ.
    Returns vertex candidates only on a real OPEN triangle boundary.
    """
    mesh.validate()
    faces = mesh.faces.long()
    comp = mesh.vertex_component_ids
    face_comp = mesh.face_component_ids
    if not bool((comp[faces] == face_comp[:, None]).all()):
        raise ValueError("RSS_VNEXT_PIN_MSTAR_CROSS_COMPONENT_FACE")
    edges = torch.cat(
        (faces[:, [0,1]],faces[:, [1,2]],faces[:, [2,0]]),dim=0
    )
    edge_components = face_comp.repeat(3)
    edge_sorted = edges.sort(dim=-1).values
    # Component ID is part of the key: two coincident, disjoint sheets
    # never share an open edge merely because their XYZ overlap.
    v = mesh.rest_vertices_iris.shape[1]
    if v > 262144 or len(edges) > 1572864:
        raise ValueError("RSS_VNEXT_PIN_HEM_TOPOLOGY_BUDGET")
    ids = (
        edge_components.to(torch.int64) * (v*v)
        + edge_sorted[:, 0] * v + edge_sorted[:, 1]
    )
    unique, inv, counts = torch.unique(
        ids, sorted=True, return_inverse=True, return_counts=True,
    )
    if bool((counts > 2).any()):
        raise ValueError("RSS_VNEXT_PIN_HEM_NONMANIFOLD_EDGE")
    boundary_edges = edges[counts[inv] == 1]
    mask = torch.zeros((v,),device=faces.device,dtype=torch.bool)
    if boundary_edges.numel():
        mask[boundary_edges.reshape(-1)] = True
    return mask


@dataclass(frozen=True)
class OpenPinVertexNeuralOutputV1:
    anchor_event_indices: tuple[int, ...]
    pin_logits: Tensor                 # [A,V]
    open_boundary_mask: Tensor         # [V]; only legal candidates
    face_component_ids: Tensor         # [F]
    vertex_component_ids: Tensor       # [V]
    mesh_topology_digest: str
    axis_sequence_digest: str
    authority: str = AUTHORITY
    source_pin_truth_qualified: bool = False
    automatic_physics_activation: bool = False


class OpenPinVertexHeadV1(nn.Module):
    """Separate vertex-level pin head, trained on authentic source hem pins."""

    def __init__(
        self, *, field_width: int, axis_width: int, hidden_width: int,
        max_anchors: int = 64, max_vertices: int = 65536,
    ):
        super().__init__()
        if min(field_width,axis_width,hidden_width,max_anchors,max_vertices)<1:
            raise ValueError("RSS_VNEXT_PIN_VERTEX_HEAD_DIMENSIONS")
        self.max_anchors=max_anchors
        self.max_vertices=max_vertices
        self.vertex=nn.Sequential(
            nn.LayerNorm(field_width + 7),
            nn.Linear(field_width + 7,hidden_width),nn.GELU(),
            nn.Linear(hidden_width,hidden_width),
        )
        self.anchor=nn.Sequential(
            nn.LayerNorm(2*axis_width+3),
            nn.Linear(2*axis_width+3,hidden_width),nn.GELU(),
            nn.Linear(hidden_width,hidden_width),
        )
        self.query=nn.Linear(hidden_width,hidden_width,bias=False)
        self.key=nn.Linear(hidden_width,hidden_width,bias=False)
        self.scale=hidden_width**(-.5)

    def forward(
        self, *, mesh: RSSvNextStructuralMeshCandidate,
        axis_hidden: Tensor, axis_sequence: AXISSequenceV6,
    )->OpenPinVertexNeuralOutputV1:
        mesh.validate()
        indices,parents,_ = (
            RSSvNextSocketFaceCorrespondenceHeadV1._socket_anchor_positions(
                axis_sequence
            )
        )
        v=int(mesh.rest_vertices_iris.shape[1])
        if (
            v>self.max_vertices or len(indices)>self.max_anchors
            or v*len(indices)>2_097_152
            or axis_hidden.ndim!=3
            or axis_hidden.shape[:2]!=axis_sequence.kind_ids.shape
            or mesh.local_features.shape!=(1,v,self.vertex[1].in_features-7)
            or axis_hidden.shape[-1]*2+3!=self.anchor[1].in_features
            or any(x.device!=mesh.rest_vertices_iris.device for x in
                   (axis_hidden,axis_sequence.kind_ids))
            or not bool(torch.isfinite(axis_hidden).all())
        ):
            raise ValueError("RSS_VNEXT_PIN_VERTEX_HEAD_ADDRESS_INVALID")
        boundary=open_hem_boundary_mask_v1(mesh)
        position=mesh.rest_vertices_iris
        normal=mesh.normals
        features=mesh.mechanics_local_features
        if features is None:
            features=mesh.local_features
        if features.shape!=mesh.local_features.shape:
            raise ValueError("RSS_VNEXT_PIN_VERTEX_MECHANICS_H_ABI")
        state=self.vertex(torch.cat(
            (features,position,normal,boundary[None,:,None].float()),dim=-1
        ))[0]
        if indices:
            anchor=axis_hidden[0,list(indices)]
            parent=axis_hidden[0,list(parents)]
            coords=(
                axis_sequence.coord_indices[0,list(indices)].to(anchor.dtype)
                /255.*2.-1.
            )
            context=self.anchor(torch.cat((anchor,parent,coords),dim=-1))
            logits=(self.query(context) @ self.key(state).T)*self.scale
        else:
            logits=state.new_zeros((0,v))
        return OpenPinVertexNeuralOutputV1(
            anchor_event_indices=indices,pin_logits=logits,
            open_boundary_mask=boundary,
            face_component_ids=mesh.face_component_ids,
            vertex_component_ids=mesh.vertex_component_ids,
            mesh_topology_digest=mesh_topology_digest_v1(mesh),
            axis_sequence_digest=axis_sequence_digest_v1(axis_sequence),
        )


@dataclass(frozen=True)
class SourceNativeOpenPinVertexTargetV1:
    source_asset_sha256: str
    source_qualification_key: str
    native_open_hem_sha256: str
    native_pin_witness_sha256: str
    heldout_host_only_intervention_sha256: str
    source_mstar_transport_sha256: str
    pin_targets: Tensor       # [A,V] bool; true pin on a qualified hem
    known_mask: Tensor        # [A,V] bool; false = UNKNOWN, NOT NO_PIN
    source_authority: str = "QUALIFIED_SOURCE_NATIVE_OPEN_HEM_AND_PIN_VERTEX"
    generated_views_as_pin_truth: bool = False

    def validate(self, output: OpenPinVertexNeuralOutputV1) -> None:
        a,v=output.pin_logits.shape
        if (
            any(not _sha(x) for x in (
                self.source_asset_sha256,self.native_open_hem_sha256,
                self.native_pin_witness_sha256,
                self.heldout_host_only_intervention_sha256,
                self.source_mstar_transport_sha256,
            ))
            or not self.source_qualification_key
            or self.source_authority !=
                "QUALIFIED_SOURCE_NATIVE_OPEN_HEM_AND_PIN_VERTEX"
            or self.generated_views_as_pin_truth is not False
            or output.authority!=AUTHORITY
            or output.source_pin_truth_qualified is not False
            or output.automatic_physics_activation is not False
            or self.pin_targets.shape!=(a,v)
            or self.known_mask.shape!=(a,v)
            or self.pin_targets.dtype!=torch.bool
            or self.known_mask.dtype!=torch.bool
            or self.pin_targets.device!=output.pin_logits.device
            or self.known_mask.device!=output.pin_logits.device
            or output.open_boundary_mask.shape!=(v,)
            or not bool(self.known_mask.any())
            or bool((self.pin_targets & ~self.known_mask).any())
            or bool((self.known_mask & ~output.open_boundary_mask[None]).any())
            or bool((self.pin_targets.sum(-1)>8).any())
            or bool((self.known_mask.any(-1)
                     & (self.pin_targets.sum(-1)<1)).any())
        ):
            raise ValueError("RSS_VNEXT_NATIVE_PIN_VERTEX_TEACHER_DOMAIN")


def source_native_open_pin_vertex_loss_v1(
    *, output: OpenPinVertexNeuralOutputV1 | None,
    teacher: SourceNativeOpenPinVertexTargetV1 | None,
    admission_key: str,
    trusted_registry: Mapping[str,Mapping[str,object]],
    dataset_manifest_sha256: str,
    candidate_mstar_topology_address_sha256: str,
    source_asset_sha256: str, source_qualification_key: str,
    original_rgba_sha256: str, original_camera_sha256: str,
    combined_generation_checkpoint_sha256: str,
) -> tuple[Tensor|None,int]:
    if teacher is None and not admission_key:
        return None,0
    if (
        not isinstance(teacher,SourceNativeOpenPinVertexTargetV1)
        or not isinstance(output,OpenPinVertexNeuralOutputV1)
        or not admission_key or not isinstance(trusted_registry,Mapping)
        or not all(_sha(x) for x in (
            dataset_manifest_sha256,candidate_mstar_topology_address_sha256,
            source_asset_sha256,original_rgba_sha256,original_camera_sha256,
            combined_generation_checkpoint_sha256,
        ))
        or teacher.source_asset_sha256 != source_asset_sha256
        or teacher.source_qualification_key != source_qualification_key
    ):
        raise ValueError("RSS_VNEXT_NATIVE_PIN_VERTEX_SOURCE_REQUIRED")
    teacher.validate(output)
    expected={
        "admission_status":"QUALIFIED_SOURCE_NATIVE_OPEN_PIN_VERTEX_REINDEX",
        "schema":TEACHER_SCHEMA,
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_asset_sha256":source_asset_sha256,
        "source_qualification_key":source_qualification_key,
        "original_rgba_sha256":original_rgba_sha256,
        "original_camera_sha256":original_camera_sha256,
        "combined_generation_checkpoint_sha256":
            combined_generation_checkpoint_sha256,
        "candidate_mstar_topology_address_sha256":
            candidate_mstar_topology_address_sha256,
        "mesh_topology_digest":output.mesh_topology_digest,
        "axis_sequence_digest":output.axis_sequence_digest,
        "native_open_hem_sha256":teacher.native_open_hem_sha256,
        "native_pin_witness_sha256":teacher.native_pin_witness_sha256,
        "heldout_host_only_intervention_sha256":
            teacher.heldout_host_only_intervention_sha256,
        "source_mstar_transport_sha256":teacher.source_mstar_transport_sha256,
        "source_pin_targets_sha256":_digest_tensor(teacher.pin_targets),
        "source_known_mask_sha256":_digest_tensor(teacher.known_mask),
        "source_pin_support_from_independent_native_boundary":True,
        "source_native_host_intervention_verified":True,
        "source_to_final_mstar_vertex_transport_verified":True,
        "student_nearest_vertex_as_pin_truth":False,
        "generated_views_as_pin_truth":False,
        "physical_material_or_xpbd_authority":False,
    }
    row=trusted_registry.get(admission_key)
    if row is None or any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError("RSS_VNEXT_NATIVE_PIN_VERTEX_RECEIPT_DRIFT")
    if not bool(torch.isfinite(output.pin_logits).all()):
        raise ValueError("RSS_VNEXT_PIN_VERTEX_PREDICTION_NONFINITE")
    target=teacher.pin_targets[teacher.known_mask].float()
    pred=output.pin_logits[teacher.known_mask].float()
    if not pred.numel():
        raise RuntimeError("RSS_VNEXT_NATIVE_PIN_VERTEX_EMPTY_LOSS")
    loss=F.binary_cross_entropy_with_logits(pred,target)
    return loss,int(target.numel())


@dataclass(frozen=True)
class OpenPinVertexNeuralProposalV1:
    anchor_event_index: int
    candidate_pin_vertex_indices: tuple[int,...]
    available_boundary_vertex_count: int
    state: str
    authority: str=AUTHORITY
    source_witness_claimed: bool=False
    executable_xpbd_claimed: bool=False


@torch.no_grad()
def propose_open_pin_vertex_indices_v1(
    output: OpenPinVertexNeuralOutputV1,
    *,min_probability:float=.85,max_pins:int=8,
)->tuple[OpenPinVertexNeuralProposalV1,...]:
    """A prediction for editor review, NEVER an auto-executable pin."""
    if (
        type(output) is not OpenPinVertexNeuralOutputV1
        or output.authority!=AUTHORITY
        or output.source_pin_truth_qualified
        or output.automatic_physics_activation
        or type(min_probability) not in (int,float)
        or not math.isfinite(min_probability)
        or not .5<=min_probability<1.
        or type(max_pins) is not int or not 1<=max_pins<=8
        or output.pin_logits.ndim!=2
        or output.open_boundary_mask.shape!=(output.pin_logits.shape[1],)
        or not bool(torch.isfinite(output.pin_logits).all())
    ):
        raise ValueError("RSS_VNEXT_PIN_VERTEX_PREVIEW_DOMAIN")
    boundary=output.open_boundary_mask
    props=[]
    for i,anchor in enumerate(output.anchor_event_indices):
        if not bool(boundary.any()):
            ids=()
        else:
            probs=torch.sigmoid(output.pin_logits[i])
            ids=tuple(
                int(j) for j in torch.where(
                    (probs>=min_probability)&boundary
                )[0].tolist()
            )
        status=(
            "UNQUALIFIED_REVIEW_PIN_VERTICES"
            if 1<=len(ids)<=max_pins else
            "ABSTAIN_NO_CONFIDENT_OPEN_PIN_SET"
        )
        props.append(OpenPinVertexNeuralProposalV1(
            anchor_event_index=anchor,
            candidate_pin_vertex_indices=(
                ids if status=="UNQUALIFIED_REVIEW_PIN_VERTICES" else ()
            ),
            available_boundary_vertex_count=int(boundary.sum()),
            state=status,
        ))
    return tuple(props)


__all__=[
    "AUTHORITY","TEACHER_SCHEMA","open_hem_boundary_mask_v1",
    "OpenPinVertexNeuralOutputV1","OpenPinVertexHeadV1",
    "SourceNativeOpenPinVertexTargetV1","source_native_open_pin_vertex_loss_v1",
    "OpenPinVertexNeuralProposalV1","propose_open_pin_vertex_indices_v1",
]
