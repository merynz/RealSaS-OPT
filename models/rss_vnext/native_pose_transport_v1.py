from __future__ import annotations

"""Source-native face/joint pose transport to EXACT TESSA candidate vertex IDs.

A bare 'source_candidate_skin' or candidate posed XYZ tensor is NOT proof
that a deforming student M* is animation-ready. This witness replays poses
from original native face/vertex IDs and independently qualified skin + joint
matrices, then transports to each candidate vertex using a specifically
admitted source-native barycentric correspondence.

The source mesh/skin/motion NEVER enters student H(x). Candidate parent
addresses are teacher-only and must be independently registered; nearest
student points, semantics, source projection, SAM masks and fabricated vertex
orders cannot qualify this witness. This court does NOT prove actual learned
MIRA weights or G3/G3B ProductRevision acceptance.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import Tensor

from .animation_ready_mesh import posed_lbs_from_joint_matrices
from .seeded_mstar_teacher_address_v1 import (
    require_qualified_mstar_teacher_address_v1,
)
from .native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1,tensor_sha256_v1,
)
from .native_layer_open_boundary_v1 import exact_open_boundary_edges_v1

SCHEMA="RealSaS.RSSvNext.NativeCandidatePoseTransport.v1"
SOURCE_RECEIPT_SCHEMA="RealSaS.RSSvNext.IndependentNativeCandidatePoseSupport.v1"


def _sha(data:Mapping[str,object])->str:
    return sha256(json.dumps(
        dict(data),sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class SourceNativeCandidatePoseTransportV1:
    source_asset_sha256: str
    native_face_witness_sha256: str
    source_qualification_key: str
    native_skin: Tensor             # [N,J] source indexed vertices
    native_skin_matrices: Tensor    # [1,P,J,4,4] qualified source pose
    candidate_native_face_ids: Tensor # [V] native face ID for EXACT student M*
    candidate_barycentric: Tensor   # [V,3] native face support
    candidate_pose_mask: Tensor     # [1,P,V] qualified source observation
    source_candidate_mesh_face_sha256: str  # target EXACT M* faces/index order
    candidate_mstar_topology_address_sha256: str  # M0 parent support + M* order
    source_candidate_support_derivation: str = (
        "INDEPENDENT_SOURCE_NATIVE_FACE_BARYCENTRIC_NOT_STUDENT_NEAREST"
    )

    def fields(self) -> dict[str,Tensor]:
        return {k:v for k,v in vars(self).items() if isinstance(v,Tensor)}

    def receipt_sha256(self)->str:
        return _sha({
            "schema":SCHEMA,
            **{k:tensor_sha256_v1(t) for k,t in self.fields().items()},
            **{k:v for k,v in vars(self).items()
               if not isinstance(v,Tensor)},
        })

    def replay(
        self,*,witness:NativeFaceLayerSamplesV1,
        candidate_faces:Tensor,
        candidate_rest:Tensor,
        active_mstar_topology_address_sha256:str,
        trusted_source_registry:Mapping[str,Mapping[str,object]],
        projection_tolerance_ratio:float=.07,
    )->tuple[Tensor,Tensor,Tensor,Tensor]:
        """Return exact candidate skin, target poses, pose mask, rest support."""
        witness.replay()
        require_qualified_mstar_teacher_address_v1(
            transport_address_sha256=self.candidate_mstar_topology_address_sha256,
            active_mstar_address_sha256=active_mstar_topology_address_sha256,
        )
        if (
            type(projection_tolerance_ratio) not in (float,int)
            or not 0<projection_tolerance_ratio<=.07
            or not isinstance(candidate_faces,Tensor)
            or not isinstance(candidate_rest,Tensor)
            or candidate_rest.ndim!=3 or candidate_rest.shape[0]!=1
            or candidate_rest.shape[-1]!=3
            or candidate_faces.ndim!=2 or candidate_faces.shape[-1]!=3
            or candidate_faces.dtype!=torch.long
            or not len(candidate_faces)
            or self.source_asset_sha256!=witness.source_asset_sha256
            or self.native_face_witness_sha256!=witness.receipt_sha256()
            or self.source_candidate_support_derivation!=
                "INDEPENDENT_SOURCE_NATIVE_FACE_BARYCENTRIC_NOT_STUDENT_NEAREST"
            or self.source_candidate_mesh_face_sha256
                !=tensor_sha256_v1(candidate_faces)
            or any(type(x) is not str or not x for x in (
                self.source_qualification_key,self.source_candidate_support_derivation))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_POSE_TRANSPORT_SOURCE_IDENTITY")
        v=len(candidate_rest[0])
        n=len(witness.vertices)
        f=len(witness.faces)
        j=self.native_skin.shape[1] if self.native_skin.ndim==2 else -1
        p=(self.native_skin_matrices.shape[1]
           if self.native_skin_matrices.ndim==5 else -1)
        b=self.candidate_barycentric
        ids=self.candidate_native_face_ids
        if (
            self.native_skin.shape!=(n,j) or j<1
            or not 1<=p<=1024
            or self.native_skin_matrices.shape!=(1,p,j,4,4)
            or ids.shape!=(v,) or ids.dtype!=torch.long
            or b.shape!=(v,3)
            or self.candidate_pose_mask.shape!=(1,p,v)
            or self.candidate_pose_mask.dtype!=torch.bool
            or not bool(self.candidate_pose_mask.any())
            or any(t.device!=candidate_rest.device for t in (
                candidate_faces,witness.vertices,
                self.native_skin,self.native_skin_matrices,
                ids,b,self.candidate_pose_mask,
            ))
            or any(t.dtype!=candidate_rest.dtype for t in (
                witness.vertices,self.native_skin,self.native_skin_matrices,b,
            ))
            or any(not bool(torch.isfinite(t).all()) for t in (
                candidate_rest,witness.vertices,self.native_skin,
                self.native_skin_matrices,b,
            ))
            or bool((ids<0).any()) or int(ids.max())>=f
            or bool((candidate_faces<0).any())
            or int(candidate_faces.max())>=v
            or bool((b<-1.e-6).any()) or bool((b>1+1.e-6).any())
            or not bool(torch.allclose(
                b.sum(-1),torch.ones_like(b[:,0]),atol=1.e-6,rtol=0,
            ))
            or bool((self.native_skin<0).any())
            or not bool(torch.allclose(
                self.native_skin.sum(-1),
                torch.ones_like(self.native_skin[:,0]),
                atol=1.e-5,rtol=0,
            ))
            or not bool(torch.allclose(
                self.native_skin_matrices[...,3,:],
                self.native_skin_matrices.new_tensor(
                    [0.,0.,0.,1.],
                ).expand_as(self.native_skin_matrices[...,3,:]),
                atol=1.e-5,rtol=0,
            ))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_POSE_TRANSPORT_TENSOR_DOMAIN")
        # The source face graph is authoritative, but this candidate M* graph
        # must be independently qualified before exact pose transport.
        exact_open_boundary_edges_v1(witness.faces)
        if not bool(self.candidate_pose_mask.any(dim=-1).all()):
            raise ValueError("RSS_VNEXT_NATIVE_POSE_TRANSPORT_UNOBSERVED_POSE")

        row=trusted_source_registry.get(self.source_qualification_key)
        expected={
            "schema":SOURCE_RECEIPT_SCHEMA,
            "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_FACE_TO_CANDIDATE_SUPPORT",
            "source_asset_sha256":witness.source_asset_sha256,
            "native_face_witness_sha256":witness.receipt_sha256(),
            "pose_transport_receipt_sha256":self.receipt_sha256(),
            "candidate_faces_sha256":tensor_sha256_v1(candidate_faces),
            "candidate_mstar_topology_address_sha256":active_mstar_topology_address_sha256,
            "candidate_source_support_not_student_nearest":True,
            "source_rights_independently_verified":True,
            "generated_views_as_source_pose_truth":False,
            "source_skin_is_native_vertex_order":True,
        }
        if not isinstance(row,Mapping) or any(
            row.get(k)!=value for k,value in expected.items()
        ):
            raise ValueError("RSS_VNEXT_NATIVE_POSE_TRANSPORT_RECEIPT_REQUIRED")

        tri=witness.vertices[witness.faces[ids]]
        rest=(tri*b[...,None]).sum(dim=1)
        source_span=(
            witness.vertices.max(dim=0).values
            -witness.vertices.min(dim=0).values
        )
        scene_scale=torch.linalg.vector_norm(source_span).clamp_min(1.e-6)
        distances=torch.linalg.vector_norm(
            candidate_rest[0].detach()-rest,dim=-1,
        )
        if (
            not bool(torch.isfinite(distances).all())
            or bool((distances>scene_scale*projection_tolerance_ratio).any())
        ):
            raise ValueError("RSS_VNEXT_NATIVE_POSE_TRANSPORT_REST_SUPPORT_DRIFT")
        # Native support interpolation is on SOURCE triangle face IDs; no
        # student positional nearest neighbor creates a source truth label.
        skin=(self.native_skin[witness.faces[ids]]
              *b[...,None]).sum(dim=1)
        native_pose=posed_lbs_from_joint_matrices(
            witness.vertices[None],self.native_skin[None],
            self.native_skin_matrices,
        )
        moved=native_pose[0,:,witness.faces[ids],:]
        target=(moved*b[None,:,:,None]).sum(dim=2)
        return (
            skin[None],
            target[None],
            self.candidate_pose_mask,
            rest[None],
        )


def validate_source_native_pose_candidate_teacher_v1(
    *,support:SourceNativeCandidatePoseTransportV1 | None,
    native_face_witness:NativeFaceLayerSamplesV1 | None,
    candidate_faces:Tensor,candidate_rest:Tensor,
    active_mstar_topology_address_sha256:str,
    source_candidate_skin:Tensor,
    source_candidate_posed:Tensor,
    source_pose_mask:Tensor,
    source_pose_matrices:Tensor,
    teacher_source_asset_sha256:str,
    teacher_receipt:Mapping[str,object],
    trusted_source_registry:Mapping[str,Mapping[str,object]],
) -> None:
    if not isinstance(support,SourceNativeCandidatePoseTransportV1):
        raise ValueError("RSS_VNEXT_TESSA_NATIVE_POSE_TRANSPORT_REQUIRED")
    if not isinstance(native_face_witness,NativeFaceLayerSamplesV1):
        raise ValueError("RSS_VNEXT_TESSA_NATIVE_FACE_POSE_WITNESS_REQUIRED")
    skin,target,mask,_=support.replay(
        witness=native_face_witness,candidate_faces=candidate_faces,
        candidate_rest=candidate_rest,
        active_mstar_topology_address_sha256=active_mstar_topology_address_sha256,
        trusted_source_registry=trusted_source_registry,
    )
    if (
        teacher_source_asset_sha256!=support.source_asset_sha256
        or teacher_receipt.get("native_pose_transport_schema")!=SCHEMA
        or teacher_receipt.get("native_pose_transport_receipt_sha256")
            !=support.receipt_sha256()
        or teacher_receipt.get("native_pose_transport_independently_audited")
            is not True
        or source_candidate_skin.shape!=skin.shape
        or source_candidate_posed.shape!=target.shape
        or source_pose_mask.shape!=mask.shape
        or source_pose_matrices.shape!=support.native_skin_matrices.shape
        or not torch.allclose(source_candidate_skin,skin,atol=1.e-5,rtol=0)
        or not torch.allclose(source_candidate_posed,target,atol=1.e-5,rtol=0)
        or not torch.equal(source_pose_mask,mask)
        or not torch.equal(source_pose_matrices,support.native_skin_matrices)
    ):
        raise ValueError("RSS_VNEXT_TESSA_NATIVE_POSE_REPLAY_OR_RECEIPT_DRIFT")


__all__=[
    "SCHEMA","SOURCE_RECEIPT_SCHEMA",
    "SourceNativeCandidatePoseTransportV1",
    "validate_source_native_pose_candidate_teacher_v1",
]
