from __future__ import annotations

"""Source-native skin/mesh correspondence WITHOUT source animation clips.

This source-only teacher court transports audited source rig skin from exact
native triangles onto an independently registered candidate M* address. It
never matches student vertices to source mesh by nearest position, never
writes into student H(x), never infers attachment/material/cloth, and never
asserts a real source pose exists. Source motion has its own separate court.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

import torch
from torch import Tensor

from .seeded_mstar_teacher_address_v1 import (
    require_qualified_mstar_teacher_address_v1,
)
from .native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1, tensor_sha256_v1,
)

SCHEMA = "RealSaS.RSSvNext.NativeCandidateSkinTransport.v1"
RECEIPT_SCHEMA = "RealSaS.RSSvNext.IndependentNativeCandidateSkinSupport.v1"
DERIVATION = "INDEPENDENT_SOURCE_NATIVE_FACE_BARYCENTRIC_NOT_STUDENT_NEAREST"


@dataclass(frozen=True)
class SourceNativeCandidateSkinTransportV1:
    source_asset_sha256: str
    source_qualification_key: str
    native_face_witness_sha256: str
    source_candidate_mesh_face_sha256: str
    # Source foundry proves that native skin columns are the SAME ordered
    # JOINT events as AXIS's source teacher; a matching tensor J is not proof.
    axis_teacher_sequence_sha256: str
    native_skin_joint_order_sha256: str
    native_skin: Tensor                 # [N,J], indexed on original vertices
    candidate_native_face_ids: Tensor  # [V], exact source face address
    candidate_barycentric: Tensor      # [V,3], independently admitted
    # Frozen M* indexed face order + source-M0 parent identity, NOT live XYZ.
    candidate_mstar_topology_address_sha256: str
    source_candidate_support_derivation: str = DERIVATION

    def receipt_sha256(self) -> str:
        value = {
            "schema":SCHEMA,
            "source_asset_sha256":self.source_asset_sha256,
            "source_qualification_key":self.source_qualification_key,
            "native_face_witness_sha256":self.native_face_witness_sha256,
            "source_candidate_mesh_face_sha256":self.source_candidate_mesh_face_sha256,
            "axis_teacher_sequence_sha256":self.axis_teacher_sequence_sha256,
            "native_skin_joint_order_sha256":self.native_skin_joint_order_sha256,
            "source_candidate_support_derivation":self.source_candidate_support_derivation,
            "native_skin_sha256":tensor_sha256_v1(self.native_skin),
            "candidate_native_face_ids_sha256":tensor_sha256_v1(self.candidate_native_face_ids),
            "candidate_barycentric_sha256":tensor_sha256_v1(self.candidate_barycentric),
            "candidate_mstar_topology_address_sha256":self.candidate_mstar_topology_address_sha256,
        }
        return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def replay(
        self, *, witness:NativeFaceLayerSamplesV1,
        candidate_faces:Tensor, candidate_rest:Tensor,
        trusted_source_registry:Mapping[str,Mapping[str,object]],
        dataset_manifest_sha256:str,
        expected_axis_teacher_sequence_sha256:str,
        active_mstar_topology_address_sha256:str,
        projection_tolerance_ratio:float=.07,
        chunk_size:int=4096,
    ) -> Tensor:
        """Return [1,V,J] barycentric source skin on the exact candidate M*.

        Source and candidate triangle *identities* are different domains;
        independently authorized barycentric native-face witnesses bind them.
        The geometric tolerance is only a fail-closed drift check, never used
        to create the correspondence or sample a nearest source face.
        """
        if not isinstance(witness,NativeFaceLayerSamplesV1):
            raise ValueError("RSS_VNEXT_NATIVE_SKIN_SOURCE_FACE_REQUIRED")
        witness.replay()
        require_qualified_mstar_teacher_address_v1(
            transport_address_sha256=self.candidate_mstar_topology_address_sha256,
            active_mstar_address_sha256=active_mstar_topology_address_sha256,
        )
        if (
            type(projection_tolerance_ratio) not in (float,int)
            or not 0 < projection_tolerance_ratio <= .07
            or type(chunk_size) is not int or not 1 <= chunk_size <= 16384
            or candidate_rest.ndim!=3 or candidate_rest.shape[0]!=1
            or candidate_rest.shape[-1]!=3
            or candidate_faces.ndim!=2 or candidate_faces.shape[-1]!=3
            or candidate_faces.dtype!=torch.long
            or len(candidate_faces)<1
            or type(dataset_manifest_sha256) is not str
            or len(dataset_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in dataset_manifest_sha256)
            or type(expected_axis_teacher_sequence_sha256) is not str
            or len(expected_axis_teacher_sequence_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_axis_teacher_sequence_sha256)
            or self.axis_teacher_sequence_sha256 != expected_axis_teacher_sequence_sha256
            or type(self.native_skin_joint_order_sha256) is not str
            or len(self.native_skin_joint_order_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.native_skin_joint_order_sha256)
            or self.source_asset_sha256 != witness.source_asset_sha256
            or self.native_face_witness_sha256 != witness.receipt_sha256()
            or self.source_candidate_support_derivation != DERIVATION
            or self.source_candidate_mesh_face_sha256 != tensor_sha256_v1(candidate_faces)
        ):
            raise ValueError("RSS_VNEXT_NATIVE_SKIN_SOURCE_IDENTITY")
        n=len(witness.vertices)
        f=len(witness.faces)
        v=int(candidate_rest.shape[1])
        j=self.native_skin.shape[1] if self.native_skin.ndim==2 else -1
        ids,b=self.candidate_native_face_ids,self.candidate_barycentric
        values=(witness.vertices,self.native_skin,b,candidate_rest)
        if (
            self.native_skin.shape!=(n,j) or j<1
            or ids.shape!=(v,) or ids.dtype!=torch.long
            or b.shape!=(v,3)
            or any(t.device!=candidate_rest.device for t in (
                candidate_faces,witness.vertices,witness.faces,
                self.native_skin,ids,b,
            ))
            or len({t.dtype for t in values})!=1
            or any(not bool(torch.isfinite(t).all()) for t in values)
            or bool((self.native_skin < 0).any())
            or bool((ids < 0).any()) or int(ids.max())>=f
            or bool((candidate_faces < 0).any())
            or int(candidate_faces.max())>=v
            or bool((b < -1.e-6).any()) or bool((b > 1+1.e-6).any())
            or not bool(torch.allclose(
                b.sum(-1),torch.ones_like(b[:,0]),atol=1.e-6,rtol=0,
            ))
            or not bool(torch.allclose(
                self.native_skin.sum(-1),
                torch.ones_like(self.native_skin[:,0]),
                atol=1.e-5,rtol=0,
            ))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_SKIN_TENSOR_DOMAIN")
        row=trusted_source_registry.get(self.source_qualification_key)
        expected={
            "schema":RECEIPT_SCHEMA,
            "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_FACE_TO_CANDIDATE_SKIN",
            "dataset_manifest_sha256":dataset_manifest_sha256,
            "source_asset_sha256":witness.source_asset_sha256,
            "native_face_witness_sha256":witness.receipt_sha256(),
            "skin_transport_receipt_sha256":self.receipt_sha256(),
            "axis_teacher_sequence_sha256":expected_axis_teacher_sequence_sha256,
            "native_skin_joint_order_sha256":self.native_skin_joint_order_sha256,
            "native_skin_columns_match_axis_joint_order":True,
            "candidate_faces_sha256":tensor_sha256_v1(candidate_faces),
            "candidate_mstar_topology_address_sha256":active_mstar_topology_address_sha256,
            "candidate_source_support_not_student_nearest":True,
            "source_rights_independently_verified":True,
            "generated_views_as_source_skin_truth":False,
            "source_skin_is_native_vertex_order":True,
        }
        if not isinstance(row,Mapping) or any(row.get(k)!=value for k,value in expected.items()):
            raise ValueError("RSS_VNEXT_NATIVE_SKIN_SOURCE_RECEIPT_REQUIRED")
        projected_skin=[]
        scene_span=torch.linalg.vector_norm(
            witness.vertices.max(dim=0).values-witness.vertices.min(dim=0).values,
        ).clamp_min(1.e-6)
        max_deviation=scene_span*float(projection_tolerance_ratio)
        for start in range(0,v,chunk_size):
            stop=min(start+chunk_size,v)
            triangles=witness.faces[ids[start:stop]]
            weights=b[start:stop]
            source_xyz=(witness.vertices[triangles]*weights[...,None]).sum(dim=1)
            deviation=torch.linalg.vector_norm(
                candidate_rest[0,start:stop].detach()-source_xyz,dim=-1,
            )
            if bool((~torch.isfinite(deviation)).any()) or bool((deviation > max_deviation).any()):
                raise ValueError("RSS_VNEXT_NATIVE_SKIN_REST_SUPPORT_DRIFT")
            part=(self.native_skin[triangles]*weights[...,None]).sum(dim=1)
            projected_skin.append(part)
        skin=torch.cat(projected_skin,dim=0)[None]
        if not bool(torch.allclose(
            skin.sum(dim=-1),torch.ones_like(skin[...,0]),
            atol=1.e-5,rtol=0,
        )):
            raise ValueError("RSS_VNEXT_NATIVE_SKIN_TRANSPORT_SIMPLEX_DRIFT")
        return skin


__all__=["SCHEMA","RECEIPT_SCHEMA","DERIVATION","SourceNativeCandidateSkinTransportV1"]
