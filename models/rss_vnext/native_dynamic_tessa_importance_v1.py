from __future__ import annotations

"""Independent source-native *dynamic* importance targets for seeded TESSA.

The teacher's ORIGINAL indexed face IDs, source-native vertex skin and verified
pose matrices create a per-face difficulty target. Native barycentric sample
addresses transfer this scalar to TESSA training samples. Never use predicted
IRIS/TESSA mesh faces, output skin, generated views, XYZ nearest neighbors or
student IDs as teacher authority.

This DOES NOT certify animation readiness: the reference may itself fail G3,
and some critical poses may be absent. Compiler G3/G3B/Stage35 remain hard
independent admission criteria. All source faces are evaluated; batches are
bounded to avoid large F x P tensors.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import Tensor

from .native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1, tensor_sha256_v1,
)
from .seeded_dynamic_full_face_court_v1 import SeededDynamicLimitsV1
from models.tessa.v1.mechanical_objective_v1 import (
    triangle_deformation_metrics_v1,
)
from .animation_ready_mesh import posed_lbs_from_joint_matrices

SCHEMA = "RealSaS.RSSvNext.NativeTESSADynamicFaceImportance.v1"
SOURCE_RECEIPT_SCHEMA = "RealSaS.RSSvNext.NativeSkinPoseEvidence.v1"


@dataclass(frozen=True)
class NativeTESSADynamicImportanceV1:
    source_asset_sha256: str
    source_face_witness_sha256: str
    source_qualification_key: str
    native_skin_sha256: str
    native_pose_sha256: str
    native_pose_mask_sha256: str
    sample_importance: Tensor  # [M] [0,1] from ORIGINAL source face ids
    sample_mask: Tensor        # [M] bool, independently qualified
    face_importance: Tensor    # [F] all original native face IDs
    qualified_pose_count: int
    source_absolute_g3_within_limits: bool
    source_failure_face_count: int
    report_sha256: str
    schema: str = SCHEMA
    source_generated_views_as_pose_truth: bool = False
    animation_ready_qualified: bool = False

    def digest(self) -> str:
        return self.report_sha256

    def validate(self, witness: NativeFaceLayerSamplesV1) -> None:
        witness.replay()
        if (
            self.schema!=SCHEMA
            or self.source_asset_sha256!=witness.source_asset_sha256
            or self.source_face_witness_sha256!=witness.receipt_sha256()
            or self.sample_importance.shape!=witness.sample_face_ids.shape
            or self.sample_mask.shape!=witness.sample_face_ids.shape
            or self.face_importance.shape!=(len(witness.faces),)
            or self.sample_mask.dtype!=torch.bool
            or not bool(self.sample_mask.any())
            or self.sample_mask.device!=witness.vertices.device
            or self.sample_importance.device!=witness.vertices.device
            or self.face_importance.device!=witness.vertices.device
            or any(x.dtype!=witness.vertices.dtype
                   for x in (self.sample_importance,self.face_importance))
            or any(not bool(torch.isfinite(x).all())
                   for x in (self.sample_importance,self.face_importance))
            or any(bool((x<0).any() or (x>1).any())
                   for x in (self.sample_importance,self.face_importance))
            or not torch.equal(
                self.sample_importance,
                self.face_importance[witness.sample_face_ids],
            )
            or type(self.qualified_pose_count) is not int
            or self.qualified_pose_count<1
            or type(self.source_absolute_g3_within_limits) is not bool
            or type(self.source_failure_face_count) is not int
            or not 0<=self.source_failure_face_count<=len(witness.faces)
            or self.source_generated_views_as_pose_truth
            or self.animation_ready_qualified
            or type(self.source_qualification_key) is not str
            or not self.source_qualification_key
            or any(type(h) is not str or len(h)!=64
                   or any(c not in "0123456789abcdef" for c in h)
                   for h in (self.native_skin_sha256,self.native_pose_sha256,
                             self.native_pose_mask_sha256,self.report_sha256))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_IMPORTANCE_DOMAIN")
        proof=_report_data(
            source_asset_sha256=self.source_asset_sha256,
            source_face_witness_sha256=self.source_face_witness_sha256,
            source_qualification_key=self.source_qualification_key,
            native_skin_sha256=self.native_skin_sha256,
            native_pose_sha256=self.native_pose_sha256,
            native_pose_mask_sha256=self.native_pose_mask_sha256,
            sample_importance=self.sample_importance,
            sample_mask=self.sample_mask,
            face_importance=self.face_importance,
            qualified_pose_count=self.qualified_pose_count,
            source_absolute_g3_within_limits=self.source_absolute_g3_within_limits,
            source_failure_face_count=self.source_failure_face_count,
        )
        if _sha_report(proof)!=self.report_sha256:
            raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_IMPORTANCE_RECEIPT_DRIFT")


def _report_data(**fields):
    return {
        "schema":SCHEMA,
        **{k:(tensor_sha256_v1(v) if isinstance(v,Tensor) else v)
           for k,v in fields.items()},
        "source_generated_views_as_pose_truth":False,
        "animation_ready_qualified":False,
    }


def _sha_report(data) -> str:
    return sha256(json.dumps(
        data,sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()


def build_source_native_tessa_dynamic_importance_v1(
    *, witness: NativeFaceLayerSamplesV1,
    native_skin: Tensor, native_pose_matrices: Tensor,
    qualified_pose_mask: Tensor,
    source_qualification_key: str,
    trusted_source_registry: Mapping[str,Mapping[str,object]],
    limits: SeededDynamicLimitsV1 = SeededDynamicLimitsV1(),
    face_chunk: int = 32768,
) -> NativeTESSADynamicImportanceV1:
    """Measure exact SOURCE-face G3/G3B-like strain over sealed source poses.

    Importance is a monotonic, nonnegative, saturated risk indicator.
    Soft geometric margins are for *teaching refinement*, NOT G3 admission.
    Skin discontinuity is an independent risk factor. Model-predicted poses
    or student TESSA vertex identities are forbidden as source authority.
    """
    limits.validate()
    if (not isinstance(witness,NativeFaceLayerSamplesV1)
            or type(source_qualification_key) is not str
            or not source_qualification_key
            or type(face_chunk) is not int
            or not 1024<=face_chunk<=131072):
        raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_SOURCE_POLICY")
    witness.replay()
    v,f=witness.vertices,witness.faces
    # Native face sampling validates sampled triangles; the motion label
    # operates on ALL original faces. Never let an unobserved degenerate,
    # duplicate or non-manifold source face silently teach the student.
    from .native_layer_open_boundary_v1 import exact_open_boundary_edges_v1
    exact_open_boundary_edges_v1(f)
    if (
        torch.unique(f.sort(dim=-1).values,dim=0).shape[0]!=len(f)
        or bool((
            torch.linalg.vector_norm(torch.cross(
                v[f[:,1]]-v[f[:,0]],
                v[f[:,2]]-v[f[:,0]],dim=-1,
            ),dim=-1)<=1.e-11
        ).any())
    ):
        raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_FULL_SOURCE_FACE_INVALID")
    n=len(v)
    j=native_skin.shape[1] if native_skin.ndim==2 else -1
    p=native_pose_matrices.shape[1] if native_pose_matrices.ndim==5 else -1
    if (
        not isinstance(native_skin,Tensor)
        or not isinstance(native_pose_matrices,Tensor)
        or not isinstance(qualified_pose_mask,Tensor)
        or not 1<=len(f)<=limits.max_faces
        or not 1<=p<=limits.max_poses
        or j<1 or native_skin.shape!=(n,j)
        or native_pose_matrices.shape!=(1,p,j,4,4)
        or qualified_pose_mask.shape!=(1,p)
        or qualified_pose_mask.dtype!=torch.bool
        or not bool(qualified_pose_mask.any())
        or any(x.device!=v.device for x in (
            native_skin,native_pose_matrices,qualified_pose_mask))
        or native_skin.dtype!=v.dtype
        or native_pose_matrices.dtype!=v.dtype
        or not bool(torch.isfinite(native_skin).all())
        or not bool(torch.isfinite(native_pose_matrices).all())
        or bool((native_skin<0).any())
        or not bool(torch.allclose(
            native_skin.sum(-1),torch.ones_like(native_skin[:,0]),
            atol=1.e-5,rtol=0,
        ))
        or not bool(torch.allclose(
            native_pose_matrices[...,3,:],
            native_pose_matrices.new_tensor([0.,0.,0.,1.]).expand_as(
                native_pose_matrices[...,3,:],
            ),
            atol=1.e-5,rtol=0,
        ))
    ):
        raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_SOURCE_SKIN_POSE_DOMAIN")
    keys={
        "schema":SOURCE_RECEIPT_SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_NATIVE_SKIN_AND_POSES",
        "source_asset_sha256":witness.source_asset_sha256,
        "source_native_face_witness_sha256":witness.receipt_sha256(),
        "source_skin_native_vertex_order":True,
        "generated_views_as_teacher":False,
        "source_rights_independently_verified":True,
        "native_skin_sha256":tensor_sha256_v1(native_skin),
        "native_pose_matrices_sha256":tensor_sha256_v1(native_pose_matrices),
        "native_pose_mask_sha256":tensor_sha256_v1(qualified_pose_mask),
    }
    receipt=trusted_source_registry.get(source_qualification_key)
    if not isinstance(receipt,Mapping) or any(
        receipt.get(k)!=value for k,value in keys.items()
    ):
        raise ValueError("RSS_VNEXT_NATIVE_DYNAMIC_TRUSTED_RECEIPT_REQUIRED")

    with torch.no_grad():
        from .animation_ready_mesh import posed_lbs_from_joint_matrices
        # Positive, bounded mechanical risk. At no deformation, all
        # normalized ratios sit below thresholds (score=0); the L1 field
        # still marks the skin transitions where *unseen* motion may stress.
        wf=native_skin[f]
        skin_jump=torch.stack((
            (wf[:,0]-wf[:,1]).abs().sum(-1),
            (wf[:,1]-wf[:,2]).abs().sum(-1),
            (wf[:,2]-wf[:,0]).abs().sum(-1),
        )).amax(dim=0)
        skin_risk=(skin_jump/2.).clamp(0.,1.)
        face_risk=skin_risk.clone()
        failure_faces=torch.zeros(len(f),dtype=torch.bool,device=v.device)
        rest=v.unsqueeze(0)
        indices=qualified_pose_mask[0].nonzero(as_tuple=False).flatten()
        for pose_id in indices.tolist():
            posed=posed_lbs_from_joint_matrices(
                rest,native_skin.unsqueeze(0),
                native_pose_matrices[:,pose_id:pose_id+1],
            )[:,0]
            for start in range(0,len(f),face_chunk):
                stop=min(len(f),start+face_chunk)
                met=triangle_deformation_metrics_v1(
                    rest,posed,f[start:stop],
                )
                # Directional bending does not imply flipped topology.
                # These scores are frame-invariant singular metrics.
                area=met.area_ratio[0]
                cond=met.condition_number[0]
                stretch=met.max_edge_ratio[0]
                if not bool(torch.isfinite(area).all()
                            and torch.isfinite(cond).all()
                            and torch.isfinite(stretch).all()):
                    raise ValueError(
                        "RSS_VNEXT_NATIVE_DYNAMIC_NONFINITE_SOURCE_FACE"
                    )
                # 0 is rigid baseline, 1 is the frozen numerical G3
                # threshold. Do not silently change acceptance thresholds.
                cond_risk=((cond-1.)/(limits.max_condition_number-1.)).clamp(0.,1.)
                edge_risk=((stretch-1.)/(limits.max_edge_ratio-1.)).clamp(0.,1.)
                area_low=((1.-area)/(1.-limits.min_area_ratio)).clamp(0.,1.)
                area_high=((area-1.)/(limits.max_area_ratio-1.)).clamp(0.,1.)
                pose_risk=torch.stack((
                    cond_risk,edge_risk,area_low,area_high,
                )).amax(dim=0)
                face_risk[start:stop]=torch.maximum(
                    face_risk[start:stop],pose_risk,
                )
                failure_faces[start:stop]|=(
                    (area<limits.min_area_ratio)
                    | (area>limits.max_area_ratio)
                    | (cond>limits.max_condition_number)
                    | (stretch>limits.max_edge_ratio)
                )
        out=NativeTESSADynamicImportanceV1(
            source_asset_sha256=witness.source_asset_sha256,
            source_face_witness_sha256=witness.receipt_sha256(),
            source_qualification_key=source_qualification_key,
            native_skin_sha256=keys["native_skin_sha256"],
            native_pose_sha256=keys["native_pose_matrices_sha256"],
            native_pose_mask_sha256=keys["native_pose_mask_sha256"],
            sample_importance=face_risk[witness.sample_face_ids].clone(),
            sample_mask=torch.ones_like(
                witness.sample_face_ids,dtype=torch.bool,
            ),
            face_importance=face_risk,
            qualified_pose_count=len(indices),
            source_absolute_g3_within_limits=not bool(failure_faces.any()),
            source_failure_face_count=int(failure_faces.sum().item()),
            report_sha256="",
        )
        data=_report_data(**{
            name:value for name,value in vars(out).items()
            if name not in {"schema","report_sha256",
                            "source_generated_views_as_pose_truth",
                            "animation_ready_qualified"}
        })
        from dataclasses import replace
        out=replace(out,report_sha256=_sha_report(data))
        out.validate(witness)
        return out


def verify_dynamic_importance_bound_to_teacher_v1(
    risk: NativeTESSADynamicImportanceV1 | None,
    *,
    witness: NativeFaceLayerSamplesV1 | None,
    source_asset_sha256: str,
    source_importance: Tensor,
    source_importance_mask: Tensor,
    receipt: Mapping[str,object],
) -> None:
    if not isinstance(risk,NativeTESSADynamicImportanceV1) or witness is None:
        raise ValueError("RSS_VNEXT_TESSA_NATIVE_DYNAMIC_IMPORTANCE_REQUIRED")
    risk.validate(witness)
    if (
        risk.source_asset_sha256!=source_asset_sha256
        or receipt.get("native_dynamic_importance_schema")!=SCHEMA
        or receipt.get("native_dynamic_importance_receipt_sha256")!=risk.report_sha256
        or source_importance.shape!=risk.sample_importance.shape
        or source_importance_mask.shape!=risk.sample_mask.shape
        or not bool(source_importance_mask[risk.sample_mask].all())
        or bool((
            source_importance[risk.sample_mask]+1.e-7
            <risk.sample_importance[risk.sample_mask]
        ).any())
    ):
        raise ValueError("RSS_VNEXT_TESSA_NATIVE_DYNAMIC_IMPORTANCE_RECEIPT_DRIFT")


__all__=[
    "SCHEMA","SOURCE_RECEIPT_SCHEMA",
    "NativeTESSADynamicImportanceV1",
    "build_source_native_tessa_dynamic_importance_v1",
    "verify_dynamic_importance_bound_to_teacher_v1",
]
