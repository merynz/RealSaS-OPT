from __future__ import annotations

"""Pre-admission full-face dynamics court for ONE final TESSA indexed mesh.

This is a model-side numerical diagnostic inspired by Compiler G3/G3B. It is
NEVER a substitute for the Compiler's canonical G3, G3B or Stage35 actual clip
proofs: those use independent frames, binding hashes, qualified envelopes and
application-specific admission policy. No average training loss can mint this
report into a ProductRevision.

Critically the court examines EVERY face under EVERY supplied pose. No
sampling, batch average or detached mean can hide a collapsed local triangle.
Both signed-solid and open sheet faces share the same vertex address domain.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor

from models.tessa.v1.mechanical_objective_v1 import (
    triangle_deformation_metrics_v1,
)

SCHEMA = "RealSaS.RSSvNext.SeededDynamicFullFaceDiagnostic.v1"


@dataclass(frozen=True)
class SeededDynamicLimitsV1:
    min_area_ratio: float = .05
    max_area_ratio: float = 20.
    max_condition_number: float = 16.
    max_edge_ratio: float = 4.
    g3b_skin_l1_risk_threshold: float = .5
    max_faces: int = 2_000_000
    max_poses: int = 1024

    def validate(self) -> None:
        if (
            not all(type(v) in (int,float) and math.isfinite(v)
                    for v in (
                        self.min_area_ratio,self.max_area_ratio,
                        self.max_condition_number,self.max_edge_ratio,
                        self.g3b_skin_l1_risk_threshold,
                    ))
            or not 0 < self.min_area_ratio < 1
            or not 1 < self.max_area_ratio <= 100
            or not 1 < self.max_condition_number <= 1000
            or not 1 < self.max_edge_ratio <= 100
            or not 0 <= self.g3b_skin_l1_risk_threshold <= 2
            or type(self.max_faces) is not int or not 1 <= self.max_faces <= 2_000_000
            or type(self.max_poses) is not int or not 1 <= self.max_poses <= 1024
        ):
            raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_LIMIT_POLICY")


@dataclass(frozen=True)
class SeededDynamicReportV1:
    mesh_sha256: str
    skin_sha256: str
    pose_sha256: str
    face_count: int
    observed_pose_count: int
    risky_face_count: int
    minimum_area_ratio: float
    maximum_area_ratio: float
    maximum_condition_number: float
    maximum_edge_ratio: float
    maximum_skin_l1_on_face: float
    g3b_max_condition_on_risky_faces: float
    g3b_max_edge_on_risky_faces: float
    failures: tuple[str,...]
    passed_numerical_preflight: bool
    report_sha256: str
    schema: str = SCHEMA
    compiler_g3_g3b_or_stage35_pass_claimed: bool = False
    animation_ready_qualified: bool = False

    def require_numerical_preflight(self) -> None:
        if not self.passed_numerical_preflight or self.failures:
            raise ValueError(
                "RSS_VNEXT_SEEDED_DYNAMIC_NUMERICAL_REJECT:" +
                ",".join(self.failures)
            )


def _sha_tensor(x: Tensor) -> str:
    a=x.detach().contiguous().cpu().numpy()
    h=sha256()
    h.update(str((str(a.dtype),a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def run_seeded_dynamic_full_face_court_v1(
    *, rest_xyz: Tensor, faces: Tensor,
    skin_weights: Tensor, joint_matrices: Tensor,
    qualified_pose_mask: Tensor,
    limits: SeededDynamicLimitsV1 = SeededDynamicLimitsV1(),
) -> SeededDynamicReportV1:
    """All-face G3 and skin-risk G3B-like metrics on exact SAME M.

    Joint matrices and pose mask are supplied by caller; this court DOES NOT
    qualify their provenance or generate arbitrary joint probes. A training
    evaluation can pass source-qualified poses; product Compiler still MUST
    independently execute its frozen G3/G3B and actual motion gate.
    """
    limits.validate()
    if not all(isinstance(x,Tensor) for x in (
        rest_xyz,faces,skin_weights,joint_matrices,qualified_pose_mask,
    )):
        raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_TENSORS_REQUIRED")
    v=rest_xyz.shape[1] if rest_xyz.ndim==3 else -1
    p=joint_matrices.shape[1] if joint_matrices.ndim==5 else -1
    j=joint_matrices.shape[2] if joint_matrices.ndim==5 else -1
    if (
        rest_xyz.shape != (1,v,3) or v < 3
        or faces.ndim!=2 or faces.shape[-1]!=3
        or faces.dtype!=torch.long or not 1 <= len(faces) <= limits.max_faces
        or skin_weights.shape!=(1,v,j) or j<1
        or joint_matrices.shape!=(1,p,j,4,4)
        or not 1<=p<=limits.max_poses
        or qualified_pose_mask.shape!=(1,p)
        or qualified_pose_mask.dtype!=torch.bool
        or not bool(qualified_pose_mask.any())
        or any(x.device!=rest_xyz.device for x in (
            faces,skin_weights,joint_matrices,qualified_pose_mask,
        ))
        or any(x.dtype!=rest_xyz.dtype for x in (
            skin_weights,joint_matrices,
        ))
        or not torch.is_floating_point(rest_xyz)
        or bool((faces<0).any()) or int(faces.max())>=v
        or any(not bool(torch.isfinite(x).all()) for x in (
            rest_xyz,skin_weights,joint_matrices,
        ))
        or bool((skin_weights<0).any())
        or not bool(torch.allclose(
            skin_weights.sum(-1),torch.ones_like(skin_weights[...,0]),
            atol=1.e-5,rtol=0,
        ))
    ):
        raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_EXACT_M_POSE_DOMAIN")

    # An index graph is not an independent position graph; reject fabricated
    # zero-area/duplicate face corners before the SVD clamps small values.
    if bool(((faces[:,0]==faces[:,1]) | (faces[:,1]==faces[:,2]) |
             (faces[:,2]==faces[:,0])).any()):
        raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_DUPLICATE_FACE_CORNER")
    if torch.unique(faces.sort(dim=-1).values,dim=0).shape[0]!=len(faces):
        raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_DUPLICATE_TRIANGLES")
    with torch.no_grad():
        tri=rest_xyz[0,faces]
        area_vec=torch.cross(tri[:,1]-tri[:,0],
                             tri[:,2]-tri[:,0],dim=-1)
        if bool((torch.linalg.vector_norm(area_vec,dim=-1)<=1.e-11).any()):
            raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_DEGENERATE_REST")

        # G3B-like risk: the actual model skin field, not semantically
        # approximated layer membership or a second independent skin mesh.
        wf=skin_weights[0,faces]
        l1=torch.stack((
            (wf[:,0]-wf[:,1]).abs().sum(-1),
            (wf[:,1]-wf[:,2]).abs().sum(-1),
            (wf[:,2]-wf[:,0]).abs().sum(-1),
        )).amax(dim=0)
        risky=l1>limits.g3b_skin_l1_risk_threshold
        max_skin_l1=float(l1.max().item())

        # Finite affine skin-matrices are required; reject projective input.
        target=joint_matrices.new_tensor([0.,0.,0.,1.])
        if not bool(torch.allclose(
            joint_matrices[...,3,:],target.expand_as(
                joint_matrices[...,3,:],
            ),atol=1.e-5,rtol=0,
        )):
            raise ValueError("RSS_VNEXT_SEEDED_DYNAMIC_JOINT_AFFINE_REQUIRED")

        homogeneous=torch.cat((
            rest_xyz[0],torch.ones((v,1),dtype=rest_xyz.dtype,
                                   device=rest_xyz.device),
        ),dim=-1)
        min_area=float("inf");max_area=0.;max_cond=0.;max_edge=0.
        g3b_cond=0.;g3b_edge=0.
        reasons=set()
        pose_ids=qualified_pose_mask[0].nonzero(as_tuple=False).flatten().tolist()
        for pose_id in pose_ids:
            pose_m=joint_matrices[0,pose_id]
            transformed=torch.einsum(
                "vc,jkc->vjk",homogeneous,pose_m[:,:3,:],
            )
            # einsum above produces v,j,3 with no materialized v,j,4 points.
            posed=(transformed*skin_weights[0,:, :,None]).sum(dim=1)
            metrics=triangle_deformation_metrics_v1(
                rest_xyz,posed.unsqueeze(0),faces,
            )
            ar=metrics.area_ratio[0]
            cond=metrics.condition_number[0]
            edge=metrics.max_edge_ratio[0]
            if not bool(torch.isfinite(ar).all() and
                        torch.isfinite(cond).all() and
                        torch.isfinite(edge).all()):
                reasons.add("NONFINITE_POSE_FACE")
            min_area=min(min_area,float(ar.min().item()))
            max_area=max(max_area,float(ar.max().item()))
            max_cond=max(max_cond,float(cond.max().item()))
            max_edge=max(max_edge,float(edge.max().item()))
            if bool((ar<limits.min_area_ratio).any()):
                reasons.add("G3_COLLAPSED_FACE_AREA")
            if bool((ar>limits.max_area_ratio).any()):
                reasons.add("G3_EXPLODED_FACE_AREA")
            if bool((cond>limits.max_condition_number).any()):
                reasons.add("G3_ILL_CONDITIONED_FACE")
            if bool((edge>limits.max_edge_ratio).any()):
                reasons.add("G3_STRETCHED_FACE_EDGE")
            if bool(risky.any()):
                g3b_cond=max(g3b_cond,float(cond[risky].max().item()))
                g3b_edge=max(g3b_edge,float(edge[risky].max().item()))
                if bool((cond[risky]>limits.max_condition_number).any()
                        or (ar[risky]<limits.min_area_ratio).any()
                        or (ar[risky]>limits.max_area_ratio).any()
                        or (edge[risky]>limits.max_edge_ratio).any()):
                    reasons.add("G3B_SKIN_TOPOLOGY_RISK_FACE")
        # No release admission: G3 and G3B must additionally qualify actual
        # joint frames/envelopes. Stage35 needs a real clip, not these poses.
        seal={
            "schema":SCHEMA,"limits":vars(limits),
            "rest":_sha_tensor(rest_xyz),"faces":_sha_tensor(faces),
            "skin":_sha_tensor(skin_weights),
            "pose":_sha_tensor(joint_matrices),
            "mask":_sha_tensor(qualified_pose_mask),
            "min_area":min_area,"max_area":max_area,
            "max_cond":max_cond,"max_edge":max_edge,
            "max_skin_l1":max_skin_l1,"g3b_cond":g3b_cond,
            "g3b_edge":g3b_edge,"failures":sorted(reasons),
        }
        digest=sha256(json.dumps(
            seal,sort_keys=True,separators=(",",":"),
            allow_nan=False,
        ).encode()).hexdigest()
        return SeededDynamicReportV1(
            mesh_sha256=sha256((seal["rest"]+seal["faces"]).encode()).hexdigest(),
            skin_sha256=seal["skin"],pose_sha256=seal["pose"],
            face_count=len(faces),observed_pose_count=len(pose_ids),
            risky_face_count=int(risky.sum().item()),
            minimum_area_ratio=min_area,maximum_area_ratio=max_area,
            maximum_condition_number=max_cond,maximum_edge_ratio=max_edge,
            maximum_skin_l1_on_face=max_skin_l1,
            g3b_max_condition_on_risky_faces=g3b_cond,
            g3b_max_edge_on_risky_faces=g3b_edge,
            failures=tuple(sorted(reasons)),
            passed_numerical_preflight=not reasons,
            report_sha256=digest,
        )


__all__ = [
    "SCHEMA","SeededDynamicLimitsV1","SeededDynamicReportV1",
    "run_seeded_dynamic_full_face_court_v1",
]
