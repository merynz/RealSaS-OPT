from __future__ import annotations

"""Production-target seeded TESSA objective, not legacy triangle-token CE.

Targets come from independently admitted SOURCE geometry and optionally
SOURCE-native bind/skin/pose. Candidate M* is the student's IRIS-derived mesh.
No source mesh/weights/pose enter the student forward; no per-edge edit label.
The trainer requires source-bound exact tensor-digest receipts and refuses
missing supervision rather than turning a requested loss into a fake zero.

Initial coefficients are VERSIONED engineering starts, not empirical optima.
This module does no dataset processing and does not start training itself.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .animation_ready_mesh import RSSvNextStructuralMeshCandidate
from .seeded_surface_sampling_v1 import (
    SeededSurfaceSamplingPolicyV1,
    component_spatial_sample_indices_v1,
)
from .tessa_seeded_surface_v1 import TESSASeededSurfaceV1
from .seeded_layer_source_court_v1 import qualify_layer_source_court_v1
from .native_face_layer_truth_v1 import NativeFaceLayerSamplesV1
from .native_dynamic_tessa_importance_v1 import (
    NativeTESSADynamicImportanceV1,
    verify_dynamic_importance_bound_to_teacher_v1,
)
from .native_layer_open_boundary_v1 import (
    NativeLayerOpenBoundaryV1, exact_open_boundary_edges_v1,
    validate_native_open_boundary_v1,
)

SCHEMA = "RealSaS.RSSvNext.SeededTESSA.SourceSupervision.v1"
PHASE = "RSSVNEXT_V1_SEEDED_CONTINUOUS_TOPOLOGY"


def _digest(t: Tensor) -> str:
    if not isinstance(t, Tensor):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_TEACHER_TENSOR_REQUIRED")
    c = t.detach().contiguous().cpu()
    raw = c.reshape(-1).view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"shape": list(c.shape), "dtype": str(c.dtype)},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return sha256(header+b"\x00"+raw).hexdigest()


def _sha(x: object) -> bool:
    return type(x) is str and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x
    )


@dataclass(frozen=True)
class SeededTESSALossWeightsV1:
    rest_surface: float = 1.0
    normal: float = 0.2
    boundary: float = 0.0  # signed-solid v1 has no open hem; explicitly enable for native open sources
    quality: float = 0.1
    layer_penetration: float = 0.0
    face_budget: float = 0.01
    deformation: float = 0.0
    pose_strain: float = 0.0
    structural_importance: float = 0.25
    clip_gradient_norm: float = 1.0

    def validate(self) -> None:
        if any(
            type(v) not in (float, int) or not math.isfinite(v)
            or v < 0 for v in asdict(self).values()
        ) or self.clip_gradient_norm <= 0 or sum(
            v for k, v in asdict(self).items() if k!="clip_gradient_norm"
        ) <= 0:
            raise ValueError("RSS_VNEXT_TESSA_V1_LOSS_WEIGHT_POLICY_INVALID")


@dataclass(frozen=True)
class SeededTESSASourceTeacherV1:
    source_asset_sha256: str
    source_qualification_key: str
    source_xyz: Tensor                    # [M,3] native 3D sampled surface
    source_normals: Tensor                # [M,3] native outward normals
    source_point_mask: Tensor             # [M] verified point + normal
    source_importance: Tensor             # [M] curvature/pose-derived [0,1]
    source_importance_mask: Tensor        # [M] native valid
    source_component_ids: Tensor | None = None  # [M], native connectivity; no student ID claim
    source_layer_ids: Tensor | None = None  # [M], audited native face/material partition
    native_face_layer_samples: NativeFaceLayerSamplesV1 | None = None  # source-only indexed triangle + barycentric witness
    native_open_boundary: NativeLayerOpenBoundaryV1 | None = None  # independently replayed complete native open hems
    # Optional native-face G3/G3B-style risk label. Never from student M/skin.
    native_dynamic_importance: NativeTESSADynamicImportanceV1 | None = None
    source_boundary_xyz: Tensor | None = None  # [K,3], exact known open seams
    source_boundary_mask: Tensor | None = None # [K] bool
    # Only if a trusted source→candidate barycentric/skin joint reindex exists.
    source_pose_matrices: Tensor | None = None # [1,P,J,4,4]
    source_candidate_skin: Tensor | None = None # [1,V,J], teacher only
    source_candidate_posed: Tensor | None = None # [1,P,V,3]
    source_pose_mask: Tensor | None = None # [1,P,V]
    source_layer_penetration_target: Tensor | None = None
    source_layer_penetration_mask: Tensor | None = None
    source_alignment_identity: str = "SOURCE_NATIVE_SURFACE_NOT_GENERATED"

    def tensor_fields(self) -> dict[str, Tensor]:
        return {
            k: v for k, v in vars(self).items() if isinstance(v, Tensor)
        }


def _validate_source_teacher(t: SeededTESSASourceTeacherV1) -> None:
    if (
        not isinstance(t, SeededTESSASourceTeacherV1)
        or not _sha(t.source_asset_sha256)
        or not t.source_qualification_key
        or t.source_alignment_identity !=
            "SOURCE_NATIVE_SURFACE_NOT_GENERATED"
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_NATIVE_PROVENANCE_REQUIRED")
    xyz,normals,mask,importance,importance_mask = (
        t.source_xyz,t.source_normals,t.source_point_mask,
        t.source_importance,t.source_importance_mask,
    )
    m=xyz.shape[0] if xyz.ndim == 2 else -1
    if (
        xyz.shape != (m,3) or m < 3
        or normals.shape != xyz.shape
        or mask.shape != (m,) or mask.dtype != torch.bool
        or importance.shape != (m,)
        or importance_mask.shape != (m,)
        or importance_mask.dtype != torch.bool
        or not bool(mask.any()) or not bool(importance_mask.any())
        or any(not bool(torch.isfinite(v).all()) for v in (
            xyz, normals, importance,
        ))
        or bool((importance < 0).any() or (importance > 1).any())
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_SURFACE_DOMAIN")
    if t.source_component_ids is not None and (
        t.source_component_ids.shape != (m,)
        or t.source_component_ids.dtype != torch.long
        or bool((t.source_component_ids < 0).any())
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_COMPONENT_DOMAIN")
    if t.source_layer_ids is not None and (
        t.source_layer_ids.shape != (m,)
        or t.source_layer_ids.dtype != torch.long
        or t.source_layer_ids.device != xyz.device
        or bool((t.source_layer_ids<0).any())
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_LAYER_ID_DOMAIN")
    if (t.source_boundary_xyz is None) != (t.source_boundary_mask is None):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_BOUNDARY_INCOMPLETE")
    if t.source_boundary_xyz is not None:
        b, bm=t.source_boundary_xyz,t.source_boundary_mask
        if (b.ndim != 2 or b.shape[-1] != 3 or b.shape[0] < 1
                or bm.shape != (b.shape[0],)
                or bm.dtype != torch.bool or not bool(bm.any())
                or not bool(torch.isfinite(b).all())):
            raise ValueError("RSS_VNEXT_TESSA_SOURCE_BOUNDARY_DOMAIN")
    pose_fields=(
        t.source_pose_matrices,t.source_candidate_skin,
        t.source_candidate_posed,t.source_pose_mask,
    )
    if any(v is not None for v in pose_fields):
        if any(v is None for v in pose_fields):
            raise ValueError("RSS_VNEXT_TESSA_SOURCE_POSE_INCOMPLETE")
        mats,skin,target,pm=pose_fields
        if (mats.ndim != 5 or mats.shape[0] != 1
                or mats.shape[-2:] != (4,4)
                or skin.shape != (1,target.shape[2],mats.shape[2])
                or target.shape != (1,mats.shape[1],skin.shape[1],3)
                or pm.shape != target.shape[:3] or pm.dtype != torch.bool
                or not bool(pm.any())
                or any(not bool(torch.isfinite(v).all()) for v in (
                    mats,skin,target,
                ))
                or bool((skin < 0).any())
                or not bool(torch.allclose(
                    skin.sum(-1),torch.ones_like(skin[...,0]),
                    atol=1e-5,rtol=0,
                ))):
            raise ValueError("RSS_VNEXT_TESSA_SOURCE_POSE_DOMAIN")
    if (t.source_layer_penetration_target is None) != (
        t.source_layer_penetration_mask is None
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_LAYER_INCOMPLETE")


def _boundary_vertices(faces: Tensor, verts: Tensor) -> Tensor:
    edges=torch.cat((
        faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]],
    )).sort(dim=1).values
    u,c=torch.unique(edges,dim=0,return_counts=True)
    if bool((c>2).any()):
        raise ValueError("RSS_VNEXT_TESSA_NONMANIFOLD_BOUNDARY")
    vi=torch.unique(u[c==1].reshape(-1),sorted=True)
    return verts[0,vi]


@torch.no_grad()
def _bounded_nearest_source_indices_v1(
    predicted_xyz: Tensor, source_xyz: Tensor,
) -> tuple[Tensor, Tensor]:
    """Source-label association has no gradient; match without NxM storage.

    Nearest-neighbor indices are not teacher-forced geometry. Ties choose
    the lowest source index independent of source batch chunking.
    """
    if (predicted_xyz.ndim != 2 or predicted_xyz.shape[-1] != 3
            or source_xyz.ndim != 2 or source_xyz.shape[-1] != 3
            or not 1 <= predicted_xyz.shape[0] <= 2_000_000
            or not 1 <= source_xyz.shape[0] <= 2_000_000):
        raise ValueError("RSS_VNEXT_TESSA_NEAREST_SOURCE_DOMAIN")
    output_indices, output_distances = [], []
    for start in range(0, predicted_xyz.shape[0], 256):
        p = predicted_xyz[start:start + 256].detach().float()
        best = torch.full(
            (len(p),), float("inf"), device=p.device, dtype=p.dtype,
        )
        best_id = torch.zeros(len(p), device=p.device, dtype=torch.long)
        for source_start in range(0, source_xyz.shape[0], 1024):
            tile = torch.cdist(
                p, source_xyz[source_start:source_start+1024].detach().float(),
            )
            distance, local_id = tile.min(dim=-1)
            choose = distance < best
            best = torch.where(choose, distance, best)
            best_id = torch.where(choose, local_id + source_start, best_id)
        output_indices.append(best_id)
        output_distances.append(best)
    return torch.cat(output_indices), torch.cat(output_distances)


def _bounded_bidirectional_dist(
    prediction: Tensor, target: Tensor, *,
    scale: Tensor, exact_zero: bool = False,
) -> Tensor:
    if (prediction.ndim != 2 or target.ndim != 2
            or prediction.shape[-1] != 3 or target.shape[-1] != 3
            or not 1 <= prediction.shape[0] <= 65536
            or not 1 <= target.shape[0] <= 65536):
        raise ValueError("RSS_VNEXT_TESSA_SURFACE_LOSS_DOMAIN")
    # Chunked exact nearest-source distances; no O(V*N) full matrix in memory.
    def _nearest(a: Tensor,b: Tensor) -> Tensor:
        d=[]
        for start in range(0,a.shape[0],512):
            mins=None
            for other in range(0,b.shape[0],1024):
                tile=torch.cdist(
                    a[start:start+512].float(),
                    b[other:other+1024].float(),
                    compute_mode=(
                        "donot_use_mm_for_euclid_dist" if exact_zero
                        else "use_mm_for_euclid_dist_if_necessary"
                    ),
                )
                best=tile.min(dim=1).values
                mins=best if mins is None else torch.minimum(mins,best)
            d.append(mins)
        return torch.cat(d)
    return (_nearest(prediction,target).mean()
            +_nearest(target,prediction).mean())/(2*scale)


class RSSvNextSeededTESSATrainerV1:
    """Versioned seeded geometry stage; no legacy TESSA token CE or dataset IO."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        weights: SeededTESSALossWeightsV1,
        trusted_source_registry: Mapping[str, Mapping[str, object]],
        max_source_projection_distance_ratio: float = 0.07,
        sampling_policy: SeededSurfaceSamplingPolicyV1 = SeededSurfaceSamplingPolicyV1(),
    ):
        weights.validate()
        sampling_policy.validate()
        if (not 0 < max_source_projection_distance_ratio <= 0.2
                or not trusted_source_registry):
            raise ValueError("RSS_VNEXT_TESSA_TRAIN_SOURCE_REGISTRY_REQUIRED")
        from .training_optimizer_ownership_v1 import (
            qualify_rss_vnext_optimizer_ownership_v1,
        )
        self.model=model
        self.optimizer=optimizer
        self.weights=weights
        self.sampling_policy=sampling_policy
        self.trusted_registry=MappingProxyType({
            str(k): MappingProxyType(dict(v))
            for k,v in trusted_source_registry.items()
        })
        self.max_match_ratio=max_source_projection_distance_ratio
        self.ownership=qualify_rss_vnext_optimizer_ownership_v1(
            model,optimizer,
        )
        self.policy_sha256=sha256(json.dumps(
            {"schema":SCHEMA,"phase":PHASE,"weights":asdict(weights),
             "max_match_ratio":self.max_match_ratio,
             "sampling_policy_sha256":self.sampling_policy.digest()},
            sort_keys=True,separators=(",",":"),allow_nan=False,
        ).encode()).hexdigest()

    def objective(
        self, mesh: RSSvNextStructuralMeshCandidate,
        teacher: SeededTESSASourceTeacherV1,
        *,
        source_face_budget: int,
        sampler_step: int = 0,
    ) -> dict[str,Tensor]:
        if (not isinstance(mesh,RSSvNextStructuralMeshCandidate)
                or mesh.seeded_surface is None
                or mesh.source_authority
                    != "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"
                or type(source_face_budget) is not int
                or not 1 <= source_face_budget <= 2_000_000):
            raise ValueError("RSS_VNEXT_TESSA_TRAIN_SEEDED_MESH_REQUIRED")
        if type(sampler_step) is not int or sampler_step < 0:
            raise ValueError("RSS_VNEXT_SEEDED_SAMPLER_STEP_INVALID")
        mesh.validate()
        _validate_source_teacher(teacher)
        receipt=self.trusted_registry.get(teacher.source_qualification_key)
        if not receipt or any(
            receipt.get(k) != v for k,v in {
                "schema":SCHEMA,
                "admission_status":"QUALIFIED_INDEPENDENT_SOURCE_NATIVE_3D",
                "source_asset_sha256":teacher.source_asset_sha256,
                "source_alignment_identity":teacher.source_alignment_identity,
                "required_phase":PHASE,
                "source_rights_independently_verified":True,
                "generated_view_as_3d_teacher":False,
                **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
            }.items()
        ):
            raise ValueError("RSS_VNEXT_TESSA_TRAIN_SOURCE_RECEIPT_DRIFT")
        # Dynamic importance must originate in ORIGINAL native faces/skin/poses.
        # The teacher's curvature+motion importance may be a max-combination,
        # but it cannot downweight any independently sealed pose-risk target.
        if teacher.native_dynamic_importance is not None:
            verify_dynamic_importance_bound_to_teacher_v1(
                teacher.native_dynamic_importance,
                witness=teacher.native_face_layer_samples,
                source_asset_sha256=teacher.source_asset_sha256,
                source_importance=teacher.source_importance,
                source_importance_mask=teacher.source_importance_mask,
                receipt=receipt,
            )
        w=self.weights
        xyz=mesh.rest_vertices_iris
        source=teacher.source_xyz
        if source.device != xyz.device or source.dtype != xyz.dtype:
            raise ValueError("RSS_VNEXT_TESSA_TRAIN_DEVICE_OR_DTYPE")
        layer_court=qualify_layer_source_court_v1(
            mesh,source_layer_ids=teacher.source_layer_ids,
            source_xyz=source,source_component_ids=teacher.source_component_ids,
            receipt=receipt,source_asset_sha256=teacher.source_asset_sha256,
            tensor_digest=_digest,
            native_face_layer_samples=teacher.native_face_layer_samples,
        )
        # Open hems require independently qualified per-layer seam target.
        # Legacy single-set boundary chamfer could bridge a body/cape edge.
        if layer_court is not None and (
            w.deformation>0 or w.pose_strain>0 or w.layer_penetration>0
        ):
            raise ValueError(
                "RSS_VNEXT_SEEDED_LAYER_POSE_OR_PENETRATION_REINDEX_NOT_QUALIFIED"
            )
        policy=self.sampling_policy
        source_all_indices=torch.nonzero(
            teacher.source_point_mask,as_tuple=False,
        ).flatten()
        full_gt=source[source_all_indices]
        span=full_gt.max(dim=0).values-full_gt.min(dim=0).values
        scale=torch.linalg.vector_norm(span).clamp_min(1.e-5).detach()
        student_idx=component_spatial_sample_indices_v1(
            xyz[0],mesh.vertex_component_ids.long(),
            budget=policy.max_student_vertices,step=sampler_step,
            minimum=policy.min_samples_per_component,
        )
        sampled_xyz=xyz[0,student_idx]
        native_comps=(
            teacher.source_component_ids[source_all_indices]
            if teacher.source_component_ids is not None
            else torch.zeros_like(source_all_indices)
        )
        important=source_all_indices[
            teacher.source_importance_mask[source_all_indices]
        ] if w.structural_importance>0 else source_all_indices[:0]
        reserved=(
            min(len(important),max(policy.min_samples_per_component,
                                   policy.max_source_points//4))
            if len(source_all_indices)>policy.max_source_points else 0
        )
        # Reserve source label budget BEFORE component-stratified sampling.
        # Truncating an already stratified list could erase a whole cape,
        # tail or small hair strand. Fail closed if the remaining budget
        # cannot cover every qualified source component.
        core_budget=policy.max_source_points-reserved
        source_idx=source_all_indices[
            component_spatial_sample_indices_v1(
                full_gt,native_comps,
                budget=core_budget,step=sampler_step,
                minimum=policy.min_samples_per_component,
            )
        ]
        if reserved:
            # Importance labels may cover only a few native patches; keep
            # those candidates without displacing small source components.
            imp_comps=(
                teacher.source_component_ids[important]
                if teacher.source_component_ids is not None
                else torch.zeros_like(important)
            )
            imp_idx=important[component_spatial_sample_indices_v1(
                source[important],imp_comps,
                budget=reserved,step=sampler_step,
                minimum=policy.min_samples_per_component,
            )]
            source_idx=torch.unique(
                torch.cat((source_idx,imp_idx)),sorted=True,
            )
        gt=source[source_idx]
        vals={}
        layer_groups=(
            layer_court.grouped(student_idx,source_idx)
            if layer_court is not None else None
        )
        if w.rest_surface>0:
            if layer_groups is None:
                vals["rest_surface"]=_bounded_bidirectional_dist(
                    sampled_xyz,gt,scale=scale,
                )
            else:
                # Equal shell weight: a small cape cannot vanish inside
                # a large body's source-native point cloud.
                vals["rest_surface"]=torch.stack([
                    _bounded_bidirectional_dist(
                        xyz[0,student],source[native],scale=scale,
                    )
                    for student,native in layer_groups
                ]).mean()
        if w.normal>0 or w.structural_importance>0:
            # Teacher-only nearest labels are restricted to the SAME SOURCE
            # layer. The student source XYZ may coincide exactly with another
            # shell; coordinates never confer semantic membership.
            if layer_groups is None:
                local_nearest, minimum_distance = (
                    _bounded_nearest_source_indices_v1(sampled_xyz,gt)
                )
                nearest=source_idx[local_nearest]
                eligible=(
                    minimum_distance <= self.max_match_ratio*scale
                ) & teacher.source_point_mask[nearest]
            else:
                nearest=torch.empty(
                    len(student_idx),device=xyz.device,dtype=torch.long,
                )
                eligible=torch.zeros(
                    len(student_idx),device=xyz.device,dtype=torch.bool,
                )
                for layer_id,(student,native) in enumerate(layer_groups):
                    sample_positions=torch.nonzero(
                        layer_court.student_layer_ids[student_idx]==layer_id,
                        as_tuple=False,
                    ).flatten()
                    local_nearest, distances=(
                        _bounded_nearest_source_indices_v1(
                            xyz[0,student],source[native],
                        )
                    )
                    nearest[sample_positions]=native[local_nearest]
                    eligible[sample_positions]=(
                        distances <= self.max_match_ratio*scale
                    ) & teacher.source_point_mask[native[local_nearest]]
                    if not bool(eligible[sample_positions].any()):
                        raise ValueError(
                            "RSS_VNEXT_SEEDED_LAYER_SOURCE_PROJECTION_ABSTAIN"
                        )
            if not bool(eligible.any()):
                raise ValueError("RSS_VNEXT_TESSA_TRAIN_NO_MATCHED_SOURCE_NORMAL")
            if w.normal>0:
                if layer_groups is None:
                    expected=F.normalize(
                        teacher.source_normals[nearest[eligible]],dim=-1,
                    )
                    actual=F.normalize(
                        mesh.normals[0,student_idx[eligible]],dim=-1,
                    )
                    vals["normal"]=(1-(expected*actual).sum(-1)).mean()
                else:
                    normal_parts=[]
                    for layer_id in range(layer_court.layer_count):
                        valid=eligible & (
                            layer_court.student_layer_ids[student_idx]==layer_id
                        )
                        expected=F.normalize(
                            teacher.source_normals[nearest[valid]],dim=-1,
                        )
                        actual=F.normalize(
                            mesh.normals[0,student_idx[valid]],dim=-1,
                        )
                        normal_parts.append(
                            (1-(expected*actual).sum(-1)).mean()
                        )
                    vals["normal"]=torch.stack(normal_parts).mean()
            if w.structural_importance>0:
                metric_mask=eligible & teacher.source_importance_mask[nearest]
                if not bool(metric_mask.any()) or mesh.tessa_metric_logits is None:
                    raise ValueError("RSS_VNEXT_TESSA_TRAIN_METRIC_TRUTH_MISSING")
                if layer_groups is None:
                    vals["structural_importance"]=(
                        F.binary_cross_entropy_with_logits(
                            mesh.tessa_metric_logits[0,student_idx[metric_mask]],
                            teacher.source_importance[nearest[metric_mask]],
                        )
                    )
                else:
                    metric_parts=[]
                    for layer_id in range(layer_court.layer_count):
                        valid=metric_mask & (
                            layer_court.student_layer_ids[student_idx]==layer_id
                        )
                        if not bool(valid.any()):
                            raise ValueError(
                                "RSS_VNEXT_SEEDED_LAYER_METRIC_TRUTH_ABSTAIN"
                            )
                        metric_parts.append(F.binary_cross_entropy_with_logits(
                            mesh.tessa_metric_logits[0,student_idx[valid]],
                            teacher.source_importance[nearest[valid]],
                        ))
                    vals["structural_importance"]=torch.stack(metric_parts).mean()
        if w.boundary>0:
            if mesh.faces.shape[0]>200000:
                raise ValueError("RSS_VNEXT_TESSA_BOUNDARY_HIGHRES_RESOURCE_GATE")
            if layer_court is None:
                if teacher.source_boundary_xyz is None:
                    raise ValueError("RSS_VNEXT_TESSA_TRAIN_BOUNDARY_TRUTH_REQUIRED")
                predicted=_boundary_vertices(mesh.faces,xyz)
                if not len(predicted):
                    raise ValueError("RSS_VNEXT_TESSA_TRAIN_BOUNDARY_LOST")
                vals["boundary"]=_bounded_bidirectional_dist(
                    predicted,teacher.source_boundary_xyz[
                        teacher.source_boundary_mask],scale=scale,
                    exact_zero=True,
                )
            else:
                # OPEN cloth is not a signed-solid hole. Reconstruct the
                # full independent SOURCE hem from its native face graph;
                # a model-predicted mask or nearest XYZ cannot serve truth.
                native_boundary,source_boundary_roles,open_layers=(
                    validate_native_open_boundary_v1(
                        teacher.native_open_boundary,
                        teacher.native_face_layer_samples,
                        source_boundary_xyz=teacher.source_boundary_xyz,
                        source_boundary_mask=teacher.source_boundary_mask,
                        source_asset_sha256=teacher.source_asset_sha256,
                        layer_namespace_sha256=layer_court.namespace_sha256,
                        receipt=receipt,
                    )
                )
                # EXACT predicted face edges, not a semantic/visibility
                # pseudo-boundary. Reject a hole in a declared closed layer
                # even when an adjacent open cloth layer overlaps in XYZ.
                predicted_edges=exact_open_boundary_edges_v1(mesh.faces)
                predicted_ids=torch.unique(
                    predicted_edges.reshape(-1),sorted=True,
                ) if len(predicted_edges) else mesh.faces.new_empty((0,))
                predicted_roles=layer_court.student_layer_ids[predicted_ids]
                allowed=set(int(v) for v in open_layers.detach().cpu().tolist())
                parts=[]
                for layer_id in range(layer_court.layer_count):
                    pred=xyz[0,predicted_ids[predicted_roles==layer_id]]
                    src=native_boundary[source_boundary_roles==layer_id]
                    if layer_id not in allowed:
                        if len(pred):
                            raise ValueError(
                                "RSS_VNEXT_TESSA_UNQUALIFIED_CLOSED_LAYER_HOLE"
                            )
                        continue
                    if len(src)<3 or len(pred)<3:
                        raise ValueError(
                            "RSS_VNEXT_TESSA_OPEN_LAYER_HEM_MISSING"
                        )
                    parts.append(_bounded_bidirectional_dist(
                        pred,src,scale=scale,exact_zero=True,
                    ))
                if not parts:
                    raise ValueError("RSS_VNEXT_TESSA_OPEN_HEM_TRUTH_EMPTY")
                vals["boundary"]=torch.stack(parts).mean()
        quality_faces=mesh.faces
        if w.quality>0:
            if mesh.faces.shape[0]>policy.max_quality_faces:
                # Face centroids are a detached LOCATION for spatial sampling
                # only; computed face quality keeps live vertex gradients.
                all_centers=xyz[0,mesh.faces].detach().mean(dim=1)
                face_idx=component_spatial_sample_indices_v1(
                    all_centers,mesh.face_component_ids.long(),
                    budget=policy.max_quality_faces,step=sampler_step,
                    minimum=policy.min_samples_per_component,
                )
                quality_faces=mesh.faces[face_idx]
            triangle=xyz[0,quality_faces]
            sides=torch.stack((
                (triangle[:,1]-triangle[:,0]).norm(dim=-1),
                (triangle[:,2]-triangle[:,1]).norm(dim=-1),
                (triangle[:,0]-triangle[:,2]).norm(dim=-1),
            ),dim=-1)
            doublearea=torch.linalg.vector_norm(
                torch.cross(
                    triangle[:,1]-triangle[:,0],
                    triangle[:,2]-triangle[:,0],dim=-1,
                ),dim=-1,
            )
            # Scale invariant normalized sliver measure.
            quality=4*(3**0.5)*0.5*doublearea/(sides.square().sum(-1).clamp_min(1.e-12))
            vals["quality"]=(1-quality.clamp(0,1)).mean()
        if w.face_budget>0:
            excess=max(0,mesh.faces.shape[0]-source_face_budget)
            vals["face_budget"]=xyz.new_tensor(
                float(excess)/source_face_budget,
            )
        if w.layer_penetration>0:
            raise ValueError("RSS_VNEXT_TESSA_LAYER_PENETRATION_NOT_YET_QUALIFIED")
        if w.deformation>0 or w.pose_strain>0:
            if len(xyz[0])>policy.max_student_vertices:
                raise ValueError("RSS_VNEXT_TESSA_POSE_HIGHRES_RESOURCE_GATE")
            fields=(
                teacher.source_pose_matrices,teacher.source_candidate_skin,
                teacher.source_candidate_posed,teacher.source_pose_mask,
            )
            if any(v is None for v in fields):
                raise ValueError("RSS_VNEXT_TESSA_TRAIN_POSE_TRUTH_REQUIRED")
            matrices,skin,target,mask=fields
            if (skin.shape[1] != xyz.shape[1]
                    or target.shape[2] != xyz.shape[1]):
                raise ValueError("RSS_VNEXT_TESSA_TRAIN_POSE_VERTEX_CORRESPONDENCE")
            from .animation_ready_mesh import posed_lbs_from_joint_matrices
            posed=posed_lbs_from_joint_matrices(xyz,skin,matrices)
            if w.deformation>0:
                vals["deformation"]=F.smooth_l1_loss(
                    posed[mask]/scale,target[mask]/scale,
                )
            if w.pose_strain>0:
                # Preserve local edge-length changes relative to the EXACT
                # source-qualified target trajectory; do not punish all
                # non-rigid deformations as if cloth were a rigid body.
                faces=mesh.faces
                pred_e=(posed[:,:,faces[:,1]]-posed[:,:,faces[:,0]]).norm(dim=-1)
                true_e=(target[:,:,faces[:,1]]-target[:,:,faces[:,0]]).norm(dim=-1)
                valid=mask[:,:,faces[:,0]] & mask[:,:,faces[:,1]]
                if not bool(valid.any()):
                    raise ValueError("RSS_VNEXT_TESSA_TRAIN_POSE_EDGE_MASK_EMPTY")
                vals["pose_strain"]=F.smooth_l1_loss(
                    pred_e[valid]/scale,true_e[valid]/scale,
                )
        keys={k:v for k,v in asdict(w).items() if (
            k != "clip_gradient_norm" and v>0
        )}
        if set(vals) != set(keys):
            raise RuntimeError("RSS_VNEXT_TESSA_ACTIVE_LOSS_WITHOUT_TRUTH")
        total=sum(vals[k]*keys[k] for k in keys)
        if not bool(torch.isfinite(total).all()):
            raise ValueError("RSS_VNEXT_TESSA_TRAIN_NONFINITE_TOTAL")
        return {
            **vals,"total":total,
            "qualified_source_points":gt.new_tensor(len(full_gt)),
            "sampled_source_points":gt.new_tensor(len(gt)),
            "sampled_student_vertices":gt.new_tensor(len(student_idx)),
            "sampled_quality_faces":gt.new_tensor(
                len(quality_faces) if w.quality>0 else 0
            ),
            "sampling_step":gt.new_tensor(sampler_step),
            "qualified_source_layer_count":gt.new_tensor(
                layer_court.layer_count if layer_court is not None else 0
            ),
        }

    def train_step(
        self, mesh: RSSvNextStructuralMeshCandidate,
        teacher: SeededTESSASourceTeacherV1,
        *, source_face_budget: int,
        sampler_step: int = 0,
    ) -> dict[str,Tensor]:
        from .training_optimizer_ownership_v1 import (
            qualify_rss_vnext_optimizer_ownership_v1,
        )
        actual=qualify_rss_vnext_optimizer_ownership_v1(
            self.model,self.optimizer,
        )
        if actual.binding_sha256 != self.ownership.binding_sha256:
            raise RuntimeError("RSS_VNEXT_TESSA_TRAIN_OPTIMIZER_OWNERSHIP_DRIFT")
        self.optimizer.zero_grad(set_to_none=True)
        parts=self.objective(
            mesh,teacher,source_face_budget=source_face_budget,
            sampler_step=sampler_step,
        )
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.weights.clip_gradient_norm,error_if_nonfinite=True,
        )
        self.optimizer.step()
        return parts


__all__ = [
    "SCHEMA","PHASE","SeededTESSALossWeightsV1",
    "SeededTESSASourceTeacherV1","RSSvNextSeededTESSATrainerV1",
]
