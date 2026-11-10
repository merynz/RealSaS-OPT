from __future__ import annotations

"""RSS.vNext animation-ready mesh *candidate* bridge.

TESSA's causal tokens are decoded with the existing strict component-safe
topology grammar, not treated as xyz points. The same indexed vertices are
sampled from the live shared H(x), refined continuously, and remain indexed
for the control/skin and posed-deformation losses. Compiler qualification is
a separate mandatory step; this module cannot issue canonical puppet assets.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.iris.vnext.coordinate_contract_v1 import tessa_xyz_to_iris_v1
from models.iris.vnext.scene_core_v1 import IRISVNextSceneCoreV1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import _codec_config, evaluate_decoded_topology_v2, TESSADecodedTopologyCourtV2
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from .source_ray_evidence import RSSvNextSourceRayEvidence
from models.tessa.v1.mechanical_objective_v1 import (
    triangle_deformation_metrics_v1, mechanical_consequence_loss_v1,
)


@dataclass
class RSSvNextStructuralMeshCandidate:
    topology_token_ids: tuple[int, ...]
    source_authority: str
    topology_court: TESSADecodedTopologyCourtV2
    base_vertices_iris: Tensor            # [1,V,3], source-independent decoded proposal
    rest_vertices_iris: Tensor            # [1,V,3], differentiably refined
    faces: Tensor                         # [F,3], exact same index domain as skin
    vertex_component_ids: Tensor          # [V], distinct ownership, NEVER global-welded
    face_component_ids: Tensor            # [F]
    local_features: Tensor                # [1,V,H], queried from shared carrier
    normals: Tensor                       # [1,V,3], differentiable
    geometry: dict[str, Tensor]           # includes UDF/sign *proposals*
    semantic_logits: Tensor               # [1,V,R+1], UNKNOWN last
    presentation_role_logits: Tensor      # [1,V,K], StdGEN-style visible material strata
    presentation_role_schema_id: str
    affordance_logits: Tensor             # [1,V,A+1]
    graphics_logits: Tensor               # [1,V,G]
    static_quality_logits: Tensor         # [1,V,4]
    chart_identity_status: str = "NOT_DECODED_FROM_V1_GRAMMAR"
    signed_solid_authority: str = "UNQUALIFIED"
    canonical_mesh_authority: bool = False
    # Original view rays can be queried at these exact candidate vertices,
    # but have no depth/visibility or point-texture ownership authority.
    source_ray_evidence: RSSvNextSourceRayEvidence | None = None

    def validate(self) -> None:
        xyz = self.rest_vertices_iris
        if xyz.ndim != 3 or xyz.shape[0] != 1 or xyz.shape[2] != 3:
            raise ValueError("RSS_VNEXT_MESH_VERTEX_SHAPE")
        v = xyz.shape[1]
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or not self.faces.numel():
            raise ValueError("RSS_VNEXT_MESH_FACE_SHAPE")
        if bool((self.faces < 0).any()) or int(self.faces.max()) >= v:
            raise ValueError("RSS_VNEXT_MESH_FACE_RANGE")
        if (
            self.presentation_role_logits.ndim != 3
            or self.presentation_role_logits.shape[:2] != (1, v)
            or not self.presentation_role_schema_id
            or not torch.isfinite(self.presentation_role_logits).all()
        ):
            raise ValueError("RSS_VNEXT_MESH_PRESENTATION_ROLES_NOT_BOUND")
        if self.local_features.shape[:2] != (1, v) or self.normals.shape != (1, v, 3):
            raise ValueError("RSS_VNEXT_MESH_FEATURE_IDENTITY")
        if self.vertex_component_ids.shape != (v,) or self.face_component_ids.shape != (self.faces.shape[0],):
            raise ValueError("RSS_VNEXT_MESH_COMPONENT_IDENTITY")
        if not torch.isfinite(xyz).all():
            raise ValueError("RSS_VNEXT_MESH_NONFINITE")
        if self.canonical_mesh_authority:
            raise ValueError("RSS_VNEXT_MESH_CANONICAL_PREMATURE")
        if self.source_ray_evidence is not None:
            self.source_ray_evidence.validate(batch=1, points=v)


def _normals_indexed(vertices: Tensor, faces: Tensor) -> Tensor:
    tri = vertices[:, faces]  # B,F,3,3
    fn = torch.cross(tri[:, :, 1] - tri[:, :, 0],
                     tri[:, :, 2] - tri[:, :, 0], dim=-1)
    out = torch.zeros_like(vertices)
    for corner in range(3):
        out = out.index_add(1, faces[:, corner], fn)
    return F.normalize(out, dim=-1, eps=1e-8)


class RSSvNextAnimationReadyMesh(nn.Module):
    """Component-safe discrete topology + shared-field continuous vertex path.

    The grammar is non-differentiable: token CE, relation supervision and
    candidate-ranking train face decisions. Gradients for continuous rest
    geometry and pose-deformation propagate through the *decoded* face set.
    """

    def __init__(self, iris: IRISVNextSceneCoreV1, cfg: TESSAStructuralizerConfigV3):
        super().__init__()
        self.iris = iris
        self.cfg = cfg

    def forward(
        self,
        scene_planes: Tensor,
        topology_token_ids: Sequence[int] | Tensor,
        *,
        refine: nn.Module,
        semantics: nn.Module,
        affordances: nn.Module,
        graphics: nn.Module,
        quality: nn.Module,
        udf: nn.Module,
        sign_logit: nn.Module,
        role_head: nn.Module,
        role_schema_id: str,
        source_authority: str = "LEARNED_PREDICTED_TOPOLOGY",
    ) -> RSSvNextStructuralMeshCandidate:
        if source_authority not in (
            "LEARNED_PREDICTED_TOPOLOGY", "TEACHER_FORCED_TRAINING_ONLY",
        ):
            raise ValueError("RSS_VNEXT_MESH_AUTHORITY_INVALID")
        if scene_planes.ndim != 5 or scene_planes.shape[0] != 1:
            raise ValueError("RSS_VNEXT_MESH_SINGLE_ASSET_REQUIRED")
        if isinstance(topology_token_ids, Tensor):
            if topology_token_ids.ndim != 1:
                raise ValueError("RSS_VNEXT_MESH_TOKEN_RANK")
            ids = tuple(int(x) for x in topology_token_ids.detach().cpu().tolist())
        else:
            ids = tuple(map(int, topology_token_ids))
        court = evaluate_decoded_topology_v2(ids, config=self.cfg)
        if not court.grammar_valid:
            raise ValueError("RSS_VNEXT_MESH_GRAMMAR_INVALID:" + str(court.decode_failure))
        if (court.zero_area_face_count or court.nonmanifold_edge_count
                or not court.finite_geometry or not court.within_scale_policy):
            raise ValueError("RSS_VNEXT_MESH_TOPOLOGY_COURT_REJECTED")
        decoded = decode_tessa_asset_sequence_v1(ids, cfg=_codec_config(self.cfg))
        base = torch.as_tensor(
            decoded.vertices_normalized, device=scene_planes.device,
            dtype=scene_planes.dtype,
        )[None]
        base = tessa_xyz_to_iris_v1(base)  # explicit [-0.5,+0.5] -> [-1,+1]
        local = self.iris.query(scene_planes, base)
        h = local["H"]
        # Perturbations are bounded per axis to remain inside admitted cube.
        free_margin = (1 - base.abs()).clamp_min(0)
        learned_delta = 0.03 * torch.tanh(refine(h))
        rest = base + free_margin * (local["extraction_delta"] + learned_delta)
        faces = torch.as_tensor(decoded.faces, device=base.device, dtype=torch.long)
        geometry = {
            **local,
            "unsigned_distance": F.softplus(udf(h)).squeeze(-1),
            "sign_logit": sign_logit(h).squeeze(-1),
        }
        proposal = RSSvNextStructuralMeshCandidate(
            topology_token_ids=ids,
            source_authority=source_authority,
            topology_court=court,
            base_vertices_iris=base,
            rest_vertices_iris=rest,
            faces=faces,
            vertex_component_ids=torch.tensor(decoded.vertex_component_indices, device=base.device, dtype=torch.long),
            face_component_ids=torch.tensor(decoded.face_component_indices, device=base.device, dtype=torch.long),
            local_features=h,
            normals=_normals_indexed(rest, faces),
            geometry=geometry,
            semantic_logits=semantics(h),
            presentation_role_logits=role_head(h),
            presentation_role_schema_id=role_schema_id,
            affordance_logits=affordances(h),
            graphics_logits=graphics(h),
            static_quality_logits=quality(h),
        )
        proposal.validate()
        return proposal


def posed_lbs_from_joint_matrices(
    rest_vertices: Tensor, weights: Tensor, skin_matrices: Tensor
) -> Tensor:
    """Differentiable exact-vertex LBS, no implied solver authority.

    rest [B,V,3], weights [B,V,J], skin_matrices [B,P,J,4,4].
    Matrices must be explicitly qualified teacher-only in training or built
    by the compiled runtime solver. NEVER student inference image input.
    """
    if rest_vertices.ndim != 3 or rest_vertices.shape[-1] != 3:
        raise ValueError("RSS_VNEXT_LBS_VERTEX_RANK")
    b, v, _ = rest_vertices.shape
    if weights.ndim != 3 or weights.shape[:2] != (b, v):
        raise ValueError("RSS_VNEXT_LBS_WEIGHT_SHAPE")
    if skin_matrices.ndim != 5 or skin_matrices.shape[:2] != (b, skin_matrices.shape[1]) or skin_matrices.shape[2:] != (weights.shape[-1], 4, 4):
        raise ValueError("RSS_VNEXT_LBS_MATRIX_SHAPE")
    if skin_matrices.shape[0] != b:
        raise ValueError("RSS_VNEXT_LBS_BATCH")
    if not (torch.isfinite(rest_vertices).all() and torch.isfinite(weights).all() and torch.isfinite(skin_matrices).all()):
        raise ValueError("RSS_VNEXT_LBS_NUMERIC_INVALID")
    if torch.any(weights < -1e-7) or torch.any((weights.sum(dim=-1) - 1).abs() > 1e-4):
        raise ValueError("RSS_VNEXT_LBS_WEIGHT_SIMPLEX_INVALID")
    ones = torch.ones((b, v, 1), device=rest_vertices.device, dtype=rest_vertices.dtype)
    homogeneous = torch.cat((rest_vertices, ones), dim=-1)
    # [B,P,J,4,4] x [B,V,4] -> [B,P,V,J,4]
    moved = torch.einsum("bpjkc,bvc->bpvjk", skin_matrices, homogeneous)[..., :3]
    return torch.einsum("bvj,bpvjk->bpvk", weights, moved)



def _area_weighted_surface_witnesses(
    vertices: Tensor,
    faces: Tensor,
    *,
    face_budget: int = 384,
) -> Tensor:
    """Deterministic area-aware surface quadrature, same indexed faces.

    Sampling locations are discrete (source face selection not differentiable),
    but barycentric point XYZ and resulting reconstruction losses ARE
    differentiable with respect to the continuous rest/posed vertices.
    This is a training surrogate, not hard collision/visibility proof.
    """
    if vertices.ndim != 3 or vertices.shape[0] != 1 or vertices.shape[2] != 3:
        raise ValueError("RSS_VNEXT_SURFACE_WITNESS_VERTEX_SHAPE")
    if faces.ndim != 2 or faces.shape[-1] != 3 or not faces.numel():
        raise ValueError("RSS_VNEXT_SURFACE_WITNESS_FACE_SHAPE")
    if face_budget < 2 or face_budget > 4096:
        raise ValueError("RSS_VNEXT_SURFACE_WITNESS_BUDGET")
    tri = vertices[0, faces]
    area = 0.5 * torch.cross(
        tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
        dim=-1,
    ).norm(dim=-1)
    if bool((area < 1e-10).any()) or not bool(torch.isfinite(area).all()):
        raise ValueError("RSS_VNEXT_SURFACE_WITNESS_DEGENERATE_FACE")
    # Uniform in cumulative surface area, independent of original face count.
    with torch.no_grad():
        cdf = area.detach().double().cumsum(0)
        fractions = (torch.arange(
            min(face_budget, faces.shape[0]), device=area.device,
            dtype=torch.float64,
        ) + .5) / min(face_budget, faces.shape[0])
        selected = torch.searchsorted(
            cdf, fractions * cdf[-1], right=False,
        ).clamp_max(len(faces) - 1)
    tris = tri[selected]
    barycentric = torch.tensor(
        [[1/3, 1/3, 1/3], [.6, .2, .2],
         [.2, .6, .2], [.2, .2, .6]],
        device=vertices.device, dtype=vertices.dtype,
    )
    return torch.einsum("qjc,sj->qsc", tris, barycentric).reshape(-1, 3)


def _two_way_surface_chamfer(predicted: Tensor, reference: Tensor) -> Tensor:
    if (reference.ndim != 2 or reference.shape[1] != 3
            or not reference.numel() or not torch.isfinite(reference).all()):
        raise ValueError("RSS_VNEXT_POSE_REFERENCE_SURFACE_INVALID")
    distance = torch.cdist(predicted[None].float(), reference[None].float())[0]
    return distance.amin(-1).mean() + distance.amin(-2).mean()


def source_pose_surface_consequence_loss(
    mesh: RSSvNextStructuralMeshCandidate,
    weights: Tensor,
    source_skin_matrices: Tensor,
    source_rest_samples: Tensor,
    source_pose_samples: Tensor,
    *,
    valid_pose_mask: Tensor,
    max_sampled_faces: int = 384,
) -> dict[str, Tensor]:
    """Differentiable source-conditioned motion geometry loss.

    Teacher-side only. Source rest and posed samples + motion matrices must be
    qualified to the same original asset/frame/pose. This function has no
    authority to approve source truth; only accepts the arrays supplied by a
    separately verifying source-preparation court.

    Learns *rest topology geometry and skin together*: if an edge across a
    bend makes the posed triangle surface fail to match source motion, this
    loss penalizes it. Gradients reach vertex XYZ and predicted skin, while
    discrete face choices must additionally use motion-aware policy learning.
    """
    mesh.validate()
    if source_rest_samples.ndim != 2 or source_rest_samples.shape[1] != 3:
        raise ValueError("RSS_VNEXT_POSE_REST_REFERENCE_SHAPE")
    if (source_pose_samples.ndim != 3 or source_pose_samples.shape[2] != 3
            or source_pose_samples.shape[0] != source_skin_matrices.shape[1]):
        raise ValueError("RSS_VNEXT_POSE_REFERENCE_SEQUENCE_SHAPE")
    if (valid_pose_mask.shape != (source_skin_matrices.shape[1],)
            or valid_pose_mask.dtype != torch.bool or not bool(valid_pose_mask.any())):
        raise ValueError("RSS_VNEXT_POSE_REFERENCE_MASK_INVALID")
    posed = posed_lbs_from_joint_matrices(
        mesh.rest_vertices_iris, weights, source_skin_matrices,
    )
    static_points = _area_weighted_surface_witnesses(
        mesh.rest_vertices_iris, mesh.faces, face_budget=max_sampled_faces,
    )
    static_error = _two_way_surface_chamfer(static_points, source_rest_samples)
    pose_losses = []
    for pose_idx in range(posed.shape[1]):
        if not bool(valid_pose_mask[pose_idx]):
            continue
        points = _area_weighted_surface_witnesses(
            posed[:, pose_idx], mesh.faces, face_budget=max_sampled_faces,
        )
        pose_losses.append(_two_way_surface_chamfer(
            points, source_pose_samples[pose_idx],
        ))
    dynamic_error = torch.stack(pose_losses).mean()
    return {
        "static_surface": static_error,
        "posed_surface": dynamic_error,
        "total": static_error + dynamic_error,
        "qualified_pose_count": valid_pose_mask.sum(),
    }


def deformation_bridge_loss(
    mesh: RSSvNextStructuralMeshCandidate,
    weights: Tensor,
    skin_matrices: Tensor,
    *,
    pose_valid_mask: Tensor,
    refined_vertices: Tensor | None = None,
    source_reference_jacobian: Tensor | None = None,
    source_reference_normal: Tensor | None = None,
) -> dict[str, Tensor]:
    """G3-derived differentiable training surrogate over actual mesh faces.

    Each pose may be teacher-supervised only when a qualified source matrix and
    correspondence exists. The hard G3/G3B/Stage35 courts remain mandatory.
    This loss does not differentiate discrete face selection.
    """
    mesh.validate()
    rest = mesh.rest_vertices_iris if refined_vertices is None else refined_vertices
    if rest.shape != mesh.rest_vertices_iris.shape:
        raise ValueError("RSS_VNEXT_DEFORM_REFINED_VERTEX_IDENTITY")
    if not bool(torch.isfinite(rest).all()):
        raise ValueError("RSS_VNEXT_DEFORM_REFINED_VERTEX_NONFINITE")
    if weights.shape != (1, rest.shape[1], skin_matrices.shape[2]):
        raise ValueError("RSS_VNEXT_DEFORM_WEIGHT_MESH_IDENTITY")
    if pose_valid_mask.shape != skin_matrices.shape[:2] or pose_valid_mask.dtype != torch.bool:
        raise ValueError("RSS_VNEXT_DEFORM_POSE_MASK")
    if not bool(pose_valid_mask.any()):
        raise ValueError("RSS_VNEXT_DEFORM_NO_QUALIFIED_POSE")
    posed = posed_lbs_from_joint_matrices(rest, weights, skin_matrices)
    terms = []
    for p in range(skin_matrices.shape[1]):
        if not bool(pose_valid_mask[0, p]):
            continue
        metrics = triangle_deformation_metrics_v1(rest, posed[:, p], mesh.faces)
        ref_j = None if source_reference_jacobian is None else source_reference_jacobian[:, p]
        ref_n = None if source_reference_normal is None else source_reference_normal[:, p]
        terms.append(mechanical_consequence_loss_v1(
            metrics, reference_jacobian=ref_j, reference_normal=ref_n,
        )["total"])
    return {
        "loss": torch.stack(terms).mean(),
        "posed_vertices": posed,
        "pose_valid_mask": pose_valid_mask,
        "authority": "TRAINING_SURROGATE_ONLY",
    }


__all__ = [
    "RSSvNextStructuralMeshCandidate", "RSSvNextAnimationReadyMesh",
    "posed_lbs_from_joint_matrices", "deformation_bridge_loss",
    "source_pose_surface_consequence_loss",
]
