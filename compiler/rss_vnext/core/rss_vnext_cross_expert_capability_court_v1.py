from __future__ import annotations

"""Typed expert ownership on the one RSS.vNext neural carrier.

StdGEN informs semantic presentation roles, AniGen informs geometry/mechanics
coherence, SkinTokens informs joint/weight coupling. Learned roles, adjacency,
RGB logits and attachment affinities are NOT executable capabilities.
The full model evidence remains immutable in CAS; this court references it.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
)
from compiler.realsas_compiler_core.rss_vnext_face_relation_schedule_v1 import (
    RSSvNextSparseFaceRelationScheduleV1,
)
from compiler.realsas_compiler_core.types import QualificationError

SCHEMA = "RealSaS.RSSvNext.CrossExpertCapabilityCourt.v1"
_CAPABILITIES = (
    ("INDEXED_SURFACE", ("Geometry", "Topology/TESSA"),
     ("SOURCE_SURFACE_FIDELITY", "STATIC_MESH_QUALIFICATION")),
    ("SEMANTIC_REGION", ("Semantic", "Geometry", "Topology/TESSA"),
     ("SOURCE_CORRESPONDENCE", "PERSISTENT_REGION_OWNERSHIP")),
    ("GRAPHICS_2D", ("Graphics", "Semantic", "Geometry"),
     ("SURFACE_APPEARANCE_OWNERSHIP", "FIRST_HIT_VISIBILITY", "POSE_COMPOSITING")),
    ("RIG_CONTROL", ("Mechanics/AXIS", "Geometry", "Topology/TESSA"),
     ("QUALIFIED_RIG", "CONTROL_INTERVENTION")),
    ("SKIN_DEFORMATION", ("Mechanics/MIRA", "Geometry", "Topology/TESSA"),
     ("DISCRETE_SKIN_REPLAY", "POSE_DEFORMATION_QUALITY", "CANONICAL_RIG")),
    ("DIRECTED_ATTACHMENT", ("Semantic", "Mechanics/AXIS", "Graphics"),
     ("SOURCE_CONTACT", "PARENT_ANCHOR", "EXECUTABLE_CONTROL")),
    ("INDEPENDENT_DRAWABLE", ("Semantic", "Graphics", "Topology/TESSA"),
     ("SOURCE_SEPARABILITY", "EDIT_CONSEQUENCE", "COMPOSITING")),
    ("ANIMATION_READY_MESH", ("Topology/TESSA", "Geometry", "Mechanics/AXIS", "Mechanics/MIRA"),
     ("STATIC_SOURCE_QUALITY", "MULTI_POSE_BEHAVIOR", "RUNTIME_RIG_SKIN")),
)


@dataclass(frozen=True)
class RSSvNextTypedCapabilityClaimV1:
    capability: str
    producers: tuple[str, ...]
    state: str
    missing_witnesses: tuple[str, ...]
    executable_authority: bool = False
    source_truth_authority: bool = False


@dataclass(frozen=True)
class RSSvNextCrossExpertCapabilityCourtV1:
    inference_evidence_hash: str
    neural_scene_carrier_sha256: str
    unified_model_config_sha256: str
    candidate_mesh_sha256: str | None
    computed_surface_sha256: str
    sparse_relation_sha256: str | None
    state: str
    face_count: int
    min_refined_base_area_ratio: float | None
    min_refined_base_normal_alignment: float | None
    max_semantic_unknown_probability: float | None
    source_ray_projection_valid_count: int
    directed_attachment_pair_count: int
    claims: tuple[RSSvNextTypedCapabilityClaimV1, ...]
    schema: str = SCHEMA
    canonical_mesh_authority: bool = False
    independent_drawable_authority: bool = False
    source_visibility_authority: bool = False
    executable_attachment_authority: bool = False
    animation_ready_authority: bool = False


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_CROSS_EXPERT_" + code)


def compile_rss_vnext_cross_expert_capability_court_v1(
    *, evidence: Any, surface: RSSvNextComputedCandidateSurfaceV1,
    model_config: Any, relation: RSSvNextSparseFaceRelationScheduleV1 | None = None,
) -> RSSvNextCrossExpertCapabilityCourtV1:
    from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
        RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
        candidate_face_region_readout_v1,
    )
    from models.rss_vnext.contracts import RSSvNextConfig
    from models.rss_vnext.model_config_archive_v1 import _digest
    if (type(evidence) is not RSSvNextInferenceEvidenceIRV1
        or type(surface) is not RSSvNextComputedCandidateSurfaceV1
        or type(model_config) is not RSSvNextConfig):
        _fail("TYPED_PARENT_REQUIRED")
    model_config.validate()
    validate_inference_evidence_v1(evidence)
    if (surface.source_inference_evidence_hash != evidence.evidence_hash
        or surface.unified_model_config_sha256 != _digest(model_config)
        or any((surface.canonical_mesh_authority, surface.signed_solid_authority,
                surface.source_visibility_authority, surface.independent_editability_authority))):
        _fail("SOURCE_OR_AUTHORITY_DRIFT")
    if (relation is not None and
        (type(relation) is not RSSvNextSparseFaceRelationScheduleV1
         or relation.inference_evidence_hash != evidence.evidence_hash
         or relation.unified_model_config_sha256 != surface.unified_model_config_sha256
         or relation.neural_scene_carrier_sha256 != surface.neural_scene_carrier_sha256
         or relation.candidate_mesh_sha256 != surface.candidate_mesh_sha256
         or any((relation.attachment_authority, relation.independent_drawable_authority,
                 relation.pose_conditioned_presentation_authority)))):
        _fail("SPARSE_RELATION_AUTHORITY_OR_PARENT_DRIFT")
    count = rays = attachments = 0
    area_ratio = normal_alignment = max_unknown = None
    if evidence.mesh_status == "ABSTAIN":
        if (evidence.mesh is not None or surface.state != "ABSTAIN_NO_VALID_TOPOLOGY"
            or surface.faces or surface.candidate_mesh_sha256 is not None
            or relation is not None):
            _fail("ABSTAIN_TOPOLOGY_MUST_NOT_PROMOTE")
        state = "ABSTAIN_TOPOLOGY_EXPERTS_RETAINED"
    elif evidence.mesh_status == "UNQUALIFIED_MESH_CANDIDATE":
        mesh = evidence.mesh
        if (surface.state != "UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE"
            or surface.candidate_mesh_sha256 != content_sha256(mesh)
            or surface.topology_token_sequence_sha256 != content_sha256(mesh["topology_tokens"])):
            _fail("TESSA_MESH_BINDING_DRIFT")
        base, rest = (np.asarray(mesh[k], dtype=np.float64)
                      for k in ("base_xyz", "rest_xyz"))
        face_idx = np.asarray(mesh["faces"], dtype=np.int64)
        vertex_components = np.asarray(mesh["vertex_components"], dtype=np.int64)
        face_components = np.asarray(mesh["face_components"], dtype=np.int64)
        count = len(face_idx)
        if (base.shape != rest.shape or base.ndim != 2 or base.shape[1] != 3
            or not count or count != len(surface.faces)
            or face_components.shape != (count,)):
            _fail("INDEXED_GEOMETRY_DOMAIN_DRIFT")
        if any(np.any(vertex_components[tri] != face_components[i])
               for i, tri in enumerate(face_idx)):
            _fail("FACE_COMPONENT_OWNERSHIP_DRIFT")
        tri_base, tri_rest = base[face_idx], rest[face_idx]
        b = np.cross(tri_base[:, 1]-tri_base[:, 0], tri_base[:, 2]-tri_base[:, 0])
        r = np.cross(tri_rest[:, 1]-tri_rest[:, 0], tri_rest[:, 2]-tri_rest[:, 0])
        bn, rn = np.linalg.norm(b, axis=1), np.linalg.norm(r, axis=1)
        if (not np.isfinite(bn).all() or not np.isfinite(rn).all()
            or np.any(bn <= 1.e-12) or np.any(rn <= 1.e-12)):
            _fail("REFINED_TRIANGLE_DEGENERATE")
        align = (b * r).sum(axis=1) / (bn * rn)
        if np.any(align <= 0):
            _fail("REFINED_TOPOLOGY_FACE_ORIENTATION_INVERTED")
        area_ratio, normal_alignment = float(np.min(rn / bn)), float(np.min(align))
        predicted = candidate_face_region_readout_v1(evidence)
        unknown = 0.
        for i, face in enumerate(surface.faces):
            tri = face_idx[i]
            expected_membership = tuple(predicted[i]["candidate_region_probabilities"]) + (
                predicted[i]["unknown_probability"],)
            if (face.candidate_face_index != i
                or face.candidate_vertex_indices != tuple(map(int, tri))
                or face.candidate_component_index != int(face_components[i])
                or not math.isclose(face.area, float(rn[i]*.5), abs_tol=1.e-8)
                or not np.allclose(face.unit_normal_xyz, r[i]/rn[i], rtol=0, atol=1.e-6)
                or not np.allclose(face.region_probabilities, expected_membership, atol=1.e-6, rtol=0)
                or len(face.presentation_role_probabilities) != len(model_config.semantic_role_names)
                or any((face.source_visibility_qualified, face.source_appearance_texture_qualified,
                        face.independent_drawable_qualified, face.mechanical_anchor_qualified))):
                _fail("FACE_SEMANTIC_TOPOLOGY_ADDRESS_OR_AUTHORITY_DRIFT")
            for j, vi in enumerate(tri):
                if (not np.array_equal(face.graphics_logits_by_corner[j], mesh["graphics_logits"][vi])
                    or not np.array_equal(face.affordance_logits_by_corner[j], mesh["affordance_logits"][vi])
                    or not np.array_equal(face.original_rgba_by_corner[j], mesh["source_rays"]["rgba"][vi])
                    or tuple(face.original_projection_valid_by_corner[j]) != tuple(mesh["source_rays"]["valid"][vi])):
                    _fail("GRAPHICS_MECHANICS_SOURCE_ADDRESS_DRIFT")
                rays += sum(bool(x) for x in face.original_projection_valid_by_corner[j])
            unknown = max(unknown, float(face.region_probabilities[-1]))
        max_unknown = unknown
        if relation is not None:
            for row in relation.face_relations:
                if (not 0 <= row.from_face_index < count
                    or not 0 <= row.to_face_index < count
                    or row.from_face_index == row.to_face_index
                    or any((row.source_drawable_qualified, row.source_separability_qualified,
                            row.source_contact_qualified, row.mechanical_parent_anchor_qualified,
                            row.source_visibility_qualified))):
                    _fail("RELATION_CANNOT_PROMOTE_ATTACHMENT")
                if row.directed_attachment_affinity is not None:
                    attachments += 1
        state = "TYPED_CROSS_EXPERT_PROPOSALS_PRESERVED"
    else:
        _fail("MESH_STATUS_INVALID")
    claims = tuple(RSSvNextTypedCapabilityClaimV1(
        capability=cap, producers=owners,
        state=("TYPED_PROPOSAL_ONLY" if count and cap in
               ("INDEXED_SURFACE", "SEMANTIC_REGION") else "ABSTAIN_MISSING_EXECUTABLE_WITNESS"),
        missing_witnesses=missing,
    ) for cap, owners, missing in _CAPABILITIES)
    return RSSvNextCrossExpertCapabilityCourtV1(
        inference_evidence_hash=evidence.evidence_hash,
        neural_scene_carrier_sha256=surface.neural_scene_carrier_sha256,
        unified_model_config_sha256=surface.unified_model_config_sha256,
        candidate_mesh_sha256=surface.candidate_mesh_sha256,
        computed_surface_sha256=content_sha256(asdict(surface)),
        sparse_relation_sha256=None if relation is None else content_sha256(asdict(relation)),
        state=state, face_count=count,
        min_refined_base_area_ratio=area_ratio,
        min_refined_base_normal_alignment=normal_alignment,
        max_semantic_unknown_probability=max_unknown,
        source_ray_projection_valid_count=rays,
        directed_attachment_pair_count=attachments, claims=claims,
    )


__all__ = ["RSSvNextTypedCapabilityClaimV1", "RSSvNextCrossExpertCapabilityCourtV1",
           "compile_rss_vnext_cross_expert_capability_court_v1"]
