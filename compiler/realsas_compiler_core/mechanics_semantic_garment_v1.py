from __future__ import annotations

"""SAM3-free semantic garments on the existing canonical TESSA/AXIS/MIRA mesh.

Semantic evidence is a hypothesis from the SAME shared H(x), never a physical
material measurement. Explicit flexible-garment intent and joint anchors are
mandatory. The current surface XPBD is the sole solver; no topology rebind,
separate cloth mesh, or invented attachment is performed.
"""

from dataclasses import dataclass
from typing import Mapping
import math
import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1, XPBDSurfaceDomainV1,
    XPBDSurfacePolicyV1, prepare_declared_xpbd_surface_domain_v1,
)
from .types import QualificationError


FOUR_WAY_ROLES = (
    "secondary_form", "focal_detail", "primary_form", "outer_covering",
)
ROLE_SCHEMA = "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1"
# These are artist-specified *starting points*, not measured material truth.
MATERIAL_STARTS = {
    "LIGHT_DRAPE": (1e-7, 4e-4, 1e-6, 3e-2, 0.8),
    "HEAVY_DRAPE": (3e-8, 6e-5, 3e-7, 1.5e-2, 2.0),
    "STRUCTURED_FABRIC": (1e-9, 8e-6, 1e-7, 8e-3, 2.8),
}


def _reject(reason: str) -> None:
    raise QualificationError("GARMENT_XPBD_" + reason)


@dataclass(frozen=True)
class SemanticGarmentEvidenceV1:
    mesh_lineage_hash: str
    scene_sha256: str
    semantic_role_schema_id: str
    role_probabilities_by_face: tuple[tuple[float, ...], ...]
    # IRIS produced candidate semantics, not source-native cloth truth.
    authority: str = "SHARED_H_CANONICAL_FACE_ROLE_CANDIDATE"


@dataclass(frozen=True)
class GarmentDynamicsRequestV1:
    material_profile: str
    anchor_joint_ids: tuple[str, ...]
    scene_units_per_meter: float
    explicitly_approved_flexible_surface: bool = False
    semantic_role: str = "outer_covering"
    minimum_role_probability: float = 0.75
    minimum_role_margin: float = 0.15
    minimum_anchor_weight: float = 0.2
    maximum_anchor_distance_edges: float = 4.0
    maximum_pinned_fraction: float = 0.5
    pins_per_joint: int = 2


@dataclass(frozen=True)
class GarmentXPBDProposalV1:
    domain: XPBDSurfaceDomainV1
    policy: XPBDSurfacePolicyV1
    audit: dict


def garment_evidence_from_uv_atlas_v1(
    mesh: CanonicalMeshIR, atlas_cas: bytes,
) -> SemanticGarmentEvidenceV1:
    """Read the REAL typed Graphics/UV artifact, not a hand-labeled mask."""
    from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
        reopen_rss_vnext_uv_texture_atlas_v1,
    )
    arrays, manifest = reopen_rss_vnext_uv_texture_atlas_v1(atlas_cas)
    if (manifest["canonical_mesh_lineage_hash"] != mesh.mesh_lineage_hash
            or manifest.get("semantic_role_schema_id") != ROLE_SCHEMA
            or manifest.get("semantic_role_count") != len(FOUR_WAY_ROLES)
            or manifest.get("semantic_role_sublayer_is_model_candidate_not_source_truth") is not True
            or "face_presentation_role_probabilities_f32" not in arrays):
        _reject("ATLAS_MESH_OR_SEMANTIC_ABI")
    digest = manifest.get("neural_graphics_scene_content_sha256")
    if (type(digest) is not str or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        _reject("ATLAS_SHARED_H_SCENE_REQUIRED")
    posterior = arrays["face_presentation_role_probabilities_f32"]
    if posterior.shape != (len(mesh.faces), len(FOUR_WAY_ROLES)):
        _reject("ATLAS_FACE_ADDRESS_DRIFT")
    return SemanticGarmentEvidenceV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        scene_sha256=digest,
        semantic_role_schema_id=ROLE_SCHEMA,
        role_probabilities_by_face=tuple(
            tuple(float(x) for x in row) for row in posterior
        ),
    )


def prepare_semantic_garment_xpbd_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    evidence: SemanticGarmentEvidenceV1,
    request: GarmentDynamicsRequestV1,
) -> GarmentXPBDProposalV1:
    """Conservatively turn an explicitly selected flexible outer layer into XPBD.

    A semantic classification alone NEVER activates simulation: this operation
    requires explicitly_approved_flexible_surface and approved anchor joints.
    Non-admissible, cut-through, disconnected or overpinned garments ABSTAIN.
    """
    if (not isinstance(evidence, SemanticGarmentEvidenceV1)
            or evidence.authority != "SHARED_H_CANONICAL_FACE_ROLE_CANDIDATE"
            or evidence.mesh_lineage_hash != mesh.mesh_lineage_hash
            or type(evidence.scene_sha256) is not str
            or len(evidence.scene_sha256) != 64
            or any(c not in "0123456789abcdef" for c in evidence.scene_sha256)
            or evidence.semantic_role_schema_id != ROLE_SCHEMA):
        _reject("UNQUALIFIED_SEMANTIC_MESH_BINDING")
    if (not isinstance(request, GarmentDynamicsRequestV1)
            or request.explicitly_approved_flexible_surface is not True
            or request.material_profile not in MATERIAL_STARTS
            or request.semantic_role != "outer_covering"
            or not isinstance(request.anchor_joint_ids, tuple)
            or not request.anchor_joint_ids
            or len(set(request.anchor_joint_ids)) != len(request.anchor_joint_ids)
            or any(type(j) is not str or not j for j in request.anchor_joint_ids)
            or type(request.pins_per_joint) is not int
            or not 1 <= request.pins_per_joint <= 8):
        _reject("EXPLICIT_FLEXIBLE_GARMENT_POLICY_REQUIRED")
    numerics = (
        request.scene_units_per_meter, request.minimum_role_probability,
        request.minimum_role_margin, request.minimum_anchor_weight,
        request.maximum_anchor_distance_edges, request.maximum_pinned_fraction,
    )
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in numerics)
            or not 0 < request.scene_units_per_meter <= 1000
            or not 0.5 <= request.minimum_role_probability <= 1
            or not 0 < request.minimum_role_margin <= 0.5
            or not 0 < request.minimum_anchor_weight <= 1
            or not 0 < request.maximum_anchor_distance_edges <= 100
            or not 0 < request.maximum_pinned_fraction <= 0.5):
        _reject("FLEXIBLE_GARMENT_POLICY_NUMERIC")

    probs = np.asarray(evidence.role_probabilities_by_face, dtype=float)
    if (probs.shape != (len(mesh.faces), len(FOUR_WAY_ROLES))
            or not np.isfinite(probs).all()
            or np.any(probs < 0) or np.any(probs > 1)
            or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        _reject("FACE_SEMANTIC_POSTERIOR_INVALID")
    order = np.argsort(-probs, axis=1, kind="stable")
    best = probs[np.arange(len(mesh.faces)), order[:, 0]]
    second = probs[np.arange(len(mesh.faces)), order[:, 1]]
    selected_ids = np.flatnonzero(
        (order[:, 0] == FOUR_WAY_ROLES.index(request.semantic_role))
        & (best >= request.minimum_role_probability)
        & (best - second >= request.minimum_role_margin)
    )
    if len(selected_ids) < 2:
        _reject("NO_CONFIDENT_GARMENT_PATCH")
    faces = tuple(mesh.faces[int(i)] for i in selected_ids)
    verts = {v.canonical_mesh_vertex_id: np.asarray(v.P, dtype=float)
             for v in mesh.vertices}
    selected_vertices = {v for face in faces for v in face}
    if len({v.component_id for v in mesh.vertices
            if v.canonical_mesh_vertex_id in selected_vertices}) != 1:
        _reject("CROSS_COMPONENT_GARMENT_PATCH")
    edge_count: dict[tuple[str, str], int] = {}
    lengths = []
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((x, y)))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            lengths.append(float(np.linalg.norm(verts[x] - verts[y])))
    if not lengths or min(lengths) <= 1e-10:
        _reject("INVALID_GARMENT_REST_EDGES")
    boundary = {v for edge, n in edge_count.items() if n == 1 for v in edge}
    if len(boundary) < 2:
        _reject("NO_FREE_GARMENT_BOUNDARY")
    other_faces = set(mesh.faces) - set(faces)
    cut_vertices = {
        v for face in other_faces for v in face if v in selected_vertices
    }
    if len(cut_vertices) / len(selected_vertices) > request.maximum_pinned_fraction:
        _reject("OVERCONSTRAINED_CUT_BOUNDARY")
    weights = {r.canonical_mesh_vertex_id: dict(r.influences) for r in skin.rows}
    joints = {j.canonical_joint_id: np.asarray(j.P, dtype=float)
              for j in rig.joints}
    if (not selected_vertices <= weights.keys()
            or not set(request.anchor_joint_ids) <= joints.keys()):
        _reject("MISSING_CANONICAL_SKIN_OR_JOINT")
    pins = set(cut_vertices)
    anchors = {}
    max_dist = float(np.median(lengths)) * request.maximum_anchor_distance_edges
    for joint_id in request.anchor_joint_ids:
        choices = []
        for v in boundary:
            influence = float(weights[v].get(joint_id, 0.0))
            distance = float(np.linalg.norm(verts[v] - joints[joint_id]))
            if influence >= request.minimum_anchor_weight and distance <= max_dist:
                choices.append((distance, -influence, v))
        choices.sort()
        if not choices:
            _reject("NO_SUPPORTED_GARMENT_PIN")
        chosen = tuple(row[2] for row in choices[:request.pins_per_joint])
        pins.update(chosen)
        anchors[joint_id] = chosen
    if (len(pins) < 2 or len(pins) >= len(selected_vertices)
            or len(pins) / len(selected_vertices) > request.maximum_pinned_fraction):
        _reject("OVERPINNED_OR_RIGID_GARMENT")
    # The real XPBD court rechecks component connectivity, all cut seams,
    # full mesh-rig-skin lineage, and topology. No second physics system.
    domain = prepare_declared_xpbd_surface_domain_v1(
        mesh, rig, skin, mechanics, DeclaredXPBDSurfaceRequestV1(
            selected_faces=faces, pinned_vertex_ids=tuple(sorted(pins)),
        ),
    )
    scale = float(np.median(lengths)) / (0.1 * request.scene_units_per_meter)
    if not 0.025 <= scale <= 40:
        _reject("UNSUPPORTED_SCENE_SCALE")
    stretch, bend, area, follow, damping = MATERIAL_STARTS[request.material_profile]
    policy = XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        stretch_compliance=stretch * scale**2,
        bend_compliance=bend * scale**2,
        area_compliance=area * scale**4,
        follow_compliance=follow * scale**2,
        gravity_xyz=(0.0, -9.81 * request.scene_units_per_meter, 0.0),
        damping_per_second=damping,
        fixed_step_seconds=1.0 / 120.0,
        iterations=24,
    )
    return GarmentXPBDProposalV1(
        domain=domain, policy=policy,
        audit={
            "status": "CANDIDATE_WITH_EXPLICIT_DYNAMICS_POLICY",
            "source_physics_truth_claimed": False,
            "sam3_used": False,
            "material_source": "DECLARED_STARTING_POLICY_NOT_LEARNED",
            "source_geometry_role_is_unqualified_neural_candidate": True,
            "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
            "shared_scene_sha256": evidence.scene_sha256,
            "selected_face_indices": tuple(int(x) for x in selected_ids),
            "pinned_vertex_ids": tuple(sorted(pins)),
            "dynamic_vertex_count": len(domain.dynamic_vertex_ids),
            "anchor_joint_pins": anchors,
            "activation_automatic": False,
        },
    )


__all__ = [
    "SemanticGarmentEvidenceV1", "GarmentDynamicsRequestV1",
    "GarmentXPBDProposalV1", "garment_evidence_from_uv_atlas_v1",
    "prepare_semantic_garment_xpbd_v1",
]
