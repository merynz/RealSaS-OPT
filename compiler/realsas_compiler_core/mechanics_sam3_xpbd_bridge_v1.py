from __future__ import annotations

"""SAM 3 observed-face evidence -> *explicitly approved* XPBD surface proposal.

This bridge consumes the exact Knight SAM 3 V4 experimental report contract.
It does not identify cloth materials, infer attachments from a name, or turn a
segmentation into autonomous dynamics. A SOURCE-MESH to CANONICAL-MESH face
reindex and an authored anchor-joint request are mandatory. Missing mapping,
ambiguous visibility, excessive cut boundaries or unsupported anchors ABSTAIN.
The existing canonical XPBD surface solver and its validation stay authoritative.

The caller owns proof that the face reindex was independently established from
the original source mesh. A mapping's presence/hash cannot prove correctness.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .hashing import content_sha256
from .mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1,
    XPBDSurfaceDomainV1,
    XPBDSurfacePolicyV1,
    prepare_declared_xpbd_surface_domain_v1,
)
from .types import QualificationError


Face3 = tuple[str, str, str]

V4_SCHEMA = "BUSTED.KnightSAM3.MultiviewFaceEvidence.v4"
SOURCE_CAMERA = "FROZEN_8_OBSERVED_KNIGHT_VIEWS_NOT_MODEL_CONSTRUCTION"
MATERIAL_PROFILES_V1: dict[str, tuple[float, float, float, float, float]] = {
    # stretch, bend, area, follow (normalized XPBD compliances), damping / s
    # ARTIST STARTING POINTS, not recovered material measurements.
    "LIGHT_DRAPE": (1e-7, 4e-4, 1e-6, 3e-2, 0.8),
    "HEAVY_DRAPE": (3e-8, 6e-5, 3e-7, 1.5e-2, 2.0),
    "STRUCTURED_FABRIC": (1e-9, 8e-6, 1e-7, 8e-3, 2.8),
}


@dataclass(frozen=True)
class SAM3CanonicalFaceBindingV1:
    source_mesh_sha256: str
    canonical_mesh_lineage_hash: str
    # Original source-face index -> exact canonical triangle, not a new mesh.
    indexed_faces: tuple[tuple[int, Face3], ...]
    mapping_authority: str = "SOURCE_NATIVE_EXPLICIT_FACE_REINDEX"


@dataclass(frozen=True)
class SAM3XPBDAuthoringRequestV1:
    concept: str
    # ORIGINAL_FRONT_BACK only uses V0/V4; ALL_EIGHT_OBSERVED uses source
    # cameras from the research report, NEVER generated product views.
    observation_scope: str
    anchor_joint_ids: tuple[str, ...]
    material_profile: str
    # Explicit user/editor calibration. Normalized mesh units are not meters.
    scene_units_per_meter: float
    user_approved_surface_dynamics: bool = False
    min_anchor_skin_weight: float = 0.2
    max_joint_distance_edges: float = 4.0
    pins_per_anchor: int = 2
    max_pinned_fraction: float = 0.55


@dataclass(frozen=True)
class SAM3XPBDSurfaceActivationV1:
    domain: XPBDSurfaceDomainV1
    policy: XPBDSurfacePolicyV1
    audit: dict[str, Any]


def _reject(code: str) -> None:
    raise QualificationError("SAM3_XPBD_" + code)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value.lower()
    )


def _source_indices(value: object, key: str) -> set[int]:
    if (not isinstance(value, list)
        or any(type(x) is not int or x < 0 for x in value)
        or len(value) != len(set(value))):
        _reject("SOURCE_FACE_IDS_INVALID:" + key)
    return set(value)


def _observed_selection(
    report: Mapping[str, Any], *, concept: str, observation_scope: str,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if (not isinstance(report, Mapping)
        or report.get("schema") != V4_SCHEMA
        or report.get("status") != "REAL_GPU_SAM3_8VIEW_FACE_FUSION_DONE"
        or report.get("camera_source") != SOURCE_CAMERA
        or report.get("source_camera_count") != 8
        or report.get("compiler_or_training_mutation") is not False
        or not _sha256(report.get("source_mesh_sha256"))):
        _reject("UNQUALIFIED_OBSERVED_SOURCE_REPORT")
    if observation_scope not in ("ORIGINAL_FRONT_BACK", "ALL_EIGHT_OBSERVED"):
        _reject("UNSUPPORTED_OBSERVATION_SCOPE")
    concepts = report.get("per_concept")
    if not isinstance(concepts, Mapping) or concept not in concepts:
        _reject("UNKNOWN_CONCEPT")
    item = concepts[concept]
    if not isinstance(item, Mapping):
        _reject("CONCEPT_INVALID")
    key = "only_original_front_back" if observation_scope == "ORIGINAL_FRONT_BACK" else "all_eight_views"
    section = item.get(key)
    per_view = item.get("per_view")
    if not isinstance(section, Mapping) or not isinstance(per_view, Mapping):
        _reject("OBSERVATION_MISSING")
    confirmed = _source_indices(section.get("confirmed_2view_faces"), "confirmed")
    contradicted = _source_indices(section.get("contradicted_face_ids"), "contradicted")
    observed_ids = section.get("independent_views_with_masks")
    allowed = {0, 4} if key == "only_original_front_back" else set(range(8))
    if (not isinstance(observed_ids, list)
        or len(observed_ids) < 2
        or any(type(i) is not int or i not in allowed for i in observed_ids)
        or len(observed_ids) != len(set(observed_ids))):
        _reject("INSUFFICIENT_INDEPENDENT_SOURCE_VIEWS")
    # Independently rebuild 2-view confirmation from per-view source masks:
    # the "confirmed" summary itself is not an authority to invent support.
    hits: dict[int, int] = {}
    actual_observed: set[int] = set()
    for view_id in sorted(allowed):
        row = per_view.get(str(view_id))
        if not isinstance(row, Mapping) or row.get("status") != "MASK_SELECTED":
            continue
        actual_observed.add(view_id)
        positive = _source_indices(row.get("positive_faces"), "view_positive")
        for f in positive:
            hits[f] = hits.get(f, 0) + 1
    if actual_observed != set(observed_ids):
        _reject("SOURCE_VIEW_SELECTION_DRIFT")
    recomputed = {f for f, count in hits.items() if count >= 2}
    if recomputed != confirmed:
        _reject("FACE_VOTE_SUMMARY_DRIFT")
    safe = confirmed - contradicted
    if not safe:
        _reject("NO_UNCONTRADICTED_SURFACE")
    return tuple(sorted(safe)), {
        "reported_confirmed_faces": len(confirmed),
        "contradicted_confirmed_faces": len(confirmed & contradicted),
        "safe_source_face_count": len(safe),
        "observed_independent_view_ids": sorted(actual_observed),
    }


def _mapped_faces(
    mesh: CanonicalMeshIR, binding: SAM3CanonicalFaceBindingV1,
    report_sha: str, source_face_ids: tuple[int, ...],
) -> tuple[Face3, ...]:
    if (not isinstance(binding, SAM3CanonicalFaceBindingV1)
        or binding.mapping_authority != "SOURCE_NATIVE_EXPLICIT_FACE_REINDEX"
        or binding.source_mesh_sha256 != report_sha
        or binding.canonical_mesh_lineage_hash != mesh.mesh_lineage_hash
        or not isinstance(binding.indexed_faces, tuple)):
        _reject("SOURCE_TO_CANONICAL_BINDING_REQUIRED")
    source_set = set(source_face_ids)
    seen_source: set[int] = set()
    seen_canonical: set[Face3] = set()
    mapping: dict[int, Face3] = {}
    legal = set(mesh.faces)
    for row in binding.indexed_faces:
        if (not isinstance(row, tuple) or len(row) != 2
            or type(row[0]) is not int or row[0] < 0
            or not isinstance(row[1], tuple) or len(row[1]) != 3):
            _reject("REINDEX_FORMAT_INVALID")
        source_index, face = row
        if (source_index in seen_source or face in seen_canonical
            or face not in legal):
            _reject("REINDEX_DUPLICATE_OR_FOREIGN_FACE")
        seen_source.add(source_index)
        seen_canonical.add(face)
        mapping[source_index] = face
    if not source_set <= seen_source:
        _reject("REINDEX_MISSING_CONFIRMED_FACES")
    return tuple(mapping[i] for i in source_face_ids)


def _pin_vertices(
    mesh: CanonicalMeshIR, rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2, faces: tuple[Face3, ...],
    request: SAM3XPBDAuthoringRequestV1,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if (not isinstance(request.anchor_joint_ids, tuple)
        or not request.anchor_joint_ids
        or len(set(request.anchor_joint_ids)) != len(request.anchor_joint_ids)
        or not all(isinstance(j, str) and j for j in request.anchor_joint_ids)):
        _reject("EXPLICIT_ANCHOR_JOINTS_REQUIRED")
    if (type(request.pins_per_anchor) is not int
        or not 1 <= request.pins_per_anchor <= 8):
        _reject("PIN_COUNT_INVALID")
    for x in (
        request.min_anchor_skin_weight, request.max_joint_distance_edges,
        request.max_pinned_fraction,
    ):
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
            _reject("PIN_PARAMETERS_INVALID")
    if (not 0 < request.min_anchor_skin_weight <= 1
        or not 0 < request.max_joint_distance_edges <= 100
        or not 0 < request.max_pinned_fraction < 1):
        _reject("PIN_PARAMETERS_INVALID")
    joint_xyz = {j.canonical_joint_id: np.asarray(j.P, dtype=float) for j in rig.joints}
    if not set(request.anchor_joint_ids) <= joint_xyz.keys():
        _reject("UNKNOWN_ANCHOR_JOINT")
    vertices = {v.canonical_mesh_vertex_id: np.asarray(v.P, dtype=float) for v in mesh.vertices}
    weights = {
        row.canonical_mesh_vertex_id: dict(row.influences) for row in skin.rows
    }
    selected_vertices = {v for f in faces for v in f}
    if not selected_vertices <= vertices.keys() or not selected_vertices <= weights.keys():
        _reject("CANONICAL_VERTEX_OR_SKIN_ADDRESS_DRIFT")
    edge_counts: dict[tuple[str, str], int] = {}
    edge_lengths = []
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((x, y)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_lengths.append(float(np.linalg.norm(vertices[x] - vertices[y])))
    if not edge_lengths or min(edge_lengths) <= 1e-10:
        _reject("DEGENERATE_REST_EDGES")
    rest_edge = float(np.median(edge_lengths))
    # A selection that cuts through the body mesh must keep all its cut
    # vertices fixed in the EXISTING XPBD contract. If this consumes most of
    # the patch, reject rather than pinning an entire "cape" into a rigid sheet.
    selected_set = set(faces)
    cut = set()
    for face in mesh.faces:
        if face not in selected_set:
            cut.update(set(face) & selected_vertices)
    if len(cut) / len(selected_vertices) > request.max_pinned_fraction:
        _reject("OVERCONSTRAINED_CUT_BOUNDARY")
    # Prefer true open-patch boundary vertices for joint attachments;
    # interior vertices are not selected as convenient artificial seams.
    boundary = {
        x for e, count in edge_counts.items() if count == 1 for x in e
    }
    if not boundary:
        _reject("NO_OPEN_ATTACHMENT_BOUNDARY")
    chosen = set(cut)
    selected_by_anchor: dict[str, tuple[str, ...]] = {}
    max_distance = rest_edge * request.max_joint_distance_edges
    for joint_id in request.anchor_joint_ids:
        candidates: list[tuple[float, float, str]] = []
        for vertex_id in boundary:
            weight = float(weights[vertex_id].get(joint_id, 0.0))
            if weight < request.min_anchor_skin_weight:
                continue
            distance = float(np.linalg.norm(vertices[vertex_id] - joint_xyz[joint_id]))
            if distance <= max_distance:
                candidates.append((distance, -weight, vertex_id))
        candidates.sort()
        if not candidates:
            _reject("NO_SKIN_SUPPORTED_ATTACHMENT_PIN")
        # Do not let a declared joint silently attach far-away unrelated
        # cloth. Exactly local anchor candidates are selected.
        row = tuple(x[2] for x in candidates[:request.pins_per_anchor])
        selected_by_anchor[joint_id] = row
        chosen.update(row)
    if len(chosen) < 2 or len(chosen) >= len(selected_vertices):
        _reject("INSUFFICIENT_DYNAMIC_SURFACE")
    if len(chosen) / len(selected_vertices) > request.max_pinned_fraction:
        _reject("OVERCONSTRAINED_ANCHOR_PINS")
    return tuple(sorted(chosen)), {
        "pinned_vertex_count": len(chosen),
        "cut_boundary_pin_count": len(cut),
        "anchor_pin_ids_by_joint": selected_by_anchor,
        "median_rest_edge_length": rest_edge,
        "pinned_fraction": len(chosen) / len(selected_vertices),
    }


def _explicit_material_policy(
    domain: XPBDSurfaceDomainV1,
    request: SAM3XPBDAuthoringRequestV1,
) -> XPBDSurfacePolicyV1:
    if request.material_profile not in MATERIAL_PROFILES_V1:
        _reject("EXPLICIT_MATERIAL_PROFILE_REQUIRED")
    u = request.scene_units_per_meter
    if (isinstance(u, bool) or not isinstance(u, (float, int))
        or not math.isfinite(u) or not 0 < u <= 1000):
        _reject("SCENE_UNIT_CALIBRATION_REQUIRED")
    lengths = [row[2] for row in domain.stretch_edges]
    if not lengths or min(lengths) <= 0:
        _reject("MISSING_REST_LENGTHS")
    # XPBD distance constraint compliances are scaled by length^2 and
    # area constraint compliance by length^4, avoiding one hardcoded
    # "cape stiffness" across different source-mesh coordinate scales.
    # Artist defaults only: NOT measured constitutive material constants.
    scale = float(np.median(lengths)) / (0.1 * float(u))
    if not 0.025 <= scale <= 40.0:
        _reject("UNSUPPORTED_MATERIAL_SCALE")
    stretch, bend, area, follow, damping = MATERIAL_PROFILES_V1[request.material_profile]
    return XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        inverse_mass=1.0,
        stretch_compliance=stretch * scale**2,
        bend_compliance=bend * scale**2,
        area_compliance=area * scale**4,
        follow_compliance=follow * scale**2,
        gravity_xyz=(0.0, -9.81 * float(u), 0.0),
        damping_per_second=damping,
        iterations=24,
        fixed_step_seconds=1.0 / 120.0,
        maximum_substeps=32,
    )


def prepare_sam3_assisted_xpbd_surface_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    report: Mapping[str, Any],
    binding: SAM3CanonicalFaceBindingV1,
    request: SAM3XPBDAuthoringRequestV1,
) -> SAM3XPBDSurfaceActivationV1:
    if (not isinstance(request, SAM3XPBDAuthoringRequestV1)
        or request.user_approved_surface_dynamics is not True):
        _reject("USER_APPROVAL_REQUIRED")
    source_face_ids, proof = _observed_selection(
        report, concept=request.concept, observation_scope=request.observation_scope,
    )
    faces = _mapped_faces(
        mesh, binding, report["source_mesh_sha256"], source_face_ids,
    )
    if len(faces) < 2:
        _reject("INSUFFICIENT_CONNECTED_FACES")
    # Do not silently discard ambiguous/disconnected islands: request an
    # edit to the source mask or surface, rather than inventing topology.
    adjacency: dict[str, set[str]] = {v: set() for f in faces for v in f}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            adjacency[x].add(y)
            adjacency[y].add(x)
    visited = set()
    pending = [faces[0][0]]
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(adjacency[current] - visited)
    if visited != set(adjacency):
        _reject("DISCONNECTED_SEMANTIC_PATCH")
    pins, pin_audit = _pin_vertices(mesh, rig, skin, faces, request)
    domain = prepare_declared_xpbd_surface_domain_v1(
        mesh, rig, skin, mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=faces,
            pinned_vertex_ids=pins,
            activation_authority="EXPLICIT_RUNTIME_POLICY",
        ),
    )
    policy = _explicit_material_policy(domain, request)
    # A preview choice is not a physical measurement: no source material
    # truth, body collision proxy, contact or cloth type is fabricated.
    audit: dict[str, Any] = {
        "schema": "RealSaS.SAM3AssistedXPBDSurfaceAudit.v1",
        "status": "EXPLICIT_USER_APPROVED_XPBD_SURFACE_PREPARED",
        "source_mesh_sha256": binding.source_mesh_sha256,
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "source_observation_scope": request.observation_scope,
        "concept_is_identity_not_material": request.concept,
        "source_face_ids": source_face_ids,
        "sam3_report_sha256": content_sha256(dict(report)),
        "source_to_canonical_reindex_digest": content_sha256(
            {"source_mesh_sha256": binding.source_mesh_sha256,
             "canonical_mesh_lineage_hash": binding.canonical_mesh_lineage_hash,
             "indexed_faces": binding.indexed_faces},
        ),
        "selected_face_count": len(faces),
        "domain_hash": domain.domain_hash,
        "policy_profile": request.material_profile,
        "source_material_parameters_qualified": False,
        "source_face_reindex_verified_independently_by_bridge": False,
        "xpbd_activated": False,
        **proof, **pin_audit,
    }
    return SAM3XPBDSurfaceActivationV1(domain, policy, audit)


__all__ = [
    "SAM3CanonicalFaceBindingV1", "SAM3XPBDAuthoringRequestV1",
    "SAM3XPBDSurfaceActivationV1", "MATERIAL_PROFILES_V1",
    "prepare_sam3_assisted_xpbd_surface_v1",
]
