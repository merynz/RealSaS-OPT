from __future__ import annotations

"""Qualification bridge for render/deformation meshes derived from a support substrate
that is intentionally not the canonical scientific MechanicalStateIR surface.

The canonical mechanical S/G/W authority stays untouched.  A render mesh may be
qualified against a different, sealed support substrate only when its exact mesh and
exact qualified skin binding are independently SHA-bound and revalidated here.  This
is the product render-support seam; it is not a surface relabel or historical weight
transfer.
"""

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Any, Mapping

from .hashing import content_sha256
from .mesh.direct_model_skin import DIRECT_MODEL_TRANSFER_METHOD
from .mesh.mwb2_skin import SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD
from .mesh.rigid_attachment_skin import RIGID_ATTACHMENT_TRANSFER_METHOD
from .mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from .types import QualificationError


_SURFACE_SUPPORT_REPLAY_L1_TOL = 1.0e-9


@dataclass(frozen=True)
class ExternalRenderSupportQualificationIR:
    view_index: int
    render_support_surface_hash: str
    mesh_lineage_hash: str
    mesh_skin_lineage_hash: str
    skeleton_lineage_hash: str
    source_skin_authority_lineage_hash: str
    materialization_manifest_sha256: str
    direct_binding_manifest_sha256: str
    transfer_method: str
    qualification_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "RealSaS.ExternalRenderSupportQualificationIR.v1"

    def to_dict(self):
        return asdict(self)


def external_render_support_hash(value: ExternalRenderSupportQualificationIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def _required_evidence_hash(name: str, value: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() in {"latest", "current", "newest"}:
        raise QualificationError(f"EXTERNAL_RENDER_SUPPORT_INVALID_{name}")
    return value


def _validate_detached_mesh(mesh) -> dict[str, tuple[tuple[str, float], ...]]:
    if mesh.mesh_lineage_hash != mesh_lineage_hash(mesh):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_MESH_LINEAGE_HASH_MISMATCH")
    if not str(mesh.surface_binding_hash or "").strip():
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_MISSING_SURFACE_HASH")
    if int(mesh.view_index) not in range(8):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_INVALID_VIEW")
    ids: set[str] = set()
    support_by_vertex: dict[str, tuple[tuple[str, float], ...]] = {}
    for vertex in mesh.vertices:
        vid = str(vertex.canonical_mesh_vertex_id or "")
        if not vid or vid in ids:
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_DUPLICATE_MESH_VERTEX")
        ids.add(vid)
        if len(vertex.P) != 3 or any(not isfinite(float(x)) for x in vertex.P):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_NONFINITE_VERTEX")
        coeffs = tuple((str(sid), float(w)) for sid, w in vertex.support_binding.coefficients)
        if not coeffs or any((not sid) or (not isfinite(w)) or w < 0.0 for sid, w in coeffs):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_INVALID_SUPPORT_COEFFICIENTS")
        if abs(sum(w for _sid, w in coeffs) - 1.0) > 1.0e-8:
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SUPPORT_SIMPLEX_MISMATCH")
        support_by_vertex[vid] = coeffs
    for face in mesh.faces:
        if len(face) < 3 or len(set(face)) != len(face) or any(str(vid) not in ids for vid in face):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_INVALID_FACE")
    for edge in mesh.edges:
        if len(edge) != 2 or edge[0] == edge[1] or any(str(vid) not in ids for vid in edge):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_INVALID_EDGE")
    return support_by_vertex


def _replay_surface_support_row(source_skin_by_surface, coeffs):
    accum: dict[str, float] = {}
    for sid, coeff in coeffs:
        source_row = source_skin_by_surface.get(str(sid))
        if source_row is None:
            raise QualificationError(f"EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_SOURCE_ROW_MISSING:{sid}")
        c = float(coeff)
        for jid, weight in source_row.influences:
            accum[str(jid)] = accum.get(str(jid), 0.0) + c * float(weight)
    total = float(sum(accum.values()))
    if total <= 0.0 or not isfinite(total):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_ZERO_OR_NONFINITE_MASS")
    return {jid: value / total for jid, value in accum.items() if value > 0.0}


def _validate_detached_mesh_skin(mesh_skin, mesh, mechanical) -> None:
    support_by_vertex = _validate_detached_mesh(mesh)
    if mesh_skin.mesh_skin_lineage_hash != mesh_skin_lineage_hash(mesh_skin):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_MESH_SKIN_LINEAGE_HASH_MISMATCH")
    if mesh_skin.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_MESH_BINDING_MISMATCH")
    if mesh_skin.surface_binding_hash != mesh.surface_binding_hash:
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_BINDING_MISMATCH")
    if mesh_skin.skeleton_binding_hash != mechanical.skeleton.skeleton_lineage_hash:
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_SKELETON_BINDING_MISMATCH")
    if mesh_skin.skin_binding_hash != mechanical.skin.skin_lineage_hash:
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_SOURCE_SKIN_AUTHORITY_MISMATCH")

    method = str(mesh_skin.transfer_method)
    allowed = {
        DIRECT_MODEL_TRANSFER_METHOD,
        SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
        RIGID_ATTACHMENT_TRANSFER_METHOD,
    }
    if method not in allowed:
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_UNSUPPORTED_BINDING_METHOD")
    if bool(mesh_skin.metadata.get("historical_weight_transfer_used", True)):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_HISTORICAL_WEIGHT_TRANSFER_FORBIDDEN")

    if method == DIRECT_MODEL_TRANSFER_METHOD:
        if not bool(mesh_skin.metadata.get("direct_model_query", False)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_DIRECT_MODEL_QUERY_REQUIRED")
    elif method == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD:
        if bool(mesh_skin.metadata.get("direct_model_query", True)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_CANNOT_CLAIM_DIRECT_MODEL_QUERY")
        if not bool(mesh_skin.metadata.get("surface_support_convex_transfer", False)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_PROOF_REQUIRED")
        if not bool(mesh_skin.metadata.get("source_skin_rows_consumed", False)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_SOURCE_SKIN_CONSUMPTION_REQUIRED")
        if bool(mesh_skin.metadata.get("semantic_skin_synthesis", True)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_SEMANTIC_SYNTHESIS_FORBIDDEN")
    else:
        if bool(mesh_skin.metadata.get("direct_model_query", True)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_RIGID_CARRY_CANNOT_CLAIM_DIRECT_MODEL_QUERY")
        if not bool(mesh_skin.metadata.get("runtime_representation_only", False)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_RIGID_CARRY_MUST_BE_RUNTIME_REPRESENTATION")
        if mesh_skin.metadata.get("attachment_truth_source") != "QUALIFIED_COMPONENT_ATTACHMENT":
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_RIGID_CARRY_ATTACHMENT_TRUTH_REQUIRED")
        if not bool(mesh_skin.metadata.get("one_hot_carry_compiled_from_attachment", False)):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_RIGID_CARRY_PROOF_REQUIRED")
        for field in ("component_assembly_hash", "component_lineage_hash", "bind_state_authority_hash", "parent_joint_id"):
            _required_evidence_hash("RIGID_CARRY_" + field.upper(), str(mesh_skin.metadata.get(field, "")))

    legal_joints = {str(j.canonical_joint_id) for j in mechanical.skeleton.joints}
    source_skin_by_surface = {str(row.surface_id): row for row in mechanical.skin.rows}
    seen: set[str] = set()
    for row in mesh_skin.rows:
        vid = str(row.canonical_mesh_vertex_id)
        if vid not in support_by_vertex or vid in seen:
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_MESH_SKIN_VERTEX_ID_MISMATCH")
        seen.add(vid)
        influences = tuple((str(jid), float(w)) for jid, w in row.influences)
        if not influences or any(jid not in legal_joints or not isfinite(w) or w < 0.0 for jid, w in influences):
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_INVALID_WEIGHT_ROW")
        if abs(sum(w for _jid, w in influences) - 1.0) > 1.0e-8:
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_WEIGHT_SIMPLEX_MISMATCH")
        if method == RIGID_ATTACHMENT_TRANSFER_METHOD:
            parent = str(mesh_skin.metadata.get("parent_joint_id", ""))
            if influences != ((parent, 1.0),):
                raise QualificationError("EXTERNAL_RENDER_SUPPORT_RIGID_CARRY_NOT_EXACT_ONE_HOT")

        source_coeffs = tuple((str(sid), float(w)) for sid, w in row.source_support_coefficients)
        if source_coeffs != support_by_vertex[vid]:
            raise QualificationError("EXTERNAL_RENDER_SUPPORT_SUPPORT_BINDING_DRIFT")

        if method == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD:
            expected = _replay_surface_support_row(source_skin_by_surface, source_coeffs)
            actual = dict(influences)
            l1 = sum(
                abs(float(actual.get(jid, 0.0)) - float(expected.get(jid, 0.0)))
                for jid in set(actual) | set(expected)
            )
            if l1 > _SURFACE_SUPPORT_REPLAY_L1_TOL:
                raise QualificationError(
                    f"EXTERNAL_RENDER_SUPPORT_SURFACE_CONVEX_REPLAY_MISMATCH:{vid}:{l1}"
                )
    if seen != set(support_by_vertex):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_INCOMPLETE_WEIGHT_ROWS")


def qualify_external_render_support(
    mesh,
    mesh_skin,
    mechanical,
    *,
    materialization_manifest_sha256: str,
    direct_binding_manifest_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> ExternalRenderSupportQualificationIR:
    _validate_detached_mesh_skin(mesh_skin, mesh, mechanical)
    method = str(mesh_skin.transfer_method)
    value = ExternalRenderSupportQualificationIR(
        view_index=int(mesh.view_index),
        render_support_surface_hash=str(mesh.surface_binding_hash),
        mesh_lineage_hash=str(mesh.mesh_lineage_hash),
        mesh_skin_lineage_hash=str(mesh_skin.mesh_skin_lineage_hash),
        skeleton_lineage_hash=str(mechanical.skeleton.skeleton_lineage_hash),
        source_skin_authority_lineage_hash=str(mechanical.skin.skin_lineage_hash),
        materialization_manifest_sha256=_required_evidence_hash("MATERIALIZATION_MANIFEST_SHA256", materialization_manifest_sha256),
        direct_binding_manifest_sha256=_required_evidence_hash("DIRECT_BINDING_MANIFEST_SHA256", direct_binding_manifest_sha256),
        transfer_method=method,
        qualification_hash="",
        metadata={
            "scientific_mechanical_surface_relabelled": False,
            "external_support_surface_promoted_to_scientific_authority": False,
            "historical_weight_transfer_used": False,
            "exact_mesh_vertex_direct_model_query": method == DIRECT_MODEL_TRANSFER_METHOD,
            "surface_support_convex_transfer": method == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
            "surface_support_convex_replay_verified": method == SURFACE_SUPPORT_CONVEX_TRANSFER_METHOD,
            "rigid_attachment_runtime_carry": method == RIGID_ATTACHMENT_TRANSFER_METHOD,
            **dict(metadata or {}),
        },
    )
    value = replace(value, qualification_hash=external_render_support_hash(value))
    validate_external_render_support_qualification(value, mesh, mesh_skin, mechanical)
    return value


def validate_external_render_support_qualification(value, mesh, mesh_skin, mechanical) -> None:
    if isinstance(value, Mapping):
        fields = dict(value)
        fields.setdefault("metadata", {})
        fields.setdefault("schema_version", "RealSaS.ExternalRenderSupportQualificationIR.v1")
        value = ExternalRenderSupportQualificationIR(**fields)
    if not isinstance(value, ExternalRenderSupportQualificationIR):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_QUALIFICATION_REQUIRED")
    _validate_detached_mesh_skin(mesh_skin, mesh, mechanical)
    expected = {
        "view_index": int(mesh.view_index),
        "render_support_surface_hash": str(mesh.surface_binding_hash),
        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
        "mesh_skin_lineage_hash": str(mesh_skin.mesh_skin_lineage_hash),
        "skeleton_lineage_hash": str(mechanical.skeleton.skeleton_lineage_hash),
        "source_skin_authority_lineage_hash": str(mechanical.skin.skin_lineage_hash),
        "transfer_method": str(mesh_skin.transfer_method),
    }
    for name, wanted in expected.items():
        if getattr(value, name) != wanted:
            raise QualificationError(f"EXTERNAL_RENDER_SUPPORT_QUALIFICATION_DRIFT:{name}")
    _required_evidence_hash("MATERIALIZATION_MANIFEST_SHA256", value.materialization_manifest_sha256)
    _required_evidence_hash("DIRECT_BINDING_MANIFEST_SHA256", value.direct_binding_manifest_sha256)
    if value.qualification_hash != external_render_support_hash(value):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_QUALIFICATION_HASH_MISMATCH")
    if bool(value.metadata.get("scientific_mechanical_surface_relabelled", True)):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_SCIENTIFIC_RELABEL_FORBIDDEN")
    if bool(value.metadata.get("historical_weight_transfer_used", True)):
        raise QualificationError("EXTERNAL_RENDER_SUPPORT_HISTORICAL_WEIGHT_TRANSFER_FORBIDDEN")


__all__ = [
    "ExternalRenderSupportQualificationIR",
    "external_render_support_hash",
    "qualify_external_render_support",
    "validate_external_render_support_qualification",
]
