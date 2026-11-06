from __future__ import annotations

"""Canonical Skin v2 for control-ready rigs.

Skin authority binds Mesh + Rig deform-structure identity, not the full animator
control program. A controller-only edit therefore does not invalidate weights;
rest/deform structure changes still do. Learned RealSaS skin codes remain proposal
evidence and are decoded before this Compiler boundary by the admitted MIRA codec.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2, canonical_rig_lineage_hash_v2
from .carrier_skin_v1 import QualifiedCarrierSkinIR
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CanonicalSkinRowIRV2:
    canonical_mesh_vertex_id: str
    influences: tuple[tuple[str, float], ...]
    source_carrier_vertex_id: str
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalSkinIRV2:
    rows: tuple[CanonicalSkinRowIRV2, ...]
    mesh_lineage_hash: str
    deform_structure_hash: str
    mesh_carrier_evidence_hash: str
    mesh_carrier_topology_hash: str
    mesh_carrier_geometry_hash: str
    carrier_skin_qualification_hash: str
    skin_code_proposal_hash: str
    skin_codec_architecture_id: str
    skin_codec_checkpoint_hash: str
    skin_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalSkinIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_skin_lineage_hash_v2(value: CanonicalSkinIRV2) -> str:
    payload = value.to_dict()
    payload.pop("skin_lineage_hash", None)
    return content_sha256(payload)


def _qualified_skin_hash(value: QualifiedCarrierSkinIR) -> str:
    return content_sha256({
        "carrier_evidence_hash": value.carrier_evidence_hash,
        "carrier_topology_hash": value.carrier_topology_hash,
        "carrier_geometry_hash": value.carrier_geometry_hash,
        "skeleton_binding_hash": value.skeleton_binding_hash,
        "rows": [row.to_dict() for row in value.rows],
        "qualification_report": value.qualification_report,
        "metadata": value.metadata,
    })


def mint_canonical_skin_v2(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    *,
    qualified_skin: QualifiedCarrierSkinIR,
    skin_code_proposal_hash: str,
    skin_codec_architecture_id: str,
    skin_codec_checkpoint_hash: str,
) -> CanonicalSkinIRV2:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("CANONICAL_SKIN_V2_MESH_LINEAGE_DRIFT")
    if mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("CANONICAL_SKIN_V2_MESH_AUTHORITY_DRIFT")
    if rig.rig_lineage_hash != canonical_rig_lineage_hash_v2(rig):
        raise QualificationError("CANONICAL_SKIN_V2_RIG_LINEAGE_DRIFT")
    if rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("CANONICAL_SKIN_V2_RIG_AUTHORITY_DRIFT")
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("CANONICAL_SKIN_V2_RIG_MESH_PARENT_DRIFT")

    qualified_hash = _qualified_skin_hash(qualified_skin)
    if qualified_skin.skin_lineage_hash != qualified_hash:
        raise QualificationError("CANONICAL_SKIN_V2_QUALIFIED_HASH_DRIFT")
    if qualified_skin.carrier_evidence_hash != mesh.carrier_evidence_hash:
        raise QualificationError("CANONICAL_SKIN_V2_CARRIER_EVIDENCE_DRIFT")
    if qualified_skin.carrier_topology_hash != mesh.carrier_topology_hash:
        raise QualificationError("CANONICAL_SKIN_V2_CARRIER_TOPOLOGY_DRIFT")
    if qualified_skin.carrier_geometry_hash != mesh.carrier_geometry_hash:
        raise QualificationError("CANONICAL_SKIN_V2_CARRIER_GEOMETRY_DRIFT")
    if qualified_skin.skeleton_binding_hash != rig.deform_structure_hash:
        raise QualificationError("CANONICAL_SKIN_V2_DEFORM_STRUCTURE_BINDING_DRIFT")
    report = dict(qualified_skin.qualification_report or {})
    if report.get("status") != "PASS_DIRECT_CARRIER_SKIN":
        raise QualificationError("CANONICAL_SKIN_V2_QUALIFICATION_NOT_PASS")

    code_hash = str(skin_code_proposal_hash).strip()
    codec_id = str(skin_codec_architecture_id).strip()
    codec_hash = str(skin_codec_checkpoint_hash).strip()
    if len(code_hash) < 8 or not codec_id or len(codec_hash) < 8:
        raise QualificationError("CANONICAL_SKIN_V2_CODEC_PROVENANCE_REQUIRED")

    mesh_by_source = {str(v.source_candidate_vertex_id): v for v in mesh.vertices}
    joint_ids = {str(j.canonical_joint_id) for j in rig.joints}
    if len(mesh_by_source) != len(mesh.vertices) or not joint_ids:
        raise QualificationError("CANONICAL_SKIN_V2_PARENT_DOMAIN_INVALID")
    rows: list[CanonicalSkinRowIRV2] = []
    seen: set[str] = set()
    for row in qualified_skin.rows:
        source_id = str(row.carrier_vertex_id)
        vertex = mesh_by_source.get(source_id)
        if vertex is None:
            raise QualificationError("CANONICAL_SKIN_V2_CARRIER_VERTEX_UNKNOWN")
        vid = str(vertex.canonical_mesh_vertex_id)
        if vid in seen:
            raise QualificationError("CANONICAL_SKIN_V2_DUPLICATE_VERTEX")
        seen.add(vid)
        influences = tuple((str(jid), float(w)) for jid, w in row.influences)
        if not influences or any(jid not in joint_ids for jid, _ in influences):
            raise QualificationError("CANONICAL_SKIN_V2_JOINT_UNKNOWN")
        if any((not math.isfinite(w)) or w < 0.0 for _, w in influences):
            raise QualificationError("CANONICAL_SKIN_V2_WEIGHT_INVALID")
        if abs(sum(w for _, w in influences) - 1.0) > 1e-6:
            raise QualificationError("CANONICAL_SKIN_V2_SIMPLEX_DRIFT")
        rows.append(CanonicalSkinRowIRV2(
            canonical_mesh_vertex_id=vid,
            influences=influences,
            source_carrier_vertex_id=source_id,
            metadata={"decoded_from_discrete_skin_representation": True},
        ))
    if seen != {str(v.canonical_mesh_vertex_id) for v in mesh.vertices}:
        raise QualificationError("CANONICAL_SKIN_V2_MESH_COVERAGE_DRIFT")

    value = CanonicalSkinIRV2(
        rows=tuple(rows),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash=qualified_hash,
        skin_code_proposal_hash=code_hash,
        skin_codec_architecture_id=codec_id,
        skin_codec_checkpoint_hash=codec_hash,
        skin_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_SKIN",
            "canonical_parents": ["CANONICAL_MESH", "CANONICAL_RIG_DEFORM_STRUCTURE"],
            "full_rig_lineage_is_not_skin_parent": True,
            "control_program_mutation_invalidates_skin": False,
            "deform_structure_mutation_invalidates_skin": True,
            "mesh_mutation_invalidates_skin": True,
            "skin_code_decode_is_model_owned": True,
            "compiler_code_search_or_repair_performed": False,
        },
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v2(value))


def canonical_skin_from_dict_v2(payload: Mapping[str, Any]) -> CanonicalSkinIRV2:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalSkinIR.v2":
        raise QualificationError("CANONICAL_SKIN_V2_SCHEMA_DRIFT")
    rows = tuple(CanonicalSkinRowIRV2(
        canonical_mesh_vertex_id=str(x["canonical_mesh_vertex_id"]),
        influences=tuple((str(j), float(w)) for j, w in x.get("influences") or ()),
        source_carrier_vertex_id=str(x["source_carrier_vertex_id"]),
        metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("rows") or ())
    value = CanonicalSkinIRV2(
        rows=rows,
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        deform_structure_hash=str(payload["deform_structure_hash"]),
        mesh_carrier_evidence_hash=str(payload["mesh_carrier_evidence_hash"]),
        mesh_carrier_topology_hash=str(payload["mesh_carrier_topology_hash"]),
        mesh_carrier_geometry_hash=str(payload["mesh_carrier_geometry_hash"]),
        carrier_skin_qualification_hash=str(payload["carrier_skin_qualification_hash"]),
        skin_code_proposal_hash=str(payload["skin_code_proposal_hash"]),
        skin_codec_architecture_id=str(payload["skin_codec_architecture_id"]),
        skin_codec_checkpoint_hash=str(payload["skin_codec_checkpoint_hash"]),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.skin_lineage_hash != canonical_skin_lineage_hash_v2(value):
        raise QualificationError("CANONICAL_SKIN_V2_LINEAGE_HASH_DRIFT")
    return value


__all__ = [
    "CanonicalSkinIRV2", "CanonicalSkinRowIRV2", "canonical_skin_from_dict_v2",
    "canonical_skin_lineage_hash_v2", "mint_canonical_skin_v2",
]
