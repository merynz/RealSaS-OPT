from __future__ import annotations

"""Carrier-native MIRA skin proposal and qualification.

The learned skin field is predicted directly on the exact Stage19 mechanical
carrier M. GSA/RiggingSurfaceIR remains evidence only; there is no semantic
surface-skin authority and no post-hoc skin transport in this path.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CarrierSkinInfluenceProposal:
    carrier_vertex_id: str
    canonical_joint_id: str
    weight: float
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CarrierSkinProposalIR:
    influences: tuple[CarrierSkinInfluenceProposal, ...]
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    skeleton_binding_hash: str
    model_provenance: str = ""
    schema_version: str = "RealSaS.CarrierSkinProposalIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedCarrierSkinRow:
    carrier_vertex_id: str
    influences: tuple[tuple[str, float], ...]
    simplex_residual_before: float
    correction_l1: float
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedCarrierSkinIR:
    rows: tuple[QualifiedCarrierSkinRow, ...]
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    skeleton_binding_hash: str
    qualification_report: Json
    skin_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedCarrierSkinIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)


def carrier_skin_proposal_from_dict(payload: Mapping[str, Any]) -> CarrierSkinProposalIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CarrierSkinProposalIR.v1":
        raise QualificationError("CARRIER_SKIN_PROPOSAL_SCHEMA_DRIFT")
    return CarrierSkinProposalIR(
        influences=tuple(
            CarrierSkinInfluenceProposal(
                carrier_vertex_id=str(row["carrier_vertex_id"]),
                canonical_joint_id=str(row["canonical_joint_id"]),
                weight=float(row["weight"]),
            )
            for row in payload.get("influences") or ()
        ),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        model_provenance=str(payload.get("model_provenance") or ""),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )


def qualified_carrier_skin_from_dict(payload: Mapping[str, Any]) -> QualifiedCarrierSkinIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.QualifiedCarrierSkinIR.v1":
        raise QualificationError("QUALIFIED_CARRIER_SKIN_SCHEMA_DRIFT")
    value = QualifiedCarrierSkinIR(
        rows=tuple(
            QualifiedCarrierSkinRow(
                carrier_vertex_id=str(row["carrier_vertex_id"]),
                influences=tuple(
                    (str(jid), float(weight))
                    for jid, weight in row.get("influences") or ()
                ),
                simplex_residual_before=float(row["simplex_residual_before"]),
                correction_l1=float(row["correction_l1"]),
            )
            for row in payload.get("rows") or ()
        ),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    expected = content_sha256(
        {
            "carrier_evidence_hash": value.carrier_evidence_hash,
            "carrier_topology_hash": value.carrier_topology_hash,
            "carrier_geometry_hash": value.carrier_geometry_hash,
            "skeleton_binding_hash": value.skeleton_binding_hash,
            "rows": [row.to_dict() for row in value.rows],
            "qualification_report": value.qualification_report,
            "metadata": value.metadata,
        }
    )
    if value.skin_lineage_hash != expected:
        raise QualificationError("QUALIFIED_CARRIER_SKIN_LINEAGE_HASH_MISMATCH")
    return value


def qualify_carrier_skin_v1(
    carrier,
    skeleton,
    proposal: CarrierSkinProposalIR,
    *,
    max_simplex_repair_l1: float,
    max_total_correction_l1: float,
    negative_tolerance: float,
    max_influences: int | None = None,
) -> QualifiedCarrierSkinIR:
    if proposal.carrier_evidence_hash != carrier.carrier_evidence_hash:
        raise QualificationError("CARRIER_SKIN_EVIDENCE_BINDING_DRIFT")
    if proposal.carrier_topology_hash != carrier.topology_hash:
        raise QualificationError("CARRIER_SKIN_TOPOLOGY_BINDING_DRIFT")
    if proposal.carrier_geometry_hash != carrier.geometry_hash:
        raise QualificationError("CARRIER_SKIN_GEOMETRY_BINDING_DRIFT")
    if proposal.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("CARRIER_SKIN_SKELETON_BINDING_DRIFT")

    meta = dict(proposal.metadata or {})
    if str(meta.get("output_domain") or "") != "MECHANICAL_CARRIER_M":
        raise QualificationError("CARRIER_SKIN_OUTPUT_DOMAIN_INVALID")
    if meta.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("CARRIER_SKIN_TRANSFER_FORBIDDEN")
    if meta.get("gsa_skin_field_authority_minted") is not False:
        raise QualificationError("CARRIER_SKIN_GSA_AUTHORITY_FORBIDDEN")
    if meta.get("direct_simplex_readout") is not True:
        raise QualificationError("CARRIER_SKIN_DIRECT_SIMPLEX_AUTHORITY_REQUIRED")
    if meta.get("source_skin_runtime_authority") is not False:
        raise QualificationError("CARRIER_SKIN_SOURCE_RUNTIME_AUTHORITY_FORBIDDEN")
    if meta.get("mesh_mutation_invalidates_prediction") is not True:
        raise QualificationError("CARRIER_SKIN_MESH_MUTATION_INVALIDATION_REQUIRED")

    vertex_ids = tuple(map(str, carrier.ordered_vertex_ids))
    vertex_set = set(vertex_ids)
    joint_ids = {str(j.canonical_joint_id) for j in skeleton.joints}
    if not vertex_ids or not joint_ids:
        raise QualificationError("CARRIER_SKIN_EMPTY_BINDING_DOMAIN")

    grouped: dict[str, dict[str, float]] = {vid: {} for vid in vertex_ids}
    for influence in proposal.influences:
        vid = str(influence.carrier_vertex_id)
        jid = str(influence.canonical_joint_id)
        weight = float(influence.weight)
        if vid not in vertex_set:
            raise QualificationError("CARRIER_SKIN_UNKNOWN_VERTEX")
        if jid not in joint_ids:
            raise QualificationError("CARRIER_SKIN_UNKNOWN_JOINT")
        if jid in grouped[vid]:
            raise QualificationError("CARRIER_SKIN_DUPLICATE_VERTEX_JOINT")
        if not math.isfinite(weight) or weight < -float(negative_tolerance):
            raise QualificationError("CARRIER_SKIN_WEIGHT_INVALID")
        grouped[vid][jid] = max(0.0, weight)

    rows: list[QualifiedCarrierSkinRow] = []
    total_correction = 0.0
    max_row_correction = 0.0
    for vid in vertex_ids:
        values = grouped[vid]
        if not values:
            raise QualificationError("CARRIER_SKIN_VERTEX_UNCOVERED")
        ordered = sorted(values.items())
        raw_sum = sum(weight for _, weight in ordered)
        if not math.isfinite(raw_sum) or raw_sum <= 0.0:
            raise QualificationError("CARRIER_SKIN_ROW_ZERO")
        residual = abs(raw_sum - 1.0)
        normalized = [(jid, weight / raw_sum) for jid, weight in ordered]
        correction = sum(
            abs(after - before)
            for (_, before), (_, after) in zip(ordered, normalized)
        )
        if correction > float(max_simplex_repair_l1) + 1e-15:
            raise QualificationError("CARRIER_SKIN_ROW_REPAIR_BUDGET_EXCEEDED")
        if max_influences is not None:
            positive = sum(weight > 0.0 for _, weight in normalized)
            if positive > int(max_influences):
                raise QualificationError("CARRIER_SKIN_MAX_INFLUENCES_EXCEEDED")
        total_correction += correction
        max_row_correction = max(max_row_correction, correction)
        rows.append(
            QualifiedCarrierSkinRow(
                carrier_vertex_id=vid,
                influences=tuple((jid, float(weight)) for jid, weight in normalized),
                simplex_residual_before=float(residual),
                correction_l1=float(correction),
            )
        )
    if total_correction > float(max_total_correction_l1) + 1e-15:
        raise QualificationError("CARRIER_SKIN_TOTAL_REPAIR_BUDGET_EXCEEDED")

    report = {
        "status": "PASS_DIRECT_CARRIER_SKIN",
        "output_domain": "MECHANICAL_CARRIER_M",
        "carrier_vertex_count": len(vertex_ids),
        "joint_count": len(joint_ids),
        "prediction_carrier_coverage": 1.0,
        "product_skin_evidence_complete": True,
        "semantic_skin_transfer_performed": False,
        "gsa_skin_field_authority_minted": False,
        "simplex_max_abs_residual_before": max(
            (row.simplex_residual_before for row in rows), default=0.0
        ),
        "max_row_correction_l1": max_row_correction,
        "total_correction_l1": total_correction,
        "max_influences": max_influences,
    }
    metadata = {
        **meta,
        "carrier_identity_in_skin_lineage": True,
        "mesh_mutation_invalidates_skin": True,
        "normal_path_skin_transport": "FORBIDDEN",
    }
    base = {
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "carrier_geometry_hash": carrier.geometry_hash,
        "skeleton_binding_hash": skeleton.skeleton_lineage_hash,
        "rows": [row.to_dict() for row in rows],
        "qualification_report": report,
        "metadata": metadata,
    }
    return QualifiedCarrierSkinIR(
        rows=tuple(rows),
        carrier_evidence_hash=carrier.carrier_evidence_hash,
        carrier_topology_hash=carrier.topology_hash,
        carrier_geometry_hash=carrier.geometry_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        qualification_report=report,
        skin_lineage_hash=content_sha256(base),
        metadata=metadata,
    )


def carrier_joint_influence_vectors(value: QualifiedCarrierSkinIR) -> dict[str, tuple[float, ...]]:
    """Return one exact per-carrier-vertex influence vector per joint.

    Unlike a raw column sum, these vectors preserve carrier-cardinality-independent
    observability statistics (mean/max) for generic Compiler qualification.
    """
    joint_ids = sorted({jid for row in value.rows for jid, _ in row.influences})
    out = {jid: [0.0] * len(value.rows) for jid in joint_ids}
    for row_index, row in enumerate(value.rows):
        for joint_id, weight in row.influences:
            out.setdefault(joint_id, [0.0] * len(value.rows))[row_index] = float(weight)
    return {jid: tuple(values) for jid, values in out.items()}


def carrier_joint_mass(value: QualifiedCarrierSkinIR) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in value.rows:
        for joint_id, weight in row.influences:
            out[joint_id] = out.get(joint_id, 0.0) + float(weight)
    return out


__all__ = [
    "CarrierSkinInfluenceProposal",
    "CarrierSkinProposalIR",
    "QualifiedCarrierSkinRow",
    "QualifiedCarrierSkinIR",
    "carrier_skin_proposal_from_dict",
    "qualified_carrier_skin_from_dict",
    "qualify_carrier_skin_v1",
    "carrier_joint_mass",
    "carrier_joint_influence_vectors",
]
