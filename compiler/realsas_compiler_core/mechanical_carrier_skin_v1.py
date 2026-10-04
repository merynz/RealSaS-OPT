from __future__ import annotations

"""Qualified skin state defined directly on one mechanical carrier.

This IR is deliberately separate from QualifiedSkinIR, whose rows live on the
GSA/RiggingSurface domain. A carrier-native MIRA query must not be silently
re-labelled as a transferred surface skin field.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class QualifiedMechanicalCarrierSkinRowIR:
    carrier_vertex_id: str
    influences: tuple[tuple[str, float], ...]
    simplex_residual_before: float
    correction_l1: float
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedMechanicalCarrierSkinIR:
    rows: tuple[QualifiedMechanicalCarrierSkinRowIR, ...]
    candidate_mesh_binding_hash: str
    carrier_evidence_binding_hash: str
    skeleton_binding_hash: str
    qualification_report: Json
    skin_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedMechanicalCarrierSkinIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def mechanical_carrier_skin_lineage_hash(
    value: QualifiedMechanicalCarrierSkinIR,
) -> str:
    payload = value.to_dict()
    payload.pop("skin_lineage_hash", None)
    return content_sha256(payload)


def qualify_mechanical_carrier_skin_v1(
    *,
    candidate,
    carrier_evidence,
    skeleton,
    vertex_ids: Sequence[str],
    joint_ids: Sequence[str],
    weights: np.ndarray,
    max_simplex_repair_l1: float = 1e-5,
    max_total_correction_l1: float = 0.05,
    negative_tolerance: float = 1e-8,
) -> QualifiedMechanicalCarrierSkinIR:
    if (
        str(carrier_evidence.candidate_mesh_binding_hash)
        != str(candidate.candidate_lineage_hash)
    ):
        raise QualificationError("CARRIER_SKIN_CANDIDATE_BINDING_DRIFT")
    if min(max_simplex_repair_l1, max_total_correction_l1, negative_tolerance) < 0.0:
        raise ValueError("carrier skin budgets must be nonnegative")

    expected_vertices = tuple(
        str(v.candidate_vertex_id)
        for v in sorted(candidate.vertices, key=lambda x: str(x.candidate_vertex_id))
    )
    vertex_ids = tuple(map(str, vertex_ids))
    if vertex_ids != expected_vertices or vertex_ids != tuple(carrier_evidence.ordered_vertex_ids):
        raise QualificationError("CARRIER_SKIN_VERTEX_DOMAIN_DRIFT")

    expected_joints = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    joint_ids = tuple(map(str, joint_ids))
    if set(joint_ids) != set(expected_joints) or len(joint_ids) != len(expected_joints):
        raise QualificationError("CARRIER_SKIN_JOINT_DOMAIN_DRIFT")
    joint_reorder = [joint_ids.index(jid) for jid in expected_joints]

    raw = np.asarray(weights, np.float64)
    if raw.shape != (len(vertex_ids), len(joint_ids)) or not np.isfinite(raw).all():
        raise QualificationError("CARRIER_SKIN_WEIGHT_MATRIX_INVALID")
    raw = raw[:, joint_reorder]
    if np.any(raw < -float(negative_tolerance)):
        raise QualificationError("CARRIER_SKIN_MATERIAL_NEGATIVE_WEIGHT")

    rows = []
    total_correction = 0.0
    max_residual = 0.0
    corrected = 0
    tiny_negative_mass = 0.0
    for vi, vertex_id in enumerate(vertex_ids):
        source = raw[vi].copy()
        tiny = source < 0.0
        if np.any(tiny):
            tiny_negative_mass += float((-source[tiny]).sum())
            source[tiny] = 0.0
        total = float(source.sum())
        if total <= 1e-12:
            raise QualificationError("CARRIER_SKIN_ZERO_ROW")
        residual = abs(total - 1.0)
        max_residual = max(max_residual, residual)
        normalized = source / total
        correction = float(np.abs(normalized - raw[vi]).sum())
        if correction > float(max_simplex_repair_l1) + 1e-15:
            raise QualificationError(
                f"CARRIER_SKIN_ROW_CORRECTION_BUDGET_EXCEEDED:{vertex_id}:{correction}"
            )
        total_correction += correction
        if total_correction > float(max_total_correction_l1) + 1e-15:
            raise QualificationError(
                f"CARRIER_SKIN_TOTAL_CORRECTION_BUDGET_EXCEEDED:{total_correction}"
            )
        if correction > 1e-12:
            corrected += 1
        influences = tuple(
            (jid, float(normalized[ji]))
            for ji, jid in enumerate(expected_joints)
            if float(normalized[ji]) > 0.0
        )
        rows.append(
            QualifiedMechanicalCarrierSkinRowIR(
                carrier_vertex_id=vertex_id,
                influences=influences,
                simplex_residual_before=float(residual),
                correction_l1=float(correction),
                metadata={"direct_carrier_inference": True},
            )
        )

    report = {
        "status": "PASS_DIRECT_CARRIER_SKIN",
        "row_count": len(rows),
        "prediction_carrier_coverage": 1.0,
        "max_simplex_residual_before": float(max_residual),
        "corrected_row_count": int(corrected),
        "total_correction_l1": float(total_correction),
        "tiny_negative_clipped_mass": float(tiny_negative_mass),
        "surface_skin_transfer_used": False,
        "direct_carrier_inference": True,
    }
    value = QualifiedMechanicalCarrierSkinIR(
        rows=tuple(rows),
        candidate_mesh_binding_hash=str(candidate.candidate_lineage_hash),
        carrier_evidence_binding_hash=str(carrier_evidence.carrier_evidence_hash),
        skeleton_binding_hash=str(skeleton.skeleton_lineage_hash),
        qualification_report=report,
        skin_lineage_hash="",
        metadata={
            "weight_domain": "EXACT_MECHANICAL_CARRIER_VERTICES",
            "teacher_product_authority": False,
        },
    )
    return replace(value, skin_lineage_hash=mechanical_carrier_skin_lineage_hash(value))


def validate_mechanical_carrier_skin_v1(
    value: QualifiedMechanicalCarrierSkinIR,
    *,
    candidate,
    carrier_evidence,
    skeleton,
) -> None:
    if value.schema_version != "RealSaS.QualifiedMechanicalCarrierSkinIR.v1":
        raise QualificationError("CARRIER_SKIN_SCHEMA_DRIFT")
    if value.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("CARRIER_SKIN_CANDIDATE_BINDING_DRIFT")
    if value.carrier_evidence_binding_hash != carrier_evidence.carrier_evidence_hash:
        raise QualificationError("CARRIER_SKIN_EVIDENCE_BINDING_DRIFT")
    if value.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("CARRIER_SKIN_SKELETON_BINDING_DRIFT")

    expected_vertices = {
        str(v.candidate_vertex_id) for v in candidate.vertices
    }
    joint_ids = {str(j.canonical_joint_id) for j in skeleton.joints}
    seen_vertices = set()
    for row in value.rows:
        vid = str(row.carrier_vertex_id)
        if vid in seen_vertices or vid not in expected_vertices:
            raise QualificationError("CARRIER_SKIN_ROW_VERTEX_INVALID")
        seen_vertices.add(vid)
        seen_joints = set()
        total = 0.0
        for jid, weight in row.influences:
            jid = str(jid)
            w = float(weight)
            if jid in seen_joints or jid not in joint_ids or not math.isfinite(w) or w < 0.0:
                raise QualificationError("CARRIER_SKIN_INFLUENCE_INVALID")
            seen_joints.add(jid)
            total += w
        if abs(total - 1.0) > 1e-8:
            raise QualificationError("CARRIER_SKIN_SIMPLEX_DRIFT")
    if seen_vertices != expected_vertices:
        raise QualificationError("CARRIER_SKIN_INCOMPLETE_VERTEX_ACCOUNTING")
    if value.skin_lineage_hash != mechanical_carrier_skin_lineage_hash(value):
        raise QualificationError("CARRIER_SKIN_LINEAGE_HASH_MISMATCH")


def mechanical_carrier_skin_from_dict(
    payload: Mapping[str, Any],
) -> QualifiedMechanicalCarrierSkinIR:
    value = QualifiedMechanicalCarrierSkinIR(
        rows=tuple(
            QualifiedMechanicalCarrierSkinRowIR(
                carrier_vertex_id=str(row["carrier_vertex_id"]),
                influences=tuple(
                    (str(jid), float(weight))
                    for jid, weight in row.get("influences") or ()
                ),
                simplex_residual_before=float(row.get("simplex_residual_before", 0.0)),
                correction_l1=float(row.get("correction_l1", 0.0)),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in payload.get("rows") or ()
        ),
        candidate_mesh_binding_hash=str(payload["candidate_mesh_binding_hash"]),
        carrier_evidence_binding_hash=str(payload["carrier_evidence_binding_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=str(
            payload.get("schema_version")
            or payload.get("schema")
            or "RealSaS.QualifiedMechanicalCarrierSkinIR.v1"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.skin_lineage_hash != mechanical_carrier_skin_lineage_hash(value):
        raise QualificationError("CARRIER_SKIN_LINEAGE_HASH_MISMATCH")
    return value


__all__ = [
    "QualifiedMechanicalCarrierSkinIR",
    "QualifiedMechanicalCarrierSkinRowIR",
    "mechanical_carrier_skin_from_dict",
    "mechanical_carrier_skin_lineage_hash",
    "qualify_mechanical_carrier_skin_v1",
    "validate_mechanical_carrier_skin_v1",
]
