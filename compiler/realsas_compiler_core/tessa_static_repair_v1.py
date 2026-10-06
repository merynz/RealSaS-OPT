from __future__ import annotations

"""Compiler-owned static repair transaction for TESSA learned carrier proposals.

The learned proposal is evidence, not final mechanical-carrier authority. This
module promotes the static survivor semantics proven by the Knight TESSA courts:

1. preserve the decoded TESSA proposal as an immutable geometric reference,
2. apply only the existing generic Compiler quality operators,
3. preserve GSA material support separately from learned geometry authority,
4. preserve an explicit final-vertex -> original-TESSA reference binding, and
5. fail closed unless the frozen static G3 policy is fully satisfied.

The stable TESSA producer id is intentionally retained for downstream schema/
consumer compatibility. The candidate metadata carries the explicit Compiler-
repaired lane. No teacher geometry, teacher skin, same-index transport, or
threshold relaxation is permitted here.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .canonical_mesh_quality_repair_v1 import (
    mechanical_quality_protected_surface_ids_v1,
    repair_candidate_endpoint_collapses_v1,
    repair_candidate_fixed_vertex_flips_v1,
    repair_candidate_projected_relaxation_v1,
)
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    MechanicalPartitionIR,
    MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError

Json = dict[str, Any]

_TESSA_PRODUCER = "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"
_INITIAL_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"
_REPAIRED_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_COMPILER_REPAIRED_V1"


@dataclass(frozen=True)
class TESSAReferenceBindingRowIR:
    candidate_vertex_id: str
    coefficients: tuple[tuple[str, float], ...]
    schema_version: str = "RealSaS.TESSAReferenceBindingRowIR.v1"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TESSAReferenceBindingIR:
    input_candidate_lineage_hash: str
    output_candidate_lineage_hash: str
    proposal_geometry_hash: str
    rows: tuple[TESSAReferenceBindingRowIR, ...]
    binding_hash: str
    schema_version: str = "RealSaS.TESSAReferenceBindingIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TESSAStaticRepairEvidenceIR:
    input_candidate_lineage_hash: str
    output_candidate_lineage_hash: str
    mesh_policy_hash: str
    partition_binding_hash: str
    proposal_geometry_hash: str
    material_support_field_hash: str
    reference_binding_hash: str
    input_vertex_count: int
    input_face_count: int
    output_vertex_count: int
    output_face_count: int
    accepted_flip_count: int
    accepted_collapse_count: int
    accepted_relaxation_count: int
    final_policy_violating_face_count: int
    final_min_angle_deg: float
    final_max_aspect: float
    evidence_hash: str
    schema_version: str = "RealSaS.TESSAStaticRepairEvidenceIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _reference_binding_hash(value: TESSAReferenceBindingIR) -> str:
    """Content hash for the mapping itself, independent of the final candidate hash.

    ``output_candidate_lineage_hash`` is a checked reverse link. Excluding only that
    field avoids a candidate-hash <-> evidence-hash cycle while the final candidate
    still commits to ``binding_hash`` in its own metadata.
    """
    payload = value.to_dict()
    payload.pop("binding_hash", None)
    payload.pop("output_candidate_lineage_hash", None)
    return content_sha256(payload)


def _repair_evidence_hash(value: TESSAStaticRepairEvidenceIR) -> str:
    """Hash the repair receipt without its reverse candidate-lineage link."""
    payload = value.to_dict()
    payload.pop("evidence_hash", None)
    payload.pop("output_candidate_lineage_hash", None)
    return content_sha256(payload)


def tessa_reference_binding_from_dict_v1(
    payload: Mapping[str, Any],
) -> TESSAReferenceBindingIR:
    if str(payload.get("schema_version") or payload.get("schema") or "") != "RealSaS.TESSAReferenceBindingIR.v1":
        raise QualificationError("TESSA_REFERENCE_BINDING_SCHEMA_INVALID")
    rows = tuple(
        TESSAReferenceBindingRowIR(
            candidate_vertex_id=str(row["candidate_vertex_id"]),
            coefficients=tuple(
                (str(pid), float(coeff)) for pid, coeff in row.get("coefficients") or ()
            ),
        )
        for row in payload.get("rows") or ()
    )
    return TESSAReferenceBindingIR(
        input_candidate_lineage_hash=str(payload.get("input_candidate_lineage_hash") or ""),
        output_candidate_lineage_hash=str(payload.get("output_candidate_lineage_hash") or ""),
        proposal_geometry_hash=str(payload.get("proposal_geometry_hash") or ""),
        rows=rows,
        binding_hash=str(payload.get("binding_hash") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def tessa_static_repair_evidence_from_dict_v1(
    payload: Mapping[str, Any],
) -> TESSAStaticRepairEvidenceIR:
    if str(payload.get("schema_version") or payload.get("schema") or "") != "RealSaS.TESSAStaticRepairEvidenceIR.v1":
        raise QualificationError("TESSA_STATIC_REPAIR_EVIDENCE_SCHEMA_INVALID")
    return TESSAStaticRepairEvidenceIR(
        input_candidate_lineage_hash=str(payload.get("input_candidate_lineage_hash") or ""),
        output_candidate_lineage_hash=str(payload.get("output_candidate_lineage_hash") or ""),
        mesh_policy_hash=str(payload.get("mesh_policy_hash") or ""),
        partition_binding_hash=str(payload.get("partition_binding_hash") or ""),
        proposal_geometry_hash=str(payload.get("proposal_geometry_hash") or ""),
        material_support_field_hash=str(payload.get("material_support_field_hash") or ""),
        reference_binding_hash=str(payload.get("reference_binding_hash") or ""),
        input_vertex_count=int(payload.get("input_vertex_count", 0)),
        input_face_count=int(payload.get("input_face_count", 0)),
        output_vertex_count=int(payload.get("output_vertex_count", 0)),
        output_face_count=int(payload.get("output_face_count", 0)),
        accepted_flip_count=int(payload.get("accepted_flip_count", 0)),
        accepted_collapse_count=int(payload.get("accepted_collapse_count", 0)),
        accepted_relaxation_count=int(payload.get("accepted_relaxation_count", 0)),
        final_policy_violating_face_count=int(payload.get("final_policy_violating_face_count", -1)),
        final_min_angle_deg=float(payload.get("final_min_angle_deg", float("nan"))),
        final_max_aspect=float(payload.get("final_max_aspect", float("nan"))),
        evidence_hash=str(payload.get("evidence_hash") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def _validate_initial_tessa_candidate(candidate: CanonicalMeshCandidateIR) -> None:
    if candidate.producer_id != _TESSA_PRODUCER:
        raise QualificationError("TESSA_STATIC_REPAIR_INPUT_PRODUCER_INVALID")
    md = dict(candidate.metadata or {})
    if md.get("geometry_authority_class") != _INITIAL_AUTHORITY_CLASS:
        raise QualificationError("TESSA_STATIC_REPAIR_INPUT_AUTHORITY_INVALID")
    if md.get("material_support_is_not_geometry_support") is not True:
        raise QualificationError("TESSA_STATIC_REPAIR_SUPPORT_SEPARATION_MISSING")
    if md.get("product_geometry_authority_claimed") is not False:
        raise QualificationError("TESSA_STATIC_REPAIR_PREMATURE_PRODUCT_AUTHORITY")
    if not str(md.get("proposal_geometry_hash") or ""):
        raise QualificationError("TESSA_STATIC_REPAIR_PROPOSAL_HASH_MISSING")
    if not str(md.get("material_support_field_hash") or ""):
        raise QualificationError("TESSA_STATIC_REPAIR_SUPPORT_FIELD_HASH_MISSING")
    for vertex in candidate.vertices:
        vmd = dict(vertex.metadata or {})
        if not str(vmd.get("proposal_vertex_id") or ""):
            raise QualificationError("TESSA_STATIC_REPAIR_PROPOSAL_VERTEX_ID_MISSING")
        if vmd.get("material_support_is_not_geometry_support") is not True:
            raise QualificationError("TESSA_STATIC_REPAIR_VERTEX_SUPPORT_SEPARATION_MISSING")


def _simplex(coefficients: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    rows = [
        (str(key), float(value))
        for key, value in coefficients.items()
        if float(value) > 1e-14
    ]
    total = sum(value for _, value in rows)
    if total <= 0.0:
        raise QualificationError("TESSA_REFERENCE_BINDING_EMPTY")
    output = tuple((key, value / total) for key, value in sorted(rows))
    if abs(sum(value for _, value in output) - 1.0) > 1e-10:
        raise QualificationError("TESSA_REFERENCE_BINDING_SIMPLEX_INVALID")
    return output


def _build_reference_binding_unsealed(
    *,
    original: CanonicalMeshCandidateIR,
    repaired: CanonicalMeshCandidateIR,
) -> TESSAReferenceBindingIR:
    original_to_proposal = {}
    for vertex in original.vertices:
        proposal_id = str(dict(vertex.metadata or {}).get("proposal_vertex_id") or "")
        if not proposal_id:
            raise QualificationError("TESSA_REFERENCE_ORIGINAL_PROPOSAL_ID_MISSING")
        original_to_proposal[str(vertex.candidate_vertex_id)] = proposal_id

    rows = []
    for vertex in repaired.vertices:
        metadata = dict(vertex.metadata or {})
        relaxed = dict(metadata.get("projected_quality_relaxation") or {})
        coefficients: dict[str, float] = {}
        if relaxed:
            face = tuple(map(str, relaxed.get("reference_face") or ()))
            bary = tuple(float(x) for x in relaxed.get("barycentric") or ())
            if len(face) != 3 or len(bary) != 3:
                raise QualificationError("TESSA_REFERENCE_RELAXATION_PROVENANCE_INVALID")
            for original_id, weight in zip(face, bary):
                if original_id not in original_to_proposal:
                    raise QualificationError("TESSA_REFERENCE_FACE_VERTEX_UNKNOWN")
                proposal_id = original_to_proposal[original_id]
                coefficients[proposal_id] = coefficients.get(proposal_id, 0.0) + weight
        else:
            proposal_id = str(metadata.get("proposal_vertex_id") or "")
            if not proposal_id:
                raise QualificationError("TESSA_REFERENCE_SURVIVOR_PROPOSAL_ID_MISSING")
            coefficients[proposal_id] = 1.0
        rows.append(
            TESSAReferenceBindingRowIR(
                candidate_vertex_id=str(vertex.candidate_vertex_id),
                coefficients=_simplex(coefficients),
            )
        )

    proposal_geometry_hash = str(dict(original.metadata or {}).get("proposal_geometry_hash") or "")
    provisional = TESSAReferenceBindingIR(
        input_candidate_lineage_hash=str(original.candidate_lineage_hash),
        output_candidate_lineage_hash="",
        proposal_geometry_hash=proposal_geometry_hash,
        rows=tuple(sorted(rows, key=lambda row: row.candidate_vertex_id)),
        binding_hash="",
        metadata={
            "authority": "COMPILER_REPAIR_PROVENANCE_TO_IMMUTABLE_TESSA_PROPOSAL",
            "same_index_skin_transport_authorized": False,
            "material_support_is_geometry_authority": False,
            "teacher_geometry_used": False,
            "output_candidate_lineage_is_reverse_link_not_hash_input": True,
        },
    )
    return replace(provisional, binding_hash=_reference_binding_hash(provisional))


def validate_tessa_reference_binding_v1(
    value: TESSAReferenceBindingIR,
    *,
    candidate: CanonicalMeshCandidateIR,
) -> None:
    if value.output_candidate_lineage_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_REFERENCE_BINDING_CANDIDATE_MISMATCH")
    if len(value.rows) != len(candidate.vertices):
        raise QualificationError("TESSA_REFERENCE_BINDING_ROW_COUNT_MISMATCH")
    expected_ids = {str(vertex.candidate_vertex_id) for vertex in candidate.vertices}
    actual_ids = {row.candidate_vertex_id for row in value.rows}
    if actual_ids != expected_ids:
        raise QualificationError("TESSA_REFERENCE_BINDING_VERTEX_ACCOUNTING_MISMATCH")
    for row in value.rows:
        if not row.coefficients or any(float(weight) < -1e-12 for _, weight in row.coefficients):
            raise QualificationError("TESSA_REFERENCE_BINDING_COEFFICIENT_INVALID")
        if abs(sum(float(weight) for _, weight in row.coefficients) - 1.0) > 1e-9:
            raise QualificationError("TESSA_REFERENCE_BINDING_SIMPLEX_INVALID")
    if value.binding_hash != _reference_binding_hash(value):
        raise QualificationError("TESSA_REFERENCE_BINDING_HASH_MISMATCH")
    md = dict(candidate.metadata or {})
    if md.get("tessa_reference_binding_hash") != value.binding_hash:
        raise QualificationError("TESSA_REFERENCE_BINDING_CANDIDATE_COMMITMENT_MISMATCH")


def validate_tessa_static_repair_evidence_v1(
    value: TESSAStaticRepairEvidenceIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    reference_binding: TESSAReferenceBindingIR,
) -> None:
    if value.output_candidate_lineage_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_STATIC_REPAIR_EVIDENCE_CANDIDATE_MISMATCH")
    if value.reference_binding_hash != reference_binding.binding_hash:
        raise QualificationError("TESSA_STATIC_REPAIR_REFERENCE_BINDING_MISMATCH")
    if value.output_vertex_count != len(candidate.vertices) or value.output_face_count != len(candidate.faces):
        raise QualificationError("TESSA_STATIC_REPAIR_EVIDENCE_COUNT_MISMATCH")
    if value.final_policy_violating_face_count != 0:
        raise QualificationError("TESSA_STATIC_REPAIR_EVIDENCE_RESIDUAL_G3")
    if value.evidence_hash != _repair_evidence_hash(value):
        raise QualificationError("TESSA_STATIC_REPAIR_EVIDENCE_HASH_MISMATCH")
    md = dict(candidate.metadata or {})
    if md.get("tessa_static_repair_evidence_hash") != value.evidence_hash:
        raise QualificationError("TESSA_STATIC_REPAIR_CANDIDATE_COMMITMENT_MISMATCH")


def repair_tessa_candidate_static_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    partition: MechanicalPartitionIR,
    collapse_batch_size: int = 32,
    collapse_limit: int = 256,
    relaxation_batch_size: int = 32,
    relaxation_limit: int = 512,
) -> tuple[CanonicalMeshCandidateIR, TESSAStaticRepairEvidenceIR, TESSAReferenceBindingIR]:
    """Repair one TESSA proposal under the frozen generic Compiler policy."""
    _validate_initial_tessa_candidate(candidate)
    if candidate.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_STATIC_REPAIR_PARTITION_BINDING_MISMATCH")
    if min(collapse_batch_size, collapse_limit, relaxation_batch_size, relaxation_limit) < 1:
        raise QualificationError("TESSA_STATIC_REPAIR_BUDGET_INVALID")

    original = candidate
    current, flip_report = repair_candidate_fixed_vertex_flips_v1(
        original,
        policy,
        max_passes=12,
    )

    collapse_total = 0
    collapse_reports = []
    while collapse_total < int(collapse_limit):
        budget = min(int(collapse_batch_size), int(collapse_limit) - collapse_total)
        next_candidate, report = repair_candidate_endpoint_collapses_v1(
            current,
            policy,
            max_collapses=budget,
            seed_only_violating_faces=True,
        )
        accepted = int(report["accepted_collapse_count"])
        collapse_reports.append(report)
        current = next_candidate
        collapse_total += accepted
        if int(report["after"]["policy_violating_face_count"]) == 0 or accepted == 0:
            break

    protected = mechanical_quality_protected_surface_ids_v1(partition)
    relaxation_total = 0
    relaxation_reports = []
    final_report = dict(collapse_reports[-1]["after"] if collapse_reports else flip_report["after"])
    while relaxation_total < int(relaxation_limit) and int(final_report["policy_violating_face_count"]) > 0:
        budget = min(int(relaxation_batch_size), int(relaxation_limit) - relaxation_total)
        next_candidate, report = repair_candidate_projected_relaxation_v1(
            current,
            original,
            policy,
            protected_surface_ids=protected,
            max_moves=budget,
        )
        accepted = int(report["accepted_move_count"])
        relaxation_reports.append(report)
        current = next_candidate
        relaxation_total += accepted
        final_report = dict(report["after"])
        if int(final_report["policy_violating_face_count"]) == 0 or accepted == 0:
            break

    if int(final_report["policy_violating_face_count"]) != 0:
        raise QualificationError(
            "TESSA_STATIC_REPAIR_RESIDUAL_G3:"
            + str(int(final_report["policy_violating_face_count"]))
        )

    changed_vertices = []
    for vertex in current.vertices:
        md = dict(vertex.metadata or {})
        md.update(
            {
                "geometry_authority_class": _REPAIRED_AUTHORITY_CLASS,
                "compiler_static_repair_applied": True,
                "material_support_is_not_geometry_support": True,
                "geometry_position_derived_from_material_support": False,
                "teacher_vertex_index_used": False,
            }
        )
        changed_vertices.append(replace(vertex, metadata=md))

    producer_policy_hash = content_sha256(
        {
            "schema": "RealSaS.TESSACompilerStaticRepairPolicy.v1",
            "input_candidate_lineage_hash": original.candidate_lineage_hash,
            "mesh_policy_hash": policy.qualification_policy_lineage_hash,
            "partition_binding_hash": partition.partition_lineage_hash,
            "operators": [
                "FIXED_VERTEX_MONOTONE_EDGE_FLIP_V1",
                "ENDPOINT_HALFEDGE_COLLAPSE_LINK_CONDITION_V1",
                "PROJECTED_LOCAL_RELAXATION_V1",
            ],
            "max_flip_passes": 12,
            "collapse_batch_size": int(collapse_batch_size),
            "collapse_limit": int(collapse_limit),
            "relaxation_batch_size": int(relaxation_batch_size),
            "relaxation_limit": int(relaxation_limit),
            "extra_protected_surface_ids": sorted(protected),
            "topological_boundary_protection": "OWNED_BY_RELAXATION_OPERATOR",
            "threshold_relaxation": False,
            "teacher_geometry_used": False,
        }
    )
    repaired_md = dict(current.metadata or {})
    repaired_md.update(
        {
            "geometry_authority_class": _REPAIRED_AUTHORITY_CLASS,
            "compiler_static_repair_applied": True,
            "static_repair_input_candidate_lineage_hash": original.candidate_lineage_hash,
            "material_support_is_not_geometry_support": True,
            "learned_xyz_preserved_exactly": True,
            "learned_xyz_preservation_scope": "POST_COMPILER_REPAIR_CANDIDATE_TO_DOWNSTREAM",
            "learned_proposal_xyz_preserved_exactly": False,
            "learned_proposal_is_immutable_reference": True,
            "legacy_g1_convex_lift_claimed": False,
            "product_geometry_authority_claimed": False,
            "requires_stage19_static_qualification": True,
            "threshold_relaxation_performed": False,
            "teacher_geometry_used_to_repair": False,
        }
    )

    # First form the repaired geometry with a temporary lineage so we can derive
    # non-circular evidence content over stable vertex ids and provenance rows.
    temporary = replace(
        current,
        vertices=tuple(changed_vertices),
        producer_id=_TESSA_PRODUCER,
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata=repaired_md,
    )
    temporary = replace(
        temporary,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(temporary),
    )
    reference_unsealed = _build_reference_binding_unsealed(
        original=original,
        repaired=temporary,
    )

    proposal_geometry_hash = str(dict(original.metadata or {}).get("proposal_geometry_hash") or "")
    material_support_field_hash = str(dict(original.metadata or {}).get("material_support_field_hash") or "")
    repair_unsealed = TESSAStaticRepairEvidenceIR(
        input_candidate_lineage_hash=str(original.candidate_lineage_hash),
        output_candidate_lineage_hash="",
        mesh_policy_hash=str(policy.qualification_policy_lineage_hash),
        partition_binding_hash=str(partition.partition_lineage_hash),
        proposal_geometry_hash=proposal_geometry_hash,
        material_support_field_hash=material_support_field_hash,
        reference_binding_hash=reference_unsealed.binding_hash,
        input_vertex_count=len(original.vertices),
        input_face_count=len(original.faces),
        output_vertex_count=len(temporary.vertices),
        output_face_count=len(temporary.faces),
        accepted_flip_count=int(flip_report["accepted_flip_count"]),
        accepted_collapse_count=int(collapse_total),
        accepted_relaxation_count=int(relaxation_total),
        final_policy_violating_face_count=int(final_report["policy_violating_face_count"]),
        final_min_angle_deg=float(final_report["min_angle_deg"]),
        final_max_aspect=float(final_report["max_aspect_longest_over_min_altitude"]),
        evidence_hash="",
        metadata={
            "authority": "COMPILER_STATIC_REPAIR_TRANSACTION_EVIDENCE",
            "static_geometry_qualification_only": True,
            "stage19_remeasurement_required": True,
            "product_authority_claimed": False,
            "motion_capability_claimed": False,
            "generalization_claimed": False,
            "teacher_geometry_used": False,
            "same_index_skin_transport_authorized": False,
            "threshold_relaxation_performed": False,
            "semantic_boundary_protection_count": len(protected),
            "collapse_batch_count": len(collapse_reports),
            "relaxation_batch_count": len(relaxation_reports),
            "output_candidate_lineage_is_reverse_link_not_hash_input": True,
        },
    )
    repair_unsealed = replace(
        repair_unsealed,
        evidence_hash=_repair_evidence_hash(repair_unsealed),
    )

    # Candidate commits to both evidence hashes. Their reverse links are filled only
    # after this final candidate lineage exists and are deliberately excluded from
    # the corresponding evidence content hashes.
    final_md = dict(temporary.metadata or {})
    final_md.update(
        {
            "tessa_reference_binding_hash": reference_unsealed.binding_hash,
            "tessa_static_repair_evidence_hash": repair_unsealed.evidence_hash,
        }
    )
    final_provisional = replace(
        temporary,
        candidate_lineage_hash="",
        metadata=final_md,
    )
    repaired = replace(
        final_provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(final_provisional),
    )
    reference_binding = replace(
        reference_unsealed,
        output_candidate_lineage_hash=repaired.candidate_lineage_hash,
    )
    evidence = replace(
        repair_unsealed,
        output_candidate_lineage_hash=repaired.candidate_lineage_hash,
    )

    validate_tessa_reference_binding_v1(reference_binding, candidate=repaired)
    validate_tessa_static_repair_evidence_v1(
        evidence,
        candidate=repaired,
        reference_binding=reference_binding,
    )
    return repaired, evidence, reference_binding
