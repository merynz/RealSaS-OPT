from __future__ import annotations

"""Deterministic structural partition over immutable RiggingSurfaceIR."""

from dataclasses import replace
from typing import Iterable

from .hashing import content_sha256
from .product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    ComponentRegionIR,
    MechanicalPartitionIR,
    build_mechanical_partition,
)
from .types import QualificationError, RiggingSurfaceIR

_UNSAFE_TOKENS = ("UNKNOWN", "AMBIGUOUS", "OCCLUDED", "UNOBSERVED", "UNSUPPORTED")


def _pair(a: str, b: str) -> tuple[str, str]:
    if not a or not b or a == b:
        raise QualificationError("PARTITION_BOUNDARY_PAIR_INVALID")
    return (a, b) if a < b else (b, a)


def _relation_is_unknown(relation) -> bool:
    kind = str(relation.relation_kind).upper()
    if any(token in kind for token in _UNSAFE_TOKENS):
        return True
    md = dict(getattr(relation, "metadata", {}) or {})
    if bool(md.get("crosses_unknown", False)) or bool(md.get("unknown_bridge", False)):
        return True
    try:
        return float(relation.score) <= 0.0
    except Exception as exc:
        raise QualificationError("PARTITION_RELATION_SCORE_INVALID") from exc


def propose_relation_boundary_constraints(
    surface: RiggingSurfaceIR,
) -> tuple[ComponentBoundaryConstraintIR, ...]:
    """Safe S relation => PRESERVE_CONTINUITY; unsafe relation => UNKNOWN.

    Missing adjacency never becomes an invented SEPARATE edge.
    """
    known = {node.surface_id for node in surface.surface_nodes}
    by_pair: dict[tuple[str, str], list] = {}
    for relation in surface.local_relations:
        pair = _pair(relation.a_surface_id, relation.b_surface_id)
        if pair[0] not in known or pair[1] not in known:
            raise QualificationError("PARTITION_RELATION_REFERENCES_UNKNOWN_SURFACE")
        by_pair.setdefault(pair, []).append(relation)

    rows = []
    for pair in sorted(by_pair):
        relations = by_pair[pair]
        unknown = any(_relation_is_unknown(r) for r in relations)
        decision = "UNKNOWN" if unknown else "PRESERVE_CONTINUITY"
        evidence_refs = tuple(sorted(r.relation_id for r in relations))
        confidence = 0.0 if unknown else min(max(float(r.score), 0.0) for r in relations)
        cid = "BOUNDARY:" + content_sha256({
            "surface": surface.geometry_lineage_hash,
            "pair": pair,
            "decision": decision,
            "evidence_refs": evidence_refs,
        })[:20]
        rows.append(ComponentBoundaryConstraintIR(
            constraint_id=cid,
            a_surface_id=pair[0],
            b_surface_id=pair[1],
            decision=decision,
            evidence_refs=evidence_refs,
            confidence=confidence,
            metadata={
                "evidence_class": "STRUCTURAL_SURFACE_RELATION",
                "categorical_identity_used": False,
                "provisional_unknown_behavior": "PRESERVE_CONTINUITY_FOR_CANDIDATE_ONLY",
            },
        ))
    return tuple(rows)


def resolve_boundary_constraints(
    surface: RiggingSurfaceIR,
    *,
    overrides: Iterable[ComponentBoundaryConstraintIR] = (),
) -> tuple[ComponentBoundaryConstraintIR, ...]:
    """Apply explicit qualified evidence only on existing local-relation adjacency."""
    defaults = propose_relation_boundary_constraints(surface)
    default_by_pair = {_pair(x.a_surface_id, x.b_surface_id): x for x in defaults}
    override_by_pair: dict[tuple[str, str], ComponentBoundaryConstraintIR] = {}

    for row in overrides:
        pair = _pair(row.a_surface_id, row.b_surface_id)
        if pair not in default_by_pair:
            raise QualificationError("PARTITION_OVERRIDE_REQUIRES_EXISTING_LOCAL_RELATION")
        if pair in override_by_pair:
            raise QualificationError("PARTITION_DUPLICATE_BOUNDARY_OVERRIDE")
        if row.decision not in {"SEPARATE", "PRESERVE_CONTINUITY", "UNKNOWN"}:
            raise QualificationError("PARTITION_BOUNDARY_DECISION_INVALID")
        if not row.evidence_refs:
            raise QualificationError("PARTITION_OVERRIDE_REQUIRES_EVIDENCE")
        override_by_pair[pair] = row

    out = []
    for pair in sorted(default_by_pair):
        base = default_by_pair[pair]
        row = override_by_pair.get(pair)
        if row is None:
            out.append(base)
        else:
            out.append(replace(
                row,
                a_surface_id=pair[0],
                b_surface_id=pair[1],
                metadata={
                    **dict(row.metadata or {}),
                    "overrides_relation_default": base.decision,
                    "categorical_identity_used": False,
                },
            ))
    return tuple(out)


def _component_regions(
    surface: RiggingSurfaceIR,
    constraints: tuple[ComponentBoundaryConstraintIR, ...],
) -> tuple[ComponentRegionIR, ...]:
    ids = tuple(sorted(node.surface_id for node in surface.surface_nodes))
    parent = {sid: sid for sid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # UNKNOWN is only a candidate-construction default. It remains explicitly
    # UNKNOWN in the partition and G4 must reject it later if consequential.
    for row in constraints:
        if row.decision != "SEPARATE":
            union(row.a_surface_id, row.b_surface_id)

    for row in constraints:
        if row.decision == "SEPARATE" and find(row.a_surface_id) == find(row.b_surface_id):
            raise QualificationError("PARTITION_SEPARATE_CONSTRAINT_NOT_A_CUT")

    groups: dict[str, list[str]] = {}
    for sid in ids:
        groups.setdefault(find(sid), []).append(sid)

    result = []
    for members in sorted(tuple(sorted(v)) for v in groups.values()):
        cid = "CMP:" + content_sha256({
            "surface": surface.geometry_lineage_hash,
            "members": members,
        })[:20]
        internal_refs = tuple(sorted(
            row.constraint_id
            for row in constraints
            if row.a_surface_id in members and row.b_surface_id in members
        ))
        result.append(ComponentRegionIR(
            component_id=cid,
            surface_ids=members,
            evidence_refs=internal_refs,
            metadata={
                "partition_method": "LOCAL_RELATION_GRAPH_WITH_EXPLICIT_CUTS_V1",
                "unknown_provisionally_preserved": True,
                "categorical_identity": None,
            },
        ))
    return tuple(result)


def build_structural_partition(
    surface: RiggingSurfaceIR,
    *,
    boundary_overrides: Iterable[ComponentBoundaryConstraintIR] = (),
) -> MechanicalPartitionIR:
    """Build structural components without object-category recognition or S mutation."""
    constraints = resolve_boundary_constraints(surface, overrides=boundary_overrides)
    components = _component_regions(surface, constraints)
    return build_mechanical_partition(
        surface=surface,
        components=components,
        boundary_constraints=constraints,
        metadata={
            "producer": "RealSaS.StructuralPartition.LocalRelationBoundaryV1",
            "source_surface_mutated": False,
            "categorical_recognition_used": False,
            "unknown_boundary_count": sum(x.decision == "UNKNOWN" for x in constraints),
            "separate_boundary_count": sum(x.decision == "SEPARATE" for x in constraints),
            "preserve_boundary_count": sum(x.decision == "PRESERVE_CONTINUITY" for x in constraints),
        },
    )
