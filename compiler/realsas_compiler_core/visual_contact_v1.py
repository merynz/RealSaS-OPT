from __future__ import annotations

"""Qualified visual contact relations derived from sealed presentation evidence.

This module does not treat coincident source coordinates as contact truth. It
combines source-chart proximity with canonical mechanical support, frozen skin
ownership and explicit target-attachment slot ownership. The result is a
small, deterministic relation set that Stage42 may consume as a constraint and
Stage45 may prove independently.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .hashing import content_sha256
from .types import QualificationError


SCHEMA = "RealSaS.QualifiedVisualContactSet.v1"
MATERIAL_CONTINUITY = 0
ARTICULATED_CONTINUITY = 1
ATTACHMENT_CONTACT = 2
RELATION_NAMES = {
    MATERIAL_CONTINUITY: "MATERIAL_CONTINUITY",
    ARTICULATED_CONTINUITY: "ARTICULATED_CONTINUITY",
    ATTACHMENT_CONTACT: "ATTACHMENT_CONTACT",
}
POLICY = {
    "schema": "RealSaS.VisualContactQualificationPolicy.v1",
    "maximum_source_boundary_distance_px": 12.0,
    "maximum_material_skin_l1": 0.60,
    "maximum_pairs_per_domain_relation": 4,
    "material_requires_shared_canonical_edge": True,
    "articulated_requires_shared_canonical_vertex": True,
    "attachment_requires_target_slot_owner": True,
    "coincident_source_coordinates_are_truth": False,
    "mechanical_state_mutation_authorized": False,
}


@dataclass(frozen=True)
class QualifiedVisualContactSet:
    view_index: int
    pairs: np.ndarray
    relation_codes: np.ndarray
    domain_pairs: np.ndarray
    rest_distances_px: np.ndarray
    evidence_scores: np.ndarray
    contact_hash: str

    def summary(self) -> dict:
        codes = np.asarray(self.relation_codes, dtype=np.int32)
        return {
            "schema": SCHEMA,
            "view_index": int(self.view_index),
            "contact_pair_count": int(len(self.pairs)),
            "material_continuity_pair_count": int(np.count_nonzero(codes == MATERIAL_CONTINUITY)),
            "articulated_continuity_pair_count": int(np.count_nonzero(codes == ARTICULATED_CONTINUITY)),
            "attachment_contact_pair_count": int(np.count_nonzero(codes == ATTACHMENT_CONTACT)),
            "contact_hash": str(self.contact_hash),
            "policy_hash": content_sha256(POLICY),
        }


def _validate_inputs(
    rest_positions,
    visual_faces,
    domain_id,
    anchor_vertex,
    anchor_mechanical_vertices,
    motion_blend_coefficients,
    vertex_attachment_owner,
):
    rest = np.asarray(rest_positions, dtype=np.float64)
    faces = np.asarray(visual_faces, dtype=np.int64)
    domains = np.asarray(domain_id, dtype=np.int32)
    anchors = np.asarray(anchor_vertex, dtype=np.int64)
    ancestry = np.asarray(anchor_mechanical_vertices, dtype=np.int64)
    blend = np.asarray(motion_blend_coefficients, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    if (
        rest.ndim != 2
        or rest.shape[1] != 2
        or not np.isfinite(rest).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(rest))
        or domains.shape != (len(rest),)
        or owners.shape != (len(rest),)
        or np.any(domains < 0)
        or np.any(owners < 0)
        or anchors.ndim != 1
        or ancestry.shape != (len(anchors), 3)
        or np.any(anchors < 0)
        or np.any(anchors >= len(rest))
        or ancestry.dtype.kind not in "iu"
        or np.any(ancestry < 0)
        or blend.ndim != 2
        or blend.shape[0] != len(rest)
        or not blend.shape[1]
        or not np.isfinite(blend).all()
        or np.any(blend < -1e-12)
        or not np.allclose(blend.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise QualificationError("VISUAL_CONTACT_INPUT_INVALID")
    if np.any(domains[faces] != domains[faces[:, :1]]):
        raise QualificationError("VISUAL_CONTACT_FACE_CROSSES_DOMAIN")
    return rest, faces, domains, anchors, ancestry, blend, owners


def _boundary_vertices(faces: np.ndarray, domains: np.ndarray) -> dict[int, np.ndarray]:
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    out = {}
    for domain in np.unique(domains):
        vertices = np.unique(boundary[domains[boundary[:, 0]] == domain])
        if not len(vertices):
            vertices = np.flatnonzero(domains == domain)
        out[int(domain)] = np.asarray(vertices, dtype=np.int64)
    return out


def _domain_mechanical_support(
    domains: np.ndarray, anchors: np.ndarray, ancestry: np.ndarray
) -> dict[int, set[int]]:
    support: dict[int, set[int]] = {int(d): set() for d in np.unique(domains)}
    for vertex, tri in zip(anchors.tolist(), ancestry.tolist()):
        support[int(domains[int(vertex)])].update(map(int, tri))
    if any(not values for values in support.values()):
        raise QualificationError("VISUAL_CONTACT_DOMAIN_CANONICAL_SUPPORT_EMPTY")
    return support


def _nearest_boundary_pairs(
    rest: np.ndarray,
    a_vertices: np.ndarray,
    b_vertices: np.ndarray,
    *,
    maximum_distance_px: float,
    maximum_pairs: int,
) -> list[tuple[int, int, float]]:
    if not len(a_vertices) or not len(b_vertices):
        return []
    b_tree = cKDTree(rest[b_vertices])
    distance, index = b_tree.query(rest[a_vertices], k=1)
    candidates = []
    for av, d, bi in zip(a_vertices.tolist(), np.atleast_1d(distance), np.atleast_1d(index)):
        if float(d) <= maximum_distance_px:
            candidates.append((int(av), int(b_vertices[int(bi)]), float(d)))
    a_tree = cKDTree(rest[a_vertices])
    distance, index = a_tree.query(rest[b_vertices], k=1)
    for bv, d, ai in zip(b_vertices.tolist(), np.atleast_1d(distance), np.atleast_1d(index)):
        if float(d) <= maximum_distance_px:
            candidates.append((int(a_vertices[int(ai)]), int(bv), float(d)))
    candidates.sort(key=lambda row: (row[2], row[0], row[1]))
    selected = []
    seen_a, seen_b = set(), set()
    for row in candidates:
        if row[0] in seen_a and row[1] in seen_b:
            continue
        selected.append(row)
        seen_a.add(row[0])
        seen_b.add(row[1])
        if len(selected) >= maximum_pairs:
            break
    return selected


def _attachment_target_owner_map(
    attachment_target_joint_by_owner: Mapping[int, int] | None,
) -> dict[int, int]:
    if attachment_target_joint_by_owner is None:
        return {}
    out = {}
    for owner, joint in attachment_target_joint_by_owner.items():
        oi, ji = int(owner), int(joint)
        if oi <= 0 or ji < 0:
            raise QualificationError("VISUAL_CONTACT_ATTACHMENT_TARGET_MAP_INVALID")
        out[oi] = ji
    return out


def qualify_visual_contacts(
    *,
    view_index: int,
    rest_positions,
    visual_faces,
    domain_id,
    anchor_vertex,
    anchor_mechanical_vertices,
    motion_blend_coefficients,
    vertex_attachment_owner,
    attachment_target_joint_by_owner: Mapping[int, int] | None = None,
) -> QualifiedVisualContactSet:
    rest, faces, domains, anchors, ancestry, blend, owners = _validate_inputs(
        rest_positions,
        visual_faces,
        domain_id,
        anchor_vertex,
        anchor_mechanical_vertices,
        motion_blend_coefficients,
        vertex_attachment_owner,
    )
    target_joint = _attachment_target_owner_map(attachment_target_joint_by_owner)
    boundary = _boundary_vertices(faces, domains)
    support = _domain_mechanical_support(domains, anchors, ancestry)
    dominant_joint = np.argmax(blend, axis=1).astype(np.int32)

    pairs = []
    relation_codes = []
    domain_pairs = []
    rest_distances = []
    evidence_scores = []

    max_distance = float(POLICY["maximum_source_boundary_distance_px"])
    max_pairs = int(POLICY["maximum_pairs_per_domain_relation"])
    domain_values = sorted(map(int, np.unique(domains)))
    for da, db in combinations(domain_values, 2):
        va = boundary[da]
        vb = boundary[db]
        nearest = _nearest_boundary_pairs(
            rest,
            va,
            vb,
            maximum_distance_px=max_distance,
            maximum_pairs=max_pairs,
        )
        if not nearest:
            continue

        shared = support[da].intersection(support[db])
        oa = int(np.bincount(owners[domains == da]).argmax())
        ob = int(np.bincount(owners[domains == db]).argmax())
        shared_edge = len(shared) >= 2
        shared_vertex = bool(shared)

        for av, bv, distance in nearest:
            l1 = float(np.abs(blend[av] - blend[bv]).sum())
            ja, jb = int(dominant_joint[av]), int(dominant_joint[bv])
            relation = None
            evidence = 0.0

            if oa == ob and shared_edge and l1 <= float(POLICY["maximum_material_skin_l1"]):
                relation = MATERIAL_CONTINUITY
                evidence = 3.0 + max(0.0, 1.0 - l1)
            elif oa == ob and shared_vertex:
                relation = ARTICULATED_CONTINUITY
                evidence = 2.0 + (1.0 if ja == jb else 0.0)
            elif oa != ob:
                attachment_owner = oa if oa > 0 else ob if ob > 0 else 0
                body_joint = jb if oa > 0 else ja
                if (
                    attachment_owner > 0
                    and attachment_owner in target_joint
                    and body_joint == target_joint[attachment_owner]
                ):
                    relation = ATTACHMENT_CONTACT
                    evidence = 2.5

            if relation is None:
                continue
            pairs.append((av, bv))
            relation_codes.append(relation)
            domain_pairs.append((da, db))
            rest_distances.append(distance)
            evidence_scores.append(evidence)

    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    code_array = np.asarray(relation_codes, dtype=np.int8)
    domain_array = np.asarray(domain_pairs, dtype=np.int32).reshape(-1, 2)
    distance_array = np.asarray(rest_distances, dtype=np.float64)
    score_array = np.asarray(evidence_scores, dtype=np.float64)

    payload = {
        "schema": SCHEMA,
        "view_index": int(view_index),
        "policy_hash": content_sha256(POLICY),
        "pairs": pair_array.tolist(),
        "relation_codes": code_array.astype(int).tolist(),
        "domain_pairs": domain_array.astype(int).tolist(),
        "rest_distances_px": distance_array.tolist(),
        "evidence_scores": score_array.tolist(),
    }
    digest = content_sha256(payload)
    return QualifiedVisualContactSet(
        int(view_index),
        pair_array,
        code_array,
        domain_array,
        distance_array,
        score_array,
        digest,
    )


def validate_qualified_visual_contacts(
    value: QualifiedVisualContactSet,
    *,
    vertex_count: int,
    domain_id,
) -> None:
    pairs = np.asarray(value.pairs)
    codes = np.asarray(value.relation_codes)
    domains = np.asarray(domain_id, dtype=np.int32)
    domain_pairs = np.asarray(value.domain_pairs)
    distances = np.asarray(value.rest_distances_px)
    scores = np.asarray(value.evidence_scores)
    n = len(pairs)
    if (
        pairs.shape != (n, 2)
        or pairs.dtype.kind not in "iu"
        or np.any(pairs < 0)
        or np.any(pairs >= int(vertex_count))
        or codes.shape != (n,)
        or domain_pairs.shape != (n, 2)
        or distances.shape != (n,)
        or scores.shape != (n,)
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
        or not np.isfinite(scores).all()
        or np.any(~np.isin(codes, tuple(RELATION_NAMES)))
        or domains.shape != (int(vertex_count),)
    ):
        raise QualificationError("VISUAL_CONTACT_QUALIFIED_SET_INVALID")
    if n and not np.array_equal(domain_pairs, domains[pairs]):
        raise QualificationError("VISUAL_CONTACT_DOMAIN_PAIR_DRIFT")
    payload = {
        "schema": SCHEMA,
        "view_index": int(value.view_index),
        "policy_hash": content_sha256(POLICY),
        "pairs": pairs.astype(int).tolist(),
        "relation_codes": codes.astype(int).tolist(),
        "domain_pairs": domain_pairs.astype(int).tolist(),
        "rest_distances_px": distances.astype(float).tolist(),
        "evidence_scores": scores.astype(float).tolist(),
    }
    if value.contact_hash != content_sha256(payload):
        raise QualificationError("VISUAL_CONTACT_HASH_DRIFT")


def contact_relation_metrics(
    *,
    rest_positions,
    baseline_positions,
    posed_positions,
    pairs,
    relation_codes,
    material_tolerance_px: float = 1e-6,
    articulated_tolerance_px: float = 1.5,
) -> dict:
    rest = np.asarray(rest_positions, dtype=np.float64)
    baseline = np.asarray(baseline_positions, dtype=np.float64)
    posed = np.asarray(posed_positions, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(relation_codes, dtype=np.int8)
    if (
        rest.shape != baseline.shape
        or rest.shape != posed.shape
        or rest.ndim != 2
        or rest.shape[1] != 2
        or codes.shape != (len(pairs),)
        or np.any(pairs < 0)
        or np.any(pairs >= len(rest))
        or not all(np.isfinite(x).all() for x in (rest, baseline, posed))
    ):
        raise QualificationError("VISUAL_CONTACT_METRIC_INPUT_INVALID")
    if not len(pairs):
        return {
            "qualified_contact_relations_passed": True,
            "contact_pair_count": 0,
            "contact_failure_count": 0,
            "maximum_contact_repair_delta_residual_px": 0.0,
            "maximum_contact_gap_growth_px": 0.0,
        }

    rest_distance = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1)
    baseline_distance = np.linalg.norm(
        baseline[pairs[:, 0]] - baseline[pairs[:, 1]], axis=1
    )
    posed_distance = np.linalg.norm(posed[pairs[:, 0]] - posed[pairs[:, 1]], axis=1)
    repair_delta = np.abs(posed_distance - baseline_distance)
    gap_growth = np.maximum(posed_distance - rest_distance, 0.0)
    tolerance = np.where(
        codes == MATERIAL_CONTINUITY,
        float(material_tolerance_px),
        float(articulated_tolerance_px),
    )
    failure = (repair_delta > tolerance) | (gap_growth > tolerance)
    return {
        "qualified_contact_relations_passed": bool(not np.any(failure)),
        "contact_pair_count": int(len(pairs)),
        "contact_failure_count": int(np.count_nonzero(failure)),
        "maximum_contact_repair_delta_residual_px": float(
            np.max(repair_delta, initial=0.0)
        ),
        "maximum_contact_gap_growth_px": float(np.max(gap_growth, initial=0.0)),
    }
