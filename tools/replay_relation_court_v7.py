"""Exact source-cut lineage closure court over sealed V6 presentation evidence.

This research court tests the Stage37 hypothesis isolated from mechanics:
coincident coordinates are *not* contact truth in general, but boundary vertices
created by separate visual domains at the same source-art cut become admissible
contact witnesses when canonical support / skin ownership (or an explicit target
attachment owner) independently qualifies the relation.

The existing Stage42 contact-aware safety operator and semantic-order runtime are
left unchanged.  M/G/W and motion witnesses remain sealed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core import visual_contact_v1 as contact_mod
from compiler.realsas_compiler_core.visual_contact_v1 import (
    ARTICULATED_CONTINUITY,
    ATTACHMENT_CONTACT,
    MATERIAL_CONTINUITY,
    QualifiedVisualContactSet,
)
from tools import replay_relation_court_v2 as strict
from tools import replay_relation_court_v6 as v6


court = v6.court
_original_strict_qualify = strict._strict_qualify

POLICY = {
    "schema": "RealSaS.ExactSourceCutLineageClosureCourt.v1",
    "exact_source_cut_epsilon_px": 1.0e-9,
    "material_requires_shared_canonical_edge": True,
    "material_maximum_skin_l1": 0.60,
    "articulated_requires_shared_canonical_vertex": True,
    "attachment_requires_explicit_target_joint": True,
    "coincident_coordinates_without_independent_evidence_are_truth": False,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _exact_boundary_pairs(rest, a_vertices, b_vertices, *, epsilon: float):
    rest = np.asarray(rest, dtype=np.float64)
    a_vertices = np.asarray(a_vertices, dtype=np.int64)
    b_vertices = np.asarray(b_vertices, dtype=np.int64)
    if not len(a_vertices) or not len(b_vertices):
        return []
    pairs = set()
    b_tree = cKDTree(rest[b_vertices])
    distance, index = b_tree.query(rest[a_vertices], k=1)
    for av, d, bi in zip(a_vertices.tolist(), np.atleast_1d(distance), np.atleast_1d(index)):
        if float(d) <= epsilon:
            pairs.add((int(av), int(b_vertices[int(bi)])))
    a_tree = cKDTree(rest[a_vertices])
    distance, index = a_tree.query(rest[b_vertices], k=1)
    for bv, d, ai in zip(b_vertices.tolist(), np.atleast_1d(distance), np.atleast_1d(index)):
        if float(d) <= epsilon:
            pairs.add((int(a_vertices[int(ai)]), int(bv)))
    return sorted(pairs)


def _expanded_source_cut_qualify(**kwargs) -> QualifiedVisualContactSet:
    # Preserve every relation that the existing exact-cut qualification already
    # admits, then fill only exact source-cut witnesses omitted by candidate
    # truncation / domain-pair sampling.
    parent = _original_strict_qualify(**kwargs)

    rest = np.asarray(kwargs["rest_positions"], dtype=np.float64)
    faces = np.asarray(kwargs["visual_faces"], dtype=np.int64)
    domains = np.asarray(kwargs["domain_id"], dtype=np.int32)
    anchors = np.asarray(kwargs["anchor_vertex"], dtype=np.int64)
    ancestry = np.asarray(kwargs["anchor_mechanical_vertices"], dtype=np.int64)
    blend = np.asarray(kwargs["motion_blend_coefficients"], dtype=np.float64)
    owners = np.asarray(kwargs["vertex_attachment_owner"], dtype=np.int32)
    target_map = {
        int(owner): int(joint)
        for owner, joint in dict(kwargs.get("attachment_target_joint_by_owner") or {}).items()
    }

    boundary = contact_mod._boundary_vertices(faces, domains)
    support = contact_mod._domain_mechanical_support(domains, anchors, ancestry)
    dominant_joint = np.argmax(blend, axis=1).astype(np.int32)

    rows = {}
    for pair, code, domain_pair, distance, score in zip(
        np.asarray(parent.pairs, dtype=np.int64).reshape(-1, 2).tolist(),
        np.asarray(parent.relation_codes, dtype=np.int8).tolist(),
        np.asarray(parent.domain_pairs, dtype=np.int32).reshape(-1, 2).tolist(),
        np.asarray(parent.rest_distances_px, dtype=np.float64).tolist(),
        np.asarray(parent.evidence_scores, dtype=np.float64).tolist(),
    ):
        key = tuple(map(int, pair))
        rows[key] = (
            int(code), tuple(map(int, domain_pair)), float(distance), float(score)
        )

    epsilon = float(POLICY["exact_source_cut_epsilon_px"])
    max_skin_l1 = float(POLICY["material_maximum_skin_l1"])
    domain_values = sorted(map(int, np.unique(domains)))
    for da, db in combinations(domain_values, 2):
        exact = _exact_boundary_pairs(rest, boundary[da], boundary[db], epsilon=epsilon)
        if not exact:
            continue
        shared = support[da].intersection(support[db])
        shared_edge = len(shared) >= 2
        shared_vertex = bool(shared)
        oa = int(np.bincount(owners[domains == da]).argmax())
        ob = int(np.bincount(owners[domains == db]).argmax())

        for av, bv in exact:
            key = (int(av), int(bv))
            if key in rows:
                continue
            l1 = float(np.abs(blend[av] - blend[bv]).sum())
            ja, jb = int(dominant_joint[av]), int(dominant_joint[bv])
            relation = None
            evidence = 0.0

            if oa == ob and shared_edge and l1 <= max_skin_l1:
                relation = MATERIAL_CONTINUITY
                evidence = 4.0 + max(0.0, 1.0 - l1)
            elif oa == ob and shared_vertex:
                relation = ARTICULATED_CONTINUITY
                evidence = 3.0 + (1.0 if ja == jb else 0.0)
            elif oa != ob:
                attachment_owner = oa if oa > 0 else ob if ob > 0 else 0
                body_joint = jb if oa > 0 else ja
                if (
                    attachment_owner > 0
                    and attachment_owner in target_map
                    and int(target_map[attachment_owner]) == int(body_joint)
                ):
                    relation = ATTACHMENT_CONTACT
                    evidence = 3.5

            # Exact coordinate coincidence alone remains insufficient.
            if relation is None:
                continue
            rows[key] = (relation, (int(da), int(db)), 0.0, evidence)

    ordered = sorted(rows.items(), key=lambda item: (item[1][1], item[0], item[1][0]))
    pairs = np.asarray([key for key, _ in ordered], dtype=np.int64).reshape(-1, 2)
    codes = np.asarray([value[0] for _, value in ordered], dtype=np.int8)
    domain_pairs = np.asarray([value[1] for _, value in ordered], dtype=np.int32).reshape(-1, 2)
    distances = np.asarray([value[2] for _, value in ordered], dtype=np.float64)
    scores = np.asarray([value[3] for _, value in ordered], dtype=np.float64)

    digest = content_sha256(
        {
            "schema": POLICY["schema"],
            "parent_contact_hash": parent.contact_hash,
            "policy_hash": content_sha256(POLICY),
            "pairs": pairs.astype(int).tolist(),
            "relation_codes": codes.astype(int).tolist(),
            "domain_pairs": domain_pairs.astype(int).tolist(),
        }
    )
    return QualifiedVisualContactSet(
        view_index=int(parent.view_index),
        pairs=pairs,
        relation_codes=codes,
        domain_pairs=domain_pairs,
        rest_distances_px=distances,
        evidence_scores=scores,
        contact_hash=digest,
    )


# replay_relation_court_v4's contact reducer calls strict._strict_qualify at
# execution time, so replacing this one research hook exercises the normal
# Stage42 safety and runtime composition path without introducing a second
# consumer or draw-order authority.
strict._strict_qualify = _expanded_source_cut_qualify


if __name__ == "__main__":
    court.main()
