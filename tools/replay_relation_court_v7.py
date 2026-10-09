"""Exact source-cut lineage closure court over sealed V6 presentation evidence.

Coincident coordinates are not contact truth in general.  The existing
source_cut_pairs contract first extracts exact same-owner cross-domain source
cuts; canonical support and skin ownership then decide whether a candidate is
material or articulated continuity.  Stage42 / runtime stay unchanged and
M/G/W remain sealed.
"""

from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core import visual_contact_v1 as contact_mod
from compiler.realsas_compiler_core.visual_contact_v1 import (
    ARTICULATED_CONTINUITY,
    MATERIAL_CONTINUITY,
    QualifiedVisualContactSet,
)
from compiler.realsas_compiler_core.visual_presentation_contract_v1 import (
    source_cut_pairs,
)
from tools import replay_relation_court_v2 as strict
from tools import replay_relation_court_v6 as v6


court = v6.court
_original_strict_qualify = strict._strict_qualify

POLICY = {
    "schema": "RealSaS.ExactSourceCutLineageClosureCourt.v2",
    "candidate_contract": "PRESENTATION_EXACT_SOURCE_CUT_SAME_OWNER_V1",
    "material_requires_shared_canonical_edge": True,
    "material_maximum_skin_l1": 0.60,
    "articulated_requires_shared_canonical_vertex": True,
    "coincident_coordinates_without_independent_evidence_are_truth": False,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _expanded_source_cut_qualify(**kwargs) -> QualifiedVisualContactSet:
    parent = _original_strict_qualify(**kwargs)
    rest = np.asarray(kwargs["rest_positions"], dtype=np.float64)
    faces = np.asarray(kwargs["visual_faces"], dtype=np.int64)
    domains = np.asarray(kwargs["domain_id"], dtype=np.int32)
    anchors = np.asarray(kwargs["anchor_vertex"], dtype=np.int64)
    ancestry = np.asarray(kwargs["anchor_mechanical_vertices"], dtype=np.int64)
    blend = np.asarray(kwargs["motion_blend_coefficients"], dtype=np.float64)
    owners = np.asarray(kwargs["vertex_attachment_owner"], dtype=np.int32)

    support = contact_mod._domain_mechanical_support(domains, anchors, ancestry)
    candidates = source_cut_pairs(rest, domains, owners)

    rows = {}
    for pair, code, domain_pair, distance, score in zip(
        np.asarray(parent.pairs, dtype=np.int64).reshape(-1, 2).tolist(),
        np.asarray(parent.relation_codes, dtype=np.int8).tolist(),
        np.asarray(parent.domain_pairs, dtype=np.int32).reshape(-1, 2).tolist(),
        np.asarray(parent.rest_distances_px, dtype=np.float64).tolist(),
        np.asarray(parent.evidence_scores, dtype=np.float64).tolist(),
    ):
        rows[tuple(map(int, pair))] = (
            int(code), tuple(map(int, domain_pair)), float(distance), float(score)
        )

    max_skin_l1 = float(POLICY["material_maximum_skin_l1"])
    for av, bv in np.asarray(candidates, dtype=np.int64).reshape(-1, 2).tolist():
        av, bv = int(av), int(bv)
        key = (av, bv)
        reverse = (bv, av)
        if key in rows or reverse in rows:
            continue
        da, db = int(domains[av]), int(domains[bv])
        if da == db or int(owners[av]) != int(owners[bv]):
            continue
        shared = support[da].intersection(support[db])
        l1 = float(np.abs(blend[av] - blend[bv]).sum())
        if len(shared) >= 2 and l1 <= max_skin_l1:
            relation = MATERIAL_CONTINUITY
            evidence = 4.0 + max(0.0, 1.0 - l1)
        elif len(shared) >= 1:
            relation = ARTICULATED_CONTINUITY
            evidence = 3.0
        else:
            # Exact source coincidence alone remains only a candidate.
            continue
        if da > db:
            key = (bv, av)
            da, db = db, da
        rows[key] = (relation, (da, db), 0.0, evidence)

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


# replay_relation_court_v4's reducer resolves this symbol at execution time.
strict._strict_qualify = _expanded_source_cut_qualify


if __name__ == "__main__":
    court.main()
