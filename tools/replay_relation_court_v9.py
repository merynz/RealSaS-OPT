"""Skin-supported exact source-cut qualification court.

V8 proved that material seam coefficients must be coupled before Stage42 safety.
The remaining Stage37 abstentions are exact same-owner source cuts whose sparse
canonical anchors do not intersect.  This court admits only domain-cut groups
with multiple exact source witnesses, one dominant motion owner, and a very low
skin-field disagreement.  Exact coordinate coincidence by itself is still not
truth.

The admitted material seam vertices then flow through the V8 coefficient
coupling and the existing Stage42 / semantic-order / native runtime unchanged.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_contact_v1 import (
    MATERIAL_CONTINUITY,
    QualifiedVisualContactSet,
)
from compiler.realsas_compiler_core.visual_presentation_contract_v1 import (
    source_cut_pairs,
)
from tools import replay_relation_court_v2 as strict
from tools import replay_relation_court_v8 as v8


court = v8.court
_parent_qualify = strict._strict_qualify

POLICY = {
    "schema": "RealSaS.SkinSupportedExactSourceCutQualificationCourt.v1",
    "candidate_contract": "PRESENTATION_EXACT_SOURCE_CUT_SAME_OWNER_V1",
    "minimum_exact_vertex_pairs_per_domain_cut": 2,
    "maximum_skin_l1_per_exact_pair": 1.0e-3,
    "require_same_dominant_joint_for_every_pair": True,
    "coincident_coordinates_without_skin_evidence_are_truth": False,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _skin_supported_source_cut_qualify(**kwargs) -> QualifiedVisualContactSet:
    parent = _parent_qualify(**kwargs)
    rest = np.asarray(kwargs["rest_positions"], dtype=np.float64)
    domains = np.asarray(kwargs["domain_id"], dtype=np.int32)
    blend = np.asarray(kwargs["motion_blend_coefficients"], dtype=np.float64)
    owners = np.asarray(kwargs["vertex_attachment_owner"], dtype=np.int32)
    candidates = np.asarray(
        source_cut_pairs(rest, domains, owners), dtype=np.int64
    ).reshape(-1, 2)

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

    grouped = defaultdict(list)
    for av, bv in candidates.tolist():
        av, bv = int(av), int(bv)
        da, db = int(domains[av]), int(domains[bv])
        if da == db or int(owners[av]) != int(owners[bv]):
            continue
        if da > db:
            av, bv = bv, av
            da, db = db, da
        grouped[(da, db)].append((av, bv))

    minimum_pairs = int(POLICY["minimum_exact_vertex_pairs_per_domain_cut"])
    max_l1 = float(POLICY["maximum_skin_l1_per_exact_pair"])
    dominant = np.argmax(blend, axis=1).astype(np.int32)
    admitted_domain_cuts = 0
    admitted_vertex_pairs = 0

    for domain_pair, pairs in sorted(grouped.items()):
        # De-duplicate exact vertex witnesses deterministically.
        pairs = sorted(set(pairs))
        if len(pairs) < minimum_pairs:
            continue
        a = np.asarray([row[0] for row in pairs], dtype=np.int64)
        b = np.asarray([row[1] for row in pairs], dtype=np.int64)
        l1 = np.abs(blend[a] - blend[b]).sum(axis=1)
        if float(np.max(l1, initial=0.0)) > max_l1:
            continue
        if bool(POLICY["require_same_dominant_joint_for_every_pair"]) and np.any(
            dominant[a] != dominant[b]
        ):
            continue

        inserted = 0
        for (av, bv), residual in zip(pairs, l1.tolist()):
            key = (int(av), int(bv))
            reverse = (int(bv), int(av))
            if key in rows or reverse in rows:
                continue
            rows[key] = (
                MATERIAL_CONTINUITY,
                tuple(map(int, domain_pair)),
                0.0,
                5.0 + max(0.0, 1.0 - float(residual)),
            )
            inserted += 1
        if inserted:
            admitted_domain_cuts += 1
            admitted_vertex_pairs += inserted

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
            "admitted_domain_cut_count": int(admitted_domain_cuts),
            "admitted_vertex_pair_count": int(admitted_vertex_pairs),
            "pairs": pairs.astype(int).tolist(),
            "relation_codes": codes.astype(int).tolist(),
            "domain_pairs": domain_pairs.astype(int).tolist(),
        }
    )
    result = QualifiedVisualContactSet(
        view_index=int(parent.view_index),
        pairs=pairs,
        relation_codes=codes,
        domain_pairs=domain_pairs,
        rest_distances_px=distances,
        evidence_scores=scores,
        contact_hash=digest,
    )
    return result


# The relation compiler resolves this hook at execution time; V8's material
# coefficient coupling therefore sees exactly these newly qualified seams.
strict._strict_qualify = _skin_supported_source_cut_qualify


if __name__ == "__main__":
    court.main()
