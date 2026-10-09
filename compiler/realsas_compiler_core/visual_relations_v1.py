"""Qualified source-art contact and region draw-order relations.

This module turns existing compiler evidence into presentation relations without
reopening M/G/W. Contacts are conservative: exact source-cut coincidence is only
a candidate and is admitted only when canonical support and ownership agree.
Draw order is region-level and temporal; canonical depth is evidence inside the
compiler, while the runtime consumes the resulting region permutation.
"""
from __future__ import annotations

import numpy as np

from .hashing import content_sha256
from .types import QualificationError

CONTACT_SCHEMA = "RealSaS.QualifiedVisualContactSet.v1"
COMPOSITION_SCHEMA = "RealSaS.QualifiedRegionComposition.v1"
COMPOSITION_OPERATOR_ID = "QUALIFIED_PRESENTATION_REGION_ORDER_V1"
POLICY = {
    "schema": "RealSaS.VisualRelationsPolicy.v1",
    "contact_candidate": "EXACT_SOURCE_CUT_COORDINATE_ACROSS_DISTINCT_VISUAL_DOMAINS",
    "contact_admission": "SAME_PRESENTATION_REGION_OR_CANONICAL_SUPPORT_VERTEX_ADJACENCY__UNIFORM_OWNER",
    "contact_runtime_constraint": "CONTACT_LINKED_BODY_DOMAINS_SHARE_ONE_SE2_REPAIR_GROUP",
    "draw_order_scope": "INTER_REGION",
    "same_region_occlusion": "CANONICAL_DEPTH",
    "inter_region_occlusion": "COMPILED_REGION_ORDER",
    "dynamic_order_evidence": "ROBUST_REGION_MEDIAN_CANONICAL_DEPTH",
    "near_equal_order": "SETUP_ORDER_STABLE_TIEBREAK",
    "mechanical_state_mutation_authorized": False,
    "runtime_relation_inference_authorized": False,
}


def _visual_domains(vertex_count: int, faces, vertex_region_id) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    region = np.asarray(vertex_region_id, dtype=np.int32)
    if (vertex_count <= 0 or region.shape != (vertex_count,) or f.ndim != 2
            or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= vertex_count)
            or np.any(region < 0) or np.any(region[f] != region[f[:, :1]])):
        raise QualificationError("VISUAL_RELATIONS_DOMAIN_INPUT_INVALID")
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for a, b, c in f.tolist():
        union(a, b); union(b, c); union(c, a)
    roots = np.asarray([find(i) for i in range(vertex_count)], dtype=np.int64)
    mapping = {int(root): i for i, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([mapping[int(root)] for root in roots], dtype=np.int32)


def _mechanical_face_owners(mechanical_faces, mechanical_vertex_owner) -> np.ndarray:
    faces = np.asarray(mechanical_faces, dtype=np.int64)
    owner = np.asarray(mechanical_vertex_owner, dtype=np.int32)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(owner)) or np.any(owner < 0)):
        raise QualificationError("VISUAL_RELATIONS_MECHANICAL_OWNER_INPUT_INVALID")
    rows = owner[faces]
    if np.any(rows != rows[:, :1]):
        raise QualificationError("VISUAL_RELATIONS_MECHANICAL_FACE_OWNER_AMBIGUOUS")
    return rows[:, 0]


def qualify_visual_contacts(*, rest_positions, visual_faces, vertex_region_id,
                            seed_region_labels, owner_face_index,
                            mechanical_faces, mechanical_vertex_owner) -> dict:
    """Conservatively qualify chart contacts from source and canonical evidence.

    Exact source-coordinate coincidence never qualifies a contact by itself.
    Candidate vertices must belong to distinct visual domains, resolve to one
    mechanical owner, and either belong to the same presentation region or have
    canonically adjacent support (their visible support touches a common
    mechanical vertex). Unsupported/ambiguous candidates remain abstentions.
    """
    rest = np.asarray(rest_positions, dtype=np.float64)
    faces = np.asarray(visual_faces, dtype=np.int64)
    regions = np.asarray(vertex_region_id, dtype=np.int32)
    seeds = np.asarray(seed_region_labels, dtype=np.int32)
    owners = np.asarray(owner_face_index, dtype=np.int64)
    mechanical = np.asarray(mechanical_faces, dtype=np.int64)
    if (rest.ndim != 2 or rest.shape[1] != 2 or not np.isfinite(rest).all()
            or regions.shape != (len(rest),) or seeds.shape != owners.shape
            or seeds.ndim != 2 or np.any(regions < 0)):
        raise QualificationError("VISUAL_CONTACT_QUALIFICATION_INPUT_INVALID")
    domain = _visual_domains(len(rest), faces, regions)
    face_owner = _mechanical_face_owners(mechanical, mechanical_vertex_owner)

    support_faces = {}
    support_vertices = {}
    region_owner = {}
    for region in sorted(set(map(int, regions.tolist()))):
        mask = (seeds == region) & (owners >= 0) & (owners < len(mechanical))
        rows = np.unique(owners[mask]).astype(np.int64)
        support_faces[region] = rows
        if len(rows):
            o = np.unique(face_owner[rows])
            region_owner[region] = int(o[0]) if len(o) == 1 else None
            support_vertices[region] = set(map(int, np.unique(mechanical[rows]).tolist()))
        else:
            region_owner[region] = None
            support_vertices[region] = set()

    _, inverse = np.unique(rest, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    candidates = []
    relations = []
    seen = set()
    for group in np.split(order, splits):
        if len(group) < 2:
            continue
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = int(group[ii]), int(group[jj])
                if int(domain[a]) == int(domain[b]):
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                ra, rb = int(regions[a]), int(regions[b])
                oa, ob = region_owner.get(ra), region_owner.get(rb)
                candidates.append(key)
                evidence = ["EXACT_SOURCE_COORDINATE", "DISTINCT_VISUAL_DOMAIN"]
                if oa is None or ob is None or oa != ob:
                    continue
                evidence.append("UNIFORM_SHARED_MECHANICAL_OWNER")
                same_region = ra == rb
                canonical_adjacent = bool(support_vertices[ra] & support_vertices[rb])
                if not same_region and not canonical_adjacent:
                    continue
                evidence.append("SAME_PRESENTATION_REGION" if same_region
                                else "CANONICAL_SUPPORT_VERTEX_ADJACENCY")
                relations.append({
                    "vertex_a": key[0], "vertex_b": key[1],
                    "domain_a": int(domain[key[0]]), "domain_b": int(domain[key[1]]),
                    "region_a": int(regions[key[0]]), "region_b": int(regions[key[1]]),
                    "mechanical_owner": int(oa),
                    "relation": "MATERIAL_CONTINUITY",
                    "evidence": evidence,
                })
    payload = {
        "schema": CONTACT_SCHEMA,
        "policy_hash": content_sha256(POLICY),
        "candidate_pair_count": int(len(candidates)),
        "qualified_pair_count": int(len(relations)),
        "abstained_pair_count": int(len(candidates) - len(relations)),
        "relations": relations,
        "qualification": "CANONICAL_AND_SOURCE_EVIDENCE__NO_SOURCE_COINCIDENCE_ONLY",
        "product_authority": False,
    }
    payload["contact_hash"] = content_sha256(payload)
    return payload


def validate_visual_contacts(value: dict, *, vertex_count: int) -> None:
    if value.get("schema") != CONTACT_SCHEMA or value.get("policy_hash") != content_sha256(POLICY):
        raise QualificationError("VISUAL_CONTACT_SCHEMA_OR_POLICY_DRIFT")
    relations = list(value.get("relations") or ())
    pairs = []
    for row in relations:
        a, b = int(row["vertex_a"]), int(row["vertex_b"])
        if not (0 <= a < vertex_count and 0 <= b < vertex_count and a < b
                and row.get("relation") == "MATERIAL_CONTINUITY"):
            raise QualificationError("VISUAL_CONTACT_RELATION_INVALID")
        pairs.append((a, b))
    if len(pairs) != len(set(pairs)) or int(value.get("qualified_pair_count", -1)) != len(pairs):
        raise QualificationError("VISUAL_CONTACT_RELATION_DUPLICATE_OR_COUNT_DRIFT")
    digest = dict(value); digest.pop("contact_hash", None)
    if value.get("contact_hash") != content_sha256(digest):
        raise QualificationError("VISUAL_CONTACT_HASH_DRIFT")


def contact_pairs_array(value: dict, *, vertex_count: int) -> np.ndarray:
    validate_visual_contacts(value, vertex_count=vertex_count)
    return np.asarray([(int(r["vertex_a"]), int(r["vertex_b"]))
                       for r in value["relations"]], dtype=np.int64).reshape(-1, 2)


def compile_setup_region_order(*, rest_depths, visual_faces, face_region_id) -> dict:
    z = np.asarray(rest_depths, dtype=np.float64)
    faces = np.asarray(visual_faces, dtype=np.int64)
    region = np.asarray(face_region_id, dtype=np.int32)
    if (z.ndim != 1 or not np.isfinite(z).all() or np.any(z <= 0)
            or faces.ndim != 2 or faces.shape[1] != 3 or region.shape != (len(faces),)
            or np.any(faces < 0) or np.any(faces >= len(z)) or np.any(region < 0)):
        raise QualificationError("VISUAL_COMPOSITION_SETUP_INPUT_INVALID")
    ids = np.unique(region)
    if not np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype)):
        raise QualificationError("VISUAL_COMPOSITION_REGION_IDS_MUST_BE_DENSE")
    face_depth = z[faces].mean(axis=1)
    medians = np.asarray([np.median(face_depth[region == rid]) for rid in ids], dtype=np.float64)
    order = np.lexsort((ids, medians)).astype(np.int32)
    payload = {
        "schema": COMPOSITION_SCHEMA,
        "operator_id": COMPOSITION_OPERATOR_ID,
        "policy_hash": content_sha256(POLICY),
        "region_count": int(len(ids)),
        "setup_order_front_to_back": [int(x) for x in order.tolist()],
        "setup_region_median_depth": [float(x) for x in medians.tolist()],
        "same_region_occlusion": "CANONICAL_DEPTH",
        "inter_region_occlusion": "COMPILED_REGION_ORDER",
        "product_authority": False,
    }
    payload["composition_hash"] = content_sha256(payload)
    return payload


def validate_setup_region_order(value: dict, *, region_count: int) -> None:
    if (value.get("schema") != COMPOSITION_SCHEMA
            or value.get("operator_id") != COMPOSITION_OPERATOR_ID
            or value.get("policy_hash") != content_sha256(POLICY)
            or int(value.get("region_count", -1)) != int(region_count)):
        raise QualificationError("VISUAL_COMPOSITION_CONTRACT_DRIFT")
    order = tuple(map(int, value.get("setup_order_front_to_back") or ()))
    if len(order) != region_count or set(order) != set(range(region_count)):
        raise QualificationError("VISUAL_COMPOSITION_SETUP_ORDER_INVALID")
    digest = dict(value); digest.pop("composition_hash", None)
    if value.get("composition_hash") != content_sha256(digest):
        raise QualificationError("VISUAL_COMPOSITION_HASH_DRIFT")


def compile_region_order_timeline(*, depths, visual_faces, face_region_id,
                                  setup_composition: dict) -> np.ndarray:
    z = np.asarray(depths, dtype=np.float64)
    faces = np.asarray(visual_faces, dtype=np.int64)
    region = np.asarray(face_region_id, dtype=np.int32)
    if (z.ndim != 2 or not len(z) or not np.isfinite(z).all() or np.any(z <= 0)
            or faces.ndim != 2 or faces.shape[1] != 3 or region.shape != (len(faces),)
            or np.any(faces < 0) or np.any(faces >= z.shape[1])):
        raise QualificationError("VISUAL_COMPOSITION_TIMELINE_INPUT_INVALID")
    region_count = int(np.max(region, initial=-1)) + 1
    validate_setup_region_order(setup_composition, region_count=region_count)
    setup = list(map(int, setup_composition["setup_order_front_to_back"]))
    setup_rank = np.empty(region_count, dtype=np.int32)
    setup_rank[np.asarray(setup, dtype=np.int32)] = np.arange(region_count, dtype=np.int32)
    out = np.empty((len(z), region_count), dtype=np.int32)
    for fi, row in enumerate(z):
        face_depth = row[faces].mean(axis=1)
        median = np.asarray([np.median(face_depth[region == rid])
                             for rid in range(region_count)], dtype=np.float64)
        span = float(np.max(median) - np.min(median))
        eps = max(1e-9, span * 1e-6)
        base = np.argsort(median, kind="stable")
        ordered = []
        start = 0
        while start < len(base):
            end = start + 1
            while end < len(base) and abs(float(median[base[end]] - median[base[end - 1]])) <= eps:
                end += 1
            group = list(map(int, base[start:end]))
            group.sort(key=lambda rid: (int(setup_rank[rid]), rid))
            ordered.extend(group)
            start = end
        rank = np.empty(region_count, dtype=np.int32)
        rank[np.asarray(ordered, dtype=np.int32)] = np.arange(region_count, dtype=np.int32)
        out[fi] = rank
    validate_region_order_timeline(out, region_count=region_count)
    return out


def validate_region_order_timeline(value, *, region_count: int) -> None:
    rows = np.asarray(value)
    if rows.ndim != 2 or rows.shape[1] != region_count or rows.dtype.kind not in "iu":
        raise QualificationError("VISUAL_COMPOSITION_TIMELINE_SHAPE_INVALID")
    expected = np.arange(region_count)
    if any(not np.array_equal(np.sort(row), expected) for row in rows):
        raise QualificationError("VISUAL_COMPOSITION_TIMELINE_NOT_PERMUTATION")
