"""Fast contact-aware safety court over the sealed V6 presentation evidence.

Material continuity is not repaired after independent chart safety projection.
Instead exact source-cut material relations first union body charts into safety
groups, so an unsafe material component receives one shared SE(2) projection.
Articulated relations remain separate pivots and are closed afterwards by a
translation-only graph solve.  This mirrors the intended Stage37 -> Stage42
contract boundary without reopening M/G/W or model inference.
"""

from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_contact_v1 import (
    ARTICULATED_CONTINUITY,
    ATTACHMENT_CONTACT,
    MATERIAL_CONTINUITY,
    QualifiedVisualContactSet,
)
from compiler.realsas_compiler_core.visual_motion_safety_v1 import compile_motion_repair
from tools import replay_relation_court_v2 as strict


court = strict.court
_CONTEXT = None

POLICY = {
    "schema": "RealSaS.ContactAwareSafetyCourt.v1",
    "source_contact": "EXACT_SOURCE_CUT_AFTER_CANONICAL_SKIN_QUALIFICATION",
    "material_continuity": "UNION_DOMAINS_BEFORE_SAFETY_REPAIR",
    "articulated_continuity": "ONE_PIVOT_PER_DOMAIN_PAIR_AFTER_SAFETY",
    "articulated_closure": "MINIMUM_NORM_DOMAIN_TRANSLATION_GRAPH",
    "constraint_target": "COINCIDENT_EXACT_SOURCE_CUT",
    "translation_regularization": 1.0e-10,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _qualified_contacts(**kwargs) -> QualifiedVisualContactSet:
    global _CONTEXT
    value = strict._strict_qualify(**kwargs)
    pairs = np.asarray(value.pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(value.relation_codes, dtype=np.int8)
    domains = np.asarray(value.domain_pairs, dtype=np.int32).reshape(-1, 2)
    distances = np.asarray(value.rest_distances_px, dtype=np.float64)
    scores = np.asarray(value.evidence_scores, dtype=np.float64)

    # Material seams retain every exact-cut witness.  Articulated relations are
    # pivots, not welds: retain one stable/high-evidence pivot per domain pair.
    keep = []
    for relation in (MATERIAL_CONTINUITY, ATTACHMENT_CONTACT):
        keep.extend(np.flatnonzero(codes == relation).tolist())
    articulated = np.flatnonzero(codes == ARTICULATED_CONTINUITY)
    for domain_pair in sorted({tuple(map(int, domains[i])) for i in articulated.tolist()}):
        idx = [
            int(i) for i in articulated.tolist()
            if tuple(map(int, domains[i])) == domain_pair
        ]
        if idx:
            keep.append(min(idx, key=lambda i: (-float(scores[i]), int(pairs[i, 0]), int(pairs[i, 1]))))
    keep = np.asarray(sorted(set(keep)), dtype=np.int64)
    out_pairs = pairs[keep].copy()
    out_codes = codes[keep].copy()
    out_domains = domains[keep].copy()
    out_distances = distances[keep].copy()
    out_scores = scores[keep].copy()
    digest = content_sha256({
        "schema": POLICY["schema"],
        "parent_contact_hash": value.contact_hash,
        "policy_hash": content_sha256(POLICY),
        "pairs": out_pairs.astype(int).tolist(),
        "relation_codes": out_codes.astype(int).tolist(),
        "domain_pairs": out_domains.astype(int).tolist(),
    })
    result = QualifiedVisualContactSet(
        view_index=value.view_index,
        pairs=out_pairs,
        relation_codes=out_codes,
        domain_pairs=out_domains,
        rest_distances_px=out_distances,
        evidence_scores=out_scores,
        contact_hash=digest,
    )
    _CONTEXT = {
        "rest_positions": np.asarray(kwargs["rest_positions"], dtype=np.float64).copy(),
        "visual_faces": np.asarray(kwargs["visual_faces"], dtype=np.int64).copy(),
        "domain_id": np.asarray(kwargs["domain_id"], dtype=np.int32).copy(),
        "vertex_attachment_owner": np.asarray(kwargs["vertex_attachment_owner"], dtype=np.int32).copy(),
        "contacts": result,
    }
    return result


def _material_safety_groups(domain_id, contacts: QualifiedVisualContactSet):
    domains = np.asarray(domain_id, dtype=np.int32)
    unique = sorted(map(int, np.unique(domains)))
    parent = {d: d for d in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    pairs = np.asarray(contacts.domain_pairs, dtype=np.int32).reshape(-1, 2)
    codes = np.asarray(contacts.relation_codes, dtype=np.int8)
    for (a, b), code in zip(pairs.tolist(), codes.tolist()):
        if int(code) == MATERIAL_CONTINUITY:
            union(int(a), int(b))
    roots = {d: find(d) for d in unique}
    root_to_group = {root: i for i, root in enumerate(sorted(set(roots.values())))}
    mapping = {d: root_to_group[roots[d]] for d in unique}
    return np.asarray([mapping[int(d)] for d in domains], dtype=np.int32), mapping


def _translation_close(frame, *, domain_id, pairs):
    result = np.asarray(frame, dtype=np.float64).copy()
    domains = np.asarray(domain_id, dtype=np.int32)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return result, 0.0, 0.0
    active = sorted(set(map(int, domains[pairs].reshape(-1).tolist())))
    index = {d: i for i, d in enumerate(active)}
    matrix = np.zeros((2 * len(pairs), 2 * len(active)), dtype=np.float64)
    target = np.zeros(2 * len(pairs), dtype=np.float64)
    for ri, (a, b) in enumerate(pairs.tolist()):
        da, db = int(domains[a]), int(domains[b])
        ia, ib = index[da], index[db]
        matrix[2 * ri, 2 * ia] = 1.0
        matrix[2 * ri + 1, 2 * ia + 1] = 1.0
        matrix[2 * ri, 2 * ib] = -1.0
        matrix[2 * ri + 1, 2 * ib + 1] = -1.0
        target[2 * ri : 2 * ri + 2] = -(result[a, :2] - result[b, :2])
    ridge = float(POLICY["translation_regularization"])
    augmented = np.vstack((matrix, np.sqrt(ridge) * np.eye(matrix.shape[1])))
    rhs = np.concatenate((target, np.zeros(matrix.shape[1], dtype=np.float64)))
    step, *_ = np.linalg.lstsq(augmented, rhs, rcond=None)
    if not np.isfinite(step).all():
        raise QualificationError("CONTACT_AWARE_SAFETY_TRANSLATION_SOLVE_NONFINITE")
    max_shift = 0.0
    for domain, di in index.items():
        shift = step[2 * di : 2 * di + 2]
        result[domains == domain, :2] += shift
        max_shift = max(max_shift, float(np.linalg.norm(shift)))
    gap = np.linalg.norm(result[pairs[:, 0], :2] - result[pairs[:, 1], :2], axis=1)
    return result, float(np.max(gap, initial=0.0)), max_shift


def _compile_contact_aware_safety(*, domain_id, contact_pairs, baseline_clip_fields, repaired_clip_fields):
    del repaired_clip_fields  # This court deliberately recomputes safety from its pre-safety authority.
    if _CONTEXT is None:
        raise QualificationError("CONTACT_AWARE_SAFETY_CONTEXT_MISSING")
    ctx = _CONTEXT
    original_domains = np.asarray(domain_id, dtype=np.int32)
    if not np.array_equal(original_domains, ctx["domain_id"]):
        raise QualificationError("CONTACT_AWARE_SAFETY_DOMAIN_DRIFT")
    contacts = ctx["contacts"]
    group_domains, original_to_group = _material_safety_groups(original_domains, contacts)

    repaired = compile_motion_repair(
        rest_positions=ctx["rest_positions"],
        visual_faces=ctx["visual_faces"],
        domain_id=group_domains,
        vertex_attachment_owner=ctx["vertex_attachment_owner"],
        clip_fields=baseline_clip_fields,
    )

    pairs = np.asarray(contacts.pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(contacts.relation_codes, dtype=np.int8)
    owners = ctx["vertex_attachment_owner"]
    body = (owners[pairs[:, 0]] == 0) & (owners[pairs[:, 1]] == 0) if len(pairs) else np.zeros(0, bool)
    # Material pairs are already protected by the shared safety group.  A small
    # final graph closure over all exact body contacts removes residual pivot
    # translation while keeping every material component rigid as one unit.
    closure_pairs = pairs[body]

    outputs = {}
    frame_diagnostics = {}
    max_gap = max_shift = 0.0
    for key, field in repaired["repaired_fields"].items():
        rows = []
        frames = []
        for fi in range(len(field)):
            frame, gap, shift = _translation_close(
                field[fi], domain_id=group_domains, pairs=closure_pairs
            )
            frames.append(frame)
            rows.append({
                "maximum_contact_gap_after_px": gap,
                "maximum_group_translation_px": shift,
            })
            max_gap = max(max_gap, gap)
            max_shift = max(max_shift, shift)
        outputs[key] = np.asarray(frames)
        frame_diagnostics[key] = rows
    return {
        "projected_fields": outputs,
        "frame_diagnostics": frame_diagnostics,
        "maximum_contact_delta_residual_after_px": float(max_gap),
        "maximum_domain_translation_px": float(max_shift),
        "material_group_count": int(len(np.unique(group_domains))),
        "material_union_count": int(len(set(original_to_group.values())) < len(original_to_group)),
        "safety_repair_domain_ids": repaired["repair_domain_ids"].astype(int).tolist(),
    }


def _contact_truth_metrics(
    *, rest_positions, baseline_positions, posed_positions, pairs, relation_codes,
    material_tolerance_px=1.0e-6, articulated_tolerance_px=1.5,
):
    rest = np.asarray(rest_positions, dtype=np.float64)
    baseline = np.asarray(baseline_positions, dtype=np.float64)
    posed = np.asarray(posed_positions, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(relation_codes, dtype=np.int8)
    if not len(pairs):
        return {
            "qualified_contact_relations_passed": True,
            "contact_pair_count": 0,
            "contact_failure_count": 0,
            "maximum_contact_repair_delta_residual_px": 0.0,
            "maximum_contact_gap_growth_px": 0.0,
        }
    rest_distance = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1)
    baseline_distance = np.linalg.norm(baseline[pairs[:, 0]] - baseline[pairs[:, 1]], axis=1)
    posed_distance = np.linalg.norm(posed[pairs[:, 0]] - posed[pairs[:, 1]], axis=1)
    repair_delta = np.abs(posed_distance - baseline_distance)
    gap_growth = np.maximum(posed_distance - rest_distance, 0.0)
    tolerance = np.where(codes == MATERIAL_CONTINUITY, float(material_tolerance_px), float(articulated_tolerance_px))
    failure = gap_growth > tolerance
    return {
        "qualified_contact_relations_passed": bool(not np.any(failure)),
        "contact_pair_count": int(len(pairs)),
        "contact_failure_count": int(np.count_nonzero(failure)),
        "maximum_contact_repair_delta_residual_px": float(np.max(repair_delta, initial=0.0)),
        "maximum_contact_gap_growth_px": float(np.max(gap_growth, initial=0.0)),
    }


court.qualify_visual_contacts = _qualified_contacts
court.compile_contact_projection = _compile_contact_aware_safety
court.contact_relation_metrics = _contact_truth_metrics


if __name__ == "__main__":
    court.main()
