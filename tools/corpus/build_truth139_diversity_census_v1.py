#!/usr/bin/env python3
"""Measure how much independent structural diversity the historical Truth139 actually contains.

A subject count is not an independence count. This census builds leakage-connected components from
anonymous source-family, exact skeleton/hierarchy, topology-lineage and material-lineage tokens,
then reports signature multiplicities and raw structural distributions without semantic labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

AUDIT_SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"
AUDIT_STATUS = "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
SCHEMA = "RealSaS.Truth139DiversityCensus.v1"
LEAKAGE_FIELDS = (
    "source_family_signature",
    "skeleton_hierarchy_signature",
    "topology_lineage_signature",
    "material_lineage_signature",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _tokens(row: dict, field: str) -> list[str]:
    return sorted({str(x) for x in ((row.get("lineage") or {}).get(field) or ()) if str(x)})


def _numeric_summary(subjects: list[dict], key: str) -> dict:
    values = [int((row.get("raw_structure") or {}).get(key, 0) or 0) for row in subjects]
    values = [x for x in values if x > 0]
    if not values:
        return {"observed_count": 0}
    ordered = sorted(values)
    def q(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]
    return {
        "observed_count": len(ordered),
        "min": ordered[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": ordered[-1],
        "sum": int(sum(ordered)),
    }


def build(audit: dict) -> dict:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != AUDIT_STATUS:
        raise ValueError("TRUTH139_DIVERSITY_AUDIT_NOT_PASS")
    subjects = sorted(list(audit.get("subjects") or ()), key=lambda row: str(row.get("subject_id") or ""))
    if len(subjects) != 139:
        raise ValueError("TRUTH139_DIVERSITY_SUBJECT_COUNT")
    ids = [str(row.get("subject_id") or "") for row in subjects]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("TRUTH139_DIVERSITY_SUBJECT_ID")

    dsu = DSU(len(subjects))
    token_owners: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(subjects):
        for field in LEAKAGE_FIELDS:
            for token in _tokens(row, field):
                token_owners[(field, token)].append(i)
    for owners in token_owners.values():
        if len(owners) > 1:
            root = owners[0]
            for other in owners[1:]:
                dsu.union(root, other)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(subjects)):
        members[dsu.find(i)].append(i)

    components = []
    for ordinal, indices in enumerate(sorted(members.values(), key=lambda x: (-len(x), [ids[i] for i in x])), 1):
        rows = [subjects[i] for i in indices]
        packs = Counter(str((row.get("source") or {}).get("pack_id") or "") for row in rows)
        token_counts = {
            field: len({token for row in rows for token in _tokens(row, field)})
            for field in LEAKAGE_FIELDS
        }
        clips = sum(int((row.get("raw_structure") or {}).get("animation_count", 0) or 0) for row in rows)
        components.append({
            "component_ordinal": ordinal,
            "size": len(rows),
            "subject_ids": [str(row["subject_id"]) for row in rows],
            "pack_counts": dict(sorted(packs.items())),
            "cross_pack": len(packs) > 1,
            "token_cardinality": token_counts,
            "animation_clip_count": int(clips),
        })

    multiplicity = {}
    signature_groups = {}
    for field in LEAKAGE_FIELDS:
        counts = Counter()
        owners = defaultdict(list)
        for row in subjects:
            for token in _tokens(row, field):
                counts[token] += 1
                owners[token].append(str(row["subject_id"]))
        multiplicity[field] = {
            "unique_token_count": len(counts),
            "singleton_token_count": sum(v == 1 for v in counts.values()),
            "shared_token_count": sum(v > 1 for v in counts.values()),
            "maximum_subjects_per_token": max(counts.values(), default=0),
        }
        signature_groups[field] = [
            {"token": token, "subject_count": count, "subject_ids": sorted(owners[token])}
            for token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            if count > 1
        ]

    pack_counts = Counter(str((row.get("source") or {}).get("pack_id") or "") for row in subjects)
    component_sizes = sorted((len(v) for v in members.values()), reverse=True)
    size_histogram = Counter(component_sizes)
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_TRUTH139_STRUCTURAL_DIVERSITY_CENSUS",
        "source_audit_identity_sha256": str(audit.get("audit_identity_sha256") or ""),
        "subject_count": len(subjects),
        "pack_counts": dict(sorted(pack_counts.items())),
        "leakage_fields": list(LEAKAGE_FIELDS),
        "leakage_component_summary": {
            "component_count": len(components),
            "singleton_component_count": sum(size == 1 for size in component_sizes),
            "cross_pack_component_count": sum(bool(row["cross_pack"]) for row in components),
            "largest_component_size": max(component_sizes, default=0),
            "largest_10_component_sizes": component_sizes[:10],
            "component_size_histogram": {str(k): v for k, v in sorted(size_histogram.items())},
            "subject_count_divided_by_component_count": (len(subjects) / len(components)) if components else None,
        },
        "signature_multiplicity": multiplicity,
        "largest_shared_signature_groups": signature_groups,
        "raw_structure_summary": {
            key: _numeric_summary(subjects, key)
            for key in (
                "animation_count",
                "joint_count",
                "skin_count",
                "vertex_count",
                "face_count",
                "triangle_position_count_total",
                "triangle_face_count_total",
                "skinned_vertex_count",
                "maximum_influences_per_vertex",
            )
        },
        "components": components,
        "interpretation_policy": {
            "subject_count_is_not_independence_count": True,
            "leakage_components_are_split_atomic": True,
            "pack_is_not_a_safe_split_unit": True,
            "semantic_morphology_labels_inferred": False,
            "visual_morphology_census_requires_render_or_explicit_metadata": True,
            "this_census_is_structural_not_model_qualification": True,
        },
        "training_authorized": False,
        "split_authorized": False,
        "qualification_minted": False,
    }
    payload["census_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).resolve().read_text())
    result = build(audit)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "subject_count": result["subject_count"],
        "leakage_component_summary": result["leakage_component_summary"],
        "signature_multiplicity": result["signature_multiplicity"],
        "census_identity_sha256": result["census_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
