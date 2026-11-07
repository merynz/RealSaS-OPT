"""Exact source-triangle semantic TEACHER candidate generator for Corpus370.

StdGEN-style source naming is intentionally used ONLY as a teacher annotation
hint. Stable source-mesh surface groups are measured before any class is
assigned; the learned semantic prior and actual attachment/drawable authority
are independent questions. No output produced here is a training target.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np

SCHEMA = "RealSaS.Corpus370SemanticSourceGroupCandidates.v1"
HINT_SCHEMA = "RealSaS.SemanticSourceHintPolicy.v1"


def _require(value: bool, code: str) -> None:
    if not value:
        raise ValueError("SEMANTIC_SOURCE_GROUP_V1_" + code)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tokens(value: str) -> set[str]:
    # Source asset naming is arbitrary. Tokenization is only a heuristic:
    # lowercase, separators, and PascalCase. Unmatched remains UNKNOWN.
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    split = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", split)
    return set(re.findall(r"[a-z]+", split.lower()))


def validate_hint_policy(policy: dict, *, concepts: set[str]) -> None:
    _require(isinstance(policy, dict) and policy.get("schema") == HINT_SCHEMA, "HINT_SCHEMA")
    _require(policy.get("authority") == "TEACHER_HINT_ONLY__INDEPENDENT_REVIEW_REQUIRED",
             "HINT_AUTHORITY")
    _require(policy.get("training_authorized") is False and policy.get("default_state") == "UNKNOWN",
             "HINT_MUST_ABSTAIN")
    rules = policy.get("rules")
    _require(isinstance(rules, list) and bool(rules), "HINT_RULES")
    used = set()
    for rule in rules:
        _require(isinstance(rule, dict) and rule.get("concept") in concepts, "HINT_CONCEPT")
        _require(rule["concept"] != "UNKNOWN", "HINT_UNKNOWN_RULE")
        terms = rule.get("tokens")
        _require(isinstance(terms, list) and bool(terms) and all(isinstance(t, str) and
                 re.fullmatch("[a-z]+", t) for t in terms), "HINT_TOKENS")
        for token in terms:
            _require(token not in used, "DUPLICATE_HINT_TOKEN")
            used.add(token)


def _hint(object_name: str, material_name: str | None, policy: dict) -> dict:
    obj_t = _tokens(object_name)
    mat_t = _tokens(material_name or "")
    evidence = []
    for rule in policy["rules"]:
        matched = sorted(set(rule["tokens"]) & (obj_t | mat_t))
        if matched:
            evidence.append({"concept": rule["concept"], "tokens": matched,
                             "source_fields": sorted(
                                 ["object" if obj_t.intersection(matched) else "",
                                  "material" if mat_t.intersection(matched) else ""]
                             )})
    # Remove empty fields; no raw name or teacher ID becomes a student feature.
    for row in evidence:
        row["source_fields"] = [f for f in row["source_fields"] if f]
    labels = {row["concept"] for row in evidence}
    return {
        "suggested_concept": next(iter(labels)) if len(labels) == 1 else "UNKNOWN",
        "hint_status": "ONE_UNVERIFIED_HINT" if len(labels) == 1 else (
            "CONFLICTING_HINTS" if len(labels) > 1 else "NO_HINT"),
        "hint_evidence": evidence,
        "semantic_supervision_admitted": False,
    }


def build_semantic_source_groups(*, subject_id: str, source_sha256: str,
                                 snapshot: dict, hint_policy: dict,
                                 semantic_prior_policy: dict) -> dict:
    """Build deterministic connected source domains and weak class hints.

    Input snapshot is the exact rest-state output from
    blender_authored_teacher_export_v3._extract_snapshot. The invoking Blender
    producer must FIRST compare its vertices/faces with immutable qualified
    source geometry. A domain is a proposed annotation unit, never proof that
    the original artist or a puppet renderer needs that drawable.
    """
    from models.iris.truth.semantic_prior_contract_v1 import validate_policy
    validate_policy(semantic_prior_policy)
    validate_hint_policy(hint_policy, concepts=set(semantic_prior_policy["concepts"]))
    _require(isinstance(subject_id, str) and bool(subject_id), "SUBJECT_ID")
    _require(isinstance(source_sha256, str) and
             re.fullmatch("[0-9a-f]{64}", source_sha256) is not None, "SOURCE_SHA")
    v = np.asarray(snapshot["vertices"], dtype=np.float64)
    f = np.asarray(snapshot["faces"], dtype=np.int64)
    owner = np.asarray(snapshot["face_owner"], dtype=np.int64)
    material = np.asarray(snapshot["face_material"], dtype=np.int64)
    owners = snapshot["owners"]
    _require(v.ndim == 2 and v.shape[1:] == (3,) and len(v) > 0 and
             np.isfinite(v).all(), "VERTEX_LAYOUT")
    _require(f.ndim == 2 and f.shape[1:] == (3,) and len(f) > 0 and
             np.all(f >= 0) and np.all(f < len(v)), "FACE_LAYOUT")
    _require(owner.shape == (len(f),) and material.shape == (len(f),),
             "DOMAIN_LAYOUT")
    _require(isinstance(owners, list) and bool(owners) and
             np.all(owner >= 0) and np.all(owner < len(owners)), "OWNER_LAYOUT")
    for i, row in enumerate(owners):
        _require(isinstance(row, dict) and row.get("anonymous_owner_index") == i and
                 isinstance(row.get("source_import_name"), str) and
                 isinstance(row.get("material_slot_names"), list), "OWNER_ROW")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (oi, mi) in enumerate(zip(owner, material)):
        slots = owners[int(oi)]["material_slot_names"]
        # An unmaterialized mesh can validly have zero slots and a default 0.
        _require(0 <= mi < len(slots) or (len(slots) == 0 and mi == 0),
                 "MATERIAL_SLOT_OUT_OF_RANGE")
        groups[(int(oi), int(mi))].append(fi)

    proposals = []
    occupied = set()
    for (oi, mi), face_ids in sorted(groups.items()):
        parent = {fi: fi for fi in face_ids}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            x, y = find(a), find(b)
            if x != y:
                parent[max(x, y)] = min(x, y)

        seen_edges: dict[tuple[int, int], int] = {}
        for fi in face_ids:
            a, b, c = (int(x) for x in f[fi])
            for u, w in ((a, b), (b, c), (c, a)):
                e = (min(u, w), max(u, w))
                if e in seen_edges:
                    union(fi, seen_edges[e])
                else:
                    seen_edges[e] = fi

        components: dict[int, list[int]] = defaultdict(list)
        for fi in face_ids:
            components[find(fi)].append(fi)
        obj = owners[oi]
        slots = obj["material_slot_names"]
        material_name = slots[mi] if slots else None
        for component_faces in sorted(components.values(), key=min):
            support = sorted(component_faces)
            _require(not occupied.intersection(support), "FACE_DUPLICATION")
            occupied.update(support)
            proposal_key = _digest({"source": source_sha256, "owner": oi, "material": mi,
                                    "support": support})
            proposed = _hint(obj["source_import_name"], material_name, hint_policy)
            proposals.append({
                "teacher_group_id": proposal_key,
                "source_face_ids": support,
                "source_owner_slot": oi,
                "source_material_slot": mi,
                "privileged_object_name": obj["source_import_name"],
                "privileged_material_name": material_name,
                **proposed,
                "annotation_required": True,
                "group_kind": "CONNECTED_SOURCE_DOMAIN__NOT_COMPONENT_IDENTITY",
                "training_authorized": False,
            })
    _require(len(occupied) == len(f), "FULL_FACE_COVERAGE")
    proposals.sort(key=lambda r: min(r["source_face_ids"]))
    return {
        "schema": SCHEMA,
        "status": "MEASURED_SOURCE_GROUPS_WITH_UNVERIFIED_SEMANTIC_HINTS",
        "subject_id": subject_id,
        "source_sha256": source_sha256,
        "source_face_count": len(f),
        "source_vertex_count": len(v),
        "group_count": len(proposals),
        "concept_hint_counts": {
            k: sum(p["suggested_concept"] == k for p in proposals)
            for k in sorted(semantic_prior_policy["concepts"])
        },
        "semantic_prior_policy_sha256": _digest(semantic_prior_policy),
        "hint_policy_sha256": _digest(hint_policy),
        "groups": proposals,
        "source_names_teacher_only": True,
        "semantic_review_required": True,
        "class_labels_admitted": False,
        "functional_priors_admitted": False,
        "graphics_drawable_authority": False,
        "axis_rig_program_authority": False,
        "training_authorized": False,
    }
