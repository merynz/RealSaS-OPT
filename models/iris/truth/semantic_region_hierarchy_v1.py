"""Upgrade StdGEN-style source-name hints from triangle islands to region hierarchy.

The V1 connected-island sweep is source evidence, NOT semantic instance truth.
For each source object/material domain, this annotator constructs one parent
teacher *region candidate* with exact union of child source-face addresses.
A parent is not automatically one drawable, one joint, one control, or even
one semantic component. Source-material absence stays explicitly UNKNOWN.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

GROUP_SCHEMA = "RealSaS.Corpus370SemanticSourceGroupCandidates.v1"
SCHEMA = "RealSaS.Corpus370SemanticRegionHierarchyCandidate.v1"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("CORPUS370_SEMANTIC_REGION_HIERARCHY_" + code)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build(*, source_groups: dict, semantic_policy: dict) -> dict:
    _require(source_groups.get("schema") == GROUP_SCHEMA and
             source_groups.get("training_authorized") is False and
             source_groups.get("class_labels_admitted") is False and
             source_groups.get("semantic_review_required") is True,
             "INVALID_V1_PARENT")
    _require(semantic_policy.get("schema") == "RealSaS.SemanticPriorPolicy.v1"
             and semantic_policy.get("authority") == "DESIGNED_PRIOR__NOT_SOURCE_PHYSICS",
             "INVALID_DESIGNED_PRIOR")
    sid = source_groups["subject_id"]
    sha = source_groups["source_sha256"]
    face_count = source_groups["source_face_count"]
    _require(type(face_count) is int and face_count > 0, "FACE_COUNT")
    grouped = defaultdict(list)
    covered: set[int] = set()
    for child in source_groups["groups"]:
        owner = child.get("source_owner_slot")
        material = child.get("source_material_slot")
        face_ids = child.get("source_face_ids")
        _require(type(owner) is int and owner >= 0 and
                 type(material) is int and material >= 0,
                 "OWNER_MATERIAL_SLOT")
        _require(isinstance(face_ids, list) and bool(face_ids)
                 and all(type(f) is int and 0 <= f < face_count for f in face_ids),
                 "ISLAND_FACE_SUPPORT")
        _require(len(face_ids) == len(set(face_ids))
                 and not covered.intersection(face_ids), "ISLAND_FACE_DISJOINT")
        covered.update(face_ids)
        grouped[(owner, material)].append(child)
    _require(len(covered) == face_count
             and min(covered) == 0 and max(covered) == face_count - 1,
             "ALL_SOURCE_FACES_MUST_BE_ACCOUNTED")
    parents = []
    counts = {"ONE_UNVERIFIED_HINT": 0, "CONFLICTING_HINTS": 0, "NO_HINT": 0}
    for (owner, material), children in sorted(grouped.items()):
        child_names = [str(x["privileged_object_name"]) for x in children]
        mat_names = [x["privileged_material_name"] for x in children]
        _require(len(set(child_names)) == 1, "OWNER_NAME_DRIFT")
        _require(len(set(mat_names)) == 1, "MATERIAL_NAME_DRIFT")
        unique_labels = {str(x["suggested_concept"]) for x in children}
        unique_status = {str(x["hint_status"]) for x in children}
        if len(unique_labels) == 1 and len(unique_status) == 1:
            status = next(iter(unique_status))
            concept = next(iter(unique_labels))
        else:
            status = "CONFLICTING_HINTS"
            concept = "UNKNOWN"
        _require(status in counts and concept in semantic_policy["concepts"],
                 "INVALID_NAME_HINT")
        if status != "ONE_UNVERIFIED_HINT":
            concept = "UNKNOWN"
        counts[status] += 1
        source_faces = sorted(face for item in children for face in item["source_face_ids"])
        parent_key = _hash({"source": sha, "owner": owner, "material": material,
                            "face_support": source_faces})
        parents.append({
            "teacher_region_ref": parent_key,
            "source_owner_slot": owner,
            "source_material_slot": material,
            "privileged_object_name": child_names[0],
            "privileged_material_name": mat_names[0],
            "source_face_ids": source_faces,
            "child_island_ids": sorted(x["teacher_group_id"] for x in children),
            "child_island_count": len(children),
            "semantic_concept_hint": concept,
            "hint_status": status,
            "functional_affordance_prior_hint": dict(
                semantic_policy["concepts"][concept]["prior"]),
            "region_kind": "SOURCE_OWNER_MATERIAL_DOMAIN_NOT_SEMANTIC_INSTANCE",
            "requires_independent_semantic_court": True,
            "requires_view_render_court": True,
            "training_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "subject_id": sid,
        "source_sha256": sha,
        "parent_source_group_digest": _hash(source_groups),
        "semantic_prior_policy_digest": _hash(semantic_policy),
        "source_face_count": face_count,
        "child_island_count": len(source_groups["groups"]),
        "parent_region_count": len(parents),
        "hint_status_counts_per_parent_region": counts,
        "regions": parents,
        "source_geometry_identity_preserved": True,
        "semantic_class_court_passed": False,
        "component_identity_court_passed": False,
        "final_drawable_authority": False,
        "rig_skin_control_authority": False,
        "training_authorized": False,
        "status": "HIERARCHICAL_SEMANTIC_TEACHER_REGION_CANDIDATES__NO_ADMISSION",
    }
