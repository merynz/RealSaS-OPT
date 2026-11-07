#!/usr/bin/env python3
"""Source-exact, non-semantic partition edit tasks for Corpus370 v7.

Producer and independent court independently load SHA-bound original proposal
and two-view source-hit receipts. This module produces only PRIVILEGED teacher
test addresses; no drawable identity or student training targets.
"""
from __future__ import annotations

from collections import Counter

SCHEMA = "RealSaS.Corpus370PartitionEditPlan.v7"
PROPOSAL = "RealSaS.Corpus370AllViewGeometryOperationalGroups.v3"
SIZES = (8, 16, 32, 64)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def build_plan(proposal, pair_receipt, *, min_distinct_tasks=2):
    require(proposal.get("schema") == PROPOSAL
            and proposal.get("status") ==
                "ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT"
            and proposal.get("training_authorized") is False
            and proposal.get("graphics_owner_truth_minted") is False
            and proposal.get("view4_rendered_edit_consequences_used_by_compiler") is False,
            "PARTITION_ORIGINAL_PROPOSAL_SCOPE")
    source_sha = proposal.get("source_sha256")
    require(isinstance(source_sha, str) and len(source_sha) == 64
            and pair_receipt.get("parents", {}).get("source_sha256") == source_sha
            and pair_receipt.get("training_authorized") is False,
            "PARTITION_SOURCE_HIT_PARENT_SHA_SCOPE")
    face_count = proposal.get("source_face_count")
    require(type(face_count) is int and face_count > 8, "PARTITION_FACE_COUNT")
    candidates = {}
    for row in proposal.get("candidates") or []:
        size = row.get("slot_count")
        if size not in SIZES:
            continue
        require(type(size) is int and size not in candidates
                and row.get("candidate") == f"ALLVIEW_GEOMETRY_GROUPS_{size}"
                and row.get("training_authorized") is False
                and row.get("source_faces_changed") is False,
                "PARTITION_CANDIDATE_AUTHORITY")
        labels = row.get("face_slot_teacher_only")
        require(isinstance(labels, list) and len(labels) == face_count
                and all(type(v) is int and 0 <= v < size for v in labels)
                and set(labels) == set(range(size)),
                "PARTITION_CANDIDATE_COMPLETE_LABELS")
        candidates[size] = labels
    sizes = sorted(candidates)
    require(8 in candidates and len(sizes) >= 2,
            "PARTITION_ALTERNATIVE_SOURCE_CANDIDATES_MISSING")
    # Test label-invariant refinement, not numeric slot equality.
    for a, b in zip(sizes, sizes[1:]):
        fine_to_coarse = {}
        for coarse, fine in zip(candidates[a], candidates[b]):
            previous = fine_to_coarse.setdefault(fine, coarse)
            require(previous == coarse, "PARTITION_NOT_NESTED_SOURCE_FACE_GROUPS")
    rows = pair_receipt.get("rows")
    require(isinstance(rows, list) and len(rows) == 2
            and {row.get("view_index") for row in rows} == {0, 4},
            "PARTITION_ORIGINAL_TWO_VIEW_PAIR")
    # Deterministic, author-name-free source loci. No view4 render response read.
    source_faces = []
    for row in sorted(rows, key=lambda x: x["view_index"]):
        for key in ("source_face_a_teacher_only", "source_face_b_teacher_only"):
            face = row.get(key)
            require(type(face) is int and 0 <= face < face_count,
                    "PARTITION_TASK_SOURCE_FACE_INVALID")
            if face not in source_faces:
                source_faces.append(face)
    fine = sizes[-1]
    # Exclude duplicate reference patches; do not call duplicates negative truth.
    selected = []
    seen_finest = set()
    for face in source_faces:
        fid = candidates[fine][face]
        if fid not in seen_finest:
            seen_finest.add(fid)
            selected.append(face)
    require(min_distinct_tasks <= len(selected) <= 4,
            "PARTITION_INSUFFICIENT_DISTINCT_REFERENCE_TARGETS")
    tasks = []
    for idx, face in enumerate(selected):
        tasks.append({
            "task_index": idx,
            "source_face_teacher_only": face,
            "finest_reference_slot_teacher_only": candidates[fine][face],
            "source_group_slot_teacher_only": {
                str(size): candidates[size][face] for size in sizes
            },
            "finest_reference_face_count": Counter(candidates[fine])[
                candidates[fine][face]],
        })
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_SOURCE_HIT_ANCHOR_PARTITION_EDIT_PROXY_ONLY",
        "source_sha256": source_sha,
        "source_face_count": face_count,
        "source_geometry_candidate_sizes": sizes,
        "unavailable_requested_sizes": [x for x in SIZES if x not in candidates],
        "finest_available_reference_size": fine,
        "tasks": tasks,
        "task_count": len(tasks),
        "source_names_as_labels": False,
        "author_drawable_identity_minted": False,
        "teacher_only": True,
        "graphics_owner_authorized": False,
        "training_authorized": False,
    }


def arm_names(plan):
    sizes = plan["source_geometry_candidate_sizes"]
    arms = ["SOURCE_ORIGINAL", "SOUP_FULL", "SOUP_NOOP", "REMOVE_ALL_GROUPS"]
    for task in plan["tasks"]:
        idx = task["task_index"]
        arms.extend((f"FINE_REMOVE_T{idx:02d}", f"FINE_ISOLATE_T{idx:02d}"))
        for size in sizes:
            arms.extend((f"GROUP_REMOVE_K{size:03d}_T{idx:02d}",
                         f"GROUP_ISOLATE_K{size:03d}_T{idx:02d}"))
    return arms
