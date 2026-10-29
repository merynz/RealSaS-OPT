#!/usr/bin/env python3
"""Select a deterministic cross-lineage pilot for Truth139 Graphics materialization.

This is an apparatus/materialization pilot, not a train/tune/unseen split.  It selects one subject
from each of the largest leakage-connected components so the first heavy run crosses genuinely
different structural lineages.  Dense Graphics + visibility/occlusion eligibility is required;
development/regression taint is avoided when possible; the optional geometry-normal lane is
preferred but never made a Graphics eligibility gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WORK_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"
CENSUS_SCHEMA = "RealSaS.Truth139StructuralDiversityCensus.v1"
OUTPUT_SCHEMA = "RealSaS.Truth139GraphicsMaterializationPilot.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _eligible(row: dict) -> bool:
    jobs = dict(row.get("jobs") or {})
    return bool((jobs.get("DENSE_GRAPHICS_FRONT_SURFACE") or {}).get("enabled")) and bool(
        (jobs.get("VISIBILITY_OCCLUSION_CROSS_VIEW") or {}).get("enabled")
    )


def build(work: dict, census: dict, *, pilot_size: int = 8) -> dict:
    if work.get("schema") != WORK_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_PILOT_WORK_SCHEMA")
    if census.get("schema") != CENSUS_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_PILOT_CENSUS_SCHEMA")
    work_identity = str(work.get("work_order_identity_sha256") or "")
    census_identity = str(census.get("census_identity_sha256") or "")
    if len(work_identity) != 64 or len(census_identity) != 64:
        raise ValueError("TRUTH139_GRAPHICS_PILOT_PARENT_IDENTITY")
    subjects = {str(row.get("subject_id") or ""): row for row in work.get("subjects") or ()}
    if len(subjects) != int(work.get("subject_count", -1)) or "" in subjects:
        raise ValueError("TRUTH139_GRAPHICS_PILOT_WORK_SUBJECTS")
    components = list(census.get("components") or ())
    if not 1 <= pilot_size <= len(components):
        raise ValueError("TRUTH139_GRAPHICS_PILOT_SIZE")

    ranked = sorted(components, key=lambda row: (-int(row.get("size", 0)), int(row.get("component_ordinal", 0))))
    selected = []
    for component in ranked[:pilot_size]:
        ordinal = int(component.get("component_ordinal", 0))
        candidates = []
        for subject_id in component.get("subject_ids") or ():
            subject_id = str(subject_id)
            row = subjects.get(subject_id)
            if row is None:
                raise ValueError(f"TRUTH139_GRAPHICS_PILOT_COMPONENT_SUBJECT_MISSING:{ordinal}:{subject_id}")
            if not _eligible(row):
                continue
            jobs = dict(row.get("jobs") or {})
            normal_truth = bool((jobs.get("GEOMETRY_FSTAR_DEPTH_NORMAL_TRUTH") or {}).get("enabled"))
            # Scientific priority: avoid development-tainted rows; then prefer the richer optional
            # geometry-normal lane; final tie break is stable subject id.  Normal truth is never
            # required for Graphics eligibility.
            rank = (
                1 if bool(row.get("development_or_regression_tainted")) else 0,
                0 if normal_truth else 1,
                subject_id,
            )
            candidates.append((rank, row, normal_truth))
        if not candidates:
            raise ValueError(f"TRUTH139_GRAPHICS_PILOT_COMPONENT_HAS_NO_GRAPHICS_CANDIDATE:{ordinal}")
        candidates.sort(key=lambda item: item[0])
        rank, row, normal_truth = candidates[0]
        source = dict(row.get("source") or {})
        selected.append(
            {
                "pilot_index": len(selected),
                "component_ordinal": ordinal,
                "component_size": int(component.get("size", 0)),
                "component_cross_pack": bool(component.get("cross_pack")),
                "component_pack_counts": dict(component.get("pack_counts") or {}),
                "subject_id": str(row["subject_id"]),
                "subject_identity_signature": str(row.get("subject_identity_signature") or ""),
                "pack_id": str(source.get("pack_id") or ""),
                "corpus_relative_path": str(source.get("corpus_relative_path") or ""),
                "source_sha256": str(source.get("sha256") or ""),
                "development_or_regression_tainted": bool(row.get("development_or_regression_tainted")),
                "optional_geometry_normal_truth_available": bool(normal_truth),
                "selection_rank": list(rank),
            }
        )

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": "READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY",
        "parent_work_order_identity_sha256": work_identity,
        "parent_diversity_census_identity_sha256": census_identity,
        "policy": {
            "purpose": "APPARATUS_AND_HEAVY_MATERIALIZATION_SMOKE_COURT_ONLY",
            "largest_leakage_components_first": True,
            "one_subject_per_component": True,
            "dense_graphics_and_visibility_required": True,
            "prefer_non_development_subject": True,
            "prefer_optional_geometry_normal_truth_but_do_not_require_it": True,
            "stable_final_tie_break": "SUBJECT_ID_ASCENDING",
            "semantic_class_selection_used": False,
            "subject_specific_code_allowed": False,
            "train_tune_cal_unseen_split_created": False,
        },
        "pilot_size": len(selected),
        "selected": selected,
        "coverage": {
            "leakage_component_count": len(selected),
            "cross_pack_component_count": sum(int(row["component_cross_pack"]) for row in selected),
            "optional_geometry_normal_truth_available_count": sum(
                int(row["optional_geometry_normal_truth_available"]) for row in selected
            ),
            "optional_geometry_normal_truth_missing_count": sum(
                int(not row["optional_geometry_normal_truth_available"]) for row in selected
            ),
            "development_or_regression_tainted_count": sum(
                int(row["development_or_regression_tainted"]) for row in selected
            ),
        },
        "training_authorized": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["pilot_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--diversity-census", required=True, type=Path)
    parser.add_argument("--pilot-size", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    work = json.loads(args.work_orders.resolve().read_text())
    census = json.loads(args.diversity_census.resolve().read_text())
    result = build(work, census, pilot_size=args.pilot_size)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "pilot_size": result["pilot_size"],
        "pilot_identity_sha256": result["pilot_identity_sha256"],
        "selected_subject_ids": [row["subject_id"] for row in result["selected"]],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
