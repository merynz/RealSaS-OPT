#!/usr/bin/env python3
"""Bridge a frozen Truth139 Graphics pilot into the common source import/render apparatus.

The output intentionally matches the generic roster contract already accepted by
``prepare_iris_observations_v1.py``.  It carries exact source bindings only; it creates no training
split, source admission, Graphics truth admission or model authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PILOT_SCHEMA = "RealSaS.Truth139GraphicsMaterializationPilot.v1"
WORK_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"
OUTPUT_SCHEMA = "RealSaS.Truth139ApparatusManifest.v1"
OUTPUT_STATUS = "READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build(pilot: dict, work: dict) -> dict:
    if pilot.get("schema") != PILOT_SCHEMA or pilot.get("status") != "READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY":
        raise ValueError("TRUTH139_GRAPHICS_APPARATUS_PILOT_SCHEMA_OR_STATUS")
    if work.get("schema") != WORK_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_APPARATUS_WORK_SCHEMA")
    if pilot.get("parent_work_order_identity_sha256") != work.get("work_order_identity_sha256"):
        raise ValueError("TRUTH139_GRAPHICS_APPARATUS_WORK_IDENTITY_DRIFT")
    pilot_identity = str(pilot.get("pilot_identity_sha256") or "")
    if len(pilot_identity) != 64:
        raise ValueError("TRUTH139_GRAPHICS_APPARATUS_PILOT_IDENTITY")
    rows = {str(row.get("subject_id") or ""): row for row in work.get("subjects") or ()}
    selected = []
    for frozen in pilot.get("selected") or ():
        subject_id = str(frozen.get("subject_id") or "")
        row = rows.get(subject_id)
        if row is None:
            raise ValueError("TRUTH139_GRAPHICS_APPARATUS_SUBJECT_MISSING:" + subject_id)
        source = dict(row.get("source") or {})
        if source.get("sha256") != frozen.get("source_sha256") or source.get("corpus_relative_path") != frozen.get("corpus_relative_path"):
            raise ValueError("TRUTH139_GRAPHICS_APPARATUS_SOURCE_DRIFT:" + subject_id)
        jobs = dict(row.get("jobs") or {})
        if not bool((jobs.get("SOURCE_IMPORT_AND_8VIEW_OBSERVATION") or {}).get("enabled")):
            raise ValueError("TRUTH139_GRAPHICS_APPARATUS_SOURCE_JOB_DISABLED:" + subject_id)
        if not bool((jobs.get("DENSE_GRAPHICS_FRONT_SURFACE") or {}).get("enabled")):
            raise ValueError("TRUTH139_GRAPHICS_APPARATUS_GRAPHICS_JOB_DISABLED:" + subject_id)
        selected.append({
            "subject_key": subject_id,
            "source": source,
            "source_work_order_identity_sha256": str(row.get("subject_identity_signature") or ""),
            "pilot_index": int(frozen.get("pilot_index", -1)),
            "leakage_component_ordinal": int(frozen.get("component_ordinal", -1)),
            "import_render_admission": "PENDING",
            "surface_geometry_export": "REQUIRED_RESPONSE_INDEPENDENT_REST_GEOMETRY",
            "training_authorized": False,
        })
    if len(selected) != int(pilot.get("pilot_size", -1)) or len({row["subject_key"] for row in selected}) != len(selected):
        raise ValueError("TRUTH139_GRAPHICS_APPARATUS_CARDINALITY")
    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": OUTPUT_STATUS,
        "purpose": "COMMON_EXACT_SOURCE_IMPORT_8VIEW_OBSERVATION_AND_RESPONSE_INDEPENDENT_REST_GEOMETRY_FOR_GRAPHICS_PILOT",
        "parent_pilot_identity_sha256": pilot_identity,
        "parent_work_order_identity_sha256": str(work.get("work_order_identity_sha256") or ""),
        "subject_count": len(selected),
        "selected": selected,
        "authority_boundary": {
            "prepared_source_bytes_only": True,
            "import_render_admission_pending": True,
            "derived_graphics_truth_admitted": False,
            "training_authorized": False,
            "split_created": False,
            "qualification_minted": False,
        },
        "training_authorized": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["apparatus_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(json.loads(args.pilot.resolve().read_text()), json.loads(args.work_orders.resolve().read_text()))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_count": result["subject_count"],
        "apparatus_identity_sha256": result["apparatus_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
