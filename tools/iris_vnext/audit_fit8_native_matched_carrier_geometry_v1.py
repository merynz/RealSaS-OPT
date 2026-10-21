#!/usr/bin/env python3
from __future__ import annotations

"""Geometry/topology measurement court for the fixed native V8 R24 H1 carrier.

This is a compatibility-specific measurement court, not a new scientific gate.
It preserves the measurements of audit_fit8_direct_surface_geometry_v1 while
admitting the explicit fixed-native carrier status introduced by the H1
pre-optimizer amendment. No training, TESSA inference, threshold change, or
surface-authority upgrade occurs here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import (
    SCHEMA,
    load_json,
    measure_surface_against_shard,
    sha256_file,
)

EXPECTED_BATCH_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1"
EXPECTED_BATCH_STATUS = "FIXED_NATIVE_V8_R24_CARRIER_PROJECTED_FOR_MATCHED_H1_AB__NO_AUTHORITY_UPGRADE"
EXPECTED_CARRIER_ID = "RealSaS.FixedSurfaceCarrier.ExactV8NativeFlexiCubesR24.v1"
EXPECTED_CARRIER_AUTHORITY = "INFERENCE_ADMISSIBLE_FIXED_SURFACE_CARRIER"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    prepared_root = Path(args.prepared_root).resolve()
    prepared_manifest_path = Path(args.prepared_manifest).resolve()
    surface_root = Path(args.surface_root).resolve()
    output_path = Path(args.output).resolve()

    manifest = load_json(prepared_manifest_path)
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("NATIVE_CARRIER_GEOMETRY_PREPARED_MANIFEST_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_SUBJECT_ORDER_DRIFT")

    batch_path = surface_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    batch = load_json(batch_path)
    if batch.get("schema") != EXPECTED_BATCH_SCHEMA:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_BATCH_SCHEMA_INVALID")
    if batch.get("status") != EXPECTED_BATCH_STATUS:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_BATCH_STATUS_INVALID")
    if batch.get("carrier_id") != EXPECTED_CARRIER_ID:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_CARRIER_ID_DRIFT")
    if batch.get("carrier_authority") != EXPECTED_CARRIER_AUTHORITY:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_CARRIER_AUTHORITY_DRIFT")
    policy = dict(batch.get("surface_carrier_policy") or {})
    if int(policy.get("resolution", -1)) != 24:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_RESOLUTION_DRIFT")
    if policy.get("same_carrier_for_a_and_b") is not True:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_MATCHED_CARRIER_BROKEN")
    if policy.get("raw_field_sparse_t512_t1024_invoked") is not False:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_RAW_SPARSE_LANE_REACTIVATED")
    if policy.get("high_resolution_capability_claimed") is not False:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_ILLEGAL_HIGH_RES_CLAIM")
    if batch.get("geometry_mutated") is not False or batch.get("topology_mutated") is not False:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_MUTATION_DETECTED")
    if batch.get("tessa_invoked") is not False or int(batch.get("optimizer_steps_executed", -1)) != 0:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_PREOPTIMIZER_CONTAMINATION")

    receipt_rows = list(batch.get("subjects") or ())
    if tuple(row.get("subject_id") for row in receipt_rows) != SUBJECT_IDS:
        raise ValueError("NATIVE_CARRIER_GEOMETRY_BATCH_SUBJECT_ORDER_DRIFT")

    results: list[dict[str, Any]] = []
    for subject_index, (manifest_row, receipt_row) in enumerate(zip(rows, receipt_rows, strict=True)):
        subject_id = SUBJECT_IDS[subject_index]
        shard = (prepared_root / str(manifest_row["shard_file"])).resolve()
        if not shard.is_relative_to(prepared_root) or sha256_file(shard) != str(manifest_row["shard_sha256"]):
            raise ValueError("NATIVE_CARRIER_GEOMETRY_PREPARED_SHARD_DRIFT:" + subject_id)

        subject_root = surface_root / "subjects" / subject_id
        stage12 = subject_root / "DIRECT_IRIS_SURFACE.stage12.npz"
        receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
        receipt = load_json(receipt_path)
        if sha256_file(stage12) != str(receipt_row.get("stage12_npz_sha256") or ""):
            raise ValueError("NATIVE_CARRIER_GEOMETRY_STAGE12_HASH_DRIFT:" + subject_id)
        if receipt.get("candidate_binding_hash") != receipt_row.get("candidate_binding_hash"):
            raise ValueError("NATIVE_CARRIER_GEOMETRY_CANDIDATE_BINDING_DRIFT:" + subject_id)
        if receipt.get("carrier_authority") != EXPECTED_CARRIER_AUTHORITY:
            raise ValueError("NATIVE_CARRIER_GEOMETRY_SUBJECT_AUTHORITY_DRIFT:" + subject_id)
        if receipt.get("geometry_mutated") is not False or receipt.get("topology_mutated") is not False:
            raise ValueError("NATIVE_CARRIER_GEOMETRY_SUBJECT_MUTATION:" + subject_id)
        if receipt.get("compiler_qualified") is not False or receipt.get("scientific_pass_claimed") is not False:
            raise ValueError("NATIVE_CARRIER_GEOMETRY_PREQUALIFIED_INPUT_FORBIDDEN:" + subject_id)

        metadata = manifest_row.get("metadata") or {}
        topology_teacher = metadata.get("topology_teacher") or {}
        teacher_components = topology_teacher.get("component_count")
        result = measure_surface_against_shard(
            subject_id=subject_id,
            stage12_path=stage12,
            prepared_shard_path=shard,
            teacher_component_count=int(teacher_components) if teacher_components is not None else None,
            seed=26091991 + subject_index,
        )
        result["candidate_binding_hash"] = receipt_row["candidate_binding_hash"]
        result["stage12_npz_sha256"] = receipt_row["stage12_npz_sha256"]
        result["carrier_id"] = EXPECTED_CARRIER_ID
        result["carrier_authority"] = EXPECTED_CARRIER_AUTHORITY
        results.append(result)

    summary = {
        "schema": SCHEMA,
        "status": "MEASURED__FIXED_NATIVE_V8_R24_CARRIER__NO_PASS_CLAIM",
        "surface_batch_receipt_sha256": sha256_file(batch_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": results,
        "surface_sample_count_per_subject": 65536,
        "carrier_id": EXPECTED_CARRIER_ID,
        "carrier_authority": EXPECTED_CARRIER_AUTHORITY,
        "same_carrier_for_a_and_b": True,
        "raw_field_sparse_t512_t1024_invoked": False,
        "high_resolution_capability_claimed": False,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "mask_depth_court_executed": False,
        "scientific_pass_claimed": False,
        "thresholds_changed": False,
        "next_action": "CONTINUE_LOCKED_H1_PRESENTATION_AND_CORPUS_PREFLIGHTS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--surface-root", required=True)
    p.add_argument("--output", required=True)
    return p


def main() -> int:
    result = audit(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
