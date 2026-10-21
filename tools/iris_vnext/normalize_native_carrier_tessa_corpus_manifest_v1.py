#!/usr/bin/env python3
from __future__ import annotations

"""Normalize TESSA corpus manifest surface provenance for the locked native-R24
matched-carrier H1 experiment.

The legacy corpus builder is intentionally reused because its tensor construction
is unchanged. That builder serializes a historical T512/T1024 surface label in
manifest metadata. This adapter rewrites metadata only after verifying that every
subject is bound to the fixed native-carrier candidate hash. It never reads or
mutates subject tensor payloads and performs zero optimizer/backward steps.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CARRIER_ID = "RealSaS.FixedSurfaceCarrier.ExactV8NativeFlexiCubesR24.v1"
CARRIER_AUTHORITY = "INFERENCE_ADMISSIBLE_FIXED_SURFACE_CARRIER"
RECEIPT_SCHEMA = "RealSaS.NativeCarrierTESSACorpusManifestNormalization.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NATIVE_CARRIER_CORPUS_JSON_OBJECT_REQUIRED")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    carrier_batch_path = Path(args.carrier_batch).resolve()
    receipt_path = Path(args.receipt).resolve()

    manifest_before_sha = sha256_file(manifest_path)
    manifest = load_json(manifest_path)
    carrier = load_json(carrier_batch_path)

    if manifest.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1":
        raise ValueError("NATIVE_CARRIER_CORPUS_SCHEMA_INVALID")
    if carrier.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1":
        raise ValueError("NATIVE_CARRIER_CORPUS_CARRIER_SCHEMA_INVALID")
    if carrier.get("carrier_id") != CARRIER_ID or carrier.get("carrier_authority") != CARRIER_AUTHORITY:
        raise ValueError("NATIVE_CARRIER_CORPUS_CARRIER_AUTHORITY_INVALID")
    if carrier.get("compatibility_projection_only") is not True:
        raise ValueError("NATIVE_CARRIER_CORPUS_PROJECTION_FLAG_MISSING")
    if carrier.get("geometry_mutated") is not False or carrier.get("topology_mutated") is not False:
        raise ValueError("NATIVE_CARRIER_CORPUS_CARRIER_MUTATED")
    policy = carrier.get("surface_carrier_policy") or {}
    if policy.get("same_carrier_for_a_and_b") is not True:
        raise ValueError("NATIVE_CARRIER_CORPUS_NOT_MATCHED")
    if policy.get("raw_field_sparse_t512_t1024_invoked") is not False:
        raise ValueError("NATIVE_CARRIER_CORPUS_RAW_SPARSE_LANE_REACTIVATED")

    carrier_rows = {str(row["subject_id"]): row for row in carrier.get("subjects") or ()}
    corpus_rows = {str(row["subject_id"]): row for row in manifest.get("subjects") or ()}
    if tuple(carrier_rows) != tuple(corpus_rows):
        raise ValueError("NATIVE_CARRIER_CORPUS_SUBJECT_ORDER_DRIFT")
    for subject_id, row in corpus_rows.items():
        if row.get("direct_surface_candidate_binding_hash") != carrier_rows[subject_id].get("candidate_binding_hash"):
            raise ValueError("NATIVE_CARRIER_CORPUS_BINDING_DRIFT:" + subject_id)

    conditioning = dict(manifest.get("conditioning") or {})
    if conditioning.get("surface") not in (
        "DIRECT_IRIS_T512_T1024_SURFACE_WITNESS",
        "FIXED_NATIVE_V8_R24_FLEXICUBES_CARRIER",
    ):
        raise ValueError("NATIVE_CARRIER_CORPUS_UNEXPECTED_SURFACE_LABEL")
    conditioning["surface"] = "FIXED_NATIVE_V8_R24_FLEXICUBES_CARRIER"
    conditioning["surface_carrier_authority"] = CARRIER_AUTHORITY
    conditioning["surface_carrier_id"] = CARRIER_ID
    conditioning["raw_field_sparse_t512_t1024_invoked"] = False
    conditioning["high_resolution_capability_claimed"] = False
    conditioning["product_surface_claimed"] = False
    manifest["conditioning"] = conditioning
    manifest["native_matched_carrier_amendment_applied"] = True
    manifest["carrier_batch_sha256"] = sha256_file(carrier_batch_path)
    manifest["subject_payloads_read_by_manifest_adapter"] = False
    manifest["subject_payloads_mutated_by_manifest_adapter"] = False
    manifest["optimizer_steps_executed_by_manifest_adapter"] = 0
    manifest["backward_steps_executed_by_manifest_adapter"] = 0

    # The legacy builder's value_sha256 described the pre-normalization manifest.
    # Preserve it as provenance and replace it with a hash of the normalized value.
    prior_value_sha = manifest.pop("value_sha256", None)
    if prior_value_sha is not None:
        manifest["pre_normalization_value_sha256"] = prior_value_sha
    normalized_value = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest["value_sha256"] = hashlib.sha256(normalized_value).hexdigest()
    write_json(manifest_path, manifest)
    manifest_after_sha = sha256_file(manifest_path)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS__NATIVE_CARRIER_PROVENANCE_NORMALIZED__NO_SUBJECT_PAYLOAD_MUTATION",
        "manifest_file": manifest_path.name,
        "manifest_sha256_before": manifest_before_sha,
        "manifest_sha256_after": manifest_after_sha,
        "carrier_batch_sha256": sha256_file(carrier_batch_path),
        "carrier_id": CARRIER_ID,
        "carrier_authority": CARRIER_AUTHORITY,
        "subject_payloads_read": False,
        "subject_payloads_mutated": False,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "scientific_pass_claimed": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--carrier-batch", required=True)
    p.add_argument("--receipt", required=True)
    result = normalize(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
