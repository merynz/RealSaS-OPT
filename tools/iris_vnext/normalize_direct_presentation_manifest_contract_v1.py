#!/usr/bin/env python3
from __future__ import annotations

"""Normalize the Direct IRIS presentation manifest to the trainer firewall contract.

This is a provenance-only compatibility adapter.  It never reads or rewrites any
presentation subject payload tensor.  The producer manifest already records that
teacher geometry/topology and materialized V2 truth are not inputs.  The locked
Structural-Evidence A/B trainer names the same three facts with
``*_in_conditioning`` keys.  This adapter verifies the producer facts are exactly
False and materializes the equivalent conditioning aliases without changing any
scientific value, evidence token, model parameter, optimizer state, threshold or
court policy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

SCHEMA = "RealSaS.DirectIRISPresentationMemoryManifest.v1"
RECEIPT_SCHEMA = "RealSaS.DirectIRISPresentationManifestContractNormalization.v1"

_SOURCE_TO_CONDITIONING = {
    "teacher_geometry_input": "teacher_geometry_in_conditioning",
    "teacher_topology_input": "teacher_topology_in_conditioning",
    "materialized_v2_truth_input": "materialized_v2_truth_in_conditioning",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_PRESENTATION_MANIFEST_OBJECT_REQUIRED")
    return value


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".staged",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def normalize(manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    receipt_path = receipt_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("DIRECT_PRESENTATION_MANIFEST_SCHEMA_INVALID")

    before_sha = sha256_file(manifest_path)
    aliases_added: list[str] = []
    for source_key, conditioning_key in _SOURCE_TO_CONDITIONING.items():
        if manifest.get(source_key) is not False:
            raise ValueError("DIRECT_PRESENTATION_MANIFEST_SOURCE_FIREWALL_BROKEN:" + source_key)
        if conditioning_key in manifest and manifest.get(conditioning_key) is not False:
            raise ValueError("DIRECT_PRESENTATION_MANIFEST_CONDITIONING_FIREWALL_BROKEN:" + conditioning_key)
        if conditioning_key not in manifest:
            manifest[conditioning_key] = False
            aliases_added.append(conditioning_key)

    _atomic_json(manifest, manifest_path)
    after_sha = sha256_file(manifest_path)
    normalized = load_json(manifest_path)
    for source_key, conditioning_key in _SOURCE_TO_CONDITIONING.items():
        if normalized.get(source_key) is not False or normalized.get(conditioning_key) is not False:
            raise AssertionError("DIRECT_PRESENTATION_MANIFEST_NORMALIZATION_POSTCONDITION_FAILED")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS__PROVENANCE_ALIAS_NORMALIZED__NO_EVIDENCE_PAYLOAD_MUTATION",
        "manifest_file": manifest_path.name,
        "manifest_sha256_before": before_sha,
        "manifest_sha256_after": after_sha,
        "aliases_added": aliases_added,
        "source_to_conditioning_alias": dict(_SOURCE_TO_CONDITIONING),
        "presentation_subject_payloads_read": False,
        "presentation_subject_payloads_mutated": False,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "scientific_pass_claimed": False,
    }
    _atomic_json(receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = normalize(Path(args.manifest), Path(args.receipt))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
