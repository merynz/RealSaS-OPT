#!/usr/bin/env python3
"""Read-only audit of exact existing FIT8 manifests; no tensor admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUBJECTS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")
PINS = {
    "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json": (
        "21434efbfd409504d11bd883f35041089dbe7329e829c37446eb21c59b19b9f6",
        "RealSaS.DirectIRISPresentationMemoryManifest.v1"),
    "DIRECT_IRIS_PRESENTATION_QUERIES_V1_MANIFEST.json": (
        "9bc67c6457918e43d88a2ff1158d0e8102875c9a3cf8db24788ac457237c6695",
        "RealSaS.DirectIRISPresentationQueriesManifest.v1"),
    "TESSA_V2_CORPUS_MANIFEST.json": (
        "0800e42fbc479320c4f8880db4e4b74bf4756eb72380b4f597b35102e44d5663",
        "RealSaS.Fit8TESSAV2TrainingCorpus.v1"),
}


def audit(directory):
    records, manifests = [], {}
    for name, (expected_hash, expected_schema) in PINS.items():
        raw = (Path(directory) / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if digest != expected_hash or value.get("schema") != expected_schema:
            raise ValueError("existing manifest identity/schema mismatch: " + name)
        roster = value.get("subject_order")
        # Preserve the source's exact roster strings; never infer subject aliases.
        if roster != list(SUBJECTS):
            raise ValueError("existing FIT8 roster/order mismatch: " + repr(roster))
        manifests[name] = value
        records.append({"name": name, "sha256": digest, "schema": expected_schema,
                        "execution_source_sha": value["execution_source_sha"]})
    authority = manifests["DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"]["evidence_authority"]
    for name in ("occlusion", "deformation_coherence", "attachment_persistence"):
        if authority.get(name) != "ABSTAIN":
            raise ValueError("historical abstention unexpectedly changed")
    return {"schema": "RealSaS.ExistingFIT8ManifestAudit.v1",
            "status": "PINNED_MANIFEST_IDENTITIES_VERIFIED__PAYLOADS_NOT_VERIFIED",
            "subjects": list(SUBJECTS), "manifests": records,
            "historical_evidence_authority": authority,
            "tensor_payload_bytes_verified": False,
            "teacher_causal_truth_qualified": False,
            "training_executed": False,
            "next_action": "REUSE_SOURCE_LINEAGE__BUILD_VERSIONED_TEACHER_MEASUREMENTS_THROUGH_PLATFORM"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
