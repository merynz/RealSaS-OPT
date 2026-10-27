#!/usr/bin/env python3
from __future__ import annotations

"""Build the exact external bundle consumed by Platform appearance admission.

The builder is transport-only.  It copies the already measured sparse appearance
payload and the already evaluated source-equivalence court into one deterministic
manifest section.  It does not create Registry qualification, infer PASS, widen
coverage, or authorize training/product use.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "RealSaS.SourceRayAppearanceCourtBundle.v1"
COURT_SCHEMA = "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1"
MEASUREMENT_SCHEMA = "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_JSON_OBJECT_REQUIRED")
    return value


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _record_hashes(measurement: dict[str, Any]) -> list[str]:
    hashes: list[str] = []
    for ray in measurement.get("rays") or ():
        if not isinstance(ray, dict):
            continue
        for hit in ray.get("hits") or ():
            if not isinstance(hit, dict):
                continue
            value = str(hit.get("record_hash") or "")
            if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
                hashes.append(value)
    if not hashes:
        # The aggregate source-equivalence court may carry the flattened record
        # set explicitly.  This is preferable when the measurement files are one
        # per subject and the court is the cohort aggregation boundary.
        for value in measurement.get("record_hashes") or ():
            text = str(value)
            if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
                hashes.append(text)
    if not hashes or len(set(hashes)) != len(hashes):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_RECORD_HASH_SET_INVALID")
    return hashes


def build(court: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    if court.get("schema") != COURT_SCHEMA:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_SCHEMA_DRIFT")
    if measurement.get("schema") not in {MEASUREMENT_SCHEMA, "RealSaS.SourceRayAppearanceEvidencePayload.v1"}:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_SCHEMA_DRIFT")
    if str(court.get("status") or "") != "PASS_SOURCE_EQUIVALENT_SPARSE_APPEARANCE_CANDIDATE":
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_NOT_PASS")
    record_hashes = _record_hashes(measurement)
    scope = court.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_SCOPE_REQUIRED")
    valid_count = int(scope.get("valid_hit_count", -1))
    if valid_count != len(record_hashes):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_RECORD_COUNT_DRIFT")
    evidence = {
        "schema": "RealSaS.SourceRayAppearanceEvidencePayload.v1",
        "record_hashes": record_hashes,
        "measurement_payload": measurement,
    }
    return {
        "schema": BUNDLE_SCHEMA,
        "court": court,
        "court_sha256": _digest(court),
        "evidence": evidence,
        "evidence_sha256": _digest(evidence),
        "transport_authority": "EXTERNAL_UNQUALIFIED_INPUT_ONLY",
        "registry_qualification_minted": False,
        "training_authorized": False,
        "product_authority_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--court", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = build(_load(args.court), _load(args.measurement))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": value["schema"],
        "record_count": len(value["evidence"]["record_hashes"]),
        "court_sha256": value["court_sha256"],
        "evidence_sha256": value["evidence_sha256"],
        "registry_qualification_minted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
