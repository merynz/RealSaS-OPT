#!/usr/bin/env python3
from __future__ import annotations

"""Build the exact #243 court bundle consumed by Platform appearance admission.

Input is the aggregate source-equivalence court plus one deterministic packed
measurement receipt per admitted FIT8 subject. The builder checks that each
receipt's canonical JSON hash is exactly the packed-A/B parent hash sealed by the
court. It performs no qualification itself.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "RealSaS.SourceRayAppearanceCourtBundle.v1"
COURT_SCHEMA = "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1"
MEASUREMENT_SCHEMA = "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"
PASS_COURT = "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_JSON_OBJECT_REQUIRED")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build(court: dict[str, Any], measurements: list[dict[str, Any]]) -> dict[str, Any]:
    if court.get("schema") != COURT_SCHEMA:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_SCHEMA_DRIFT")
    if str(court.get("court_status") or "") != PASS_COURT:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_NOT_PASS")
    subject_rows = court.get("subjects")
    if not isinstance(subject_rows, list) or not subject_rows:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_SUBJECTS_REQUIRED")
    expected: dict[str, str] = {}
    for row in subject_rows:
        subject_id = str(row.get("subject_id") or "")
        parents = row.get("parents")
        if not subject_id or subject_id in expected or not isinstance(parents, dict):
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_SUBJECT_INVALID")
        packed_a = str(parents.get("packed_a_sha256") or "")
        packed_b = str(parents.get("packed_b_sha256") or "")
        if packed_a != packed_b or len(packed_a) != 64:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_PACKED_REPLAY_NOT_EXACT")
        expected[subject_id] = packed_a
    if len(expected) != int(court.get("subject_count", -1)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_SUBJECT_COUNT_DRIFT")

    bound = []
    seen: set[str] = set()
    rays = hits = 0
    for payload in measurements:
        if payload.get("schema") != MEASUREMENT_SCHEMA:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_SCHEMA_DRIFT")
        subject_id = str(payload.get("subject_id") or "")
        if subject_id not in expected or subject_id in seen:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_SUBJECT_DRIFT")
        digest = _digest(payload)
        if digest != expected[subject_id]:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_PARENT_HASH_DRIFT:" + subject_id)
        seen.add(subject_id)
        ray_rows = payload.get("rays") or []
        rays += len(ray_rows)
        hits += sum(len(ray.get("canonical_hits") or []) for ray in ray_rows if isinstance(ray, dict))
        bound.append({"subject_id": subject_id, "sha256": digest, "payload": payload})
    if seen != set(expected):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_COHORT_INCOMPLETE")
    if rays != int(court.get("ray_count", -1)) or hits != int(court.get("hit_count", -1)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_COUNT_DRIFT")

    bound.sort(key=lambda row: row["subject_id"])
    return {
        "schema": BUNDLE_SCHEMA,
        "court": court,
        "court_sha256": _digest(court),
        "measurements": bound,
        "transport_authority": "EXTERNAL_UNQUALIFIED_INPUT_ONLY",
        "registry_qualification_minted": False,
        "canonical_mesh_binding_minted": False,
        "training_authorized": False,
        "product_authority_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--court", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = build(_load(args.court), [_load(path) for path in args.measurement])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": value["schema"],
        "subject_count": len(value["measurements"]),
        "ray_count": value["court"]["ray_count"],
        "hit_count": value["court"]["hit_count"],
        "court_sha256": value["court_sha256"],
        "registry_qualification_minted": False,
        "canonical_mesh_binding_minted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
