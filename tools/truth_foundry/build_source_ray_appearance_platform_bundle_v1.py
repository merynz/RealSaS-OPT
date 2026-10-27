#!/usr/bin/env python3
from __future__ import annotations

"""Build the exact #243 court bundle consumed by Platform appearance admission.

Raw JSON bytes are preserved deliberately. The aggregate court raw SHA is pinned
in source; its per-subject packed-A/B parent SHA values then anchor every sparse
measurement receipt. This builder transports those exact bytes into one manifest
section and does not mint qualification.
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
SOURCE_WORKFLOW_RUN_ID = 40544781242
SOURCE_ARTIFACT_ID = 12751917983
SOURCE_ARTIFACT_NAME = "source-ray-appearance-source-equivalence-fit8-v1-40544781242"
SOURCE_ARTIFACT_DIGEST = "sha256:a63940ef7bb9064f42941f595089b85bb120a7f53a271b7392b0db3c2dbe01dc"
SOURCE_COURT_RAW_SHA256 = "f62412457156a7e36f351fd3d89d9ad042bf80213b4749329a7cfa6ee0cf912a"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(code + "_JSON_OBJECT_REQUIRED")
    return value


def build(court_raw: bytes, measurement_raws: list[bytes]) -> dict[str, Any]:
    if _sha(court_raw) != SOURCE_COURT_RAW_SHA256:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_COURT_RAW_HASH_DRIFT")
    court = _decode(court_raw, "SOURCE_RAY_APPEARANCE_BUNDLE_COURT")
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
    for raw in measurement_raws:
        payload = _decode(raw, "SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT")
        if payload.get("schema") != MEASUREMENT_SCHEMA:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_SCHEMA_DRIFT")
        subject_id = str(payload.get("subject_id") or "")
        if subject_id not in expected or subject_id in seen:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_SUBJECT_DRIFT")
        digest = _sha(raw)
        if digest != expected[subject_id]:
            raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_PARENT_HASH_DRIFT:" + subject_id)
        seen.add(subject_id)
        ray_rows = payload.get("rays") or []
        rays += len(ray_rows)
        hits += sum(len(ray.get("canonical_hits") or []) for ray in ray_rows if isinstance(ray, dict))
        bound.append({"subject_id": subject_id, "sha256": digest, "json_text": raw.decode("utf-8")})
    if seen != set(expected):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_COHORT_INCOMPLETE")
    if rays != int(court.get("ray_count", -1)) or hits != int(court.get("hit_count", -1)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BUNDLE_MEASUREMENT_COUNT_DRIFT")

    bound.sort(key=lambda row: row["subject_id"])
    return {
        "schema": BUNDLE_SCHEMA,
        "source_artifact": {
            "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
            "artifact_id": SOURCE_ARTIFACT_ID,
            "artifact_name": SOURCE_ARTIFACT_NAME,
            "artifact_digest": SOURCE_ARTIFACT_DIGEST,
        },
        "court": {"sha256": SOURCE_COURT_RAW_SHA256, "json_text": court_raw.decode("utf-8")},
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
    value = build(args.court.read_bytes(), [path.read_bytes() for path in args.measurement])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": value["schema"],
        "subject_count": len(value["measurements"]),
        "court_sha256": value["court"]["sha256"],
        "source_artifact_digest": value["source_artifact"]["artifact_digest"],
        "registry_qualification_minted": False,
        "canonical_mesh_binding_minted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
