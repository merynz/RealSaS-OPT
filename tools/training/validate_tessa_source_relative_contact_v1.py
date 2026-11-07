"""Independent court for exact-frame source-relative TESSA contact certificates.

Does not import the certificate producer or trust its declared PASS flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.TESSASourceRelativeContactCourt.v1"
PASS = "PASS_INDEPENDENT_EXACT_FRAME_CONTACT_EQUIVALENCE"
MASKED = "MASKED_CONTACT_EQUIVALENCE_NOT_PROVEN"
PARENT_SCHEMA = "RealSaS.TESSASourceRelativeContact.v1"
PARENT_PASS = "PASS_EXACT_SOURCE_FRAME_CONTACT_EQUIVALENCE"
PARENT_MASKED = "MASKED_SOURCE_RELATIVE_CONTACT_UNPROVEN"


def bound(row: dict) -> Path:
    path = Path(str(row.get("path") or "")).resolve()
    expected = row.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("TESSA_CONTACT_COURT_PARENT_MISSING")
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected != observed:
        raise ValueError("TESSA_CONTACT_COURT_PARENT_DRIFT")
    return path


def decode(path: Path, source: bool) -> tuple[np.ndarray, ...]:
    rest_key = "rest_vertices_source_world" if source else "vertices_source_world"
    with np.load(path, allow_pickle=False) as archive:
        keys = (rest_key, "faces", "isolated_position")
        if not all(key in archive.files for key in keys):
            raise ValueError("TESSA_CONTACT_COURT_ARRAY_KEYS")
        v = np.asarray(archive[rest_key], dtype=np.float64)
        f = np.asarray(archive["faces"])
        p = np.asarray(archive["isolated_position"], dtype=np.float64)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError("TESSA_CONTACT_COURT_VERTEX_LAYOUT")
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError("TESSA_CONTACT_COURT_FACE_LAYOUT")
    if p.ndim != 3 or p.shape[1:] != v.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("TESSA_CONTACT_COURT_FRAME_LAYOUT")
    return v, f.astype(np.int64, copy=False), p


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA:
        raise ValueError("TESSA_CONTACT_COURT_SCHEMA")
    if manifest.get("contract") != "EXACT_INDEXED_REST_AND_ALL_SAMPLED_ACTION_FRAMES_ONLY":
        raise ValueError("TESSA_CONTACT_COURT_SCOPE")
    if not isinstance(manifest.get("subject_id"), str) or not manifest["subject_id"]:
        raise ValueError("TESSA_CONTACT_COURT_SUBJECT")
    parent = manifest.get("parents") or {}
    source = decode(bound(parent.get("source") or {}), True)
    candidate = decode(bound(parent.get("candidate") or {}), False)
    same = all(np.array_equal(a, b) for a, b in zip(source, candidate))
    expected_status = PARENT_PASS if same else PARENT_MASKED
    auth = manifest.get("authority") or {}
    if (manifest.get("status") != expected_status
            or manifest.get("rest_and_sampled_frames_index_identical") is not same
            or manifest.get("source_frame_count") != len(source[2])
            or manifest.get("candidate_frame_count") != len(candidate[2])
            or auth.get("source_relative_contact_equivalence_certified") is not same
            or auth.get("new_or_increased_contact_excluded_on_identical_sampled_frames") is not same):
        raise ValueError("TESSA_CONTACT_COURT_RESIGNED_FALSE_PASS")
    for key in ("physical_collision_freedom_claimed", "continuous_time_motion_qualified",
                "training_authorized", "product_authority_minted"):
        if auth.get(key) is not False:
            raise ValueError("TESSA_CONTACT_COURT_AUTHORITY_OVERCLAIM:" + key)
    return {
        "schema": SCHEMA,
        "status": PASS if same else MASKED,
        "subject_id": manifest["subject_id"],
        "independently_verified_sampled_frame_count": int(len(source[2])),
        "source_relative_contact_equivalence_certified": same,
        "authority": {
            "final_tessa_teacher_qualified": False,
            "motion_between_sampled_frames_qualified": False,
            "training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": value["status"], "subject_id": value["subject_id"]}))


if __name__ == "__main__":
    main()
