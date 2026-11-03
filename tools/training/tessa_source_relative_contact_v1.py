"""Conservative source-relative contact certificate for exact TESSA frame identity.

A source-matched candidate has exactly the same indexed triangles in rest and
EVERY sealed sampled action as its source. Its intersecting pairs and overlap
severity therefore cannot be new or increased on those frames. This does NOT
prove collision freedom or continuous-time safety. Non-identical candidates
remain unqualified; no approximate tolerance or fallback name can admit them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.TESSASourceRelativeContact.v1"
PASS = "PASS_EXACT_SOURCE_FRAME_CONTACT_EQUIVALENCE"
MASKED = "MASKED_SOURCE_RELATIVE_CONTACT_UNPROVEN"
SOURCE_KEYS = ("rest_vertices_source_world", "faces", "isolated_position")
CANDIDATE_KEYS = ("vertices_source_world", "faces", "isolated_position")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path, keys: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(set(archive.files)):
            raise ValueError("TESSA_CONTACT_ARRAY_KEYS")
        vertices = np.asarray(archive[keys[0]], np.float64)
        faces = np.asarray(archive[keys[1]])
        poses = np.asarray(archive[keys[2]], np.float64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not len(vertices) or not np.isfinite(vertices).all()):
        raise ValueError("TESSA_CONTACT_REST_LAYOUT")
    if (faces.ndim != 2 or faces.shape[1:] != (3,)
            or not len(faces) or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("TESSA_CONTACT_FACE_LAYOUT")
    if (poses.ndim != 3 or poses.shape[1:] != vertices.shape
            or not len(poses) or not np.isfinite(poses).all()):
        raise ValueError("TESSA_CONTACT_SAMPLED_FRAME_LAYOUT")
    return vertices, faces.astype(np.int64, copy=False), poses


def build(source_npz: Path, candidate_npz: Path, output: Path, *,
          source_sha256: str, candidate_sha256: str, subject_id: str) -> dict:
    """Issue a scoped candidate certificate, never model/training authority."""
    source_npz, candidate_npz = Path(source_npz).resolve(), Path(candidate_npz).resolve()
    if (sha256(source_npz) != source_sha256
            or sha256(candidate_npz) != candidate_sha256):
        raise ValueError("TESSA_CONTACT_PARENT_BYTES_DRIFT")
    if not subject_id or not isinstance(subject_id, str):
        raise ValueError("TESSA_CONTACT_SUBJECT_REQUIRED")
    source_v, source_f, source_frames = _read(source_npz, SOURCE_KEYS)
    candidate_v, candidate_f, candidate_frames = _read(candidate_npz, CANDIDATE_KEYS)
    exact = bool(
        np.array_equal(source_v, candidate_v)
        and np.array_equal(source_f, candidate_f)
        and np.array_equal(source_frames, candidate_frames)
    )
    result = {
        "schema": SCHEMA,
        "status": PASS if exact else MASKED,
        "subject_id": subject_id,
        "parents": {
            "source": {"path": str(source_npz), "sha256": source_sha256},
            "candidate": {"path": str(candidate_npz), "sha256": candidate_sha256},
        },
        "contract": "EXACT_INDEXED_REST_AND_ALL_SAMPLED_ACTION_FRAMES_ONLY",
        "source_frame_count": int(len(source_frames)),
        "candidate_frame_count": int(len(candidate_frames)),
        "rest_and_sampled_frames_index_identical": exact,
        "authority": {
            "source_relative_contact_equivalence_certified": exact,
            "new_or_increased_contact_excluded_on_identical_sampled_frames": exact,
            "physical_collision_freedom_claimed": False,
            "continuous_time_motion_qualified": False,
            "training_authorized": False,
            "product_authority_minted": False,
        },
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    partial.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_npz, args.candidate_npz, args.output,
                   source_sha256=args.source_sha256,
                   candidate_sha256=args.candidate_sha256,
                   subject_id=args.subject_id)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"]}))


if __name__ == "__main__":
    main()
