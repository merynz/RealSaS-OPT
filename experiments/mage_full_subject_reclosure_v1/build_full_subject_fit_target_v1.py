from __future__ import annotations

"""Build the corrected Mage FIT teacher only after raster/teacher scope agreement passes.

The output has the exact historical H1 teacher payload shape (vertices, faces, center,
half_extent) but is generated from the full skin-supported normalized source instead of
the old armature-modifier-only subset. No model is executed and no optimizer exists.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.audit_full_subject_fit_authority import audit


SCHEMA = "RealSaS.MageFullSubjectFitTarget.v1"
NORMALIZED_SHA256 = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args) -> dict:
    normalized = Path(args.normalized)
    if _sha(normalized) != NORMALIZED_SHA256:
        raise RuntimeError("MAGE_NORMALIZED_SOURCE_SHA_DRIFT")
    cameras = tuple(Path(x) for x in args.cameras)
    observations = tuple(Path(x) for x in args.observations)
    authority = audit(
        normalized,
        cameras,
        observations,
        alpha_threshold=8,
        min_recall=0.98,
        min_precision=0.98,
        min_iou=0.97,
        skin_mass_epsilon=1e-8,
    )
    if not authority["fit_authorized"]:
        raise RuntimeError("FULL_SUBJECT_FIT_AUTHORITY_GATE_FAILED")

    with np.load(normalized, allow_pickle=False) as d:
        source_vertices = np.asarray(d["vertices_source"], dtype=np.float32)
        source_faces = np.asarray(d["faces"], dtype=np.int64)
        source_skin = np.asarray(d["skin"], dtype=np.float64)
    supported = source_skin.sum(axis=1) > 1e-8
    used = np.flatnonzero(supported)
    remap = np.full(len(source_vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    keep_face = supported[source_faces].all(axis=1)
    faces = remap[source_faces[keep_face]]
    vertices = source_vertices[used]
    if np.any(faces < 0) or int(faces.max()) >= len(vertices):
        raise RuntimeError("FULL_SUBJECT_FACE_REMAP_INVALID")

    camera0 = json.loads(cameras[0].read_text(encoding="utf-8"))
    center = np.asarray(camera0["center"], dtype=np.float32)
    half = np.asarray([float(camera0["half_extent"])], dtype=np.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=vertices,
        faces=faces.astype(np.int32),
        center=center,
        half_extent=half,
    )
    target_sha = _sha(out)
    manifest = {
        "schema": SCHEMA,
        "status": "PASS__CORRECTED_FULL_SUBJECT_FIT_TARGET_BUILT",
        "normalized_source_sha256": NORMALIZED_SHA256,
        "fit_authority_status": authority["status"],
        "fit_authority_minimums": {
            "recall": min(v["recall"] for v in authority["views"]),
            "precision": min(v["precision"] for v in authority["views"]),
            "iou": min(v["iou"] for v in authority["views"]),
        },
        "teacher_policy": authority["teacher_policy"],
        "teacher_vertex_count": int(len(vertices)),
        "teacher_face_count": int(len(faces)),
        "excluded_zero_or_unsupported_vertex_count": int((~supported).sum()),
        "excluded_not_fully_supported_face_count": int((~keep_face).sum()),
        "normalization_center": center.astype(float).tolist(),
        "normalization_half_extent": float(half[0]),
        "camera_hashes": [v["camera_sha256"] for v in authority["views"]],
        "observation_hashes": [v["observation_sha256"] for v in authority["views"]],
        "target_file": out.name,
        "target_sha256": target_sha,
        "optimizer_constructed": False,
        "fit_executed": False,
        "product_pass_claimed": False,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest["status"])
    print("vertices=", len(vertices), "faces=", len(faces))
    print("target_sha256=", target_sha)
    print(out)
    print(manifest_path)
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--normalized", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
