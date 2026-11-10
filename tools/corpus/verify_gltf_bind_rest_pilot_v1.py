"""Independent, read-only, hash-bound bind-rest raw glTF versus importer pilot.

This diagnostic is *not* source/observation admission. It replays the historical
default-node reader and the versioned bind-rest reader against the exact original
imported.json, retaining the source and original importer bytes unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF
from tools.corpus.gltf_bind_scene_v2 import SourceGLTFBindRestV2

SCHEMA = "RealSaS.Corpus370GLTFBindRestPilot.v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_hausdorff(left, right, *, scale: float) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if (a.ndim != 2 or a.shape[1:] != (3,) or not len(a)
            or b.ndim != 2 or b.shape[1:] != (3,) or not len(b)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("BIND_PILOT_GEOMETRY_LAYOUT")
    return float(max(cKDTree(a).query(b)[0].max(),
                     cKDTree(b).query(a)[0].max()) / scale)


def inspect(source: Path, imported: Path, *, expected_source_sha256: str,
            expected_import_sha256: str, subject_id: str) -> dict:
    source = Path(source).resolve()
    imported = Path(imported).resolve()
    if source.suffix.lower() not in (".gltf", ".glb"):
        raise ValueError("BIND_PILOT_GLTF_REQUIRED")
    if not subject_id:
        raise ValueError("BIND_PILOT_SUBJECT_REQUIRED")
    if (sha256(source) != expected_source_sha256
            or sha256(imported) != expected_import_sha256):
        raise ValueError("BIND_PILOT_PARENT_DRIFT")
    importer = json.loads(imported.read_text())
    if (importer.get("import", {}).get("source_sha256") != expected_source_sha256
            or importer.get("import", {}).get("pose_policy")
                != "SOURCE_BIND_REST__NO_ANIMATION"):
        raise ValueError("BIND_PILOT_IMPORT_SCOPE")
    historical = SourceGLTF(source).geometry()
    corrected = SourceGLTFBindRestV2(source).geometry()
    if (historical["bone_count"] != corrected["bone_count"]
            or historical["bone_count"] != importer["bone_count"]):
        raise ValueError("BIND_PILOT_BONE_CARDINALITY")
    for key in ("faces", "meshes"):
        if historical[key] != corrected[key]:
            raise ValueError("BIND_PILOT_SOURCE_TOPOLOGY_CHANGED:" + key)
    a = np.asarray(historical["vertices"], dtype=np.float64)
    b = np.asarray(corrected["vertices"], dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(a, axis=0)))
    return {
        "schema": SCHEMA,
        "status": "BIND_REST_NUMERIC_DIAGNOSTIC__NO_ADMISSION",
        "subject_id": subject_id,
        "source": {"path": str(source), "sha256": expected_source_sha256},
        "admitted_import": {"path": str(imported), "sha256": expected_import_sha256},
        "historical_relative_vertex_error": relative_hausdorff(
            a, importer["vertices"], scale=diagonal),
        "corrected_relative_vertex_error": relative_hausdorff(
            b, importer["vertices"], scale=diagonal),
        "corrected_primitive_count": len(corrected["bind_rest_attachments"]),
        "raw_bone_count": historical["bone_count"],
        "source_topology_identity_preserved": True,
        "read_only": True,
        "training_authorized": False,
        "observation_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imported", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-import-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = inspect(args.source, args.imported,
                    expected_source_sha256=args.expected_source_sha256,
                    expected_import_sha256=args.expected_import_sha256,
                    subject_id=args.subject_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: value[k] for k in
                      ("subject_id", "historical_relative_vertex_error",
                       "corrected_relative_vertex_error", "corrected_primitive_count")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
