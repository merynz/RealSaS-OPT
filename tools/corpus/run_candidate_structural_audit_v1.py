#!/usr/bin/env python3
from __future__ import annotations

"""Run one fail-closed cheap structural geometry audit for a corpus candidate."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1


def _load_geometry(path: Path):
    with np.load(path, allow_pickle=False) as z:
        vertex_key = next(
            (key for key in ("rest_vertices_source_world", "vertices_source_world", "vertices") if key in z.files),
            None,
        )
        face_key = next((key for key in ("faces", "triangles") if key in z.files), None)
        owner_key = next(
            (key for key in ("face_owner_privileged", "face_owner", "triangle_owner") if key in z.files),
            None,
        )
        if not vertex_key or not face_key or not owner_key:
            raise ValueError("STRUCTURAL_AUDIT_GEOMETRY_ARRAYS_MISSING")
        return (
            np.asarray(z[vertex_key], dtype=np.float64),
            np.asarray(z[face_key], dtype=np.int64),
            np.asarray(z[owner_key], dtype=np.int32),
        )


def _load_posed(path: Path | None, vertex_count: int):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as z:
        key = next(
            (
                key
                for key in (
                    "posed_vertices_source_world",
                    "isolated_position",
                    "posed_vertices",
                )
                if key in z.files
            ),
            None,
        )
        if not key:
            raise ValueError("STRUCTURAL_AUDIT_POSED_ARRAY_MISSING")
        value = np.asarray(z[key], dtype=np.float64)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3 or value.shape[1:] != (vertex_count, 3):
        raise ValueError("STRUCTURAL_AUDIT_POSED_ARRAY_LAYOUT")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--posed-npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    geometry = args.geometry_npz.expanduser().resolve()
    if not geometry.is_file():
        raise FileNotFoundError(geometry)
    posed_path = args.posed_npz.expanduser().resolve() if args.posed_npz else None
    if posed_path is not None and not posed_path.is_file():
        raise FileNotFoundError(posed_path)

    vertices, faces, owner = _load_geometry(geometry)
    posed = _load_posed(posed_path, len(vertices))
    result = audit_structural_geometry_v1(
        vertices,
        faces,
        owner,
        posed_vertices=posed,
        subject_id=args.subject_id,
    )
    result["inputs"] = {
        "geometry_npz": str(geometry),
        "posed_npz": str(posed_path) if posed_path else None,
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "subject_id": result["subject_id"],
                "counts": result["counts"],
                "failure_classes_observed": result["failure_classes_observed"],
                "audit_identity_sha256": result["audit_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
