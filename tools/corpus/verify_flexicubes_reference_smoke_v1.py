from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SCHEMA = "RealSaS.FFlexiCubesReferenceRun.v1"


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def verify(candidate_npz: Path, run_manifest: Path) -> dict[str, int | bool]:
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("FLEXICUBES_SMOKE_OUTPUT_SCHEMA_INVALID")
    if manifest.get("subject_key") != "SYNTHETIC_SPHERE_SMOKE":
        raise ValueError("FLEXICUBES_SMOKE_SUBJECT_INVALID")

    archive = np.load(candidate_npz, allow_pickle=False)
    vertices = np.asarray(archive["vertices"])
    faces = np.asarray(archive["faces"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("FLEXICUBES_SMOKE_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("FLEXICUBES_SMOKE_FACES_INVALID")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise ValueError("FLEXICUBES_SMOKE_FACE_INDEX_OOB")

    edges = np.sort(
        np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            axis=0,
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))
    if boundary_edges != 0:
        raise ValueError("FLEXICUBES_SMOKE_BOUNDARY_EDGE_PRESENT")
    if nonmanifold_edges != 0:
        raise ValueError("FLEXICUBES_SMOKE_NONMANIFOLD_EDGE_PRESENT")

    uf = _UnionFind(vertices.shape[0])
    for a, b in unique_edges:
        uf.union(int(a), int(b))
    used_vertices = set(map(int, np.unique(faces)))
    component_count = len({uf.find(i) for i in used_vertices})
    if component_count != 1:
        raise ValueError("FLEXICUBES_SMOKE_COMPONENT_COUNT_INVALID")

    euler = int(len(used_vertices) - unique_edges.shape[0] + faces.shape[0])
    if euler != 2:
        raise ValueError("FLEXICUBES_SMOKE_EULER_INVALID")

    result = {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "component_count": component_count,
        "euler_characteristic": euler,
        "pass": True,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-npz", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    args = parser.parse_args()
    verify(args.candidate_npz, args.run_manifest)


if __name__ == "__main__":
    main()
