#!/usr/bin/env python3
"""V2 entrypoint for the V3.2 mechanical TESSA teacher.

V1's mechanics are retained byte-for-byte except for the spatial-index
construction used by self-intersection courts.  Rtree defaults to 2D while
these courts use xyz AABBs; V2 explicitly supplies a 3D index.
"""
from __future__ import annotations

import numpy as np

from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as impl
from tools.training.truth139_v32_spatial_index_v1 import new_rtree3


def has_self_intersection_3d(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    spatial = new_rtree3()

    for i, t in enumerate(tri):
        lo = t.min(axis=0)
        hi = t.max(axis=0)
        spatial.insert(
            i,
            (
                float(lo[0]), float(lo[1]), float(lo[2]),
                float(hi[0]), float(hi[1]), float(hi[2]),
            ),
        )

    for i, t in enumerate(tri):
        lo = t.min(axis=0)
        hi = t.max(axis=0)
        bounds = (
            float(lo[0]), float(lo[1]), float(lo[2]),
            float(hi[0]), float(hi[1]), float(hi[2]),
        )
        for j in spatial.intersection(bounds):
            if j <= i:
                continue
            if len(set(map(int, f[i])) & set(map(int, f[j]))):
                continue
            if impl.tri_intersect_sat(t, tri[j]):
                return True
    return False


# V1 evaluate_candidate resolves this function through its module globals.
impl.has_self_intersection = has_self_intersection_3d


if __name__ == "__main__":
    impl.main()
