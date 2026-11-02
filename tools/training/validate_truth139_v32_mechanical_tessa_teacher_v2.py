#!/usr/bin/env python3
"""V2 independent selected-teacher court entrypoint.

Keeps the V1 rebind policy while fixing its xyz self-intersection index to be
explicitly three-dimensional.
"""
from __future__ import annotations

import numpy as np

from tools.training import validate_truth139_v32_mechanical_tessa_teacher_v1 as impl
from tools.training.truth139_v32_spatial_index_v1 import new_rtree3


def rest_self_intersection_3d(vertices, faces):
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


impl.rest_self_intersection = rest_self_intersection_3d


if __name__ == "__main__":
    impl.main()
