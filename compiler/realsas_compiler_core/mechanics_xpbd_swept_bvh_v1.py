from __future__ import annotations

"""Bounded, deterministic swept-AABB BVH for canonical collision primitives.

This is a broadphase only. It never creates new canonical surface IDs,
changes MIRA/Graphics authority, or declares that an overlap is a contact.
The primitive boxes must bound *both endpoints* of each linear trajectory.
"""

from dataclasses import dataclass

import numpy as np

from .types import QualificationError

Box3 = tuple[float, float, float]


@dataclass(frozen=True)
class SweptAABBPrimitiveV1:
    minimum: Box3
    maximum: Box3


@dataclass(frozen=True)
class SweptBVHNodeV1:
    minimum: Box3
    maximum: Box3
    left: int
    right: int
    primitive_indices: tuple[int, ...]


@dataclass(frozen=True)
class SweptBVHV1:
    primitives: tuple[SweptAABBPrimitiveV1, ...]
    nodes: tuple[SweptBVHNodeV1, ...]
    root: int


@dataclass(frozen=True)
class SweptBVHQueryV1:
    candidate_indices: tuple[int, ...]
    visited_nodes: int
    examined_leaf_primitives: int


def swept_aabb_from_endpoints_v1(
    start: tuple[np.ndarray, ...],
    end: tuple[np.ndarray, ...],
    *, margin: float,
) -> SweptAABBPrimitiveV1:
    if not start or len(start) != len(end):
        raise QualificationError("XPBD_SWEPT_BVH_PRIMITIVE_ENDPOINT_DRIFT")
    coords = np.stack((*start, *end))
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise QualificationError("XPBD_SWEPT_BVH_PRIMITIVE_NONFINITE")
    if not np.isfinite(margin) or margin < 0.:
        raise QualificationError("XPBD_SWEPT_BVH_MARGIN_INVALID")
    lo = coords.min(axis=0) - margin
    hi = coords.max(axis=0) + margin
    return SweptAABBPrimitiveV1(
        tuple(map(float, lo)), tuple(map(float, hi)),
    )


def prepare_swept_bvh_v1(
    primitives: tuple[SweptAABBPrimitiveV1, ...],
    *, max_primitives: int = 16384,
    leaf_size: int = 4,
) -> SweptBVHV1:
    if (not isinstance(primitives, tuple)
            or not primitives
            or type(max_primitives) is not int
            or not 1 <= len(primitives) <= max_primitives <= 16384
            or type(leaf_size) is not int or not 1 <= leaf_size <= 16):
        raise QualificationError("XPBD_SWEPT_BVH_RESOURCE_BOUND")
    for item in primitives:
        if type(item) is not SweptAABBPrimitiveV1:
            raise QualificationError("XPBD_SWEPT_BVH_PRIMITIVE_INVALID")
        lo, hi = np.asarray(item.minimum), np.asarray(item.maximum)
        if (lo.shape != (3,) or hi.shape != (3,)
                or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                or bool(np.any(lo > hi))):
            raise QualificationError("XPBD_SWEPT_BVH_BOUNDS_INVALID")
    nodes: list[SweptBVHNodeV1] = []

    def build(indices: tuple[int, ...]) -> int:
        lows = np.asarray([primitives[i].minimum for i in indices], dtype=np.float64)
        highs = np.asarray([primitives[i].maximum for i in indices], dtype=np.float64)
        lo = tuple(map(float, lows.min(axis=0)))
        hi = tuple(map(float, highs.max(axis=0)))
        if len(indices) <= leaf_size:
            nodes.append(SweptBVHNodeV1(
                lo, hi, -1, -1, tuple(sorted(indices)),
            ))
        else:
            midpoints = .5 * (lows + highs)
            axis = int(np.argmax(midpoints.max(axis=0) - midpoints.min(axis=0)))
            ordered = sorted(indices, key=lambda i: (
                .5 * (primitives[i].minimum[axis] + primitives[i].maximum[axis]), i,
            ))
            half = len(ordered) // 2
            left = build(tuple(ordered[:half]))
            right = build(tuple(ordered[half:]))
            nodes.append(SweptBVHNodeV1(lo, hi, left, right, ()))
        if len(nodes) > 2 * len(primitives):
            raise QualificationError("XPBD_SWEPT_BVH_NODE_BOUND")
        return len(nodes) - 1

    root = build(tuple(range(len(primitives))))
    return SweptBVHV1(primitives, tuple(nodes), root)


def query_swept_bvh_v1(
    bvh: SweptBVHV1,
    query: SweptAABBPrimitiveV1,
    *, max_candidates: int,
) -> SweptBVHQueryV1:
    if (type(max_candidates) is not int or not 1 <= max_candidates <= 32768
            or not isinstance(bvh, SweptBVHV1)
            or type(query) is not SweptAABBPrimitiveV1):
        raise QualificationError("XPBD_SWEPT_BVH_QUERY_INVALID")
    qlo = np.asarray(query.minimum)
    qhi = np.asarray(query.maximum)
    if (not np.isfinite(qlo).all() or not np.isfinite(qhi).all()
            or np.any(qlo > qhi)):
        raise QualificationError("XPBD_SWEPT_BVH_QUERY_INVALID")

    def overlaps(minimum: Box3, maximum: Box3) -> bool:
        return bool(np.all(qhi >= minimum) and np.all(qlo <= maximum))

    stack = [bvh.root]
    hits: list[int] = []
    visited = 0
    examined = 0
    while stack:
        node = bvh.nodes[stack.pop()]
        visited += 1
        if not overlaps(node.minimum, node.maximum):
            continue
        if node.primitive_indices:
            for i in node.primitive_indices:
                examined += 1
                primitive = bvh.primitives[i]
                if overlaps(primitive.minimum, primitive.maximum):
                    hits.append(i)
                    if len(hits) > max_candidates:
                        raise QualificationError("XPBD_SWEPT_BVH_CANDIDATE_BOUND")
        else:
            # Stack order does not affect returned candidates; they are sorted.
            stack.append(node.right)
            stack.append(node.left)
    return SweptBVHQueryV1(tuple(sorted(hits)), visited, examined)
