from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from contracts_v0_1 import ProjectedControlV1, SkeletonTeacherProjectionV1


def _validate_parent_forest(parents: np.ndarray) -> None:
    n = len(parents)
    for i, p in enumerate(parents.tolist()):
        if p < -1 or p >= n:
            raise ValueError(f"parent index out of range at bone {i}: {p}")
        if p == i:
            raise ValueError(f"self-parent at bone {i}")

    # Every parent chain must terminate at -1. This catches arbitrary cycles,
    # including cycles that contain only helper/non-deform bones.
    for start in range(n):
        seen: set[int] = set()
        cur = start
        while cur != -1:
            if cur in seen:
                raise ValueError(f"parent cycle detected from bone {start}")
            seen.add(cur)
            cur = int(parents[cur])


def _nearest_deform_ancestor(index: int, parents: np.ndarray, deform: np.ndarray) -> int | None:
    p = int(parents[index])
    while p != -1:
        if bool(deform[p]):
            return p
        p = int(parents[p])
    return None


def _bfs_ids(controls: list[ProjectedControlV1]) -> tuple[str, ...]:
    by = {c.control_id: c for c in controls}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    source_index = {c.control_id: c.source_bone_index for c in controls}
    for c in controls:
        if c.parent_control_id is None:
            roots.append(c.control_id)
        else:
            children[c.parent_control_id].append(c.control_id)
    roots.sort(key=source_index.__getitem__)
    for k in children:
        children[k].sort(key=source_index.__getitem__)
    q = deque(roots)
    out: list[str] = []
    while q:
        x = q.popleft()
        if x not in by:
            raise AssertionError(x)
        out.append(x)
        q.extend(children.get(x, ()))
    if len(out) != len(controls):
        raise ValueError("projected deform-control graph is not a forest")
    return tuple(out)


def project_skeleton_teacher_v1(
    bone_heads,
    bone_tails,
    parents,
    deform_mask,
) -> SkeletonTeacherProjectionV1:
    """Project source rig arrays to anonymous Geppetto control truth.

    Contract:
      * one anonymous control per deform bone, located at the source bone head;
      * helper/non-deform bones do not become model targets;
      * parent relation skips helper chains to the nearest deform ancestor;
      * source bone indices/tails remain teacher-only provenance;
      * no canonical/product IDs are created here.

    This projection deliberately preserves multiple deform roots rather than
    silently inventing a super-root. A later training/data gate must quantify
    and handle multi-root assets explicitly.
    """
    H = np.asarray(bone_heads, dtype=np.float32)
    T = np.asarray(bone_tails, dtype=np.float32)
    P = np.asarray(parents, dtype=np.int64)
    D = np.asarray(deform_mask, dtype=bool)

    if H.ndim != 2 or H.shape[1] != 3 or T.shape != H.shape:
        raise ValueError(f"bad bone head/tail shapes: {H.shape}/{T.shape}")
    n = len(H)
    if P.shape != (n,) or D.shape != (n,):
        raise ValueError(f"bad parent/deform shapes: {P.shape}/{D.shape} for {n} bones")
    if not np.isfinite(H).all() or not np.isfinite(T).all():
        raise ValueError("non-finite source skeleton")
    _validate_parent_forest(P)

    deform_ids = np.flatnonzero(D).astype(int).tolist()
    if not deform_ids:
        raise ValueError("no deform controls")
    control_id = {i: f"TC:{i:05d}" for i in deform_ids}
    controls: list[ProjectedControlV1] = []
    for i in deform_ids:
        pa = _nearest_deform_ancestor(i, P, D)
        controls.append(
            ProjectedControlV1(
                control_id=control_id[i],
                source_bone_index=i,
                position=tuple(map(float, H[i])),
                teacher_tail=tuple(map(float, T[i])),
                parent_control_id=None if pa is None else control_id[pa],
                deform=True,
            )
        )

    bfs = _bfs_ids(controls)
    roots = tuple(c.control_id for c in controls if c.parent_control_id is None)
    roots = tuple(sorted(roots, key=lambda cid: int(cid.split(":")[1])))
    out = SkeletonTeacherProjectionV1(
        controls=tuple(controls),
        root_control_ids=roots,
        bfs_control_ids=bfs,
        source_bone_count=n,
        deform_control_count=len(controls),
        skipped_helper_count=n - len(controls),
        metadata={
            "control_location": "SOURCE_BONE_HEAD",
            "parent_policy": "NEAREST_DEFORM_ANCESTOR",
            "source_tail_usage": "TEACHER_AUDIT_ONLY__NOT_PRODUCTION_FEATURE",
            "canonical_ids_created": False,
            "multi_root_preserved": True,
        },
    )
    out.validate()
    return out


def bfs_training_order_v1(
    projection: SkeletonTeacherProjectionV1,
    *,
    sibling_seed: int | None = None,
) -> tuple[str, ...]:
    """Return BFS order; optionally randomize siblings without changing depth.

    Deterministic projection remains the authority. Sibling randomization is a
    training-only augmentation inspired by autoregressive rigging literature.
    """
    projection.validate()
    if sibling_seed is None:
        return projection.bfs_control_ids

    rng = np.random.default_rng(int(sibling_seed))
    by = projection.control_by_id()
    children: dict[str | None, list[str]] = defaultdict(list)
    for c in projection.controls:
        children[c.parent_control_id].append(c.control_id)
    for k, v in children.items():
        v.sort(key=lambda cid: by[cid].source_bone_index)
        rng.shuffle(v)

    current = list(children[None])
    out: list[str] = []
    while current:
        out.extend(current)
        nxt: list[str] = []
        for cid in current:
            nxt.extend(children.get(cid, ()))
        current = nxt
    if set(out) != set(by) or len(out) != len(by):
        raise AssertionError("randomized BFS corrupted control set")
    return tuple(out)


def control_skin_column_map_v1(projection: SkeletonTeacherProjectionV1) -> dict[str, int]:
    """Teacher skin column carried by each anonymous projected control."""
    projection.validate()
    return {c.control_id: c.source_bone_index for c in projection.controls}
