from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from teacher_contracts_v1 import ProjectedControlV1, SkeletonTeacherProjectionV1


HISTORICAL_SOURCE_BLOB_SHA = "3f82dd5559ab3620ed9d47005316642037d9d856"


def _validate_parent_forest(parents: np.ndarray) -> None:
    n = len(parents)
    for i, parent in enumerate(parents.tolist()):
        if parent < -1 or parent >= n:
            raise ValueError(f"parent index out of range at bone {i}: {parent}")
        if parent == i:
            raise ValueError(f"self-parent at bone {i}")
    for start in range(n):
        seen: set[int] = set()
        current = start
        while current != -1:
            if current in seen:
                raise ValueError(f"parent cycle detected from bone {start}")
            seen.add(current)
            current = int(parents[current])


def _nearest_deform_ancestor(index: int, parents: np.ndarray, deform: np.ndarray) -> int | None:
    parent = int(parents[index])
    while parent != -1:
        if bool(deform[parent]):
            return parent
        parent = int(parents[parent])
    return None


def _bfs_ids(controls: list[ProjectedControlV1]) -> tuple[str, ...]:
    by_id = {control.control_id: control for control in controls}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    source_index = {control.control_id: control.source_bone_index for control in controls}
    for control in controls:
        if control.parent_control_id is None:
            roots.append(control.control_id)
        else:
            children[control.parent_control_id].append(control.control_id)
    roots.sort(key=source_index.__getitem__)
    for child_ids in children.values():
        child_ids.sort(key=source_index.__getitem__)
    queue = deque(roots)
    output: list[str] = []
    while queue:
        control_id = queue.popleft()
        if control_id not in by_id:
            raise AssertionError(control_id)
        output.append(control_id)
        queue.extend(children.get(control_id, ()))
    if len(output) != len(controls):
        raise ValueError("projected deform-control graph is not a forest")
    return tuple(output)


def project_skeleton_teacher_v1(bone_heads, bone_tails, parents, deform_mask) -> SkeletonTeacherProjectionV1:
    """Project source-rig arrays to anonymous Geppetto teacher controls.

    This is training/evaluation truth construction only. It does not emit a
    product proposal, choose a canonical root/tree, or mint canonical IDs.
    """
    heads = np.asarray(bone_heads, dtype=np.float32)
    tails = np.asarray(bone_tails, dtype=np.float32)
    parent = np.asarray(parents, dtype=np.int64)
    deform = np.asarray(deform_mask, dtype=bool)

    if heads.ndim != 2 or heads.shape[1] != 3 or tails.shape != heads.shape:
        raise ValueError(f"bad bone head/tail shapes: {heads.shape}/{tails.shape}")
    n = len(heads)
    if parent.shape != (n,) or deform.shape != (n,):
        raise ValueError(f"bad parent/deform shapes: {parent.shape}/{deform.shape} for {n} bones")
    if not np.isfinite(heads).all() or not np.isfinite(tails).all():
        raise ValueError("non-finite source skeleton")
    _validate_parent_forest(parent)

    deform_indices = np.flatnonzero(deform).astype(int).tolist()
    if not deform_indices:
        raise ValueError("no deform controls")
    control_ids = {index: f"TC:{index:05d}" for index in deform_indices}
    controls: list[ProjectedControlV1] = []
    for index in deform_indices:
        deform_parent = _nearest_deform_ancestor(index, parent, deform)
        controls.append(
            ProjectedControlV1(
                control_id=control_ids[index],
                source_bone_index=index,
                position=tuple(map(float, heads[index])),
                teacher_tail=tuple(map(float, tails[index])),
                parent_control_id=None if deform_parent is None else control_ids[deform_parent],
                deform=True,
            )
        )

    bfs = _bfs_ids(controls)
    roots = tuple(
        sorted(
            (c.control_id for c in controls if c.parent_control_id is None),
            key=lambda cid: int(cid.split(":")[1]),
        )
    )
    projection = SkeletonTeacherProjectionV1(
        controls=tuple(controls),
        root_control_ids=roots,
        bfs_control_ids=bfs,
        source_bone_count=n,
        deform_control_count=len(controls),
        skipped_helper_count=n - len(controls),
        metadata={
            "authority_class": "TEACHER_EVALUATOR_ONLY",
            "historical_source_blob_sha": HISTORICAL_SOURCE_BLOB_SHA,
            "control_location": "SOURCE_BONE_HEAD",
            "parent_policy": "NEAREST_DEFORM_ANCESTOR",
            "source_tail_usage": "TEACHER_AUDIT_ONLY__NOT_PRODUCTION_FEATURE",
            "canonical_ids_created": False,
            "multi_root_preserved": True,
        },
    )
    projection.validate()
    return projection


def bfs_training_order_v1(
    projection: SkeletonTeacherProjectionV1,
    *,
    sibling_seed: int | None = None,
) -> tuple[str, ...]:
    """Return authoritative BFS or a training-only sibling permutation."""
    projection.validate()
    if sibling_seed is None:
        return projection.bfs_control_ids

    rng = np.random.default_rng(int(sibling_seed))
    by_id = projection.control_by_id()
    children: dict[str | None, list[str]] = defaultdict(list)
    for control in projection.controls:
        children[control.parent_control_id].append(control.control_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda cid: by_id[cid].source_bone_index)
        rng.shuffle(child_ids)

    current = list(children[None])
    output: list[str] = []
    while current:
        output.extend(current)
        next_level: list[str] = []
        for control_id in current:
            next_level.extend(children.get(control_id, ()))
        current = next_level
    if set(output) != set(by_id) or len(output) != len(by_id):
        raise AssertionError("randomized BFS corrupted control set")
    return tuple(output)


def control_skin_column_map_v1(projection: SkeletonTeacherProjectionV1) -> dict[str, int]:
    """Return explicit teacher skin-column provenance for each anonymous control."""
    projection.validate()
    return {control.control_id: control.source_bone_index for control in projection.controls}
