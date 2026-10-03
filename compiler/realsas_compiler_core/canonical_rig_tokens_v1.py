from __future__ import annotations

"""Deterministic, ID-free tokenization of QualifiedSkeletonIR for learned consumers.

This module creates an experimental conditioning representation. It does not
alter QualifiedSkeletonIR, does not mint skeleton authority, and never exposes
source proposal IDs or source bone names to learned consumers.

Serialization is parent-before-child and content ordered. Canonical joint IDs are
used only to resolve the already-qualified graph; they are not serialized into
the token payload or token-set hash.
"""

from dataclasses import dataclass, asdict
import math
from typing import Iterable

from .hashing import content_sha256
from .types import QualificationError, QualifiedSkeletonIR


_SERIALIZER = "CONTENT_ORDERED_PARENT_BEFORE_CHILD_V1"
_SCHEMA = "RealSaS.CanonicalRigTokenSetIR.v1"


@dataclass(frozen=True)
class CanonicalRigTokenV1:
    sequence_index: int
    parent_sequence_index: int | None
    is_root: bool
    position_normalized: tuple[float, float, float]
    parent_delta_normalized: tuple[float, float, float]
    parent_distance_normalized: float
    child_count: int
    token_kind: str = "JOINT"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CanonicalRigTokenSetV1:
    tokens: tuple[CanonicalRigTokenV1, ...]
    skeleton_lineage_hash: str
    normalization_center_xyz: tuple[float, float, float]
    normalization_half_extent: float
    serializer: str
    token_set_hash: str
    schema_version: str = _SCHEMA

    def to_dict(self):
        return asdict(self)


def _finite3(x: Iterable[float]) -> tuple[float, float, float]:
    row = tuple(float(v) for v in x)
    if len(row) != 3 or not all(math.isfinite(v) for v in row):
        raise QualificationError("CANONICAL_RIG_TOKEN_POSITION_INVALID")
    return row


def _validate_tree(skeleton: QualifiedSkeletonIR):
    joints = tuple(skeleton.joints)
    if not joints:
        raise QualificationError("CANONICAL_RIG_TOKEN_SKELETON_EMPTY")
    by_id = {str(j.canonical_joint_id): j for j in joints}
    if len(by_id) != len(joints):
        raise QualificationError("CANONICAL_RIG_TOKEN_DUPLICATE_JOINT_ID")
    root_id = str(skeleton.root_id)
    if root_id not in by_id:
        raise QualificationError("CANONICAL_RIG_TOKEN_ROOT_MISSING")
    roots = [j for j in joints if j.parent_canonical_id is None]
    if len(roots) != 1 or str(roots[0].canonical_joint_id) != root_id:
        raise QualificationError("CANONICAL_RIG_TOKEN_ROOT_CONTRACT_INVALID")

    children = {jid: [] for jid in by_id}
    for jid, joint in by_id.items():
        pid = joint.parent_canonical_id
        if pid is None:
            continue
        pid = str(pid)
        if pid not in by_id or pid == jid:
            raise QualificationError("CANONICAL_RIG_TOKEN_PARENT_INVALID")
        children[pid].append(jid)

    state = {jid: 0 for jid in by_id}

    def visit(jid: str):
        if state[jid] == 1:
            raise QualificationError("CANONICAL_RIG_TOKEN_CYCLE")
        if state[jid] == 2:
            return
        state[jid] = 1
        for child in children[jid]:
            visit(child)
        state[jid] = 2

    visit(root_id)
    if any(v != 2 for v in state.values()):
        raise QualificationError("CANONICAL_RIG_TOKEN_DISCONNECTED")
    return by_id, children, root_id


def _normalization(by_id):
    positions = [_finite3(j.position) for j in by_id.values()]
    mins = tuple(min(p[k] for p in positions) for k in range(3))
    maxs = tuple(max(p[k] for p in positions) for k in range(3))
    center = tuple((mins[k] + maxs[k]) * 0.5 for k in range(3))
    half = max(max(abs(p[k] - center[k]) for p in positions) for k in range(3))
    if not math.isfinite(half):
        raise QualificationError("CANONICAL_RIG_TOKEN_NORMALIZATION_INVALID")
    if half <= 1e-12:
        half = 1.0
    return center, float(half)


def _normalized_position(joint, center, half):
    p = _finite3(joint.position)
    return tuple((p[k] - center[k]) / half for k in range(3))


def build_canonical_rig_tokens_v1(skeleton: QualifiedSkeletonIR) -> CanonicalRigTokenSetV1:
    by_id, children, root_id = _validate_tree(skeleton)
    center, half = _normalization(by_id)
    pos = {jid: _normalized_position(j, center, half) for jid, j in by_id.items()}

    # Content signatures deliberately exclude canonical/source IDs.
    sig_cache = {}

    def signature(jid: str):
        if jid in sig_cache:
            return sig_cache[jid]
        child_sigs = tuple(sorted(signature(c) for c in children[jid]))
        row = (
            tuple(round(float(x), 12) for x in pos[jid]),
            child_sigs,
        )
        sig_cache[jid] = row
        return row

    signature(root_id)

    order = []

    def emit(jid: str):
        order.append(jid)
        groups = {}
        for child in children[jid]:
            groups.setdefault(signature(child), []).append(child)
        for sig in sorted(groups):
            # Exact content-tied children are semantically indistinguishable to
            # this token contract. Their identical token subtrees make tie order
            # irrelevant to the serialized payload.
            rows = groups[sig]
            for child in rows:
                emit(child)

    emit(root_id)
    if len(order) != len(by_id):
        raise QualificationError("CANONICAL_RIG_TOKEN_SERIALIZATION_INCOMPLETE")

    seq = {jid: i for i, jid in enumerate(order)}
    tokens = []
    for jid in order:
        joint = by_id[jid]
        parent = None if joint.parent_canonical_id is None else str(joint.parent_canonical_id)
        p = pos[jid]
        if parent is None:
            delta = (0.0, 0.0, 0.0)
            dist = 0.0
            parent_index = None
        else:
            pp = pos[parent]
            delta = tuple(float(p[k] - pp[k]) for k in range(3))
            dist = math.sqrt(sum(v * v for v in delta))
            parent_index = int(seq[parent])
            if parent_index >= seq[jid]:
                raise QualificationError("CANONICAL_RIG_TOKEN_PARENT_ORDER_INVALID")
        tokens.append(CanonicalRigTokenV1(
            sequence_index=int(seq[jid]),
            parent_sequence_index=parent_index,
            is_root=parent is None,
            position_normalized=tuple(float(x) for x in p),
            parent_delta_normalized=delta,
            parent_distance_normalized=float(dist),
            child_count=int(len(children[jid])),
        ))

    payload = {
        "schema_version": _SCHEMA,
        "serializer": _SERIALIZER,
        "normalization_center_xyz": tuple(float(x) for x in center),
        "normalization_half_extent": float(half),
        "tokens": tuple(t.to_dict() for t in tokens),
    }
    return CanonicalRigTokenSetV1(
        tokens=tuple(tokens),
        skeleton_lineage_hash=str(skeleton.skeleton_lineage_hash),
        normalization_center_xyz=tuple(float(x) for x in center),
        normalization_half_extent=float(half),
        serializer=_SERIALIZER,
        token_set_hash=content_sha256(payload),
    )


__all__ = [
    "CanonicalRigTokenV1",
    "CanonicalRigTokenSetV1",
    "build_canonical_rig_tokens_v1",
]
