from __future__ import annotations

"""Training-only mechanical-core projection for Geppetto FIT witnesses.

The projection is deliberately anonymous and generic:
- skin support is measured from numeric dense skin weights;
- required bridge controls are graph-theoretic ancestors strictly between
  two supported controls;
- assembly-only root chains above the highest supported control are excluded;
- source bone names, source control IDs, family names and hand-written index
  lists are never selection authority.

This module produces teacher targets only. Nothing returned here may enter
RiggingSurfaceIR or product inference.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1


SCHEMA = "RealSaS.GeppettoMechanicalCoreTeacherProjection.v1"
RULE_ID = "SKIN_SUPPORTED_PLUS_SUPPORTED_BRIDGES__ASSEMBLY_ONLY_ROOT_EXCLUDED"


def _sha(payload: object) -> str:
    def norm(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        return v
    raw = json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _validate_parent_forest(parents: np.ndarray) -> None:
    p = np.asarray(parents, dtype=np.int64)
    n = len(p)
    if p.shape != (n,) or n == 0:
        raise ValueError("parents must be non-empty [J]")
    if np.any((p < -1) | (p >= n)):
        raise ValueError("parent index out of range")
    if np.any(p == np.arange(n, dtype=np.int64)):
        raise ValueError("self-parent")
    state = np.zeros(n, dtype=np.int8)
    def visit(i: int) -> None:
        if state[i] == 1:
            raise ValueError("teacher parent graph contains a cycle")
        if state[i] == 2:
            return
        state[i] = 1
        q = int(p[i])
        if q >= 0:
            visit(q)
        state[i] = 2
    for i in range(n):
        visit(i)


def _source_positions_world(
    bone_heads_canonical: np.ndarray,
    inverse_canonical_transform: np.ndarray,
) -> np.ndarray:
    h = np.asarray(bone_heads_canonical, dtype=np.float64)
    t = np.asarray(inverse_canonical_transform, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 3 or not np.isfinite(h).all():
        raise ValueError("bone_heads must be finite [J,3]")
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("inverse_canonical_transform must be finite [4,4]")
    homo = np.concatenate([h, np.ones((len(h), 1), dtype=np.float64)], axis=1)
    out = homo @ t.T
    w = out[:, 3]
    if np.any(np.abs(w) <= 1e-12):
        raise ValueError("inverse transform produced invalid homogeneous coordinate")
    xyz = out[:, :3] / w[:, None]
    if not np.isfinite(xyz).all():
        raise ValueError("inverse transform produced non-finite source positions")
    return xyz.astype(np.float32)


def derive_mechanical_core_indices_v1(
    parents,
    deform_mask,
    skin,
    *,
    skin_mass_epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected source rows and their nearest-selected parent rows.

    A required bridge is an ancestor strictly between a skin-supported control
    and its nearest skin-supported ancestor. If a supported control has no
    supported ancestor, the unsupported chain above it is not included; this is
    the assembly-only-root exclusion.
    """
    p = np.asarray(parents, dtype=np.int64)
    d = np.asarray(deform_mask, dtype=bool)
    w = np.asarray(skin, dtype=np.float64)
    _validate_parent_forest(p)
    j = len(p)
    if d.shape != (j,):
        raise ValueError("deform_mask shape mismatch")
    if w.ndim != 2 or w.shape[1] != j:
        raise ValueError("skin must be [V,J]")
    if not np.isfinite(w).all() or np.any(w < -1e-10):
        raise ValueError("skin must be finite and non-negative")
    eps = float(skin_mass_epsilon)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("skin_mass_epsilon must be finite and non-negative")

    mass = np.maximum(w, 0.0).sum(axis=0)
    supported = d & (mass > eps)
    supported_rows = set(map(int, np.flatnonzero(supported)))
    if not supported_rows:
        raise ValueError("teacher contains no skin-supported deform controls")

    selected = set(supported_rows)
    for child in sorted(supported_rows):
        between: list[int] = []
        anc = int(p[child])
        while anc >= 0 and anc not in supported_rows:
            between.append(anc)
            anc = int(p[anc])
        if anc in supported_rows:
            selected.update(between)

    rows = np.asarray(sorted(selected), dtype=np.int64)
    row_set = set(map(int, rows.tolist()))
    source_parent = np.full(len(rows), -1, dtype=np.int64)
    for oi, src in enumerate(rows.tolist()):
        anc = int(p[src])
        while anc >= 0 and anc not in row_set:
            anc = int(p[anc])
        source_parent[oi] = anc
    return rows, source_parent


def _structural_serialize(
    positions: np.ndarray,
    source_rows: np.ndarray,
    source_parent_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Parent-before-child content serialization with no source-row tie authority.

    Exact automorphic subtrees are allowed to remain equivalent. Their output
    rows contain identical recursive content, so internal ordering among exact
    ties conveys no usable source identity.
    """
    pos = np.asarray(positions, dtype=np.float32)
    rows = np.asarray(source_rows, dtype=np.int64)
    spar = np.asarray(source_parent_rows, dtype=np.int64)
    n = len(rows)
    if pos.shape != (n, 3) or spar.shape != (n,):
        raise ValueError("structural serialization shape mismatch")
    row_to_local = {int(r): i for i, r in enumerate(rows.tolist())}
    local_parent = np.asarray(
        [-1 if int(q) < 0 else row_to_local[int(q)] for q in spar.tolist()],
        dtype=np.int64,
    )
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, q in enumerate(local_parent.tolist()):
        if q < 0:
            roots.append(i)
        else:
            children[q].append(i)
    if not roots:
        raise ValueError("mechanical core contains no root")

    def pkey(x: np.ndarray) -> tuple[float, float, float]:
        return tuple(0.0 if float(v) == 0.0 else float(v) for v in x)

    state = np.zeros(n, dtype=np.int8)
    sigs: list[tuple | None] = [None] * n
    def signature(i: int) -> tuple:
        if state[i] == 1:
            raise ValueError("mechanical core contains a cycle")
        if state[i] == 2:
            assert sigs[i] is not None
            return sigs[i]  # type: ignore[return-value]
        state[i] = 1
        child_sigs = tuple(sorted(signature(c) for c in children[i]))
        sig = (pkey(pos[i]), child_sigs)
        sigs[i] = sig
        state[i] = 2
        return sig
    for r in roots:
        signature(r)
    if np.any(state != 2):
        raise ValueError("mechanical core node unreachable from roots")

    automorphic: list[int] = []
    def grouped_order(nodes: list[int]) -> list[int]:
        groups: dict[tuple, list[int]] = {}
        for i in nodes:
            s = sigs[i]
            assert s is not None
            groups.setdefault(s, []).append(i)
        out: list[int] = []
        for s in sorted(groups):
            members = groups[s]
            if len(members) > 1:
                automorphic.append(len(members))
            out.extend(members)
        return out

    order: list[int] = []
    def emit(i: int) -> None:
        order.append(i)
        for c in grouped_order(children[i]):
            emit(c)
    for r in grouped_order(roots):
        emit(r)
    perm = np.asarray(order, dtype=np.int64)
    if len(perm) != n or set(perm.tolist()) != set(range(n)):
        raise RuntimeError("structural serialization did not emit an exact permutation")
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    parent_serial = np.asarray(
        [-1 if local_parent[old] < 0 else inv[int(local_parent[old])] for old in perm],
        dtype=np.int64,
    )
    for child, parent in enumerate(parent_serial.tolist()):
        if parent >= child:
            raise RuntimeError("parent-before-child invariant violated")
    return perm, parent_serial, tuple(sorted(automorphic, reverse=True))


@dataclass(frozen=True)
class MechanicalCoreTeacherProjectionV1:
    source_indices: tuple[int, ...]
    source_parent_indices: tuple[int, ...]
    positions_world: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    skin_mass: np.ndarray
    rule_id: str = RULE_ID
    schema_version: str = SCHEMA

    @property
    def count(self) -> int:
        return int(len(self.source_indices))

    @property
    def content_hash(self) -> str:
        return _sha({
            "schema": self.schema_version,
            "rule_id": self.rule_id,
            "source_indices": self.source_indices,
            "source_parent_indices": self.source_parent_indices,
            "positions_world": self.positions_world,
            "parent_indices": self.parent_indices,
            "root_mask": self.root_mask,
            "skin_mass": self.skin_mass,
        })


def project_mechanical_core_from_arrays_v1(
    *,
    parents,
    deform_mask,
    skin,
    bone_heads,
    inverse_canonical_transform,
    skin_mass_epsilon: float = 1e-8,
) -> MechanicalCoreTeacherProjectionV1:
    p = np.asarray(parents, dtype=np.int64)
    w = np.asarray(skin, dtype=np.float64)
    rows, source_parent = derive_mechanical_core_indices_v1(
        p, deform_mask, w, skin_mass_epsilon=skin_mass_epsilon
    )
    world_all = _source_positions_world(bone_heads, inverse_canonical_transform)
    world = world_all[rows]
    mass_all = np.maximum(w, 0.0).sum(axis=0).astype(np.float64)
    perm, parent_serial, _automorphic = _structural_serialize(world, rows, source_parent)
    rows_s = rows[perm]
    spar_s = source_parent[perm]
    world_s = world[perm]
    mass_s = mass_all[rows_s]
    root = parent_serial < 0
    if not root.any():
        raise RuntimeError("projected mechanical core has no root")
    return MechanicalCoreTeacherProjectionV1(
        source_indices=tuple(map(int, rows_s.tolist())),
        source_parent_indices=tuple(map(int, spar_s.tolist())),
        positions_world=world_s.astype(np.float32),
        parent_indices=parent_serial.astype(np.int64),
        root_mask=root.astype(bool),
        skin_mass=mass_s.astype(np.float64),
    )


def project_mechanical_core_from_normalized_npz_v1(
    path: str | Path,
    *,
    skin_mass_epsilon: float = 1e-8,
) -> MechanicalCoreTeacherProjectionV1:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = (
            "parents", "deform_mask", "skin", "bone_heads",
            "inverse_canonical_transform",
        )
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"normalized corpus record missing fields:{missing}")
        return project_mechanical_core_from_arrays_v1(
            parents=z["parents"],
            deform_mask=z["deform_mask"],
            skin=z["skin"],
            bone_heads=z["bone_heads"],
            inverse_canonical_transform=z["inverse_canonical_transform"],
            skin_mass_epsilon=skin_mass_epsilon,
        )


def to_geppetto_teacher_target_v1(
    projection: MechanicalCoreTeacherProjectionV1,
    normalization,
) -> GeppettoTeacherTargetV1:
    """Normalize only after product surface normalization is known."""
    pos = np.asarray(normalization.normalize(projection.positions_world), dtype=np.float32)
    if pos.shape != projection.positions_world.shape or not np.isfinite(pos).all():
        raise ValueError("surface normalization produced invalid teacher positions")
    return GeppettoTeacherTargetV1(
        positions_normalized=pos,
        parent_indices=np.asarray(projection.parent_indices, dtype=np.int64).copy(),
        root_mask=np.asarray(projection.root_mask, dtype=bool).copy(),
        valid=True,
    )


__all__ = [
    "SCHEMA",
    "RULE_ID",
    "MechanicalCoreTeacherProjectionV1",
    "derive_mechanical_core_indices_v1",
    "project_mechanical_core_from_arrays_v1",
    "project_mechanical_core_from_normalized_npz_v1",
    "to_geppetto_teacher_target_v1",
]
