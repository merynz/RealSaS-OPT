from __future__ import annotations

"""Mechanically meaningful Geppetto teacher target V2.

Training/evaluation only. Learner input remains the qualified RiggingSurfaceIR.

V1 intentionally projected the source armature down to a visually clean
skin-supported core. V2 replaces that aesthetic objective with a mechanical one:

1. retain every deform articulation that carries non-trivial skin mass;
2. retain zero-mass structural bridges needed to preserve those articulations;
3. retain one mechanically useful root-motion anchor when the source has one
   unique root above the supported body;
4. retain one geometry-continuous terminal extension attached to an otherwise
   terminal selected articulation (e.g. hand/tool attachment), but never chase
   detached helper/IK chains;
5. emit an explicit terminal tip for every retained terminal source bone so the
   point-skeleton represents the full final bone span rather than silently
   truncating it at the last bone head;
6. exclude zero-mass branches that neither carry skin, bridge supported
   mechanics, anchor root motion, nor continue a selected terminal span.

No bone names, family names, source row lists, or fixed target count are used.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np


SCHEMA = "RealSaS.GeppettoMechanicallyMeaningfulRigTarget.v2"
RULE = (
    "SKIN_SUPPORTED_PLUS_STRUCTURAL_BRIDGES_PLUS_ROOT_MOTION_ANCHOR_"
    "PLUS_TERMINAL_EXTENSIONS_AND_TIPS"
)
POSITION_AUTHORITY = "WORLD_SOURCE_FRAME_EXPLICIT"

ROLE_SKIN_SUPPORTED = 0
ROLE_STRUCTURAL_BRIDGE = 1
ROLE_ROOT_MOTION_ANCHOR = 2
ROLE_TERMINAL_EXTENSION_SOURCE = 3
ROLE_TERMINAL_TIP_SYNTHETIC = 4

ROLE_NAMES = {
    ROLE_SKIN_SUPPORTED: "SKIN_SUPPORTED_ARTICULATION",
    ROLE_STRUCTURAL_BRIDGE: "STRUCTURAL_BRIDGE",
    ROLE_ROOT_MOTION_ANCHOR: "ROOT_MOTION_ANCHOR",
    ROLE_TERMINAL_EXTENSION_SOURCE: "TERMINAL_EXTENSION_SOURCE",
    ROLE_TERMINAL_TIP_SYNTHETIC: "TERMINAL_TIP_SYNTHETIC",
}


@dataclass(frozen=True)
class MechanicallyMeaningfulRigTargetV2:
    positions_world: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    source_indices_provenance_only: np.ndarray
    skin_mass: np.ndarray
    role_codes: np.ndarray
    rule: str = RULE
    position_authority: str = POSITION_AUTHORITY
    schema: str = SCHEMA

    @property
    def count(self) -> int:
        return int(len(self.positions_world))

    @property
    def role_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for code, name in ROLE_NAMES.items():
            out[name] = int(np.count_nonzero(np.asarray(self.role_codes) == int(code)))
        return out


def _validate(
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
    bone_tails_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(parents, dtype=np.int64)
    d = np.asarray(deform_mask, dtype=bool)
    w = np.asarray(skin, dtype=np.float64)
    h = np.asarray(bone_heads_world, dtype=np.float64)
    t = np.asarray(bone_tails_world, dtype=np.float64)
    j = int(len(p))
    if j < 1 or d.shape != (j,) or h.shape != (j, 3) or t.shape != (j, 3):
        raise ValueError("teacher skeleton shape mismatch")
    if w.ndim != 2 or w.shape[1] != j:
        raise ValueError("teacher skin shape mismatch")
    if not np.isfinite(h).all() or not np.isfinite(t).all() or not np.isfinite(w).all():
        raise ValueError("teacher target contains non-finite values")
    if np.any(w < 0):
        raise ValueError("teacher skin contains negative weights")
    if np.any((p < -1) | (p >= j)) or np.any(p == np.arange(j)):
        raise ValueError("teacher parent array is invalid")
    for start in range(j):
        seen: set[int] = set()
        node = start
        while node >= 0:
            if node in seen:
                raise ValueError("teacher parent graph contains a cycle")
            seen.add(node)
            node = int(p[node])
    return p, d, w, h, t


def _children(parents: np.ndarray) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(len(parents))]
    for child, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent >= 0:
            out[int(parent)].append(int(child))
    return out


def _has_descendant_in(
    node: int,
    children: list[list[int]],
    members: set[int],
) -> bool:
    stack = list(children[int(node)])
    while stack:
        cur = int(stack.pop())
        if cur in members:
            return True
        stack.extend(children[cur])
    return False


def _select_supported_and_bridges(
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin_mass: np.ndarray,
    *,
    support_epsilon: float,
) -> tuple[set[int], dict[int, int]]:
    supported = set(
        np.flatnonzero(deform_mask & (skin_mass > float(support_epsilon))).tolist()
    )
    if not supported:
        raise ValueError("no skin-supported deform articulations")
    selected = set(supported)
    role: dict[int, int] = {i: ROLE_SKIN_SUPPORTED for i in supported}

    for child in tuple(sorted(supported)):
        chain: list[int] = []
        parent = int(parents[child])
        while parent >= 0:
            if parent in supported:
                for node in chain:
                    selected.add(int(node))
                    role.setdefault(int(node), ROLE_STRUCTURAL_BRIDGE)
                break
            chain.append(parent)
            parent = int(parents[parent])
    return selected, role


def _add_root_motion_anchor(
    selected: set[int],
    role: dict[int, int],
    parents: np.ndarray,
    children: list[list[int]],
    supported: set[int],
    heads: np.ndarray,
) -> None:
    roots = [int(i) for i in np.flatnonzero(parents < 0).tolist()]
    candidates = [r for r in roots if _has_descendant_in(r, children, supported)]
    if not candidates:
        return
    if len(candidates) != 1:
        raise ValueError("mechanically meaningful V2 currently requires one supported source root")
    root = candidates[0]
    if root in selected:
        return
    descendant_heads = heads[np.asarray(sorted(supported), dtype=np.int64)]
    span = float(np.linalg.norm(descendant_heads.max(axis=0) - descendant_heads.min(axis=0)))
    if not math.isfinite(span) or span <= 1e-8:
        raise ValueError("supported body span is degenerate")
    # A root coincident with the selected body is redundant. A spatially
    # distinct source root is useful for whole-body/root motion.
    nearest = min(float(np.linalg.norm(heads[root] - heads[s])) for s in supported)
    if nearest > 1e-4 * span:
        selected.add(root)
        role[root] = ROLE_ROOT_MOTION_ANCHOR


def _add_terminal_extensions(
    selected: set[int],
    role: dict[int, int],
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin_mass: np.ndarray,
    heads: np.ndarray,
    tails: np.ndarray,
    children: list[list[int]],
    supported: set[int],
    *,
    support_epsilon: float,
    continuity_ratio_max: float,
) -> None:
    span = float(np.linalg.norm(heads.max(axis=0) - heads.min(axis=0)))
    min_length = max(1e-8, 1e-5 * span)
    base = tuple(sorted(selected))
    selected_children = {
        b: [c for c in children[b] if c in selected]
        for b in base
    }
    additions: list[int] = []
    for bone in base:
        if parents[bone] < 0 or selected_children[bone]:
            continue
        plen = float(np.linalg.norm(tails[bone] - heads[bone]))
        if plen <= min_length:
            continue
        candidates: list[tuple[float, int]] = []
        for child in children[bone]:
            child = int(child)
            if child in selected:
                continue
            if not bool(deform_mask[child]):
                continue
            if float(skin_mass[child]) > float(support_epsilon):
                continue
            if _has_descendant_in(child, children, supported):
                continue
            clen = float(np.linalg.norm(tails[child] - heads[child]))
            if clen <= min_length:
                continue
            continuity = float(np.linalg.norm(heads[child] - tails[bone])) / max(plen, min_length)
            if continuity <= float(continuity_ratio_max):
                candidates.append((continuity, child))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            additions.append(int(candidates[0][1]))

    for node in additions:
        selected.add(node)
        role[node] = ROLE_TERMINAL_EXTENSION_SOURCE


def _project_source_parents(parents: np.ndarray, selected: list[int]) -> np.ndarray:
    row = {int(old): i for i, old in enumerate(selected)}
    out = np.empty(len(selected), dtype=np.int64)
    for i, old in enumerate(selected):
        parent = int(parents[old])
        while parent >= 0 and parent not in row:
            parent = int(parents[parent])
        out[i] = -1 if parent < 0 else int(row[parent])
    return out


def _position_key(x: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("invalid target position")
    return tuple(0.0 if float(v) == 0.0 else float(v) for v in a)


def _content_serialization(
    positions: np.ndarray,
    parents: np.ndarray,
    roles: np.ndarray,
) -> np.ndarray:
    n = int(len(positions))
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent < 0:
            roots.append(i)
        else:
            children[int(parent)].append(i)

    state = np.zeros(n, dtype=np.int8)
    signatures: list[tuple | None] = [None] * n

    def signature(node: int) -> tuple:
        if state[node] == 1:
            raise ValueError("projected target contains cycle")
        if state[node] == 2:
            result = signatures[node]
            assert result is not None
            return result
        state[node] = 1
        child_sigs = tuple(sorted(signature(c) for c in children[node]))
        result = (int(roles[node]), _position_key(positions[node]), child_sigs)
        signatures[node] = result
        state[node] = 2
        return result

    for root in roots:
        signature(root)
    if np.any(state != 2):
        raise ValueError("projected target contains unreachable nodes")

    order: list[int] = []

    def emit(node: int) -> None:
        order.append(node)
        groups: dict[tuple, list[int]] = {}
        for child in children[node]:
            sig = signatures[child]
            assert sig is not None
            groups.setdefault(sig, []).append(child)
        for sig in sorted(groups):
            for child in groups[sig]:
                emit(child)

    root_groups: dict[tuple, list[int]] = {}
    for root in roots:
        sig = signatures[root]
        assert sig is not None
        root_groups.setdefault(sig, []).append(root)
    for sig in sorted(root_groups):
        for root in root_groups[sig]:
            emit(root)

    permutation = np.asarray(order, dtype=np.int64)
    if permutation.shape != (n,) or set(permutation.tolist()) != set(range(n)):
        raise RuntimeError("target serialization is not an exact permutation")
    return permutation


def build_mechanically_meaningful_target_v2(
    *,
    parents: np.ndarray,
    deform_mask: np.ndarray,
    skin: np.ndarray,
    bone_heads_world: np.ndarray,
    bone_tails_world: np.ndarray,
    support_epsilon: float = 1e-8,
    continuity_ratio_max: float = 0.75,
) -> MechanicallyMeaningfulRigTargetV2:
    if support_epsilon < 0:
        raise ValueError("support_epsilon must be non-negative")
    if not (0.0 < continuity_ratio_max <= 1.5):
        raise ValueError("continuity_ratio_max outside bounded policy")

    parents, deform_mask, skin, heads, tails = _validate(
        parents, deform_mask, skin, bone_heads_world, bone_tails_world
    )
    skin_mass = skin.sum(axis=0)
    children = _children(parents)
    supported = set(
        np.flatnonzero(deform_mask & (skin_mass > float(support_epsilon))).tolist()
    )
    selected, role = _select_supported_and_bridges(
        parents, deform_mask, skin_mass, support_epsilon=support_epsilon
    )
    _add_root_motion_anchor(selected, role, parents, children, supported, heads)
    _add_terminal_extensions(
        selected,
        role,
        parents,
        deform_mask,
        skin_mass,
        heads,
        tails,
        children,
        supported,
        support_epsilon=support_epsilon,
        continuity_ratio_max=continuity_ratio_max,
    )

    source_nodes = sorted(selected)
    source_row = {old: i for i, old in enumerate(source_nodes)}
    positions = [heads[old].copy() for old in source_nodes]
    provenance = [int(old) for old in source_nodes]
    masses = [float(skin_mass[old]) for old in source_nodes]
    roles = [int(role[old]) for old in source_nodes]
    projected_parents = _project_source_parents(parents, source_nodes).tolist()

    selected_child_count = {old: 0 for old in source_nodes}
    for old in source_nodes:
        parent = int(parents[old])
        while parent >= 0 and parent not in selected:
            parent = int(parents[parent])
        if parent >= 0:
            selected_child_count[parent] += 1

    # Point-skeletons otherwise truncate every terminal source bone at its head.
    # Add the actual source-bone tail as a synthetic terminal articulation.
    span = float(np.linalg.norm(heads.max(axis=0) - heads.min(axis=0)))
    min_tip_length = max(1e-8, 1e-5 * span)
    for old in source_nodes:
        if selected_child_count[old] != 0:
            continue
        tip = tails[old].copy()
        if float(np.linalg.norm(tip - heads[old])) <= min_tip_length:
            continue
        positions.append(tip)
        provenance.append(-1)
        masses.append(0.0)
        roles.append(ROLE_TERMINAL_TIP_SYNTHETIC)
        projected_parents.append(int(source_row[old]))

    positions_a = np.asarray(positions, dtype=np.float64)
    parents_a = np.asarray(projected_parents, dtype=np.int64)
    roles_a = np.asarray(roles, dtype=np.int64)
    provenance_a = np.asarray(provenance, dtype=np.int64)
    masses_a = np.asarray(masses, dtype=np.float64)

    permutation = _content_serialization(positions_a, parents_a, roles_a)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    serialized_parents = np.asarray(
        [
            -1 if parents_a[old] < 0 else int(inverse[int(parents_a[old])])
            for old in permutation.tolist()
        ],
        dtype=np.int64,
    )
    if any(
        parent >= child
        for child, parent in enumerate(serialized_parents.tolist())
        if parent >= 0
    ):
        raise RuntimeError("target serialization violated parent-before-child invariant")
    if int(np.count_nonzero(serialized_parents < 0)) != 1:
        raise RuntimeError("mechanically meaningful target must have one root")

    target = MechanicallyMeaningfulRigTargetV2(
        positions_world=positions_a[permutation].astype(np.float32),
        parent_indices=serialized_parents,
        root_mask=(serialized_parents < 0),
        source_indices_provenance_only=provenance_a[permutation],
        skin_mass=masses_a[permutation],
        role_codes=roles_a[permutation],
    )
    if target.role_counts["SKIN_SUPPORTED_ARTICULATION"] != len(supported):
        raise RuntimeError("skin-supported articulation loss during V2 target construction")
    if target.role_counts["TERMINAL_TIP_SYNTHETIC"] < 1:
        raise RuntimeError("V2 target failed to preserve terminal mechanical span")
    return target


def target_content_sha256_v2(target: MechanicallyMeaningfulRigTargetV2) -> str:
    payload = {
        "schema": target.schema,
        "rule": target.rule,
        "position_authority": target.position_authority,
        "positions_world": np.asarray(target.positions_world, np.float32).tolist(),
        "parents": np.asarray(target.parent_indices, np.int64).tolist(),
        "root_mask": np.asarray(target.root_mask, bool).astype(int).tolist(),
        "role_codes": np.asarray(target.role_codes, np.int64).tolist(),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA",
    "RULE",
    "POSITION_AUTHORITY",
    "ROLE_SKIN_SUPPORTED",
    "ROLE_STRUCTURAL_BRIDGE",
    "ROLE_ROOT_MOTION_ANCHOR",
    "ROLE_TERMINAL_EXTENSION_SOURCE",
    "ROLE_TERMINAL_TIP_SYNTHETIC",
    "ROLE_NAMES",
    "MechanicallyMeaningfulRigTargetV2",
    "build_mechanically_meaningful_target_v2",
    "target_content_sha256_v2",
]
