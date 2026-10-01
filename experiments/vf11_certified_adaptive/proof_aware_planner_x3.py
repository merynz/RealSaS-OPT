from __future__ import annotations

"""Proof-aware fixed-lattice narrow-band planner for VF11 X3 V2.

The proof partition is coarser than the d13 extraction blocks. A terminal
strict-sign EMPTY leaf safely prunes all descendants. A GRAPH or
PROVEN_ZERO_EXISTS leaf keeps all descendants query-active, but existence is
audited at proof-leaf scope rather than incorrectly assigned to every
descendant block.

X3 remains research-only and ordinary float64 is not a shipping certificate.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from block_mc_x2 import (
    BlockMCResult,
    _edge_key_from_global_grid_vertex,
    _audit_topology,
    _require_skimage,
)


EMPTY_STATES = {"EMPTY_POSITIVE", "EMPTY_NEGATIVE"}


@dataclass(frozen=True)
class PlannedBlock:
    root_anchor_id: int
    block_xyz: tuple[int, int, int]
    octree_path: str
    proof_path: str
    proof_relative_depth: int
    proof_state: str
    proof_c0_state: str | None
    selected_axis: str | None
    planner_class: str


def _child_digit(xbit: int, ybit: int, zbit: int) -> str:
    return str(int(xbit) * 4 + int(ybit) * 2 + int(zbit))


def _decode_digit(ch: str) -> tuple[int, int, int]:
    v = int(ch)
    if not 0 <= v <= 7:
        raise ValueError(f"invalid octree digit {ch}")
    return ((v >> 2) & 1, (v >> 1) & 1, v & 1)


def block_path_d11(block_xyz: Iterable[int]) -> str:
    """Map one of 8^3 aligned d13 blocks inside a d8 root to its d11 path."""
    b = np.asarray(tuple(block_xyz), dtype=np.int64).reshape(3)
    if np.any(b < 0) or np.any(b >= 8):
        raise ValueError("block coordinates must be in [0,7]^3")
    chars = []
    for shift in (2, 1, 0):
        chars.append(
            _child_digit(
                (int(b[0]) >> shift) & 1,
                (int(b[1]) >> shift) & 1,
                (int(b[2]) >> shift) & 1,
            )
        )
    return "".join(chars)


def proof_path_cell_bounds_d13(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive-exclusive d13 cell bounds inside one d8 root.

    The root contains 32 cells per axis. Every octree path digit halves all
    three axes, so relative depths 0..3 align exactly to the d13 lattice.
    """
    if len(path) > 3:
        raise ValueError("X3 target d13 supports proof relative depth <= 3")
    lo = np.zeros(3, dtype=np.int64)
    hi = np.full(3, 32, dtype=np.int64)
    for ch in path:
        bits = np.asarray(_decode_digit(ch), dtype=np.int64)
        mid = (lo + hi) // 2
        lo = np.where(bits == 0, lo, mid)
        hi = np.where(bits == 0, mid, hi)
    if np.any(hi <= lo):
        raise RuntimeError(f"invalid proof-path bounds {path}: {lo} {hi}")
    return lo, hi


def final_partition_records(c2_v3_profile: dict, c2_v4_profile: dict) -> list[dict]:
    """Return the true leaf partition, excluding historical ancestors."""
    records = list(c2_v3_profile["records"]) + list(c2_v4_profile["new_records"])
    by_root = defaultdict(list)
    for r in records:
        by_root[int(r["root_anchor_id"])].append(r)

    final = []
    for root, rows in by_root.items():
        for r in rows:
            path = str(r["path"])
            rel = int(r["relative_depth"])
            has_descendant = any(
                int(q["relative_depth"]) > rel
                and str(q["path"]).startswith(path)
                for q in rows
            )
            if not has_descendant:
                final.append(r)

    for root, rows in by_root.items():
        leaves = [r for r in final if int(r["root_anchor_id"]) == root]
        weight = sum(1.0 / float(8 ** int(r["relative_depth"])) for r in leaves)
        if abs(weight - 1.0) > 1e-12:
            raise RuntimeError(
                f"final partition does not cover root {root}: volume weight={weight}"
            )
    return final


def planner_class_for_leaf(record: dict) -> str:
    state = str(record["state"])
    c0 = record.get("c0_state")
    if state in EMPTY_STATES:
        return "CERTIFIED_EMPTY"
    if state == "GRAPH":
        return "ACTIVE_GRAPH"
    if state != "UNRESOLVED":
        raise ValueError(f"unexpected proof state {state}")
    if c0 == "PROVEN_ZERO_EXISTS":
        return "ACTIVE_ZERO_EXISTS"
    if c0 == "UNKNOWN":
        return "ACTIVE_UNKNOWN"
    raise ValueError(f"unexpected unresolved C0 state {c0}")


def plan_d13_blocks(c2_v3_profile: dict, c2_v4_profile: dict) -> list[PlannedBlock]:
    leaves = final_partition_records(c2_v3_profile, c2_v4_profile)
    by_root = defaultdict(dict)
    for r in leaves:
        root = int(r["root_anchor_id"])
        path = str(r["path"])
        if path in by_root[root]:
            raise RuntimeError(f"duplicate final proof path root={root} path={path}")
        by_root[root][path] = r

    out = []
    for root in sorted(by_root):
        leaf_map = by_root[root]
        for bx in range(8):
            for by in range(8):
                for bz in range(8):
                    path = block_path_d11((bx, by, bz))
                    matches = [
                        leaf_map[p]
                        for n in range(0, 4)
                        for p in [path[:n]]
                        if p in leaf_map
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"block must map to exactly one proof leaf: "
                            f"root={root} block={(bx,by,bz)} path={path} matches={len(matches)}"
                        )
                    r = matches[0]
                    out.append(
                        PlannedBlock(
                            root_anchor_id=root,
                            block_xyz=(bx, by, bz),
                            octree_path=path,
                            proof_path=str(r["path"]),
                            proof_relative_depth=int(r["relative_depth"]),
                            proof_state=str(r["state"]),
                            proof_c0_state=r.get("c0_state"),
                            selected_axis=r.get("selected_axis"),
                            planner_class=planner_class_for_leaf(r),
                        )
                    )

    if len(out) != 512 * len(by_root):
        raise RuntimeError("unexpected d13 block count")
    return out


def mixed_cube_mask(values_zyx: np.ndarray) -> np.ndarray:
    """Boolean [Z-1,Y-1,X-1] mask of d13 cubes with both signs at corners."""
    f = np.asarray(values_zyx, dtype=np.float64)
    if f.ndim != 3 or min(f.shape) < 2:
        raise ValueError("values_zyx must be a 3D corner grid")
    if not np.isfinite(f).all():
        raise ValueError("non-finite target grid")
    if np.any(f == 0.0):
        raise ValueError("X3 exact-zero target corners are forbidden")
    corners = [
        f[0:-1, 0:-1, 0:-1],
        f[0:-1, 0:-1, 1:],
        f[0:-1, 1:, 0:-1],
        f[0:-1, 1:, 1:],
        f[1:, 0:-1, 0:-1],
        f[1:, 0:-1, 1:],
        f[1:, 1:, 0:-1],
        f[1:, 1:, 1:],
    ]
    mn = np.minimum.reduce(corners)
    mx = np.maximum.reduce(corners)
    return (mn < 0.0) & (mx > 0.0)


def leaf_mixed_cube_count(values_zyx: np.ndarray, proof_path: str) -> int:
    mask = mixed_cube_mask(values_zyx)
    lo, hi = proof_path_cell_bounds_d13(proof_path)
    # mask is Z,Y,X while bounds are X,Y,Z.
    sub = mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    return int(np.count_nonzero(sub))


def audit_graph_leaf_lines(
    values_zyx: np.ndarray,
    proof_path: str,
    selected_axis: str,
) -> dict:
    """Audit d13 sampled capture of a certified graph leaf.

    Because the graph certificate proves opposite-sign entry/exit faces and
    strict regularity along the selected coordinate axis, every transverse
    lattice line spanning the proof leaf should contain at least one adjacent
    opposite-sign pair on any aligned subdivision.
    """
    f = np.asarray(values_zyx, dtype=np.float64)
    if f.shape != (33, 33, 33):
        raise ValueError("X3 graph-line audit expects one 33^3 d13 root grid")
    if np.any(f == 0.0):
        raise ValueError("X3 exact-zero target corners are forbidden")
    axis_name = str(selected_axis)
    axis = {"X": 0, "Y": 1, "Z": 2}.get(axis_name)
    if axis is None:
        raise ValueError(f"GRAPH leaf missing valid selected_axis: {selected_axis}")

    lo, hi = proof_path_cell_bounds_d13(proof_path)
    node_lo = lo
    node_hi = hi + 1

    total = 0
    miss = 0
    if axis == 0:
        for z in range(node_lo[2], node_hi[2]):
            for y in range(node_lo[1], node_hi[1]):
                line = f[z, y, node_lo[0]:node_hi[0]]
                total += 1
                if not np.any(line[:-1] * line[1:] < 0.0):
                    miss += 1
    elif axis == 1:
        for z in range(node_lo[2], node_hi[2]):
            for x in range(node_lo[0], node_hi[0]):
                line = f[z, node_lo[1]:node_hi[1], x]
                total += 1
                if not np.any(line[:-1] * line[1:] < 0.0):
                    miss += 1
    else:
        for y in range(node_lo[1], node_hi[1]):
            for x in range(node_lo[0], node_hi[0]):
                line = f[node_lo[2]:node_hi[2], y, x]
                total += 1
                if not np.any(line[:-1] * line[1:] < 0.0):
                    miss += 1

    return {
        "selected_axis": axis_name,
        "transverse_line_count": int(total),
        "line_miss_count": int(miss),
        "line_capture_fraction": 1.0 if total == 0 else float((total - miss) / total),
    }


def block_corner_brackets(
    root_grid_zyx: np.ndarray,
    block_xyz: tuple[int, int, int],
    *,
    block_cells: int = 4,
) -> bool:
    f = np.asarray(root_grid_zyx)
    bx, by, bz = map(int, block_xyz)
    x0, y0, z0 = bx * block_cells, by * block_cells, bz * block_cells
    slab = f[
        z0 : z0 + block_cells + 1,
        y0 : y0 + block_cells + 1,
        x0 : x0 + block_cells + 1,
    ]
    if slab.shape != (block_cells + 1,) * 3:
        raise ValueError(f"bad block slab shape {slab.shape}")
    return float(slab.min()) < 0.0 < float(slab.max())


def unique_query_corner_count(
    blocks: Iterable[PlannedBlock],
    *,
    block_cells: int = 4,
) -> int:
    by_root = defaultdict(set)
    for b in blocks:
        if b.planner_class == "CERTIFIED_EMPTY":
            continue
        bx, by, bz = b.block_xyz
        for z in range(bz * block_cells, bz * block_cells + block_cells + 1):
            for y in range(by * block_cells, by * block_cells + block_cells + 1):
                for x in range(bx * block_cells, bx * block_cells + block_cells + 1):
                    by_root[b.root_anchor_id].add((x, y, z))
    return sum(len(v) for v in by_root.values())


def extract_selected_d13_blocks(
    root_grid_zyx: np.ndarray,
    selected_blocks_xyz: Iterable[tuple[int, int, int]],
    *,
    root_origin_xyz: tuple[float, float, float],
    root_spacing_xyz: tuple[float, float, float],
    block_cells: int = 4,
    level: float = 0.0,
) -> BlockMCResult:
    measure = _require_skimage()
    field = np.asarray(root_grid_zyx, dtype=np.float32)
    if field.shape != (33, 33, 33):
        raise ValueError("X3 local target grid must be exactly 33^3")
    if np.any(field == float(level)):
        raise ValueError("X3 exact-zero target corners are forbidden")

    origin = np.asarray(root_origin_xyz, dtype=np.float64)
    spacing = np.asarray(root_spacing_xyz, dtype=np.float64)
    key_to_vid = {}
    vertices = []
    edge_keys = []
    faces = []
    processed = 0
    skipped = 0

    for bxyz in sorted(set(tuple(map(int, b)) for b in selected_blocks_xyz)):
        bx, by, bz = bxyz
        if min(bxyz) < 0 or max(bxyz) >= 8:
            raise ValueError(f"block outside d13 root lattice: {bxyz}")
        x0, y0, z0 = bx * block_cells, by * block_cells, bz * block_cells
        slab = field[
            z0 : z0 + block_cells + 1,
            y0 : y0 + block_cells + 1,
            x0 : x0 + block_cells + 1,
        ]
        smin, smax = float(slab.min()), float(slab.max())
        if not (smin < float(level) < smax):
            skipped += 1
            continue
        processed += 1

        v_zyx, local_faces, _n, _ = measure.marching_cubes(
            slab,
            level=float(level),
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
            method="lewiner",
        )
        gxyz = np.stack(
            [
                x0 + v_zyx[:, 2],
                y0 + v_zyx[:, 1],
                z0 + v_zyx[:, 0],
            ],
            axis=-1,
        ).astype(np.float64)

        remap = {}
        for i, g in enumerate(gxyz):
            key = _edge_key_from_global_grid_vertex(g)
            world = origin + g * spacing
            vid = key_to_vid.get(key)
            if vid is None:
                vid = len(vertices)
                key_to_vid[key] = vid
                vertices.append(world)
                edge_keys.append(key)
            else:
                if float(np.linalg.norm(vertices[vid] - world)) > 2e-5 * float(np.max(spacing)):
                    raise RuntimeError(f"shared vertex mismatch for edge {key}")
            remap[int(i)] = int(vid)

        for tri in np.asarray(local_faces, dtype=np.int64):
            faces.append(tuple(remap[int(i)] for i in tri))

    result = BlockMCResult(
        vertices=(
            np.asarray(vertices, dtype=np.float64)
            if vertices else np.empty((0, 3), dtype=np.float64)
        ),
        faces=(
            np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            if faces else np.empty((0, 3), dtype=np.int64)
        ),
        edge_keys=tuple(edge_keys),
        processed_block_count=processed,
        skipped_block_count=skipped,
    )
    audit = _audit_topology(result.vertices, result.faces)
    if audit["degenerate_face_count"] or audit["duplicate_face_count"] or audit["nonmanifold_edge_count"]:
        raise RuntimeError(f"X3 selected-block topology alarm: {audit}")
    return result
