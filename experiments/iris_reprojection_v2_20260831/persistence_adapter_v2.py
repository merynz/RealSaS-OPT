from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import numpy as np

try:
    from compiler.realsas_compiler_core.types import ObservationEvidenceIR, PersistenceGroup, RiggingSurfaceIR, SurfaceRelation, QualificationError
    from compiler.realsas_compiler_core.surface import build_surface_from_persistence
    from compiler.realsas_compiler_core.hashing import content_sha256
    from compiler.realsas_compiler_core.local_geometry import (
        robust_local_plane_normals_for_rows,
        orient_normal_against_ray,
        attach_dtb_nd1_normals,
        dtb_nd1_operator_hash,
    )
except ImportError:
    from realsas_compiler_core.types import ObservationEvidenceIR, PersistenceGroup, RiggingSurfaceIR, SurfaceRelation, QualificationError
    from realsas_compiler_core.surface import build_surface_from_persistence
    from realsas_compiler_core.hashing import content_sha256
    from realsas_compiler_core.local_geometry import (
        robust_local_plane_normals_for_rows,
        orient_normal_against_ray,
        attach_dtb_nd1_normals,
        dtb_nd1_operator_hash,
    )


_LOCAL_RELATION_OPERATOR = {
    "schema": "RealSaS.IRISV2.ObservedAnchorRasterLocalRelations.v1",
    "neighborhood": "EIGHT_CONNECTED_INFERRED_NATIVE_LATTICE",
    "minimum_common_support_views": 2,
    "world_gate": "MEDIAN_PLUS_6MAD_OR_3X_MEDIAN",
    "source_mesh_used": False,
    "teacher_truth_used": False,
}
_UNSAFE_RELATION_FLAGS = ("UNKNOWN", "UNOBSERVED", "OCCLUDED", "AMBIGUOUS", "UNSUPPORTED")


def _unit(v):
    a = np.asarray(v, np.float64)
    n = float(np.linalg.norm(a))
    if n <= 1e-12 or not math.isfinite(n):
        raise QualificationError("invalid ray forward")
    return a / n


def _P(sample):
    return np.asarray(sample.ray_origin, np.float64) + float(sample.depth) * _unit(sample.ray_forward)


def build_persistence_groups_v2(evidence: ObservationEvidenceIR, *, max_common_frame_error: float = 0.003) -> tuple[PersistenceGroup, ...]:
    groups_meta = evidence.metadata.get("hypothesis_groups", {})
    if not isinstance(groups_meta, dict):
        raise QualificationError("IRIS_V2_MISSING_HYPOTHESIS_GROUPS")
    by_id = {s.observation_id: s for s in evidence.samples}
    if len(by_id) != len(evidence.samples):
        raise QualificationError("duplicate observation ids")
    groups = []
    used = set()
    for gid in sorted(groups_meta):
        ids = tuple(groups_meta[gid])
        if not ids:
            continue
        if any(x not in by_id for x in ids):
            raise QualificationError("persistence group references missing observation")
        if used.intersection(ids):
            raise QualificationError("observation reused across q groups")
        used.update(ids)
        supported = [by_id[x] for x in ids if bool(by_id[x].support)]
        if not supported:
            continue
        P = np.stack([_P(s) for s in supported], axis=0)
        center = P.mean(axis=0)
        err = np.linalg.norm(P - center[None], axis=1)
        if float(err.max(initial=0.0)) > float(max_common_frame_error):
            raise QualificationError(f"IRIS_V2_PERSISTENCE_COMMON_FRAME_ERROR:{gid}:{float(err.max())}")
        groups.append(PersistenceGroup(
            group_id=f"QPV2:{gid}",
            observation_ids=ids,
            method="IRIS_V2_Q_MODE_RECIPROCAL_COMMON_FRAME",
            diagnostics={
                "supported_count": len(supported),
                "max_common_frame_error": float(err.max(initial=0.0)),
                "threshold": float(max_common_frame_error),
                "teacher_identity_used": False,
            },
        ))
    return tuple(groups)


def _surface_gid(node) -> str | None:
    raw = str(node.persistence_group_id or "")
    return raw[len("QPV2:"):] if raw.startswith("QPV2:") else None


def _relation_safe_node(node) -> bool:
    flags = tuple(str(x).upper() for x in node.validity_flags)
    return not any(token in flag for flag in flags for token in _UNSAFE_RELATION_FLAGS)


def _axis_step(values: np.ndarray) -> float | None:
    unique = np.unique(np.round(np.asarray(values, np.float64), decimals=6))
    if len(unique) < 2:
        return None
    diff = np.diff(np.sort(unique))
    diff = diff[diff > 1e-5]
    return None if len(diff) == 0 else float(diff.min())


def attach_observed_local_relations_v2(
    evidence: ObservationEvidenceIR,
    surface: RiggingSurfaceIR,
    *,
    minimum_common_support_views: int = 2,
) -> RiggingSurfaceIR:
    """Attach conservative local topology using only observed anchor-raster locality.

    IRIS V2 emits the exact native anchor raster coordinate for every q/mode group.
    This adapter turns that already-observed locality into a sparse eight-connected
    relation complex. It never consults source mesh triangles, teacher rig/skin,
    semantic part IDs or fitted-family constants. A robust world-distance gate keeps
    locally adjacent but depth-disconnected hypotheses from becoming mesh bridges.
    """
    if minimum_common_support_views < 1:
        raise ValueError("minimum_common_support_views must be positive")
    anchors = evidence.metadata.get("hypothesis_anchor_raster", {})
    if not isinstance(anchors, dict) or not anchors:
        raise QualificationError("IRIS_V2_LOCAL_RELATIONS_REQUIRE_ANCHOR_RASTER_METADATA")
    if surface.metadata.get("raster_coordinate_system") != "PIXEL_CENTER_XY":
        raise QualificationError("IRIS_V2_LOCAL_RELATIONS_REQUIRE_PIXEL_CENTER_XY")

    entries_by_view: dict[int, list[tuple[object, str, np.ndarray]]] = defaultdict(list)
    for node in sorted(surface.surface_nodes, key=lambda n: n.surface_id):
        gid = _surface_gid(node)
        if gid is None or gid not in anchors or not _relation_safe_node(node):
            continue
        rec = anchors[gid]
        if not isinstance(rec, dict) or "view_index" not in rec or "raster_xy" not in rec:
            raise QualificationError(f"IRIS_V2_BAD_ANCHOR_RASTER_METADATA:{gid}")
        xy = np.asarray(rec["raster_xy"], np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise QualificationError(f"IRIS_V2_BAD_ANCHOR_RASTER_COORDINATE:{gid}")
        entries_by_view[int(rec["view_index"])].append((node, gid, xy))

    raw_candidates: dict[tuple[str, str], dict] = {}
    for view, entries in sorted(entries_by_view.items()):
        if len(entries) < 3:
            continue
        coords = np.stack([x[2] for x in entries], axis=0)
        sx = _axis_step(coords[:, 0])
        sy = _axis_step(coords[:, 1])
        steps = [s for s in (sx, sy) if s is not None]
        if not steps:
            continue
        step = float(min(steps))
        if not math.isfinite(step) or step <= 0.0:
            continue
        origin = coords.min(axis=0)
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, xy in enumerate(coords):
            key = tuple(np.rint((xy - origin) / step).astype(np.int64).tolist())
            reconstructed = origin + step * np.asarray(key, np.float64)
            if float(np.linalg.norm(reconstructed - xy)) <= max(0.2, 0.10 * step):
                cells[key].append(i)
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for cell in sorted(cells):
            for dx, dy in directions:
                other = (cell[0] + dx, cell[1] + dy)
                if other not in cells:
                    continue
                for ia in cells[cell]:
                    for ib in cells[other]:
                        na, _, xa = entries[ia]
                        nb, _, xb = entries[ib]
                        common = sorted(set(map(int, na.support_views)).intersection(map(int, nb.support_views)))
                        if len(common) < int(minimum_common_support_views):
                            continue
                        raster_d = float(np.linalg.norm(xa - xb))
                        if raster_d > math.sqrt(2.0) * step * 1.10 + 1e-6:
                            continue
                        world_d = float(np.linalg.norm(np.asarray(na.P, np.float64) - np.asarray(nb.P, np.float64)))
                        if not math.isfinite(world_d) or world_d <= 1e-12:
                            continue
                        a, b = sorted((na.surface_id, nb.surface_id))
                        rec = {
                            "a": a,
                            "b": b,
                            "view": int(view),
                            "raster_distance_px": raster_d,
                            "world_distance": world_d,
                            "lattice_step_px": step,
                            "common_support_views": tuple(common),
                        }
                        old = raw_candidates.get((a, b))
                        if old is None or (world_d, raster_d, view) < (old["world_distance"], old["raster_distance_px"], old["view"]):
                            raw_candidates[(a, b)] = rec

    if raw_candidates:
        d = np.asarray([rec["world_distance"] for rec in raw_candidates.values()], np.float64)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        world_gate = max(3.0 * med, med + 6.0 * mad, 1e-8)
    else:
        world_gate = 0.0

    operator_hash = content_sha256({**_LOCAL_RELATION_OPERATOR, "minimum_common_support_views": int(minimum_common_support_views)})
    relations = []
    for key in sorted(raw_candidates):
        rec = raw_candidates[key]
        if rec["world_distance"] > world_gate:
            continue
        support_fraction = len(rec["common_support_views"]) / 8.0
        raster_term = math.exp(-rec["raster_distance_px"] / max(2.0 * rec["lattice_step_px"], 1e-8))
        world_term = math.exp(-rec["world_distance"] / max(world_gate, 1e-8))
        score = float(max(1e-8, min(1.0, support_fraction * raster_term * world_term)))
        relation_id = "IRISREL:" + content_sha256({"operator": operator_hash, **rec})[:20]
        relations.append(SurfaceRelation(
            relation_id,
            rec["a"],
            rec["b"],
            "OBSERVED_LOCAL_RASTER_NEIGHBOR",
            score,
            metadata={
                "operator_hash": operator_hash,
                "anchor_view_index": rec["view"],
                "raster_distance_px": rec["raster_distance_px"],
                "world_distance": rec["world_distance"],
                "world_gate": world_gate,
                "lattice_step_px": rec["lattice_step_px"],
                "common_support_views": rec["common_support_views"],
                "crosses_unknown": False,
                "unknown_bridge": False,
                "source_mesh_used": False,
                "teacher_truth_used": False,
            },
        ))

    lineage = content_sha256({
        "base_geometry_lineage_hash": surface.geometry_lineage_hash,
        "local_relation_operator_hash": operator_hash,
        "relations": [r.to_dict() for r in relations],
    })
    metadata = dict(surface.metadata)
    metadata.update({
        "local_relation_operator": _LOCAL_RELATION_OPERATOR["schema"],
        "local_relation_operator_hash": operator_hash,
        "local_relation_count": len(relations),
        "local_relation_source_mesh_used": False,
        "local_relation_teacher_truth_used": False,
    })
    return replace(surface, local_relations=tuple(relations), geometry_lineage_hash=lineage, metadata=metadata)


def compile_surface_v2(evidence: ObservationEvidenceIR, *, max_common_frame_error: float = 0.003) -> RiggingSurfaceIR:
    groups = build_persistence_groups_v2(evidence, max_common_frame_error=max_common_frame_error)
    surface = build_surface_from_persistence(evidence, groups)
    return attach_observed_local_relations_v2(evidence, surface)


def _pixel_index(sample, resolution: int) -> int | None:
    x, y = map(float, sample.raster_xy)
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or ix >= resolution or iy < 0 or iy >= resolution:
        return None
    return iy * resolution + ix


def attach_dtb_nd1_from_evidence(evidence: ObservationEvidenceIR, surface: RiggingSurfaceIR) -> RiggingSurfaceIR:
    if evidence.metadata.get("raster_coordinate_system") != "PIXEL_CENTER_XY":
        raise QualificationError("DTB_ND1_REQUIRES_PIXEL_CENTER_XY")
    resolution = int(evidence.metadata.get("resolution", 0))
    if resolution <= 0:
        raise QualificationError("DTB_ND1_REQUIRES_RESOLUTION")
    per_view = defaultdict(list)
    for s in evidence.samples:
        if not s.support:
            continue
        pix = _pixel_index(s, resolution)
        if pix is not None:
            per_view[int(s.view_index)].append((pix, s))

    sigma_by_group = evidence.metadata.get("mode_sigma", {})
    view_rows = {}
    obs_to_row = {}
    for view, entries in per_view.items():
        best = {}
        for pix, s in entries:
            parts = s.observation_id.split(":")
            gid = ":".join(parts[2:4]) if len(parts) >= 5 else s.provenance_ref
            rank = (float(sigma_by_group.get(gid, float("inf"))), s.observation_id)
            if pix not in best or rank < best[pix][0]:
                best[pix] = (rank, s)
        ordered = [(pix, best[pix][1]) for pix in sorted(best)]
        pix = np.asarray([x for x, _ in ordered], np.int64)
        P = np.asarray([_P(s) for _, s in ordered], np.float32)
        rows = np.arange(len(ordered), dtype=np.int64)
        normals, valid, counts = robust_local_plane_normals_for_rows(pix, P, resolution, rows)
        view_rows[view] = (ordered, normals, valid, counts)
        for row, (_, s) in enumerate(ordered):
            obs_to_row[s.observation_id] = (view, row)

    normal_map = {}
    diagnostics = {}
    for node in surface.surface_nodes:
        candidates = []
        counts = []
        for oid in node.source_observation_ids:
            if oid not in obs_to_row:
                continue
            view, row = obs_to_row[oid]
            ordered, normals, valid, retained = view_rows[view]
            if not bool(valid[row]):
                continue
            sample = ordered[row][1]
            candidates.append(orient_normal_against_ray(normals[row], sample.ray_forward))
            counts.append(int(retained[row]))
        if candidates:
            mean = np.sum(np.asarray(candidates, np.float32), axis=0)
            n = float(np.linalg.norm(mean))
            if n > 1e-8:
                normal_map[node.surface_id] = mean / n
                diagnostics[node.surface_id] = {
                    "valid_view_count": len(candidates),
                    "retained_neighbor_count_min": min(counts),
                    "retained_neighbor_count_max": max(counts),
                    "operator_hash": dtb_nd1_operator_hash(),
                }
    return attach_dtb_nd1_normals(surface, normal_map, diagnostics_by_surface_id=diagnostics)
