from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

try:
    from compiler.realsas_compiler_core.types import ObservationEvidenceIR, PersistenceGroup, RiggingSurfaceIR, QualificationError
    from compiler.realsas_compiler_core.surface import build_surface_from_persistence
    from compiler.realsas_compiler_core.local_geometry import (
        robust_local_plane_normals_for_rows,
        orient_normal_against_ray,
        attach_dtb_nd1_normals,
        dtb_nd1_operator_hash,
    )
except ImportError:
    from realsas_compiler_core.types import ObservationEvidenceIR, PersistenceGroup, RiggingSurfaceIR, QualificationError
    from realsas_compiler_core.surface import build_surface_from_persistence
    from realsas_compiler_core.local_geometry import (
        robust_local_plane_normals_for_rows,
        orient_normal_against_ray,
        attach_dtb_nd1_normals,
        dtb_nd1_operator_hash,
    )


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


def compile_surface_v2(evidence: ObservationEvidenceIR, *, max_common_frame_error: float = 0.003) -> RiggingSurfaceIR:
    groups = build_persistence_groups_v2(evidence, max_common_frame_error=max_common_frame_error)
    return build_surface_from_persistence(evidence, groups)


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
