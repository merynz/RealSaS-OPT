from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .types import QualificationError, RiggingSurfaceIR


def _finite_vec(values, width: int) -> bool:
    try:
        a = np.asarray(values, dtype=np.float64)
    except Exception:
        return False
    return a.shape == (width,) and bool(np.isfinite(a).all())


def audit_rigging_surface_ir_v1(
    surface: RiggingSurfaceIR,
    *,
    require_scene_first_signed_contract: bool = False,
    raise_on_error: bool = False,
) -> dict:
    """Read-only structural audit for the RiggingSurfaceIR boundary.

    This function does not mutate, complete, repair or reinterpret the surface.
    It centralizes invariants that were previously checked only piecemeal by
    producers/consumers. Product paths do not call it implicitly; promotion of
    any fail-closed interlock is a separate decision.
    """
    errors: list[str] = []
    warnings: list[str] = []

    nodes = tuple(surface.surface_nodes)
    relations = tuple(surface.local_relations)
    if not nodes:
        errors.append("EMPTY_SURFACE")

    ids = [str(n.surface_id) for n in nodes]
    if any(not x for x in ids):
        errors.append("EMPTY_SURFACE_ID")
    if len(ids) != len(set(ids)):
        errors.append("DUPLICATE_SURFACE_ID")
    id_set = set(ids)

    support_hist = Counter()
    raster_binding_count = 0
    normal_count = 0
    completed_count = 0
    observed_count = 0

    for i, node in enumerate(nodes):
        prefix = f"NODE[{i}]"
        if not _finite_vec(node.P, 3):
            errors.append(f"{prefix}:NONFINITE_OR_BAD_P")

        views = tuple(int(v) for v in node.support_views)
        if len(views) != len(set(views)):
            errors.append(f"{prefix}:DUPLICATE_SUPPORT_VIEW")
        if any(v < 0 or v > 7 for v in views):
            errors.append(f"{prefix}:SUPPORT_VIEW_OUT_OF_RANGE")
        support_hist[len(views)] += 1

        binds = tuple(node.raster_bindings)
        bind_views = []
        for bi, binding in enumerate(binds):
            if not isinstance(binding, tuple) or len(binding) != 2:
                errors.append(f"{prefix}:BAD_RASTER_BINDING[{bi}]")
                continue
            v, xy = binding
            try:
                iv = int(v)
            except Exception:
                errors.append(f"{prefix}:BAD_RASTER_VIEW[{bi}]")
                continue
            bind_views.append(iv)
            if iv < 0 or iv > 7:
                errors.append(f"{prefix}:RASTER_VIEW_OUT_OF_RANGE[{bi}]")
            if not _finite_vec(xy, 2):
                errors.append(f"{prefix}:NONFINITE_OR_BAD_RASTER[{bi}]")
        if len(bind_views) != len(set(bind_views)):
            errors.append(f"{prefix}:DUPLICATE_RASTER_VIEW")
        if set(bind_views) != set(views):
            errors.append(f"{prefix}:RASTER_SUPPORT_VIEW_MISMATCH")
        raster_binding_count += len(binds)

        normal = node.derived_normal
        if normal is not None:
            normal_count += 1
            if not _finite_vec(normal, 3):
                errors.append(f"{prefix}:NONFINITE_OR_BAD_NORMAL")
            else:
                norm = float(np.linalg.norm(np.asarray(normal, dtype=np.float64)))
                if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
                    errors.append(f"{prefix}:NONUNIT_NORMAL:{norm}")

        flags = tuple(str(x) for x in node.validity_flags)
        completed = "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in flags
        observed = "OBSERVED_SIGNED_ZERO_SURFACE" in flags
        if completed:
            completed_count += 1
            if views or binds:
                errors.append(f"{prefix}:COMPLETED_NODE_HAS_OBSERVED_SUPPORT")
        if observed:
            observed_count += 1
            if not views:
                errors.append(f"{prefix}:OBSERVED_NODE_HAS_NO_SUPPORT")
        if completed and observed:
            errors.append(f"{prefix}:OBSERVED_AND_COMPLETED")

    relation_ids = [str(r.relation_id) for r in relations]
    if any(not x for x in relation_ids):
        errors.append("EMPTY_RELATION_ID")
    if len(relation_ids) != len(set(relation_ids)):
        errors.append("DUPLICATE_RELATION_ID")

    relation_kinds = Counter()
    degree = Counter()
    for i, rel in enumerate(relations):
        prefix = f"REL[{i}]"
        a = str(rel.a_surface_id)
        b = str(rel.b_surface_id)
        if a not in id_set or b not in id_set:
            errors.append(f"{prefix}:UNKNOWN_ENDPOINT")
        if a == b:
            errors.append(f"{prefix}:SELF_LOOP")
        if not str(rel.relation_kind):
            errors.append(f"{prefix}:EMPTY_RELATION_KIND")
        try:
            score = float(rel.score)
        except Exception:
            score = float("nan")
        if not math.isfinite(score):
            errors.append(f"{prefix}:NONFINITE_SCORE")
        relation_kinds[str(rel.relation_kind)] += 1
        if a in id_set and b in id_set and a != b:
            degree[a] += 1
            degree[b] += 1

    if not str(surface.geometry_lineage_hash):
        errors.append("MISSING_GEOMETRY_LINEAGE_HASH")

    meta = dict(surface.metadata or {})
    if require_scene_first_signed_contract:
        if surface.builder_id != "RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1":
            errors.append("SCENE_FIRST:WRONG_BUILDER_ID")
        if meta.get("scene_first_signed_geometry") is not True:
            errors.append("SCENE_FIRST:MISSING_SCENE_FIRST_FLAG")
        if meta.get("teacher_truth_used") is not False:
            errors.append("SCENE_FIRST:TEACHER_TRUTH_FLAG_NOT_FALSE")
        if meta.get("character_gen_runtime_used") is not False:
            errors.append("SCENE_FIRST:CHARACTER_GEN_RUNTIME_FLAG_NOT_FALSE")
        if meta.get("raster_coordinate_system") != "PIXEL_CENTER_XY":
            errors.append("SCENE_FIRST:BAD_RASTER_COORDINATE_SYSTEM")
        try:
            resolution = int(meta.get("resolution", 0))
        except Exception:
            resolution = 0
        if resolution <= 0:
            errors.append("SCENE_FIRST:MISSING_OR_BAD_RESOLUTION")
        for key in ("source_run_id", "source_checkpoint_sha256", "source_zero_surface_sha256", "Nd_operator_sha256"):
            if not str(meta.get(key, "")):
                errors.append(f"SCENE_FIRST:MISSING_{key.upper()}")
        if int(meta.get("compact_surface_node_count", -1)) != len(nodes):
            errors.append("SCENE_FIRST:NODE_COUNT_METADATA_DRIFT")
        if int(meta.get("observed_node_count", -1)) != sum(bool(n.support_views) for n in nodes):
            errors.append("SCENE_FIRST:OBSERVED_COUNT_METADATA_DRIFT")
        if int(meta.get("completed_node_count", -1)) != sum(not bool(n.support_views) for n in nodes):
            errors.append("SCENE_FIRST:COMPLETED_COUNT_METADATA_DRIFT")
        for i, rel in enumerate(relations):
            if rel.relation_kind != "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR":
                warnings.append(f"SCENE_FIRST:UNEXPECTED_RELATION_KIND[{i}]:{rel.relation_kind}")
            if dict(rel.metadata or {}).get("teacher_truth_used") is not False:
                errors.append(f"SCENE_FIRST:RELATION_TEACHER_TRUTH_FLAG_NOT_FALSE[{i}]")

    degree_values = np.asarray([degree.get(x, 0) for x in ids], dtype=np.float64) if ids else np.zeros(0)
    report = {
        "schema": "RealSaS.RiggingSurfaceIRBoundaryAudit.v1",
        "builder_id": str(surface.builder_id),
        "schema_version": str(surface.schema_version),
        "geometry_lineage_hash": str(surface.geometry_lineage_hash),
        "node_count": len(nodes),
        "relation_count": len(relations),
        "raster_binding_count": int(raster_binding_count),
        "normal_count": int(normal_count),
        "observed_flag_count": int(observed_count),
        "completed_flag_count": int(completed_count),
        "support_count_histogram": {str(k): int(v) for k, v in sorted(support_hist.items())},
        "relation_kind_counts": {k: int(v) for k, v in sorted(relation_kinds.items())},
        "mean_relation_degree": None if not len(degree_values) else float(degree_values.mean()),
        "min_relation_degree": None if not len(degree_values) else int(degree_values.min()),
        "max_relation_degree": None if not len(degree_values) else int(degree_values.max()),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": tuple(errors),
        "warnings": tuple(warnings),
        "passed": not errors,
        "scene_first_profile_required": bool(require_scene_first_signed_contract),
    }
    if raise_on_error and errors:
        raise QualificationError("RIGGING_SURFACE_IR_AUDIT_FAIL:" + ";".join(errors[:12]))
    return report


def validate_rigging_surface_ir_v1(
    surface: RiggingSurfaceIR,
    *,
    require_scene_first_signed_contract: bool = False,
) -> dict:
    """Fail-closed read-only validation wrapper."""
    return audit_rigging_surface_ir_v1(
        surface,
        require_scene_first_signed_contract=require_scene_first_signed_contract,
        raise_on_error=True,
    )


__all__ = ["audit_rigging_surface_ir_v1", "validate_rigging_surface_ir_v1"]
