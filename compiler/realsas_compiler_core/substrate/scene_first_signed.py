from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .hashing import content_sha256
from .surface import rigging_surface_from_d2_arrays
from .types import QualificationError, RiggingSurfaceIR, SurfaceRelation


_SCENE_FIRST_COMPACT_NORMAL_OPERATOR = {
    "schema": "RealSaS.GSA.SceneFirstCompactNormal.v1",
    "method": "ROBUST_LOCAL_PCA_ON_ZERO_SURFACE",
    "field_gradient_used": False,
    "teacher_truth_used": False,
}


def scene_first_compact_normal_operator_hash_v1() -> str:
    return content_sha256(_SCENE_FIRST_COMPACT_NORMAL_OPERATOR)


def rigging_surface_from_scene_first_compact_v1(
    points,
    normals,
    support_view_mask,
    raster_xy,
    relations: Iterable[tuple[int, int]],
    *,
    authority_label: str,
    source_run_id: str,
    source_checkpoint_sha256: str,
    source_zero_surface_sha256: str,
    resolution: int = 1024,
    metadata: dict | None = None,
) -> RiggingSurfaceIR:
    """Bridge a compact signed zero-surface carrier into canonical RiggingSurfaceIR.

    This is a deterministic GSA boundary. It consumes only geometry reconstructed
    upstream plus exact-camera observation support/raster bindings. Hidden teacher
    mesh, rig and skin labels are not accepted by the API.
    """
    p = np.asarray(points, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    support = np.asarray(support_view_mask)
    raster = np.asarray(raster_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4 or not np.isfinite(p).all():
        raise QualificationError("scene-first points must be finite [N,3] with N>=4")
    if n.shape != p.shape or not np.isfinite(n).all():
        raise QualificationError("scene-first normals must be finite [N,3]")
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths <= 1e-8):
        raise QualificationError("scene-first normal contains zero vector")
    n = n / lengths[:, None]
    if support.shape != (len(p), 8):
        raise QualificationError("scene-first support must be [N,8]")
    if not np.all(np.asarray(support, dtype=bool).any(axis=1)):
        raise QualificationError("every compact scene-first node requires observed support")
    if raster.shape != (len(p), 8, 2) or not np.isfinite(raster).all():
        raise QualificationError("scene-first raster_xy must be finite [N,8,2]")
    if int(resolution) <= 0:
        raise QualificationError("scene-first resolution must be positive")

    base = rigging_surface_from_d2_arrays(
        p,
        support,
        raster,
        authority_label=str(authority_label),
        persistence_label="SCENE_FIRST_SIGNED_ZERO_SURFACE_V1",
    )
    nodes = tuple(
        replace(
            node,
            derived_normal=tuple(map(float, n[i])),
            validity_flags=tuple(sorted(set(node.validity_flags) | {"OBSERVED_SIGNED_ZERO_SURFACE"})),
            metadata={
                **dict(node.metadata),
                "normal_operator": _SCENE_FIRST_COMPACT_NORMAL_OPERATOR["schema"],
                "normal_operator_hash": scene_first_compact_normal_operator_hash_v1(),
                "normal_field_gradient_used": False,
                "teacher_truth_used": False,
            },
        )
        for i, node in enumerate(base.surface_nodes)
    )
    ids = tuple(node.surface_id for node in nodes)
    rel_rows = []
    seen = set()
    for raw in relations:
        if len(raw) != 2:
            raise QualificationError("scene-first relation must be an index pair")
        a, b = map(int, raw)
        if a == b or min(a, b) < 0 or max(a, b) >= len(nodes):
            raise QualificationError("scene-first relation index out of range")
        a, b = sorted((a, b))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        dist = float(np.linalg.norm(p[a] - p[b]))
        rid = "SFSREL:" + content_sha256({"a": ids[a], "b": ids[b], "d": dist})[:20]
        rel_rows.append(
            SurfaceRelation(
                rid,
                ids[a],
                ids[b],
                "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
                1.0,
                {
                    "world_distance": dist,
                    "crosses_unknown": False,
                    "unknown_bridge": False,
                    "source_mesh_role": "PREDICTED_ZERO_SURFACE_ONLY",
                    "teacher_truth_used": False,
                },
            )
        )
    if not rel_rows:
        raise QualificationError("scene-first compact surface requires local topology relations")
    op_hash = scene_first_compact_normal_operator_hash_v1()
    lineage = content_sha256(
        {
            "schema": "RealSaS.SceneFirstSignedToRiggingSurface.v1",
            "base": base.geometry_lineage_hash,
            "source_run_id": str(source_run_id),
            "source_checkpoint_sha256": str(source_checkpoint_sha256),
            "source_zero_surface_sha256": str(source_zero_surface_sha256),
            "normal_operator_sha256": op_hash,
            "nodes": [node.to_dict() for node in nodes],
            "relations": [row.to_dict() for row in rel_rows],
        }
    )
    meta = {
        **dict(base.metadata),
        **dict(metadata or {}),
        "scene_first_signed_geometry": True,
        "scene_first_decoder": "SIGNED_FIELD_ZERO_LEVEL_SURFACE",
        "raster_coordinate_system": "PIXEL_CENTER_XY",
        "resolution": int(resolution),
        "Nd_operator": _SCENE_FIRST_COMPACT_NORMAL_OPERATOR["schema"],
        "Nd_operator_sha256": op_hash,
        "normal_field_gradient_used": False,
        "source_run_id": str(source_run_id),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "source_zero_surface_sha256": str(source_zero_surface_sha256),
        "teacher_truth_used": False,
        "character_gen_runtime_used": False,
        "full_hidden_mesh_completeness_hard_gate": False,
    }
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=tuple(sorted(rel_rows, key=lambda row: row.relation_id)),
        geometry_lineage_hash=lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1",
        schema_version=base.schema_version,
        metadata=meta,
    )


def load_scene_first_surface_fixture_v1(path: str | Path) -> tuple[RiggingSurfaceIR, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "RealSaS.PromotedSceneFirstSignedSurfaceFixture.v1":
        raise QualificationError("unexpected scene-first fixture schema")
    if payload.get("teacher_mesh_used_at_inference") is not False:
        raise QualificationError("fixture violates inference teacher isolation")
    surface = rigging_surface_from_scene_first_compact_v1(
        payload["points"],
        payload["normals"],
        payload["support"],
        payload["raster_xy"],
        payload["relations"],
        authority_label="IRIS_SCENE_FIRST_SIGNED_V3_FIT_WITNESS",
        source_run_id=payload["source_geometry_run_id"],
        source_checkpoint_sha256=payload["source_checkpoint_sha256"],
        source_zero_surface_sha256=payload["source_clipped_mesh_sha256"],
        resolution=int(payload["cameras"][0]["resolution"]),
        metadata={
            "fixture_family": payload.get("family", ""),
            "fit_only": bool(payload.get("fit_only", False)),
            "product_inference_inputs": payload.get("product_inference_inputs", []),
            "promotion_metrics": payload.get("metrics", {}),
            "sampling": payload.get("sampling", {}),
        },
    )
    return surface, payload


__all__ = [
    "scene_first_compact_normal_operator_hash_v1",
    "rigging_surface_from_scene_first_compact_v1",
    "load_scene_first_surface_fixture_v1",
]
