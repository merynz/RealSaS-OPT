from __future__ import annotations

"""Lossless fieldwise tensorization of RiggingSurfaceIR for Geppetto.

This adapter intentionally does *not* collapse the substrate to the legacy 24D
summary. Shipping-available evidence stays factorized:
P, robust N, per-view support, per-view raster bindings, validity flags and the
exact GSA local-relation graph.

Provenance and operator metadata are retained in the certificate/hash but are
not learner features. Teacher/oracle contamination causes fail-close.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np

from compiler.realsas_compiler_core.types import RiggingSurfaceIR


SCHEMA = "RealSaS.GeppettoRiggingSurfaceTensorization.v1"
NORMALIZATION_POLICY = "BBOX_CENTER_MAX_SPAN_V1"
EXPECTED_SCENE_FIRST_RELATION_KIND = "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR"


def _json_norm(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _json_norm(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (tuple, list)):
        return [_json_norm(x) for x in v]
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite tensorization payload")
    return v


def _hash(v) -> str:
    raw = json.dumps(_json_norm(v), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _contains_teacher_truth(v) -> bool:
    if isinstance(v, dict):
        for k, x in v.items():
            if str(k).lower() == "teacher_truth_used" and bool(x):
                return True
            if _contains_teacher_truth(x):
                return True
        return False
    if isinstance(v, (tuple, list)):
        return any(_contains_teacher_truth(x) for x in v)
    return False


def _normalize_positions(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.asarray(P, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("surface positions must be finite non-empty [N,3]")
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("surface normalization scale is degenerate")
    return ((p - center[None, :]) / scale).astype(np.float32), center.astype(np.float64), scale


def _normalise_raster_xy(surface: RiggingSurfaceIR, xy) -> np.ndarray:
    a = np.asarray(xy, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("raster binding must be finite XY")
    mode = str(surface.metadata.get("raster_coordinate_system", ""))
    if mode == "PIXEL_CENTER_XY":
        resolution = int(surface.metadata.get("resolution", 0))
        if resolution <= 0:
            raise ValueError("PIXEL_CENTER_XY requires positive metadata.resolution")
        return np.asarray(
            [
                2.0 * (a[0] + 0.5) / float(resolution) - 1.0,
                2.0 * (a[1] + 0.5) / float(resolution) - 1.0,
            ],
            dtype=np.float32,
        )
    if mode in {"GRID_XY", "GRID_SAMPLE_XY", "NORMALIZED_GRID_XY"}:
        return a.astype(np.float32)
    raise ValueError(f"unsupported raster coordinate system:{mode!r}")


@dataclass(frozen=True)
class RiggingSurfaceTensorV1:
    surface_ids: tuple[str, ...]
    positions_world: np.ndarray
    positions_normalized: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    support: np.ndarray
    raster_xy_normalized: np.ndarray
    raster_valid: np.ndarray
    observed: np.ndarray
    completed: np.ndarray
    validity_vocab: tuple[str, ...]
    validity_bits: np.ndarray
    edge_index: np.ndarray
    edge_score: np.ndarray
    edge_distance_normalized: np.ndarray
    edge_crosses_unknown: np.ndarray
    edge_unknown_bridge: np.ndarray
    relation_kind_vocab: tuple[str, ...]
    relation_kind_index: np.ndarray
    degree: np.ndarray
    normalization_center: np.ndarray
    normalization_scale: float
    source_surface_hash: str
    local_geometry_operator_hash: str
    certificate_hash: str
    tensorization_hash: str
    schema_version: str = SCHEMA

    @property
    def node_count(self) -> int:
        return int(len(self.surface_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_index))


def tensorize_rigging_surface_v1(
    surface: RiggingSurfaceIR,
    *,
    require_scene_first: bool = True,
) -> RiggingSurfaceTensorV1:
    if not isinstance(surface, RiggingSurfaceIR):
        raise TypeError("expected RiggingSurfaceIR")
    if not surface.geometry_lineage_hash:
        raise ValueError("surface geometry_lineage_hash required")
    if not surface.surface_nodes:
        raise ValueError("surface has no nodes")
    if _contains_teacher_truth(surface.metadata):
        raise ValueError("teacher truth contamination in surface metadata")

    nodes = tuple(sorted(surface.surface_nodes, key=lambda n: str(n.surface_id)))
    ids = tuple(str(n.surface_id) for n in nodes)
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("surface IDs must be non-empty and unique")
    row = {sid: i for i, sid in enumerate(ids)}

    P = np.asarray([n.P for n in nodes], dtype=np.float64)
    Pn, center, scale = _normalize_positions(P)

    normals = np.zeros((len(nodes), 3), dtype=np.float32)
    nvalid = np.zeros(len(nodes), dtype=bool)
    support = np.zeros((len(nodes), 8), dtype=bool)
    raster = np.zeros((len(nodes), 8, 2), dtype=np.float32)
    raster_valid = np.zeros((len(nodes), 8), dtype=bool)
    observed = np.zeros(len(nodes), dtype=bool)
    completed = np.zeros(len(nodes), dtype=bool)

    vocab = tuple(sorted({str(flag) for n in nodes for flag in n.validity_flags}))
    vindex = {x: i for i, x in enumerate(vocab)}
    vbits = np.zeros((len(nodes), len(vocab)), dtype=bool)

    for i, n in enumerate(nodes):
        if _contains_teacher_truth(n.metadata):
            raise ValueError(f"teacher truth contamination in node:{n.surface_id}")
        rawp = np.asarray(n.P, dtype=np.float64)
        if rawp.shape != (3,) or not np.isfinite(rawp).all():
            raise ValueError(f"invalid node position:{n.surface_id}")
        if n.derived_normal is not None:
            raw = np.asarray(n.derived_normal, dtype=np.float64)
            ln = float(np.linalg.norm(raw))
            if raw.shape != (3,) or not np.isfinite(raw).all() or ln <= 1e-12:
                raise ValueError(f"invalid derived normal:{n.surface_id}")
            normals[i] = (raw / ln).astype(np.float32)
            nvalid[i] = True

        views = tuple(map(int, n.support_views))
        if len(views) != len(set(views)) or any(v < 0 or v >= 8 for v in views):
            raise ValueError(f"invalid support views:{n.surface_id}")
        for v in views:
            support[i, v] = True

        seen_bind = set()
        for vraw, xy in n.raster_bindings:
            v = int(vraw)
            if v < 0 or v >= 8 or v in seen_bind:
                raise ValueError(f"invalid/duplicate raster view:{n.surface_id}")
            if v not in views:
                raise ValueError(f"raster binding outside support set:{n.surface_id}")
            seen_bind.add(v)
            raster[i, v] = _normalise_raster_xy(surface, xy)
            raster_valid[i, v] = True
        if require_scene_first and set(views) != seen_bind:
            raise ValueError(f"scene-first node missing raster binding for supported view:{n.surface_id}")

        flags = set(map(str, n.validity_flags))
        observed[i] = "OBSERVED_SIGNED_ZERO_SURFACE" in flags
        completed[i] = "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in flags
        if observed[i] and completed[i]:
            raise ValueError(f"node cannot be both observed and completed:{n.surface_id}")
        if require_scene_first and not (observed[i] or completed[i]):
            raise ValueError(f"scene-first node missing observed/completed validity:{n.surface_id}")
        if require_scene_first and observed[i] != bool(support[i].any()):
            raise ValueError(f"scene-first observed/support mismatch:{n.surface_id}")
        for flag in flags:
            vbits[i, vindex[flag]] = True

    relations = tuple(surface.local_relations)
    if require_scene_first and not relations:
        raise ValueError("scene-first surface must contain local relations")
    relation_kinds = tuple(sorted({str(r.relation_kind) for r in relations}))
    kind_index = {x: i for i, x in enumerate(relation_kinds)}
    edges = []
    scores = []
    distances = []
    crosses = []
    unknown_bridge = []
    kinds = []
    seen_rel_ids = set()
    seen_pairs = set()
    for rel in relations:
        if _contains_teacher_truth(rel.metadata):
            raise ValueError(f"teacher truth contamination in relation:{rel.relation_id}")
        rid = str(rel.relation_id)
        if not rid or rid in seen_rel_ids:
            raise ValueError("relation IDs must be non-empty and unique")
        seen_rel_ids.add(rid)
        a = row.get(str(rel.a_surface_id))
        b = row.get(str(rel.b_surface_id))
        if a is None or b is None or a == b:
            raise ValueError(f"invalid relation endpoints:{rid}")
        pair = tuple(sorted((int(a), int(b))))
        if pair in seen_pairs:
            raise ValueError(f"duplicate undirected relation pair:{rid}")
        seen_pairs.add(pair)
        kind = str(rel.relation_kind)
        if require_scene_first and kind != EXPECTED_SCENE_FIRST_RELATION_KIND:
            raise ValueError(f"unexpected scene-first relation kind:{kind}")
        score = float(rel.score)
        if not math.isfinite(score):
            raise ValueError(f"non-finite relation score:{rid}")
        dnorm = float(np.linalg.norm(Pn[pair[0]] - Pn[pair[1]]))
        edges.append(pair)
        scores.append(score)
        distances.append(dnorm)
        crosses.append(bool(rel.metadata.get("crosses_unknown", False)))
        unknown_bridge.append(bool(rel.metadata.get("unknown_bridge", False)))
        kinds.append(kind_index[kind])

    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    order = (
        np.argsort(edge_array[:, 0] * len(nodes) + edge_array[:, 1])
        if len(edge_array)
        else np.zeros(0, dtype=np.int64)
    )
    edge_index = edge_array[order]
    edge_score = np.asarray(scores, dtype=np.float32)[order]
    edge_dist = np.asarray(distances, dtype=np.float32)[order]
    edge_cross = np.asarray(crosses, dtype=bool)[order]
    edge_bridge = np.asarray(unknown_bridge, dtype=bool)[order]
    rel_kind_idx = np.asarray(kinds, dtype=np.int64)[order]

    degree = np.zeros(len(nodes), dtype=np.int64)
    if len(edge_index):
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)

    op_hash = str(surface.metadata.get("Nd_operator_sha256", ""))
    if require_scene_first and not op_hash:
        raise ValueError("scene-first surface missing Nd_operator_sha256")
    if require_scene_first and not bool(surface.metadata.get("scene_first_signed_geometry", False)):
        raise ValueError("surface is not declared scene-first signed geometry")

    certificate_payload = {
        "schema": SCHEMA,
        "source_surface_hash": str(surface.geometry_lineage_hash),
        "builder_id": str(surface.builder_id),
        "surface_schema_version": str(surface.schema_version),
        "surface_metadata": surface.metadata,
        "node_provenance": [
            {
                "surface_id": str(n.surface_id),
                "provenance_refs": tuple(map(str, n.provenance_refs)),
                "source_observation_ids": tuple(map(str, n.source_observation_ids)),
                "persistence_group_id": n.persistence_group_id,
                "metadata": n.metadata,
            }
            for n in nodes
        ],
        "relation_metadata": [
            {"relation_id": str(r.relation_id), "metadata": r.metadata}
            for r in relations
        ],
    }
    certificate_hash = _hash(certificate_payload)
    tensor_payload = {
        "schema": SCHEMA,
        "normalization_policy": NORMALIZATION_POLICY,
        "ids": ids,
        "positions_world": P,
        "positions_normalized": Pn,
        "normals": normals,
        "normal_valid": nvalid,
        "support": support,
        "raster_xy_normalized": raster,
        "raster_valid": raster_valid,
        "observed": observed,
        "completed": completed,
        "validity_vocab": vocab,
        "validity_bits": vbits,
        "edge_index": edge_index,
        "edge_score": edge_score,
        "edge_distance_normalized": edge_dist,
        "edge_crosses_unknown": edge_cross,
        "edge_unknown_bridge": edge_bridge,
        "relation_kind_vocab": relation_kinds,
        "relation_kind_index": rel_kind_idx,
        "degree": degree,
        "normalization_center": center,
        "normalization_scale": scale,
        "source_surface_hash": str(surface.geometry_lineage_hash),
        "local_geometry_operator_hash": op_hash,
        "certificate_hash": certificate_hash,
    }
    tensorization_hash = _hash(tensor_payload)
    return RiggingSurfaceTensorV1(
        surface_ids=ids,
        positions_world=P.astype(np.float32),
        positions_normalized=Pn,
        normals=normals,
        normal_valid=nvalid,
        support=support,
        raster_xy_normalized=raster,
        raster_valid=raster_valid,
        observed=observed,
        completed=completed,
        validity_vocab=vocab,
        validity_bits=vbits,
        edge_index=edge_index,
        edge_score=edge_score,
        edge_distance_normalized=edge_dist,
        edge_crosses_unknown=edge_cross,
        edge_unknown_bridge=edge_bridge,
        relation_kind_vocab=relation_kinds,
        relation_kind_index=rel_kind_idx,
        degree=degree,
        normalization_center=center,
        normalization_scale=scale,
        source_surface_hash=str(surface.geometry_lineage_hash),
        local_geometry_operator_hash=op_hash,
        certificate_hash=certificate_hash,
        tensorization_hash=tensorization_hash,
    )


__all__ = [
    "SCHEMA",
    "NORMALIZATION_POLICY",
    "RiggingSurfaceTensorV1",
    "tensorize_rigging_surface_v1",
]
