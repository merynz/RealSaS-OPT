from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np


def _sha(payload: object) -> str:
    def norm(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("non-finite conditioning payload")
        return v
    return sha256(json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GeometryNormalizationV1:
    center: tuple[float, float, float]
    scale: float
    policy: str = "BBOX_CENTER_MAX_EXTENT_V1"

    def normalize(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float32) - np.asarray(self.center, dtype=np.float32)) / float(self.scale)

    def denormalize(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=np.float32) * float(self.scale) + np.asarray(self.center, dtype=np.float32)


@dataclass(frozen=True)
class GeppettoConditioningBatchV1:
    surface_ids: tuple[tuple[str, ...], ...]
    features: np.ndarray
    positions_normalized: np.ndarray
    valid_mask: np.ndarray
    normalizations: tuple[GeometryNormalizationV1, ...]
    source_surface_hashes: tuple[str, ...]
    conditioning_hashes: tuple[str, ...]
    schema_version: str = "RealSaS.GeppettoConditioningBatch.v1"


@dataclass(frozen=True)
class ArachneConditioningBatchV1:
    surface_ids: tuple[tuple[str, ...], ...]
    joint_ids: tuple[tuple[str, ...], ...]
    surface_features: np.ndarray
    surface_positions_normalized: np.ndarray
    surface_mask: np.ndarray
    joint_features: np.ndarray
    joint_mask: np.ndarray
    parent_indices: np.ndarray
    normalizations: tuple[GeometryNormalizationV1, ...]
    source_surface_hashes: tuple[str, ...]
    source_skeleton_hashes: tuple[str, ...]
    conditioning_hashes: tuple[str, ...]
    schema_version: str = "RealSaS.ArachneConditioningBatch.v1"


def _normalization(points: np.ndarray) -> GeometryNormalizationV1:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("surface points must be non-empty [N,3]")
    if not np.isfinite(points).all():
        raise ValueError("surface points must be finite")
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if scale <= 1e-12:
        scale = 1.0
    return GeometryNormalizationV1(tuple(map(float, center)), scale)


def _relation_summary(surface) -> dict[str, tuple[float, float]]:
    degree: dict[str, int] = {n.surface_id: 0 for n in surface.surface_nodes}
    score_sum: dict[str, float] = {n.surface_id: 0.0 for n in surface.surface_nodes}
    for rel in getattr(surface, "local_relations", ()):
        if rel.a_surface_id in degree:
            degree[rel.a_surface_id] += 1
            score_sum[rel.a_surface_id] += float(rel.score)
        if rel.b_surface_id in degree:
            degree[rel.b_surface_id] += 1
            score_sum[rel.b_surface_id] += float(rel.score)
    max_degree = max(degree.values(), default=1) or 1
    out = {}
    for sid, d in degree.items():
        out[sid] = (float(d) / float(max_degree), score_sum[sid] / max(d, 1))
    return out


def _surface_features(surface, normalization: GeometryNormalizationV1) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    nodes = tuple(sorted(surface.surface_nodes, key=lambda n: n.surface_id))
    if not nodes:
        raise ValueError("RiggingSurfaceIR contains no surface nodes")
    ids = tuple(n.surface_id for n in nodes)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate surface_id")
    positions = np.asarray([n.P for n in nodes], dtype=np.float32)
    pn = normalization.normalize(positions)
    rel = _relation_summary(surface)
    feats = []
    for i, n in enumerate(nodes):
        bits = [1.0 if v in set(n.support_views) else 0.0 for v in range(8)]
        support_count = min(len(set(n.support_views)), 8) / 8.0
        normal = tuple(float(x) for x in n.derived_normal) if n.derived_normal is not None else (0.0, 0.0, 0.0)
        normal_valid = 1.0 if n.derived_normal is not None else 0.0
        degree, mean_score = rel[n.surface_id]
        raster_count = min(len(n.raster_bindings), 8) / 8.0
        radius = float(np.linalg.norm(pn[i]))
        row = [*map(float, pn[i]), *bits, support_count, *normal, normal_valid, degree, mean_score, raster_count, radius]
        if len(row) != 20:
            raise AssertionError(f"surface feature width drifted:{len(row)}")
        feats.append(row)
    arr = np.asarray(feats, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite surface feature")
    return ids, arr, pn.astype(np.float32)


def _pad_3d(items: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    b = len(items)
    n = max(len(x) for x in items)
    out = np.zeros((b, n, width), dtype=np.float32)
    mask = np.zeros((b, n), dtype=bool)
    for i, x in enumerate(items):
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError("feature width mismatch")
        out[i, :len(x)] = x
        mask[i, :len(x)] = True
    return out, mask


class GeppettoConditioningAdapter:
    """Deterministic S_hat -> model tensor adapter.

    This adapter consumes only admitted RiggingSurfaceIR. Teacher controls, source
    bone IDs, hidden meshes and product canonical IDs are not accepted inputs.
    """
    feature_dim = 20

    def __call__(self, surfaces: Iterable) -> GeppettoConditioningBatchV1:
        surfaces = tuple(surfaces)
        if not surfaces:
            raise ValueError("at least one surface required")
        ids_all = []
        feats_all = []
        pos_all = []
        norms = []
        hashes = []
        cond_hashes = []
        for surface in surfaces:
            points = np.asarray([n.P for n in surface.surface_nodes], dtype=np.float32)
            norm = _normalization(points)
            ids, feats, pos = _surface_features(surface, norm)
            ids_all.append(ids)
            feats_all.append(feats)
            pos_all.append(pos)
            norms.append(norm)
            source_hash = str(surface.geometry_lineage_hash)
            if not source_hash:
                raise ValueError("surface geometry_lineage_hash required")
            hashes.append(source_hash)
            cond_hashes.append(_sha({"surface": source_hash, "ids": ids, "features": feats, "normalization": norm.__dict__}))
        features, mask = _pad_3d(feats_all, self.feature_dim)
        positions, _ = _pad_3d(pos_all, 3)
        return GeppettoConditioningBatchV1(tuple(ids_all), features, positions, mask, tuple(norms), tuple(hashes), tuple(cond_hashes))


class ArachneConditioningAdapter:
    """Deterministic (S_hat, Compiler-qualified G) -> model tensor adapter."""
    surface_feature_dim = 20
    joint_feature_dim = 8

    def __call__(self, surfaces: Iterable, skeletons: Iterable) -> ArachneConditioningBatchV1:
        surfaces = tuple(surfaces)
        skeletons = tuple(skeletons)
        if not surfaces or len(surfaces) != len(skeletons):
            raise ValueError("surfaces and skeletons require equal nonzero batch")
        sid_all = []
        jid_all = []
        sf_all = []
        sp_all = []
        jf_all = []
        pi_all = []
        norms = []
        sh = []
        gh = []
        ch = []
        for surface, skeleton in zip(surfaces, skeletons):
            points = np.asarray([n.P for n in surface.surface_nodes], dtype=np.float32)
            norm = _normalization(points)
            sids, sf, sp = _surface_features(surface, norm)
            joints = tuple(sorted(skeleton.joints, key=lambda j: j.canonical_joint_id))
            if not joints:
                raise ValueError("qualified skeleton contains no joints")
            jids = tuple(j.canonical_joint_id for j in joints)
            if len(jids) != len(set(jids)):
                raise ValueError("duplicate canonical joint id")
            jmap = {jid: i for i, jid in enumerate(jids)}
            roots = set(getattr(skeleton, "deform_root_ids", ())) or {j.canonical_joint_id for j in joints if j.parent_canonical_id is None}
            depth_cache = {}

            def depth(jid, stack=()):
                if jid in depth_cache:
                    return depth_cache[jid]
                if jid in stack:
                    raise ValueError("skeleton cycle in conditioning input")
                j = joints[jmap[jid]]
                if j.parent_canonical_id is None:
                    d = 0
                else:
                    if j.parent_canonical_id not in jmap:
                        raise ValueError("unknown parent in qualified skeleton")
                    d = 1 + depth(j.parent_canonical_id, stack + (jid,))
                depth_cache[jid] = d
                return d

            max_depth = max(depth(jid) for jid in jids) or 1
            jfeats = []
            parents = []
            for j in joints:
                pos = norm.normalize(np.asarray([j.position], dtype=np.float32))[0]
                parent_index = -1 if j.parent_canonical_id is None else jmap[j.parent_canonical_id]
                support_count = min(len(set(j.support_surface_ids)), 16) / 16.0
                row = [*map(float, pos), 1.0 if j.canonical_joint_id in roots else 0.0, 1.0 if parent_index >= 0 else 0.0, support_count, depth(j.canonical_joint_id) / max_depth, 1.0]
                if len(row) != 8:
                    raise AssertionError("joint feature width drift")
                jfeats.append(row)
                parents.append(parent_index)
            source_hash = str(surface.geometry_lineage_hash)
            skeleton_hash = str(skeleton.skeleton_lineage_hash)
            if not source_hash or not skeleton_hash:
                raise ValueError("surface/skeleton lineage hashes required")
            sid_all.append(sids)
            jid_all.append(jids)
            sf_all.append(sf)
            sp_all.append(sp)
            jf_all.append(np.asarray(jfeats, np.float32))
            pi_all.append(np.asarray(parents, np.int64))
            norms.append(norm)
            sh.append(source_hash)
            gh.append(skeleton_hash)
            ch.append(_sha({"surface": source_hash, "skeleton": skeleton_hash, "sids": sids, "jids": jids, "surface_features": sf, "joint_features": jfeats, "parents": parents, "normalization": norm.__dict__}))
        surface_features, surface_mask = _pad_3d(sf_all, self.surface_feature_dim)
        surface_positions, _ = _pad_3d(sp_all, 3)
        joint_features, joint_mask = _pad_3d(jf_all, self.joint_feature_dim)
        maxj = joint_features.shape[1]
        parent_indices = np.full((len(pi_all), maxj), -1, dtype=np.int64)
        for i, p in enumerate(pi_all):
            parent_indices[i, :len(p)] = p
        return ArachneConditioningBatchV1(tuple(sid_all), tuple(jid_all), surface_features, surface_positions, surface_mask, joint_features, joint_mask, parent_indices, tuple(norms), tuple(sh), tuple(gh), tuple(ch))
