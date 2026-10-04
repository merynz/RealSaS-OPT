from __future__ import annotations

"""Task-neutral surface evidence contract shared by ATLAS and MIRA research.

This module does not define shared weights. It canonicalizes the pre-skeleton
physical evidence both model families are allowed to consume so E1/E2/E3 shared
encoder experiments compare the same information rather than accidentally
changing the input contract.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


SCHEMA = "RealSaS.SurfaceEvidenceEncoderInput.v1"


def _hash(payload: Any) -> str:
    def encode(x):
        if isinstance(x, np.ndarray):
            return {
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "sha256": hashlib.sha256(x.tobytes(order="C")).hexdigest(),
            }
        raise TypeError(type(x).__name__)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=encode)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class SurfaceEvidenceEncoderInputV1:
    positions_normalized: np.ndarray      # [N,3]
    normals: np.ndarray                   # [N,3]
    normal_valid: np.ndarray              # [N]
    support_views: np.ndarray             # [N,8]
    raster_xy: np.ndarray                 # [N,8,2]
    raster_valid: np.ndarray              # [N,8]
    observed: np.ndarray                  # [N]
    completed: np.ndarray                 # [N]
    edge_index: np.ndarray                # [E,2]
    edge_features: np.ndarray             # [E,4] score, distance, crosses_unknown, unknown_bridge
    degree: np.ndarray                    # [N], derived from edge_index
    support_fraction: np.ndarray          # [N,1], derived
    raster_fraction: np.ndarray           # [N,1], derived
    view_yaw_fourier: np.ndarray          # [8,4]
    source_surface_hash: str
    source_tensorization_hash: str
    contract_hash: str
    schema_version: str = SCHEMA

    @property
    def node_count(self) -> int:
        return int(self.positions_normalized.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[0])


def _arr(x, dtype) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def build_surface_evidence_encoder_input_v1(
    surface_tensor,
    *,
    view_yaw_fourier,
) -> SurfaceEvidenceEncoderInputV1:
    """Build the task-neutral superset from a RiggingSurfaceTensorV1-like object."""

    p = _arr(surface_tensor.positions_normalized, np.float32)
    n = _arr(surface_tensor.normals, np.float32)
    nv = _arr(surface_tensor.normal_valid, bool)
    support = _arr(surface_tensor.support, bool)
    raster = _arr(surface_tensor.raster_xy_normalized, np.float32)
    raster_valid = _arr(surface_tensor.raster_valid, bool)
    observed = _arr(surface_tensor.observed, bool)
    completed = _arr(surface_tensor.completed, bool)
    edge_index = _arr(surface_tensor.edge_index, np.int64).reshape(-1, 2)
    edge_score = _arr(surface_tensor.edge_score, np.float32)
    edge_dist = _arr(surface_tensor.edge_distance_normalized, np.float32)
    edge_cross = _arr(surface_tensor.edge_crosses_unknown, bool)
    edge_bridge = _arr(surface_tensor.edge_unknown_bridge, bool)
    yaws = _arr(view_yaw_fourier, np.float32)

    N = len(p)
    E = len(edge_index)
    if p.shape != (N, 3) or n.shape != (N, 3):
        raise ValueError("SURFACE_EVIDENCE_POSITION_NORMAL_SHAPE")
    if nv.shape != (N,) or observed.shape != (N,) or completed.shape != (N,):
        raise ValueError("SURFACE_EVIDENCE_NODE_FLAG_SHAPE")
    if support.shape != (N, 8) or raster.shape != (N, 8, 2) or raster_valid.shape != (N, 8):
        raise ValueError("SURFACE_EVIDENCE_VIEW_SHAPE")
    if yaws.shape != (8, 4):
        raise ValueError("SURFACE_EVIDENCE_YAW_FOURIER_SHAPE")
    if edge_score.shape != (E,) or edge_dist.shape != (E,) or edge_cross.shape != (E,) or edge_bridge.shape != (E,):
        raise ValueError("SURFACE_EVIDENCE_EDGE_FEATURE_SHAPE")
    if not (
        np.isfinite(p).all()
        and np.isfinite(n).all()
        and np.isfinite(raster).all()
        and np.isfinite(edge_score).all()
        and np.isfinite(edge_dist).all()
        and np.isfinite(yaws).all()
    ):
        raise ValueError("SURFACE_EVIDENCE_NONFINITE")
    if E:
        if edge_index.min() < 0 or edge_index.max() >= N:
            raise ValueError("SURFACE_EVIDENCE_EDGE_INDEX_RANGE")
        if np.any(edge_index[:, 0] == edge_index[:, 1]):
            raise ValueError("SURFACE_EVIDENCE_SELF_EDGE")

    degree = np.zeros(N, dtype=np.int64)
    if E:
        np.add.at(degree, edge_index[:, 0], 1)
        np.add.at(degree, edge_index[:, 1], 1)

    declared_degree = _arr(surface_tensor.degree, np.int64)
    if declared_degree.shape != (N,) or not np.array_equal(degree, declared_degree):
        raise ValueError("SURFACE_EVIDENCE_DERIVED_DEGREE_DRIFT")

    support_fraction = support.astype(np.float32).mean(axis=1, keepdims=True)
    raster_fraction = raster_valid.astype(np.float32).mean(axis=1, keepdims=True)
    edge_features = np.stack(
        [
            edge_score,
            edge_dist,
            edge_cross.astype(np.float32),
            edge_bridge.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    source_surface_hash = str(surface_tensor.source_surface_hash)
    source_tensorization_hash = str(surface_tensor.tensorization_hash)
    if not source_surface_hash or not source_tensorization_hash:
        raise ValueError("SURFACE_EVIDENCE_LINEAGE_MISSING")

    payload = {
        "schema": SCHEMA,
        "positions_normalized": p,
        "normals": n,
        "normal_valid": nv.astype(np.uint8),
        "support_views": support.astype(np.uint8),
        "raster_xy": raster,
        "raster_valid": raster_valid.astype(np.uint8),
        "observed": observed.astype(np.uint8),
        "completed": completed.astype(np.uint8),
        "edge_index": edge_index,
        "edge_features": edge_features,
        "degree": degree,
        "support_fraction": support_fraction,
        "raster_fraction": raster_fraction,
        "view_yaw_fourier": yaws,
        "source_surface_hash": source_surface_hash,
        "source_tensorization_hash": source_tensorization_hash,
    }
    contract_hash = _hash(payload)
    return SurfaceEvidenceEncoderInputV1(
        positions_normalized=p,
        normals=n,
        normal_valid=nv,
        support_views=support,
        raster_xy=raster,
        raster_valid=raster_valid,
        observed=observed,
        completed=completed,
        edge_index=edge_index,
        edge_features=edge_features,
        degree=degree,
        support_fraction=support_fraction,
        raster_fraction=raster_fraction,
        view_yaw_fourier=yaws,
        source_surface_hash=source_surface_hash,
        source_tensorization_hash=source_tensorization_hash,
        contract_hash=contract_hash,
    )


def assert_no_downstream_authority_fields_v1(payload: dict) -> None:
    """Fail if skeleton/skin/teacher authority is smuggled into the shared input."""

    forbidden_tokens = (
        "teacher",
        "skin",
        "weight",
        "joint_id",
        "bone",
        "parent_index",
        "qualified_skeleton",
        "skeleton_binding",
    )
    bad = sorted(
        str(k) for k in payload
        if any(token in str(k).lower() for token in forbidden_tokens)
    )
    if bad:
        raise ValueError("SURFACE_EVIDENCE_DOWNSTREAM_AUTHORITY_FORBIDDEN:" + ",".join(bad))


__all__ = [
    "SCHEMA",
    "SurfaceEvidenceEncoderInputV1",
    "build_surface_evidence_encoder_input_v1",
    "assert_no_downstream_authority_fields_v1",
]
