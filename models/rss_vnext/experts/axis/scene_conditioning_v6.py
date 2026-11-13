from __future__ import annotations

"""Replaceable product-evidence adapter for AXIS V6.

The adapter consumes admitted RiggingSurface evidence and emits per-node scene
features. It is not a new geometry authority and deliberately contains no learned
mesh encoder. Future IRIS shared-scene evidence can replace/extend this adapter
without changing the RigProgram output contract.
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor

from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    RiggingSurfaceTensorV1,
    tensorize_rigging_surface_v1,
)

AXIS_V6_SURFACE_SCENE_DIM = 47
AXIS_V6_SCENE_ADAPTER_ID = "RealSaS.AXIS.RiggingSurfaceSceneAdapter.v6_0"


@dataclass(frozen=True)
class AXISSceneConditioningV6:
    scene_memory: Tensor
    surface_tensor: RiggingSurfaceTensorV1
    scene_adapter_id: str = AXIS_V6_SCENE_ADAPTER_ID


def _edge_aggregates(t: RiggingSurfaceTensorV1) -> np.ndarray:
    n = t.node_count
    # mean score, mean normalized distance, crosses-unknown fraction,
    # unknown-bridge fraction, normalized degree.
    out = np.zeros((n, 5), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    for row, (a, b) in enumerate(np.asarray(t.edge_index, dtype=np.int64)):
        for node in (int(a), int(b)):
            out[node, 0] += float(t.edge_score[row])
            out[node, 1] += float(t.edge_distance_normalized[row])
            out[node, 2] += float(bool(t.edge_crosses_unknown[row]))
            out[node, 3] += float(bool(t.edge_unknown_bridge[row]))
            counts[node] += 1.0
    valid = counts > 0
    out[valid, :4] /= counts[valid, None]
    max_degree = max(float(np.max(t.degree)) if len(t.degree) else 0.0, 1.0)
    out[:, 4] = np.asarray(t.degree, dtype=np.float32) / max_degree
    return out


def scene_features_from_surface_tensor_v6(t: RiggingSurfaceTensorV1) -> np.ndarray:
    support = np.asarray(t.support, dtype=np.float32)
    raster = np.asarray(t.raster_xy_normalized, dtype=np.float32).reshape(t.node_count, -1)
    raster_valid = np.asarray(t.raster_valid, dtype=np.float32)
    features = np.concatenate(
        [
            np.asarray(t.positions_normalized, dtype=np.float32),                 # 3
            np.asarray(t.normals, dtype=np.float32),                              # 3
            np.asarray(t.normal_valid, dtype=np.float32)[:, None],                # 1
            support,                                                               # 8
            raster,                                                                # 16
            raster_valid,                                                          # 8
            np.asarray(t.observed, dtype=np.float32)[:, None],                     # 1
            np.asarray(t.completed, dtype=np.float32)[:, None],                    # 1
            np.asarray(t.degree > 0, dtype=np.float32)[:, None],                   # 1
            _edge_aggregates(t),                                                   # 5
        ],
        axis=1,
    )
    if features.shape != (t.node_count, AXIS_V6_SURFACE_SCENE_DIM):
        raise RuntimeError(f"AXIS_V6_SCENE_FEATURE_DIM_DRIFT:{features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("AXIS_V6_SCENE_FEATURE_NONFINITE")
    return features


def condition_axis_v6_from_rigging_surface(surface, *, device: torch.device | str) -> AXISSceneConditioningV6:
    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=False)
    features = scene_features_from_surface_tensor_v6(tensor)
    memory = torch.as_tensor(features, device=torch.device(device), dtype=torch.float32).unsqueeze(0)
    return AXISSceneConditioningV6(scene_memory=memory, surface_tensor=tensor)


__all__ = [
    "AXISSceneConditioningV6",
    "AXIS_V6_SCENE_ADAPTER_ID",
    "AXIS_V6_SURFACE_SCENE_DIM",
    "condition_axis_v6_from_rigging_surface",
    "scene_features_from_surface_tensor_v6",
]
