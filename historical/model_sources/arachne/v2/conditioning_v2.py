from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np

from .geppetto_conditioning_v2 import (
    FEATURE_CONTRACT_V2,
    GeometryNormalizationV2,
    GeppettoConditioningAdapterV2,
    GeppettoConditioningBatchV2,
)
from .conditioning_v1 import ArachneConditioningAdapter
from .arachne_geometry_v2 import PAIR_GEOMETRY_CONTRACT_V2, arachne_pair_geometry_v2


def _hash(payload: object) -> str:
    def norm(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, dict): return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (tuple, list)): return [norm(x) for x in v]
        if isinstance(v, (np.integer, np.floating)): return v.item()
        return v
    return sha256(json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneConditioningBatchV2:
    surface_ids: tuple[tuple[str, ...], ...]
    joint_ids: tuple[tuple[str, ...], ...]
    surface_features: np.ndarray
    surface_positions_normalized: np.ndarray
    surface_mask: np.ndarray
    joint_features: np.ndarray
    joint_mask: np.ndarray
    parent_indices: np.ndarray
    pair_geometry: np.ndarray
    pair_mask: np.ndarray
    source_surface_hashes: tuple[str, ...]
    source_skeleton_hashes: tuple[str, ...]
    local_geometry_operator_hashes: tuple[str, ...]
    conditioning_hashes: tuple[str, ...]
    pair_geometry_contract: tuple[str, ...] = PAIR_GEOMETRY_CONTRACT_V2
    schema_version: str = "RealSaS.ArachneConditioningBatch.v2"


class ArachneConditioningAdapterV2:
    """Current (S, qualified G) conditioning with explicit point/segment geometry."""
    surface_feature_dim = 20
    joint_feature_dim = 8
    pair_geometry_dim = len(PAIR_GEOMETRY_CONTRACT_V2)

    def __call__(self, surfaces, skeletons) -> ArachneConditioningBatchV2:
        surfaces = tuple(surfaces); skeletons = tuple(skeletons)
        if not surfaces or len(surfaces) != len(skeletons):
            raise ValueError("surfaces/skeletons require equal nonzero batch")
        base = ArachneConditioningAdapter()(surfaces, skeletons)
        operator_hashes = []
        for surface in surfaces:
            op = str(getattr(surface, "metadata", {}).get("Nd_operator_sha256", ""))
            if not op:
                raise ValueError("Arachne V2 requires surface.metadata.Nd_operator_sha256")
            operator_hashes.append(op)
        surface_normals = np.asarray(base.surface_features[..., 12:15], dtype=np.float32)
        surface_normal_valid = np.asarray(base.surface_features[..., 15] > 0.5, dtype=bool)
        joint_positions = np.asarray(base.joint_features[..., 0:3], dtype=np.float32)
        pair_geometry, pair_mask = arachne_pair_geometry_v2(
            base.surface_positions_normalized,
            surface_normals,
            surface_normal_valid,
            joint_positions,
            base.parent_indices,
            base.surface_mask,
            base.joint_mask,
        )
        hashes = []
        for b in range(len(surfaces)):
            ns = len(base.surface_ids[b]); nj = len(base.joint_ids[b])
            hashes.append(_hash({
                "base_conditioning_hash": base.conditioning_hashes[b],
                "surface_hash": base.source_surface_hashes[b],
                "skeleton_hash": base.source_skeleton_hashes[b],
                "Nd_operator_sha256": operator_hashes[b],
                "pair_geometry_contract": PAIR_GEOMETRY_CONTRACT_V2,
                "pair_geometry": pair_geometry[b, :ns, :nj],
                "parent_indices": base.parent_indices[b, :nj],
            }))
        return ArachneConditioningBatchV2(
            surface_ids=base.surface_ids,
            joint_ids=base.joint_ids,
            surface_features=base.surface_features,
            surface_positions_normalized=base.surface_positions_normalized,
            surface_mask=base.surface_mask,
            joint_features=base.joint_features,
            joint_mask=base.joint_mask,
            parent_indices=base.parent_indices,
            pair_geometry=pair_geometry,
            pair_mask=pair_mask,
            source_surface_hashes=base.source_surface_hashes,
            source_skeleton_hashes=base.source_skeleton_hashes,
            local_geometry_operator_hashes=tuple(operator_hashes),
            conditioning_hashes=tuple(hashes),
        )


__all__ = [
    "FEATURE_CONTRACT_V2",
    "GeometryNormalizationV2",
    "GeppettoConditioningAdapterV2",
    "GeppettoConditioningBatchV2",
    "PAIR_GEOMETRY_CONTRACT_V2",
    "ArachneConditioningBatchV2",
    "ArachneConditioningAdapterV2",
]
