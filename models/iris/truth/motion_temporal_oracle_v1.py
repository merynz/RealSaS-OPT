from __future__ import annotations

"""Temporal relative-transport evidence over measured authored motion.

This wrapper applies the already-audited surface relative-transport measurement
to an explicit time sequence. It accepts only geometry, measured positions,
seconds, canonical/source surface query indices and explicit discovery/validation
frame indices. No bone ids, semantic part labels, physical DOF, impulse or
secondary-motion class enters the evidence kernel.
"""

from dataclasses import dataclass
from typing import Sequence
import re

import numpy as np

from models.iris.truth.mechanics_oracle_v1 import (
    RelativeTransportEvidenceV1,
    RelativeTransportPolicyV1,
    SurfacePairQueryV1,
    measure_relative_transport_v1,
)


@dataclass(frozen=True)
class TemporalRelativeTransportEvidenceV1:
    condition_binding_hash: str
    query: SurfacePairQueryV1
    discovery_frame_indices: tuple[int, ...]
    validation_frame_indices: tuple[int, ...]
    discovery_time_range_seconds: tuple[float, float]
    validation_time_range_seconds: tuple[float, float]
    relative_transport: RelativeTransportEvidenceV1
    schema_version: str = "RealSaS.TemporalRelativeTransportEvidence.v1"

    @property
    def final_status(self) -> str:
        return self.relative_transport.final_status

    @property
    def heldout_confirmed(self) -> bool:
        return self.relative_transport.heldout_confirmed


def _indices(value: Sequence[int], *, frame_count: int, code: str) -> tuple[int, ...]:
    out = tuple(int(i) for i in value)
    if not out or len(set(out)) != len(out) or any(i < 0 or i >= frame_count for i in out):
        raise ValueError(code)
    if tuple(sorted(out)) != out:
        raise ValueError(code + "_NOT_SORTED")
    return out


def measure_temporal_relative_transport_v1(
    rest_vertices_source_world,
    faces,
    positions_source_world,
    timestamps_seconds,
    query: SurfacePairQueryV1,
    *,
    condition_binding_hash: str,
    discovery_frame_indices: Sequence[int],
    validation_frame_indices: Sequence[int],
    subject_scale: float,
    degeneracy_tolerance: float,
    policy: RelativeTransportPolicyV1 | None = None,
) -> TemporalRelativeTransportEvidenceV1:
    positions = np.asarray(positions_source_world, dtype=np.float64)
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
        raise ValueError("MOTION_TEMPORAL_POSITIONS_INVALID")
    if timestamps.ndim != 1 or len(timestamps) != len(positions) or not np.isfinite(timestamps).all():
        raise ValueError("MOTION_TEMPORAL_TIMESTAMPS_INVALID")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("MOTION_TEMPORAL_TIME_ORDER_INVALID")
    if re.fullmatch(r"[0-9a-f]{64}", str(condition_binding_hash)) is None:
        raise ValueError("MOTION_TEMPORAL_CONDITION_HASH_REQUIRED")

    discovery = _indices(
        discovery_frame_indices,
        frame_count=len(positions),
        code="MOTION_TEMPORAL_DISCOVERY_INDICES_INVALID",
    )
    validation = _indices(
        validation_frame_indices,
        frame_count=len(positions),
        code="MOTION_TEMPORAL_VALIDATION_INDICES_INVALID",
    )
    if set(discovery) & set(validation):
        raise ValueError("MOTION_TEMPORAL_SPLIT_OVERLAP")
    if max(discovery) >= min(validation):
        raise ValueError("MOTION_TEMPORAL_HELDOUT_MUST_FOLLOW_DISCOVERY")

    measured = measure_relative_transport_v1(
        rest_vertices_source_world,
        faces,
        positions[np.asarray(discovery, dtype=np.int64)],
        positions[np.asarray(validation, dtype=np.int64)],
        query,
        subject_scale=subject_scale,
        degeneracy_tolerance=degeneracy_tolerance,
        policy=policy,
    )
    return TemporalRelativeTransportEvidenceV1(
        condition_binding_hash=str(condition_binding_hash),
        query=query,
        discovery_frame_indices=discovery,
        validation_frame_indices=validation,
        discovery_time_range_seconds=(float(timestamps[discovery[0]]), float(timestamps[discovery[-1]])),
        validation_time_range_seconds=(float(timestamps[validation[0]]), float(timestamps[validation[-1]])),
        relative_transport=measured,
    )
