from __future__ import annotations

"""Measured authored temporal-response candidates.

This module converts an already-sealed authored pose sequence plus an explicit
second-valued time axis into the Truth Foundry `TEMPORAL_RESPONSE` array layout.
It does not infer physics, admissible DOF, joint limits, impulses, secondary
motion classes, semantic parts, or student observability.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DYNAMIC_AUTHORITY = "MEASURED_AUTHORED_NATIVE_CLIP_RESPONSE__NO_INFERRED_PHYSICS"
RESPONSE_SEMANTICS = "SOURCE_WORLD_DISPLACEMENT_FROM_SEALED_REST__ROOT_MOTION_PRESERVED"


@dataclass(frozen=True)
class TemporalResponseCandidateV1:
    action_identity: str
    response: np.ndarray
    timestamps_seconds: np.ndarray
    sample_validity: np.ndarray
    interval_validity: np.ndarray
    dynamic_authority: str = DYNAMIC_AUTHORITY
    response_semantics: str = RESPONSE_SEMANTICS
    coordinate_frame: str = "SOURCE_WORLD"
    time_unit: str = "SECOND"
    condition_axis_semantics: str = "SINGLE_AUTHORED_ACTION"
    qualification_minted: bool = False
    student_target_allowed: bool = False
    compiler_consequence_allowed: bool = False

    @property
    def locus_count(self) -> int:
        return int(self.response.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.response.shape[2])

    def array_contract(self) -> dict[str, Any]:
        return {
            "response": {"shape": list(self.response.shape), "dtype": self.response.dtype.str},
            "timestamps_seconds": {
                "shape": list(self.timestamps_seconds.shape),
                "dtype": self.timestamps_seconds.dtype.str,
            },
            "sample_validity": {
                "shape": list(self.sample_validity.shape),
                "dtype": self.sample_validity.dtype.str,
            },
            "interval_validity": {
                "shape": list(self.interval_validity.shape),
                "dtype": self.interval_validity.dtype.str,
            },
        }


def build_temporal_response_candidate_v1(
    rest_vertices_source_world: np.ndarray,
    posed_vertices_source_world: np.ndarray,
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    action_identity: str,
) -> TemporalResponseCandidateV1:
    """Build one `[L,1,T,3]` measured authored-action response.

    `sample_rows` must carry the global teacher `sample_index`, action identity
    and exact `time_seconds` produced by the timebase court. Different authored
    actions are intentionally never concatenated onto one time axis.
    """
    rest = np.asarray(rest_vertices_source_world, dtype=np.float64)
    posed = np.asarray(posed_vertices_source_world, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1:] != (3,) or not np.isfinite(rest).all():
        raise ValueError("TEMPORAL_RESPONSE_REST_LAYOUT_OR_NUMERIC_INVALID")
    if posed.ndim != 3 or posed.shape[1:] != rest.shape or not np.isfinite(posed).all():
        raise ValueError("TEMPORAL_RESPONSE_POSED_LAYOUT_OR_NUMERIC_INVALID")
    action = str(action_identity)
    if not action:
        raise ValueError("TEMPORAL_RESPONSE_ACTION_REQUIRED")

    rows = [row for row in sample_rows if str(row.get("action") or "") == action]
    if not rows:
        raise ValueError("TEMPORAL_RESPONSE_ACTION_HAS_NO_SAMPLES")
    rows.sort(key=lambda row: int(row["sample_index"]))
    indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(posed)) or len(set(map(int, indices))) != len(indices):
        raise ValueError("TEMPORAL_RESPONSE_SAMPLE_INDEX_INVALID")

    timestamps = np.asarray([float(row["time_seconds"]) for row in rows], dtype=np.float64)
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all() or timestamps[0] < -1.0e-12:
        raise ValueError("TEMPORAL_RESPONSE_TIME_INVALID")
    if abs(float(timestamps[0])) > 1.0e-12:
        raise ValueError("TEMPORAL_RESPONSE_ACTION_LOCAL_ZERO_REQUIRED")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("TEMPORAL_RESPONSE_TIME_NOT_STRICTLY_INCREASING")

    selected = posed[indices]
    # Teacher storage is [T,L,3]. Truth Foundry TEMPORAL_RESPONSE is [L,C,T,D].
    displacement_tld = selected - rest[None, :, :]
    response = np.transpose(displacement_tld, (1, 0, 2))[:, None, :, :]
    sample_validity = np.isfinite(response).all(axis=-1)
    if len(timestamps) > 1:
        dt_valid = np.diff(timestamps) > 0
        interval_validity = (
            sample_validity[:, :, :-1]
            & sample_validity[:, :, 1:]
            & dt_valid[None, None, :]
        )
    else:
        interval_validity = np.empty((len(rest), 1, 0), dtype=bool)

    candidate = TemporalResponseCandidateV1(
        action_identity=action,
        response=np.asarray(response, dtype=np.float64),
        timestamps_seconds=timestamps,
        sample_validity=np.asarray(sample_validity, dtype=bool),
        interval_validity=np.asarray(interval_validity, dtype=bool),
    )
    if candidate.response.shape != (len(rest), 1, len(timestamps), 3):
        raise AssertionError("TEMPORAL_RESPONSE_INTERNAL_LAYOUT_DRIFT")
    if candidate.interval_validity.shape != (len(rest), 1, max(0, len(timestamps) - 1)):
        raise AssertionError("TEMPORAL_RESPONSE_INTERVAL_LAYOUT_DRIFT")
    return candidate
