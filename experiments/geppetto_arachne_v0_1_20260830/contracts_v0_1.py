from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

import numpy as np


class VolumeState(IntEnum):
    CERTAIN_OUTSIDE = 0
    SUPPORTED_SURFACE_BAND = 1
    POSSIBLE_INTERIOR = 2
    HIGH_CONFIDENCE_INTERIOR = 3
    UNKNOWN_CONCAVITY = 4


@dataclass(frozen=True)
class ConsumerTokenV1:
    """Authority-safe substrate token used by G0.1/A0.1 adapters.

    Fields that do not exist for a token type are represented by an explicit
    validity bit; zero is never overloaded to mean "unknown/not applicable".
    """

    token_id: str
    position: tuple[float, float, float]
    token_type: str  # SURFACE | INTERIOR
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_valid: bool = False
    support_view_mask: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)
    geometry_uncertainty: float = 0.0
    geometry_uncertainty_valid: bool = False
    medialness: float = 0.0
    local_thickness: float = 0.0
    branch_likelihood: float = 0.0
    volume_state: VolumeState = VolumeState.POSSIBLE_INTERIOR
    provenance: dict[str, Any] = field(default_factory=dict)

    def feature_vector(self) -> np.ndarray:
        if self.token_type not in {"SURFACE", "INTERIOR"}:
            raise ValueError(f"bad token_type: {self.token_type}")
        if len(self.support_view_mask) != 8:
            raise ValueError("support_view_mask must have 8 entries")
        t = [1.0, 0.0] if self.token_type == "SURFACE" else [0.0, 1.0]
        state = np.zeros(5, dtype=np.float32)
        state[int(self.volume_state)] = 1.0
        out = np.asarray(
            [
                *self.position,
                *t,
                *self.normal,
                float(self.normal_valid),
                *[float(x) for x in self.support_view_mask],
                float(self.geometry_uncertainty),
                float(self.geometry_uncertainty_valid),
                float(self.medialness),
                float(self.local_thickness),
                float(self.branch_likelihood),
                *state.tolist(),
            ],
            dtype=np.float32,
        )
        if out.shape != (27,):
            raise AssertionError(out.shape)
        if not np.isfinite(out).all():
            raise ValueError(f"non-finite feature vector for {self.token_id}")
        return out


@dataclass(frozen=True)
class ProjectedControlV1:
    """Teacher-only anonymous control used to train Geppetto.

    `source_bone_index` is provenance, not a product/canonical ID.
    `teacher_tail` is retained only for audit/weight-column transport; it must
    not become a production A0 feature because QualifiedSkeletonIR does not
    expose source bone tails.
    """

    control_id: str
    source_bone_index: int
    position: tuple[float, float, float]
    teacher_tail: tuple[float, float, float]
    parent_control_id: str | None
    deform: bool = True


@dataclass(frozen=True)
class SkeletonTeacherProjectionV1:
    controls: tuple[ProjectedControlV1, ...]
    root_control_ids: tuple[str, ...]
    bfs_control_ids: tuple[str, ...]
    source_bone_count: int
    deform_control_count: int
    skipped_helper_count: int
    schema_version: str = "RealSaS.SkeletonTeacherProjection.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def control_by_id(self) -> dict[str, ProjectedControlV1]:
        return {c.control_id: c for c in self.controls}

    def validate(self) -> None:
        by = self.control_by_id()
        if len(by) != len(self.controls):
            raise ValueError("duplicate control_id")
        if set(self.bfs_control_ids) != set(by):
            raise ValueError("BFS serialization does not cover controls exactly")
        if len(self.bfs_control_ids) != len(by):
            raise ValueError("BFS serialization contains duplicates")
        roots = {c.control_id for c in self.controls if c.parent_control_id is None}
        if roots != set(self.root_control_ids):
            raise ValueError("root set mismatch")
        for c in self.controls:
            if c.parent_control_id is not None and c.parent_control_id not in by:
                raise ValueError(f"missing parent control: {c.control_id}")


@dataclass(frozen=True)
class PathStateFeaturesV1:
    state_fraction: tuple[float, float, float, float, float]
    length_by_state: tuple[float, float, float, float, float]
    total_length: float
    certain_outside_fraction: float
    unknown_concavity_fraction: float

    def feature_vector(self) -> np.ndarray:
        out = np.asarray(
            [*self.state_fraction, *self.length_by_state, self.total_length],
            dtype=np.float32,
        )
        if out.shape != (11,) or not np.isfinite(out).all():
            raise ValueError("bad path feature vector")
        return out


def stack_feature_vectors(tokens: Iterable[ConsumerTokenV1]) -> np.ndarray:
    rows = [t.feature_vector() for t in tokens]
    return np.stack(rows, axis=0) if rows else np.empty((0, 27), np.float32)
