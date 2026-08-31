from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProjectedControlV1:
    """Teacher-only anonymous control for Geppetto supervision/evaluation.

    Source bone index and tail are provenance only. This type has no product or
    canonical skeleton authority and must never be used as QualifiedSkeletonIR.
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
        return {control.control_id: control for control in self.controls}

    def validate(self) -> None:
        by_id = self.control_by_id()
        if len(by_id) != len(self.controls):
            raise ValueError("duplicate control_id")
        if set(self.bfs_control_ids) != set(by_id):
            raise ValueError("BFS serialization does not cover controls exactly")
        if len(self.bfs_control_ids) != len(by_id):
            raise ValueError("BFS serialization contains duplicates")
        roots = {c.control_id for c in self.controls if c.parent_control_id is None}
        if roots != set(self.root_control_ids):
            raise ValueError("root set mismatch")
        for control in self.controls:
            if control.parent_control_id is not None and control.parent_control_id not in by_id:
                raise ValueError(f"missing parent control: {control.control_id}")
            if control.control_id.startswith("J:"):
                raise ValueError("teacher projection may not mint canonical J:* IDs")
