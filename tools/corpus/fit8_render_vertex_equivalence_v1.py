from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence
import math

SCHEMA_VERSION = "RealSaS.RenderVertexGeometricEquivalence.v1"


@dataclass(frozen=True)
class RenderVertexEvidenceV1:
    render_vertex_id: int
    rest_xyz: tuple[float, float, float]
    deformation_signature: str
    render_signature: str

    def validate(self) -> None:
        if (
            self.render_vertex_id < 0
            or len(self.rest_xyz) != 3
            or not all(math.isfinite(float(value)) for value in self.rest_xyz)
        ):
            raise ValueError("RENDER_EQ_VERTEX_INVALID")
        if not self.deformation_signature:
            raise ValueError("RENDER_EQ_DEFORMATION_SIGNATURE_MISSING")
        if not self.render_signature:
            raise ValueError("RENDER_EQ_RENDER_SIGNATURE_MISSING")


@dataclass(frozen=True)
class GeometricEquivalenceGroupV1:
    geometric_vertex_id: int
    rest_xyz: tuple[float, float, float]
    deformation_signature: str
    render_vertex_ids: tuple[int, ...]
    render_signature_count: int


@dataclass(frozen=True)
class RenderVertexEquivalenceReportV1:
    schema_version: str
    render_vertex_count: int
    geometric_vertex_count: int
    merged_render_vertex_count: int
    coincident_position_count: int
    mechanically_conflicted_position_count: int
    render_seam_group_count: int
    groups: tuple[GeometricEquivalenceGroupV1, ...]
    render_to_geometric: tuple[int, ...]

    @property
    def mechanical_conflict_free(self) -> bool:
        return self.mechanically_conflicted_position_count == 0

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("RENDER_EQ_SCHEMA_INVALID")
        if len(self.render_to_geometric) != self.render_vertex_count:
            raise ValueError("RENDER_EQ_MAPPING_CARDINALITY_INVALID")
        if len(self.groups) != self.geometric_vertex_count:
            raise ValueError("RENDER_EQ_GROUP_CARDINALITY_INVALID")


def qualify_render_vertex_equivalence_v1(
    vertices: Sequence[RenderVertexEvidenceV1],
) -> RenderVertexEquivalenceReportV1:
    """Create a mechanical/geometric quotient without deleting render identity.

    V1 uses exact rest-position equality. A tolerance-based position policy, if ever
    needed, must be a separate provenance-carrying contract rather than a hidden
    weld epsilon.

    Coincident vertices with different deformation signatures are deliberately kept
    separate. Normal/UV/material differences belong in `render_signature` and are
    allowed to survive as many render vertices mapping to one qualified geometric
    identity.
    """
    if not vertices:
        raise ValueError("RENDER_EQ_EMPTY")
    ids = [vertex.render_vertex_id for vertex in vertices]
    if sorted(ids) != list(range(len(vertices))):
        raise ValueError("RENDER_EQ_IDS_MUST_BE_CONTIGUOUS")
    for vertex in vertices:
        vertex.validate()

    by_position: dict[tuple[float, float, float], list[RenderVertexEvidenceV1]] = defaultdict(list)
    for vertex in vertices:
        by_position[tuple(map(float, vertex.rest_xyz))].append(vertex)

    coincident_position_count = sum(len(rows) > 1 for rows in by_position.values())
    mechanical_conflicts = sum(
        len({vertex.deformation_signature for vertex in rows}) > 1
        for rows in by_position.values()
        if len(rows) > 1
    )

    raw_groups = []
    for position, rows in by_position.items():
        by_mechanics: dict[str, list[RenderVertexEvidenceV1]] = defaultdict(list)
        for vertex in rows:
            by_mechanics[vertex.deformation_signature].append(vertex)
        for deformation_signature, group in by_mechanics.items():
            raw_groups.append(
                (
                    position,
                    deformation_signature,
                    tuple(sorted(vertex.render_vertex_id for vertex in group)),
                    len({vertex.render_signature for vertex in group}),
                )
            )
    raw_groups.sort(key=lambda row: (row[0], row[1], row[2]))

    render_to_geometric = [-1] * len(vertices)
    groups: list[GeometricEquivalenceGroupV1] = []
    seam_group_count = 0
    for geometric_vertex_id, (position, mechanics, render_ids, render_signature_count) in enumerate(raw_groups):
        for render_vertex_id in render_ids:
            render_to_geometric[render_vertex_id] = geometric_vertex_id
        if len(render_ids) > 1 and render_signature_count > 1:
            seam_group_count += 1
        groups.append(
            GeometricEquivalenceGroupV1(
                geometric_vertex_id=geometric_vertex_id,
                rest_xyz=position,
                deformation_signature=mechanics,
                render_vertex_ids=render_ids,
                render_signature_count=render_signature_count,
            )
        )

    report = RenderVertexEquivalenceReportV1(
        schema_version=SCHEMA_VERSION,
        render_vertex_count=len(vertices),
        geometric_vertex_count=len(groups),
        merged_render_vertex_count=len(vertices) - len(groups),
        coincident_position_count=coincident_position_count,
        mechanically_conflicted_position_count=mechanical_conflicts,
        render_seam_group_count=seam_group_count,
        groups=tuple(groups),
        render_to_geometric=tuple(render_to_geometric),
    )
    report.validate()
    return report


def verify_equivalence_motion_witness_v1(
    report: RenderVertexEquivalenceReportV1,
    posed_xyz_by_context: Mapping[str, Sequence[Sequence[float]]],
    tolerance: float = 1e-7,
) -> dict:
    """Independently verify that merged render identities remain coincident in motion."""
    import numpy as np

    report.validate()
    if tolerance < 0.0:
        raise ValueError("RENDER_EQ_MOTION_TOLERANCE_INVALID")
    worst_separation = 0.0
    violations = []
    for context, rows in posed_xyz_by_context.items():
        array = np.asarray(rows, dtype=np.float64)
        if array.shape != (report.render_vertex_count, 3):
            raise ValueError("RENDER_EQ_MOTION_SHAPE_INVALID")
        for group in report.groups:
            if len(group.render_vertex_ids) < 2:
                continue
            points = array[list(group.render_vertex_ids)]
            separation = float(
                np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2))
            )
            worst_separation = max(worst_separation, separation)
            if separation > tolerance:
                violations.append((context, group.geometric_vertex_id, separation))
    return {
        "context_count": len(posed_xyz_by_context),
        "worst_separation": worst_separation,
        "violations": tuple(violations),
        "pass": not violations,
    }
