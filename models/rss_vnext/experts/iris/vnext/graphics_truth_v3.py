from __future__ import annotations

"""Role-free Graphics Truth V3.

V3 makes the architectural separation explicit:
- pair truth may supervise static relations, projected boundaries and pairwise
  front/occlusion;
- global visibility is a complete ordered ray-stack target and never a pair
  target.

Unknown structural ownership/attachment continues to abstain.  No semantic name
or material-ID heuristic becomes drawable authority.
"""

from dataclasses import dataclass

from .boundary_causality_v1 import EvidenceAuthority
from .graphics_truth_v2 import QualifiedProbabilityTargetV2


PAIR_STATIC_KEYS_V3 = (
    "same_drawable_affinity",
    "cut_probability",
    "attachment_affinity",
)
PAIR_VIEW_KEYS_V3 = (
    "front_probability",
    "occlusion_probability",
    "projected_alpha_boundary_probability",
    "projected_appearance_boundary_probability",
)
PAIR_KEYS_V3 = PAIR_STATIC_KEYS_V3 + PAIR_VIEW_KEYS_V3


@dataclass(frozen=True)
class GraphicsPairTruthRowV3:
    view_index: int
    face_a: int
    face_b: int
    same_drawable_affinity: QualifiedProbabilityTargetV2
    cut_probability: QualifiedProbabilityTargetV2
    attachment_affinity: QualifiedProbabilityTargetV2
    front_probability: QualifiedProbabilityTargetV2
    occlusion_probability: QualifiedProbabilityTargetV2
    projected_alpha_boundary_probability: QualifiedProbabilityTargetV2
    projected_appearance_boundary_probability: QualifiedProbabilityTargetV2
    schema_version: str = "RealSaS.GraphicsPairTruthRow.v3"

    def __post_init__(self) -> None:
        if self.view_index < 0 or self.face_a < 0 or self.face_b < 0 or self.face_a == self.face_b:
            raise ValueError("GRAPHICS_TRUTH_V3_PAIR_ROW_INVALID")

    def training_maps(self) -> tuple[dict[str, float], dict[str, bool]]:
        values: dict[str, float] = {}
        validity: dict[str, bool] = {}
        for key in PAIR_KEYS_V3:
            target = getattr(self, key)
            values[key] = float(target.value)
            validity[key] = bool(target.valid_for_training)
        return values, validity


@dataclass(frozen=True)
class RayStackVisibilityTruthV1:
    """Global visibility truth for one complete, depth-ordered camera ray."""

    view_index: int
    ray_index: int
    face_ids: tuple[int, ...]
    depths: tuple[float, ...]
    visibility: tuple[QualifiedProbabilityTargetV2, ...]
    complete_stack: bool
    provenance: str
    schema_version: str = "RealSaS.RayStackVisibilityTruth.v1"

    def __post_init__(self) -> None:
        if self.view_index < 0 or self.ray_index < 0:
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_INDEX_INVALID")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_PROVENANCE_REQUIRED")
        if len(self.face_ids) != len(self.depths) or len(self.face_ids) != len(self.visibility):
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_CARDINALITY_MISMATCH")
        if not self.face_ids or any(face < 0 for face in self.face_ids):
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_FACE_INVALID")
        if any(not (depth > 0.0) for depth in self.depths):
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_DEPTH_INVALID")
        if any(self.depths[i] > self.depths[i + 1] for i in range(len(self.depths) - 1)):
            raise ValueError("GRAPHICS_TRUTH_V3_RAY_DEPTH_NOT_ORDERED")
        if not self.complete_stack and any(target.valid_for_training for target in self.visibility):
            raise ValueError("GRAPHICS_TRUTH_V3_INCOMPLETE_STACK_CANNOT_TRAIN_VISIBILITY")

    @classmethod
    def from_complete_depth_order(
        cls,
        *,
        view_index: int,
        ray_index: int,
        face_ids: tuple[int, ...],
        depths: tuple[float, ...],
        provenance: str,
    ) -> "RayStackVisibilityTruthV1":
        if not face_ids:
            raise ValueError("GRAPHICS_TRUTH_V3_COMPLETE_STACK_EMPTY")
        visibility = tuple(
            QualifiedProbabilityTargetV2(
                value=1.0 if rank == 0 else 0.0,
                authority=EvidenceAuthority.DETERMINISTIC_DERIVED,
                provenance=provenance + f":rank={rank}",
                valid_for_training=True,
            )
            for rank in range(len(face_ids))
        )
        return cls(
            view_index=view_index,
            ray_index=ray_index,
            face_ids=tuple(int(v) for v in face_ids),
            depths=tuple(float(v) for v in depths),
            visibility=visibility,
            complete_stack=True,
            provenance=provenance,
        )

    @classmethod
    def abstain_incomplete(
        cls,
        *,
        view_index: int,
        ray_index: int,
        face_ids: tuple[int, ...],
        depths: tuple[float, ...],
        provenance: str,
    ) -> "RayStackVisibilityTruthV1":
        visibility = tuple(
            QualifiedProbabilityTargetV2.abstain(provenance=provenance + f":rank={rank}:INCOMPLETE_STACK")
            for rank in range(len(face_ids))
        )
        return cls(
            view_index=view_index,
            ray_index=ray_index,
            face_ids=tuple(int(v) for v in face_ids),
            depths=tuple(float(v) for v in depths),
            visibility=visibility,
            complete_stack=False,
            provenance=provenance,
        )

    def training_maps(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        return (
            tuple(float(target.value) for target in self.visibility),
            tuple(bool(target.valid_for_training) for target in self.visibility),
        )
