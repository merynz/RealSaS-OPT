from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CAO_PROVENANCE_V1 = (
    "SOURCE_EXACT",
    "SOURCE_REPROJECTED",
    "LEARNED_SOURCE_CONSISTENT",
    "LEARNED_CROSSVIEW_SUPPORTED",
    "LEARNED_HIDDEN_COMPLETION",
    "COMPILER_BOUNDED_FILL",
    "UNSUPPORTED_ABSTAIN",
)

CAO_ORIGINS_V1 = (
    "OBSERVED_SOURCE",
    "LEARNED_FIELD",
    "COMPILER",
    "NONE",
)

CAO_DECOMPOSITION_RELATIONS_V1 = (
    "FRONT_OF",
    "BEHIND",
    "COPLANAR_TIE",
    "DISJOINT",
    "UNKNOWN",
)

_CA0_AUTHORITY_RANK_V1 = {
    "SOURCE_EXACT": 0,
    "SOURCE_REPROJECTED": 1,
    "LEARNED_SOURCE_CONSISTENT": 2,
    "LEARNED_CROSSVIEW_SUPPORTED": 3,
    "LEARNED_HIDDEN_COMPLETION": 4,
    "COMPILER_BOUNDED_FILL": 5,
    "UNSUPPORTED_ABSTAIN": 6,
}

_ALLOWED_ORIGIN_BY_PROVENANCE_V1 = {
    "SOURCE_EXACT": {"OBSERVED_SOURCE"},
    "SOURCE_REPROJECTED": {"OBSERVED_SOURCE"},
    "LEARNED_SOURCE_CONSISTENT": {"LEARNED_FIELD"},
    "LEARNED_CROSSVIEW_SUPPORTED": {"LEARNED_FIELD"},
    "LEARNED_HIDDEN_COMPLETION": {"LEARNED_FIELD"},
    "COMPILER_BOUNDED_FILL": {"COMPILER"},
    "UNSUPPORTED_ABSTAIN": {"NONE", "COMPILER", "LEARNED_FIELD"},
}


@dataclass(frozen=True)
class CAOAppearanceSampleV1:
    """One candidate appearance value on canonical-M surface addressing."""

    surface_address: str
    rgba: tuple[float, float, float, float] | None
    provenance: str
    origin: str
    confidence: float
    source_view: int | None = None
    owner_descriptor_hash: str = ""
    material_descriptor_hash: str = ""
    source_region_hash: str = ""

    def validate(self) -> None:
        if not self.surface_address:
            raise ValueError("CAO_SURFACE_ADDRESS_MISSING")
        if self.provenance not in CAO_PROVENANCE_V1:
            raise ValueError("CAO_PROVENANCE_INVALID")
        if self.origin not in CAO_ORIGINS_V1:
            raise ValueError("CAO_ORIGIN_INVALID")
        if self.origin not in _ALLOWED_ORIGIN_BY_PROVENANCE_V1[self.provenance]:
            raise ValueError("CAO_PROVENANCE_ORIGIN_MISMATCH")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("CAO_CONFIDENCE_OUT_OF_RANGE")
        if self.provenance == "UNSUPPORTED_ABSTAIN":
            if self.rgba is not None:
                raise ValueError("CAO_ABSTAIN_MUST_NOT_CARRY_RGBA")
        else:
            if self.rgba is None or len(self.rgba) != 4:
                raise ValueError("CAO_RGBA_REQUIRED")
            if any(not (0.0 <= float(v) <= 1.0) for v in self.rgba):
                raise ValueError("CAO_RGBA_OUT_OF_RANGE")
        if self.origin == "OBSERVED_SOURCE" and self.source_view is None:
            raise ValueError("CAO_OBSERVED_SOURCE_VIEW_MISSING")
        if self.origin != "OBSERVED_SOURCE" and self.source_view is not None:
            raise ValueError("CAO_NON_SOURCE_SAMPLE_HAS_SOURCE_VIEW")


@dataclass(frozen=True)
class CAODecompositionFragmentV1:
    """Generic appearance decomposition unit, independent of human anatomy.

    A fragment is an editable/presentable visual ownership region bound to the
    exact canonical M domain. It may correspond to one authored object, a
    learned owner cluster, a material region, or a compiler-resolved split.
    Draw-order policy consumes relations between fragment IDs rather than raw
    camera-Z alone.
    """

    fragment_id: str
    owner_descriptor_hash: str
    material_descriptor_hash: str
    mechanical_owner_hash: str = ""
    amodal_support_hash: str = ""
    attachment_relation_hash: str = ""
    semantic_confidence: float = 1.0

    def validate(self) -> None:
        if not self.fragment_id:
            raise ValueError("CAO_FRAGMENT_ID_MISSING")
        if not self.owner_descriptor_hash:
            raise ValueError("CAO_FRAGMENT_OWNER_DESCRIPTOR_MISSING")
        if not self.material_descriptor_hash:
            raise ValueError("CAO_FRAGMENT_MATERIAL_DESCRIPTOR_MISSING")
        if not (0.0 <= float(self.semantic_confidence) <= 1.0):
            raise ValueError("CAO_FRAGMENT_CONFIDENCE_OUT_OF_RANGE")


@dataclass(frozen=True)
class CAOOverlapRelationV1:
    front_fragment_id: str
    back_fragment_id: str
    view_id: int
    relation: str
    confidence: float
    pose_dependent: bool

    def validate(self) -> None:
        if self.relation not in CAO_DECOMPOSITION_RELATIONS_V1:
            raise ValueError("CAO_OVERLAP_RELATION_INVALID")
        if not self.front_fragment_id or not self.back_fragment_id:
            raise ValueError("CAO_OVERLAP_FRAGMENT_ID_MISSING")
        if self.front_fragment_id == self.back_fragment_id:
            raise ValueError("CAO_OVERLAP_SELF_RELATION")
        if not (0 <= int(self.view_id) < 8):
            raise ValueError("CAO_OVERLAP_VIEW_OUT_OF_RANGE")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("CAO_OVERLAP_CONFIDENCE_OUT_OF_RANGE")


@dataclass(frozen=True)
class CAOVisibleSurfaceCoherenceReportV1:
    """Gate joining qualified geometry to final visible appearance.

    A canonical M can be mechanically/static qualified while the final raster
    still contains holes, peppering, or disconnected alpha islands. CAO treats
    this as a first-class failure unless the missing coverage is explicitly
    explained by qualified transparency, occlusion, or intentional absence.
    """

    geometric_visible_pixels: int
    appearance_visible_pixels: int
    unexplained_missing_pixels: int
    fragmented_island_pixels: int
    maximum_unexplained_hole_area_px: int
    explained_transparent_pixels: int = 0
    explained_occluded_pixels: int = 0

    def validate(self) -> None:
        values = (
            self.geometric_visible_pixels,
            self.appearance_visible_pixels,
            self.unexplained_missing_pixels,
            self.fragmented_island_pixels,
            self.maximum_unexplained_hole_area_px,
            self.explained_transparent_pixels,
            self.explained_occluded_pixels,
        )
        if any(int(v) < 0 for v in values):
            raise ValueError("CAO_VISIBLE_COHERENCE_NEGATIVE_COUNT")
        if self.appearance_visible_pixels > self.geometric_visible_pixels:
            raise ValueError("CAO_APPEARANCE_EXCEEDS_GEOMETRIC_DOMAIN")

    @property
    def pass_fail_closed(self) -> bool:
        self.validate()
        return (
            self.unexplained_missing_pixels == 0
            and self.fragmented_island_pixels == 0
            and self.maximum_unexplained_hole_area_px == 0
        )


def resolve_cao_sample_v1(candidates: Iterable[CAOAppearanceSampleV1]) -> CAOAppearanceSampleV1:
    """Fail-closed source-wins ordering for one exact M surface address."""

    rows = tuple(candidates)
    if not rows:
        raise ValueError("CAO_CANDIDATE_SET_EMPTY")
    for row in rows:
        row.validate()
    addresses = {row.surface_address for row in rows}
    if len(addresses) != 1:
        raise ValueError("CAO_CANDIDATE_ADDRESS_DRIFT")

    # Rank by authority class first, then confidence. This makes learned evidence
    # completion-capable without allowing it to overwrite exact observed source.
    return min(
        rows,
        key=lambda row: (
            _CA0_AUTHORITY_RANK_V1[row.provenance],
            -float(row.confidence),
        ),
    )


__all__ = [
    "CAO_DECOMPOSITION_RELATIONS_V1",
    "CAO_ORIGINS_V1",
    "CAO_PROVENANCE_V1",
    "CAOAppearanceSampleV1",
    "CAODecompositionFragmentV1",
    "CAOOverlapRelationV1",
    "CAOVisibleSurfaceCoherenceReportV1",
    "resolve_cao_sample_v1",
]
