from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import hypot, isfinite
from statistics import median
from types import SimpleNamespace
from typing import Any, Iterable

from .directional_binding import DirectionalBindingPolicyV1, _fit_view_projection, project_mechanical_point
from .hashing import content_sha256
from .types import QualificationError


@dataclass(frozen=True)
class BilateralAttachmentPolicyV1:
    min_interface_samples_per_view: int = 32
    min_informative_views: int = 4
    min_view_normalized_margin: float = 0.20
    min_winner_vote_fraction: float = 0.80
    min_median_winner_margin: float = 0.30
    required_views: int = 8
    schema_version: str = "RealSaS.BilateralAttachmentPolicy.v1"

    def validate(self) -> None:
        if self.min_interface_samples_per_view < 1:
            raise ValueError("min_interface_samples_per_view must be positive")
        if not (1 <= self.min_informative_views <= self.required_views):
            raise ValueError("invalid informative-view requirement")
        if not (0.0 < self.min_view_normalized_margin < 1.0):
            raise ValueError("invalid per-view margin")
        if not (0.5 < self.min_winner_vote_fraction <= 1.0):
            raise ValueError("invalid vote fraction")
        if not (0.0 < self.min_median_winner_margin < 1.0):
            raise ValueError("invalid winner margin")
        if self.required_views != 8:
            raise ValueError("current directional product requires exactly eight views")


@dataclass(frozen=True)
class BilateralAttachmentViewEvidenceIR:
    view_index: int
    interface_sample_count: int
    interface_centroid_xy: tuple[float, float]
    left_pivot_xy: tuple[float, float]
    right_pivot_xy: tuple[float, float]
    left_median_distance_px: float
    right_median_distance_px: float
    normalized_margin: float
    vote: str
    informative: bool
    projection_binding_hash: str
    schema_version: str = "RealSaS.BilateralAttachmentViewEvidenceIR.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedBilateralAttachmentAuthorityIR:
    component_id: str
    left_candidate_joint_id: str
    right_candidate_joint_id: str
    selected_side: str
    selected_joint_id: str
    informative_view_count: int
    left_vote_count: int
    right_vote_count: int
    winner_vote_fraction: float
    median_winner_margin: float
    view_evidence: tuple[BilateralAttachmentViewEvidenceIR, ...]
    qualification_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "RealSaS.QualifiedBilateralAttachmentAuthorityIR.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _authority_hash(value: QualifiedBilateralAttachmentAuthorityIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def build_bilateral_view_evidence(
    *,
    view_index: int,
    interface_points_xy: Iterable[tuple[float, float]],
    left_pivot_xy: tuple[float, float],
    right_pivot_xy: tuple[float, float],
    projection_binding_hash: str,
    policy: BilateralAttachmentPolicyV1 = BilateralAttachmentPolicyV1(),
) -> BilateralAttachmentViewEvidenceIR:
    policy.validate()
    points = tuple((float(x), float(y)) for x, y in interface_points_xy)
    if not points:
        raise QualificationError(f"BILATERAL_ATTACHMENT_EMPTY_INTERFACE:V{view_index}")
    lx, ly = map(float, left_pivot_xy)
    rx, ry = map(float, right_pivot_xy)
    if any(not isfinite(v) for v in (lx, ly, rx, ry)):
        raise QualificationError("BILATERAL_ATTACHMENT_NONFINITE_PIVOT")
    left_dist = tuple(hypot(x - lx, y - ly) for x, y in points)
    right_dist = tuple(hypot(x - rx, y - ry) for x, y in points)
    lm = float(median(left_dist))
    rm = float(median(right_dist))
    denom = max(lm, rm, 1.0)
    margin = abs(lm - rm) / denom
    enough = len(points) >= int(policy.min_interface_samples_per_view)
    informative = bool(enough and margin >= float(policy.min_view_normalized_margin))
    vote = "AMBIGUOUS"
    if informative:
        vote = "LEFT" if lm < rm else "RIGHT"
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return BilateralAttachmentViewEvidenceIR(
        int(view_index), len(points), (float(cx), float(cy)), (lx, ly), (rx, ry),
        lm, rm, float(margin), vote, informative, str(projection_binding_hash),
    )


def qualify_bilateral_attachment_authority(
    *,
    component_id: str,
    left_candidate_joint_id: str,
    right_candidate_joint_id: str,
    view_evidence: Iterable[BilateralAttachmentViewEvidenceIR],
    policy: BilateralAttachmentPolicyV1 = BilateralAttachmentPolicyV1(),
    metadata: dict[str, Any] | None = None,
) -> QualifiedBilateralAttachmentAuthorityIR:
    policy.validate()
    rows = tuple(sorted(tuple(view_evidence), key=lambda row: int(row.view_index)))
    if len(rows) != policy.required_views or tuple(row.view_index for row in rows) != tuple(range(policy.required_views)):
        raise QualificationError("BILATERAL_ATTACHMENT_REQUIRES_EXACT_VIEWS_0_TO_7")
    left_joint = str(left_candidate_joint_id or "")
    right_joint = str(right_candidate_joint_id or "")
    if not left_joint or not right_joint or left_joint == right_joint:
        raise QualificationError("BILATERAL_ATTACHMENT_INVALID_CANDIDATE_JOINTS")
    informative = tuple(row for row in rows if row.informative)
    if len(informative) < int(policy.min_informative_views):
        raise QualificationError(
            f"BILATERAL_ATTACHMENT_INSUFFICIENT_INFORMATIVE_VIEWS:{len(informative)}"
        )
    left_votes = sum(row.vote == "LEFT" for row in informative)
    right_votes = sum(row.vote == "RIGHT" for row in informative)
    if left_votes == right_votes:
        raise QualificationError("BILATERAL_ATTACHMENT_VOTE_TIE")
    selected_side = "LEFT" if left_votes > right_votes else "RIGHT"
    winner_votes = max(left_votes, right_votes)
    vote_fraction = float(winner_votes) / float(len(informative))
    if vote_fraction < float(policy.min_winner_vote_fraction):
        raise QualificationError(
            f"BILATERAL_ATTACHMENT_VOTE_CONSENSUS_TOO_LOW:{vote_fraction}"
        )
    winner_rows = tuple(row for row in informative if row.vote == selected_side)
    winner_margin = float(median(row.normalized_margin for row in winner_rows))
    if winner_margin < float(policy.min_median_winner_margin):
        raise QualificationError(
            f"BILATERAL_ATTACHMENT_MARGIN_TOO_LOW:{winner_margin}"
        )
    selected_joint = left_joint if selected_side == "LEFT" else right_joint
    value = QualifiedBilateralAttachmentAuthorityIR(
        component_id=str(component_id),
        left_candidate_joint_id=left_joint,
        right_candidate_joint_id=right_joint,
        selected_side=selected_side,
        selected_joint_id=selected_joint,
        informative_view_count=len(informative),
        left_vote_count=int(left_votes),
        right_vote_count=int(right_votes),
        winner_vote_fraction=vote_fraction,
        median_winner_margin=winner_margin,
        view_evidence=rows,
        qualification_hash="",
        metadata={
            "authority": "MULTIVIEW_OWNER_INTERFACE_TO_CURRENT_CANONICAL_HAND_PIVOTS",
            "source_side_label_promoted_to_canonical_side": False,
            "filename_or_component_name_used_for_side_selection": False,
            "ambiguous_side_views_allowed_but_never_counted_as_votes": True,
            "policy": asdict(policy),
            **dict(metadata or {}),
        },
    )
    value = replace(value, qualification_hash=_authority_hash(value))
    validate_bilateral_attachment_authority(value, policy=policy)
    return value


def validate_bilateral_attachment_authority(
    value: QualifiedBilateralAttachmentAuthorityIR,
    *,
    policy: BilateralAttachmentPolicyV1 = BilateralAttachmentPolicyV1(),
) -> None:
    policy.validate()
    if value.qualification_hash != _authority_hash(value):
        raise QualificationError("BILATERAL_ATTACHMENT_AUTHORITY_HASH_MISMATCH")
    if value.selected_side not in {"LEFT", "RIGHT"}:
        raise QualificationError("BILATERAL_ATTACHMENT_INVALID_SELECTED_SIDE")
    expected_joint = value.left_candidate_joint_id if value.selected_side == "LEFT" else value.right_candidate_joint_id
    if value.selected_joint_id != expected_joint:
        raise QualificationError("BILATERAL_ATTACHMENT_SELECTED_JOINT_DRIFT")
    if len(value.view_evidence) != policy.required_views:
        raise QualificationError("BILATERAL_ATTACHMENT_VIEW_EVIDENCE_INCOMPLETE")
    if bool(value.metadata.get("source_side_label_promoted_to_canonical_side", True)):
        raise QualificationError("BILATERAL_ATTACHMENT_SOURCE_SIDE_PROMOTION_FORBIDDEN")
    if bool(value.metadata.get("filename_or_component_name_used_for_side_selection", True)):
        raise QualificationError("BILATERAL_ATTACHMENT_NAME_SIDE_INFERENCE_FORBIDDEN")


def fit_preproduct_view_projection(
    mechanical,
    *,
    view_index: int,
    camera_binding_hash: str,
    authority_state_hash: str,
    policy: DirectionalBindingPolicyV1 = DirectionalBindingPolicyV1(),
):
    """Reuse the exact directional-binding fitter before CanonicalPuppetGraph exists."""
    state_hash = str(authority_state_hash or "").strip()
    if not state_hash:
        raise QualificationError("BILATERAL_ATTACHMENT_PREPRODUCT_AUTHORITY_HASH_REQUIRED")
    proxy = SimpleNamespace(product_state_hash=state_hash, mechanical_state=mechanical)
    return _fit_view_projection(proxy, int(view_index), str(camera_binding_hash), policy)


def project_joint_candidate(projection, skeleton, joint_id: str) -> tuple[float, float]:
    by_id = {str(j.canonical_joint_id): j for j in skeleton.joints}
    joint = by_id.get(str(joint_id))
    if joint is None:
        raise QualificationError(f"BILATERAL_ATTACHMENT_UNKNOWN_JOINT:{joint_id}")
    return project_mechanical_point(projection, joint.position)


__all__ = [
    "BilateralAttachmentPolicyV1",
    "BilateralAttachmentViewEvidenceIR",
    "QualifiedBilateralAttachmentAuthorityIR",
    "build_bilateral_view_evidence",
    "qualify_bilateral_attachment_authority",
    "validate_bilateral_attachment_authority",
    "fit_preproduct_view_projection",
    "project_joint_candidate",
]
