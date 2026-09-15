from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.attachment_interface_authority import (
    BilateralAttachmentViewEvidenceIR,
    qualify_bilateral_attachment_authority,
)
from compiler.realsas_compiler_core.types import QualificationError


def _row(view: int, vote: str, margin: float = 0.8):
    if vote == "LEFT":
        left, right, informative = 20.0, 100.0, True
    elif vote == "RIGHT":
        left, right, informative = 100.0, 20.0, True
    else:
        left, right, informative, margin = 100.0, 101.0, False, 0.01
    return BilateralAttachmentViewEvidenceIR(
        view_index=view,
        interface_sample_count=100,
        interface_centroid_xy=(50.0, 50.0),
        left_pivot_xy=(20.0, 50.0),
        right_pivot_xy=(80.0, 50.0),
        left_median_distance_px=left,
        right_median_distance_px=right,
        normalized_margin=margin,
        vote=vote,
        informative=informative,
        projection_binding_hash=f"projection:{view}",
    )


def test_six_informative_views_can_override_opposite_source_side_label_without_promoting_names():
    rows = tuple(_row(view, "RIGHT" if view not in (2, 6) else "AMBIGUOUS") for view in range(8))
    authority = qualify_bilateral_attachment_authority(
        component_id="fixture",
        left_candidate_joint_id="J:left",
        right_candidate_joint_id="J:right",
        view_evidence=rows,
        metadata={"source_side_label": "handslot.l"},
    )
    assert authority.selected_side == "RIGHT"
    assert authority.selected_joint_id == "J:right"
    assert authority.informative_view_count == 6
    assert authority.right_vote_count == 6
    assert authority.left_vote_count == 0
    assert authority.metadata["source_side_label"] == "handslot.l"
    assert authority.metadata["source_side_label_promoted_to_canonical_side"] is False
    assert authority.metadata["filename_or_component_name_used_for_side_selection"] is False


def test_side_views_may_be_ambiguous_but_do_not_vote():
    rows = tuple(_row(view, "LEFT" if view not in (2, 6) else "AMBIGUOUS") for view in range(8))
    authority = qualify_bilateral_attachment_authority(
        component_id="fixture",
        left_candidate_joint_id="J:left",
        right_candidate_joint_id="J:right",
        view_evidence=rows,
    )
    assert authority.selected_side == "LEFT"
    assert authority.informative_view_count == 6
    assert authority.left_vote_count == 6


def test_split_multiview_evidence_fails_closed():
    rows = tuple(_row(view, "LEFT" if view < 4 else "RIGHT") for view in range(8))
    with pytest.raises(QualificationError, match="VOTE_TIE"):
        qualify_bilateral_attachment_authority(
            component_id="fixture",
            left_candidate_joint_id="J:left",
            right_candidate_joint_id="J:right",
            view_evidence=rows,
        )


def test_low_margin_evidence_fails_closed():
    rows = tuple(
        BilateralAttachmentViewEvidenceIR(
            view_index=view,
            interface_sample_count=100,
            interface_centroid_xy=(50.0, 50.0),
            left_pivot_xy=(20.0, 50.0),
            right_pivot_xy=(80.0, 50.0),
            left_median_distance_px=100.0,
            right_median_distance_px=85.0,
            normalized_margin=0.15,
            vote="AMBIGUOUS",
            informative=False,
            projection_binding_hash=f"projection:{view}",
        )
        for view in range(8)
    )
    with pytest.raises(QualificationError, match="INSUFFICIENT_INFORMATIVE_VIEWS"):
        qualify_bilateral_attachment_authority(
            component_id="fixture",
            left_candidate_joint_id="J:left",
            right_candidate_joint_id="J:right",
            view_evidence=rows,
        )
