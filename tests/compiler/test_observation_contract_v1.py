import numpy as np
import pytest

from compiler.realsas_compiler_core.observation_contract_v1 import (
    ObservationEvidence,
    aggregate_multiview_evidence_v1,
    audit_controlled_observation_set_v1,
    classify_projected_observation_evidence_v1,
    require_controlled_observation_set_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _mask():
    m = np.zeros((64, 64), dtype=bool)
    m[8:56, 10:54] = True
    return m


def test_full_subject_frame_requires_margin_all_eight_views():
    audit = require_controlled_observation_set_v1([_mask() for _ in range(8)], minimum_margin_px=8)
    assert audit.full_subject_frame_pass
    assert audit.minimum_observed_margin_px == 8


def test_border_contact_fails_even_if_other_views_are_good():
    masks = [_mask() for _ in range(8)]
    bad = _mask()
    bad[:4, 20:30] = True
    masks[3] = bad
    audit = audit_controlled_observation_set_v1(masks, minimum_margin_px=8)
    assert not audit.full_subject_frame_pass
    assert "TOP" in audit.views[3].border_contacts
    with pytest.raises(QualificationError):
        require_controlled_observation_set_v1(masks, minimum_margin_px=8)


def test_out_of_frame_is_unknown_not_negative():
    m = _mask()
    xy = np.asarray([[-1.0, 20.0], [20.0, 20.0], [2.0, 2.0], [64.0, 20.0]])
    got = classify_projected_observation_evidence_v1(xy, m)
    assert got.tolist() == [
        ObservationEvidence.UNKNOWN,
        ObservationEvidence.POSITIVE,
        ObservationEvidence.NEGATIVE,
        ObservationEvidence.UNKNOWN,
    ]


def test_multiview_positive_dominates_negative_and_unknown():
    s = np.zeros((8, 3), dtype=np.int8)
    s[:, 0] = ObservationEvidence.UNKNOWN
    s[:, 1] = ObservationEvidence.UNKNOWN
    s[2, 1] = ObservationEvidence.NEGATIVE
    s[:, 2] = ObservationEvidence.NEGATIVE
    s[6, 2] = ObservationEvidence.POSITIVE
    got = aggregate_multiview_evidence_v1(s)
    assert got.tolist() == [
        ObservationEvidence.UNKNOWN,
        ObservationEvidence.NEGATIVE,
        ObservationEvidence.POSITIVE,
    ]
