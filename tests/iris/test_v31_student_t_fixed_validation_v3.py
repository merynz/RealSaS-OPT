from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models.iris.v31.fixed_snake_evidence_student_t_fixed_validation_v3 import (
    IMPLEMENTATION_REVISION_V3,
    IrisV31FixedSnakeEvidenceStudentTFixedValidationV3,
)


def test_validation_rows_are_step_invariant_and_complete_partition():
    trainer = object.__new__(IrisV31FixedSnakeEvidenceStudentTFixedValidationV3)
    subject = SimpleNamespace(
        partitions={
            "mechanics": (
                np.asarray([0, 1, 3], dtype=np.int64),
                np.asarray([2, 4, 5, 9], dtype=np.int64),
            )
        }
    )
    a = trainer._presentation_ids(subject, "mechanics", 0, train=False)
    b = trainer._presentation_ids(subject, "mechanics", 2000, train=False)
    expected = np.asarray([2, 4, 5, 9], dtype=np.int64)
    assert np.array_equal(a, expected)
    assert np.array_equal(b, expected)
    assert a is not subject.partitions["mechanics"][1]
    assert IMPLEMENTATION_REVISION_V3.endswith("FIXED_FULL_VALIDATION_PARTITION")
