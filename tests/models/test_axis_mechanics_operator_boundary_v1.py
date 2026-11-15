from __future__ import annotations

"""Guard AXIS V6's wider research grammar against an unexecutable product rig."""

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6,
    CONSTRAINT_OPERATORS_V6,
)
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1,
    validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_wider_axis_vocabulary_is_not_an_unhandled_product_operator():
    actual = set(RIG_CONSTRAINT_OPERATOR_IDS_V1)
    research = set(CONSTRAINT_OPERATORS_V6)
    assert actual == {"IK", "AIM", "FOLLOW", "COPY_TRANSFORM", "STRETCH"}
    assert research - actual == {
        "LIMIT_ROTATION", "LIMIT_TRANSLATION", "PATH",
    }
    assert actual.issubset(research)
    for operator in research - actual:
        with pytest.raises(QualificationError, match="RIG_CONSTRAINT_OPERATOR_UNSUPPORTED"):
            validate_rig_constraint_operator_v1(
                operator_id=operator, input_control_count=1,
                output_joint_count=1, has_chain=False, parameters={},
            )


def test_model_mask_prevents_unexecutable_product_constraints():
    logits = torch.zeros(len(CONSTRAINT_OPERATORS_V6))
    masked = AXISRigProgramModelV6.capability_masked_constraint_logits(
        logits, allowed_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
    )
    for index, operator in enumerate(CONSTRAINT_OPERATORS_V6):
        assert bool(torch.isfinite(masked[index])) == (
            operator in RIG_CONSTRAINT_OPERATOR_IDS_V1
        )
    with pytest.raises(ValueError, match="AXIS_V6_EXECUTABLE_OPERATOR_UNKNOWN"):
        AXISRigProgramModelV6.capability_masked_constraint_logits(
            logits, allowed_constraint_operators=("PATH", "NOT_IMPLEMENTED"),
        )
    with pytest.raises(ValueError, match="AXIS_V6_EXECUTABLE_OPERATOR_SET_INVALID"):
        AXISRigProgramModelV6.capability_masked_constraint_logits(
            logits, allowed_constraint_operators=("IK", "IK"),
        )
