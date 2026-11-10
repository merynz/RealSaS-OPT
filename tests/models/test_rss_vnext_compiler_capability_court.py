from __future__ import annotations

"""Explicit neural rig-program vs deterministic Compiler executable-capability court.

This test does NOT create solver authority. It prevents a wider learned
vocabulary from silently being advertised as executable or product qualified.
The source architecture may PROPOSE more operators than Compiler currently
executes. The product must abstain or reject until typed capabilities exist.
"""

import pytest

from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6, CONSTRAINT_OPERATORS_V6,
)
import torch
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1,
    validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_learned_control_vocabulary_is_not_the_runtime_support_list():
    neural = set(CONSTRAINT_OPERATORS_V6)
    executable = set(RIG_CONSTRAINT_OPERATOR_IDS_V1)
    # Failing this means the frozen audit must be consciously updated:
    # either a solver landed or the model's declared action space changed.
    assert executable == {
        "IK", "AIM", "FOLLOW", "COPY_TRANSFORM", "STRETCH",
    }
    assert neural - executable == {
        "LIMIT_ROTATION", "LIMIT_TRANSLATION", "PATH",
    }
    assert not executable - neural


@pytest.mark.parametrize(
    "operator_id,inputs,outputs,chain",
    [
        ("IK", 2, 1, True),
        ("AIM", 1, 1, False),
        ("FOLLOW", 1, 1, False),
        ("COPY_TRANSFORM", 1, 1, False),
        ("STRETCH", 1, 2, True),
    ],
)
def test_executable_typed_constraint_has_deterministic_parameter_contract(
    operator_id, inputs, outputs, chain,
):
    validate_rig_constraint_operator_v1(
        operator_id=operator_id,
        input_control_count=inputs,
        output_joint_count=outputs,
        has_chain=chain,
        parameters={},
    )


@pytest.mark.parametrize(
    "unsupported", ["LIMIT_ROTATION", "LIMIT_TRANSLATION", "PATH"],
)
def test_neural_only_operator_must_not_be_claimed_executable(unsupported):
    assert unsupported in CONSTRAINT_OPERATORS_V6
    with pytest.raises(QualificationError, match="RIG_CONSTRAINT_OPERATOR_UNSUPPORTED"):
        validate_rig_constraint_operator_v1(
            operator_id=unsupported, input_control_count=1,
            output_joint_count=1, has_chain=True, parameters={},
        )


def test_even_supported_operator_cannot_run_with_missing_evidence_arity():
    with pytest.raises(QualificationError, match="RIG_CONSTRAINT_CHAIN_REQUIRED"):
        validate_rig_constraint_operator_v1(
            operator_id="IK", input_control_count=1,
            output_joint_count=1, has_chain=False, parameters={},
        )
    with pytest.raises(QualificationError, match="RIG_CONSTRAINT_CONTROL_ARITY"):
        validate_rig_constraint_operator_v1(
            operator_id="AIM", input_control_count=0,
            output_joint_count=1, has_chain=False, parameters={},
        )



def test_runtime_capability_mask_blocks_unsupported_neural_top_choice():
    logits = torch.full((len(CONSTRAINT_OPERATORS_V6),), -7.)
    path_idx = CONSTRAINT_OPERATORS_V6.index("PATH")
    ik_idx = CONSTRAINT_OPERATORS_V6.index("IK")
    logits[path_idx] = 100.
    logits[ik_idx] = 3.
    filtered = AXISRigProgramModelV6.capability_masked_constraint_logits(
        logits, allowed_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
    )
    assert int(filtered.argmax()) == ik_idx
    assert torch.isneginf(filtered[path_idx])
    assert filtered[ik_idx] == logits[ik_idx]
    with pytest.raises(ValueError, match="EXECUTABLE_OPERATOR_SET_INVALID"):
        AXISRigProgramModelV6.capability_masked_constraint_logits(
            logits, allowed_constraint_operators=(),
        )
    with pytest.raises(ValueError, match="EXECUTABLE_OPERATOR_UNKNOWN"):
        AXISRigProgramModelV6.capability_masked_constraint_logits(
            logits, allowed_constraint_operators=("FABRICATED_SOLVER",),
        )


def test_control_generation_requires_explicit_compiler_capabilities():
    from models.rss_vnext import (
        RSSvNextConfig, RSSvNextPatchEncoderForTests,
        RSSvNextUnifiedCharacterModel,
    )
    config = RSSvNextConfig.tiny_for_test()
    model = RSSvNextUnifiedCharacterModel(
        config,
        vision_encoder=RSSvNextPatchEncoderForTests(config.iris.image_token_dim),
    ).eval()
    images = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1
    # This guard runs BEFORE structural topology generation and ensures that
    # a model-only caller cannot claim runtime control support.
    with pytest.raises(ValueError, match="RIG_REQUIRES_COMPILER_CAPABILITIES"):
        model.propose_teacher_free(images, cam, generate_controls=True)
    with pytest.raises(ValueError, match="UNUSED_COMPILER_CAPABILITIES"):
        model.propose_teacher_free(
            images, cam, generate_controls=False,
            executable_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
        )
