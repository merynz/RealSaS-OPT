from __future__ import annotations

import torch

from compiler.realsas_compiler_core.skin_code_ir_v1 import validate_skin_code_proposal_v1
from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecConfig, MIRASkinFSQCodecV6
from models.mira.training_v6 import encode_axis_teacher_codes_v6, mira_v6_codec_objective


def test_mira_v6_training_objective_is_finite_and_sparse_pressure_is_model_side():
    torch.manual_seed(23)
    cfg = MIRASkinFSQCodecConfig(5, 4, hidden_dim=24, tokens_per_joint=2, levels=(3, 3, 3))
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(2, 10, 5)
    joint = torch.randn(2, 3, 4)
    weights = torch.softmax(torch.randn(2, 10, 3), dim=-1)
    result = mira_v6_codec_objective(model, weights, vertex, joint)
    for key in ("total", "reconstruction", "quantization", "entropy", "mae", "max_abs", "simplex_max_abs"):
        assert torch.isfinite(result[key])
    result["total"].backward()
    assert model.latent_head.weight.grad is not None
    assert float(model.latent_head.weight.grad.abs().sum()) > 0.0


def test_frozen_mira_codec_exports_exact_axis_teacher_code_contract():
    torch.manual_seed(29)
    cfg = MIRASkinFSQCodecConfig(5, 4, hidden_dim=24, tokens_per_joint=2, levels=(3, 3, 3))
    model = MIRASkinFSQCodecV6(cfg)
    vertex = torch.randn(1, 7, 5)
    joint = torch.randn(1, 3, 4)
    weights = torch.softmax(torch.randn(1, 7, 3), dim=-1)
    proposal = encode_axis_teacher_codes_v6(
        model, weights, vertex, joint,
        proposal_joint_ids=("J0", "J1", "J2"),
        rig_program_proposal_hash="1" * 64,
        carrier_evidence_hash="2" * 64,
        carrier_topology_hash="3" * 64,
        carrier_geometry_hash="4" * 64,
        codec_checkpoint_hash="a" * 64,
        model_provenance="mira-v6-test",
    )
    validate_skin_code_proposal_v1(
        proposal,
        expected_proposal_joint_ids=("J0", "J1", "J2"),
        expected_rig_program_proposal_hash="1" * 64,
    )
    assert proposal.codec_checkpoint_hash == "a" * 64
    assert proposal.codes_per_joint == 2
    assert len(proposal.rows) == 3
    assert all(len(row.code_ids) == 2 for row in proposal.rows)
    assert proposal.metadata["compiler_code_search_allowed"] is False
