from __future__ import annotations

import numpy as np
import torch

from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import ArachneCandidateConfigV2, ArachneCandidateV2
from experiments.geppetto_arachne_r6_20260901.eval_arachne_r6_a1_v1 import eval_arachne_r6_a1_v1
from experiments.geppetto_arachne_r6_20260901.eval_codec_r6_a0_v1 import eval_codec_r6_a0_v1
from experiments.geppetto_arachne_r6_20260901.train_arachne_r6_a1_v1 import train_arachne_r6_a1_step_v1
from experiments.geppetto_arachne_r6_20260901.test_arachne_codec_v2_source import _conditioning, _codec, _teacher, _probe_transforms


def _fixture():
    surface, skeleton, conditioning = _conditioning("TAIL")
    codec = _codec()
    model = ArachneCandidateV2(codec, ArachneCandidateConfigV2(model_dim=32, surface_encoder_layers=1, attention_heads=4, feedforward_dim=64))
    sf = torch.tensor(conditioning.surface_features)
    jf = torch.tensor(conditioning.joint_features)
    sm = torch.tensor(conditioning.surface_mask)
    jm = torch.tensor(conditioning.joint_mask)
    teacher = _teacher(6, 3)
    rest = torch.tensor([n.P for n in surface.surface_nodes], dtype=torch.float32)[None]
    transforms = _probe_transforms(3)
    _, token = eval_codec_r6_a0_v1(
        codec, sf, jf, teacher, sm, jm, rest, transforms,
        source_gate="SYNTHETIC_GENERIC_STRENGTH", optimizer_steps=0,
        max_reconstruction=100.0, max_deformation_rms=100.0, max_simplex_residual=1e-5,
    )
    return conditioning, model, teacher, rest, transforms, token


def test_base_a1_training_contract_reports_hard_tail_not_only_mean():
    conditioning, model, teacher, rest, transforms, token = _fixture()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    metrics = train_arachne_r6_a1_step_v1(
        model, opt, conditioning, teacher, rest, transforms,
        a0_token=token,
        expected_surface_hashes=conditioning.source_surface_hashes,
        expected_skeleton_hashes=conditioning.source_skeleton_hashes,
        tail_weight=1.0,
        tail_fraction=0.10,
    )
    assert "tail_row_l1" in metrics
    assert np.isfinite(metrics["tail_row_l1"])
    assert metrics["tail_row_l1"] >= 0.0


def test_a1_evaluator_exposes_mean_p95_tail_and_deformation_consequence():
    conditioning, model, teacher, rest, transforms, token = _fixture()
    metrics = eval_arachne_r6_a1_v1(
        model, conditioning, teacher, rest, transforms,
        a0_token=token,
        expected_surface_hashes=conditioning.source_surface_hashes,
        expected_skeleton_hashes=conditioning.source_skeleton_hashes,
    )
    for key in ("row_l1_mean", "row_l1_p95", "tail_row_l1", "deformation_rms", "latent_nll", "uncertainty_mean"):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics["row_l1_p95"] >= metrics["row_l1_mean"] * 0.0
    assert metrics["tail_row_l1"] >= metrics["row_l1_mean"]
