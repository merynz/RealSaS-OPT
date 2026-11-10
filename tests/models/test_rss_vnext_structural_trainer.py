from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch, RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(5)
    torch.set_num_threads(1)


def make_model():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def make_sample(net):
    xy = np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    seq = encode_adjacent_teacher_mesh_v1(
        xy, faces, component_id="toy", chart_id="toy",
        cfg=TESSAConfigV1(coordinate_bins=net.cfg.tessa.coordinate_bins),
    ).token_ids
    tok = torch.tensor(seq, dtype=torch.long)[None]
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1
    return RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=cam,
        topology_teacher_tokens=tok, topology_valid_mask=torch.tensor([True]),
        source_qualification_key="unit-test-provenance-is-synthetic-not-real",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )


def manifest_hash():
    return hashlib.sha256(b"synthetic-unit-test-only").hexdigest()


def test_one_real_two_view_structural_train_step():
    net = make_model()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    trainer = RSSvNextStructuralTrainer(net, optimizer, dataset_manifest_sha256=manifest_hash())
    result = trainer.train_step(make_sample(net))
    assert trainer.global_step == 1
    assert result.eligible_topology_examples == 1
    assert result.topology_loss is not None
    assert result.gradient_norm is not None and torch.isfinite(result.gradient_norm)
    assert net.articulation_prior[-1].weight.grad is not None


def test_training_fail_closed_without_admission_or_mask():
    net = make_model()
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters()), dataset_manifest_sha256=manifest_hash(),
    )
    batch = make_sample(net)
    batch.admission_status = "CANDIDATE_NOT_QUALIFIED"
    with pytest.raises(ValueError, match="ADMISSION_NOT_QUALIFIED"):
        trainer.objective(batch)
    batch.admission_status = "QUALIFIED_FOR_THIS_SUPERVISION_FAMILY"
    batch.topology_valid_mask = None
    with pytest.raises(ValueError, match="MASK_REQUIRED"):
        trainer.objective(batch)


def test_strict_checkpoint_resume_and_dataset_binding(tmp_path):
    net = make_model()
    train = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256=manifest_hash(),
    )
    train.train_step(make_sample(net))
    checkpoint = tmp_path / "rss-checkpoint.pt"
    train.save_checkpoint(checkpoint, sampler_cursor=7)
    assert checkpoint.exists()

    loaded = make_model()
    loaded_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    resume = RSSvNextStructuralTrainer(
        loaded, loaded_optimizer, dataset_manifest_sha256=manifest_hash(),
    )
    cursor = resume.load_checkpoint(checkpoint)
    assert cursor == 7 and resume.global_step == 1
    for p, q in zip(net.parameters(), loaded.parameters()):
        assert torch.equal(p.detach(), q.detach())

    wrong = RSSvNextStructuralTrainer(
        make_model(), torch.optim.AdamW(make_model().parameters()),
        dataset_manifest_sha256=hashlib.sha256(b"different-corpus").hexdigest(),
    )
    with pytest.raises(ValueError, match="DATASET_MANIFEST_DRIFT"):
        wrong.load_checkpoint(checkpoint)
