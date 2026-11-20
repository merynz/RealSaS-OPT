from __future__ import annotations

import pytest
import torch

from models.rss_vnext.mira_embedded_codec_source_training_v1 import (
    RSSvNextEmbeddedMIRASourceCodecTrainerV1,
)
from tests.models.test_rss_vnext_mira_embedded_codec_source_training_v1 import (
    _fixture, DATASET,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(1043)
    torch.set_num_threads(1)


def make_trainer(model,registry,weight):
    return RSSvNextEmbeddedMIRASourceCodecTrainerV1(
        model,torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),lr=1e-3,
        ),
        dataset_manifest_sha256=DATASET,
        trusted_source_skin_registry=registry,
        source_sparse_influence_weight=weight,
    )


def test_source_sparse_loss_is_true_source_supervised_and_differentiable():
    model,batch,registry=_fixture()
    trainer=make_trainer(model,registry,1.)
    result=trainer.objective(batch)
    assert result.source_sparse_balanced_influence_loss is not None
    assert torch.isfinite(result.source_sparse_balanced_influence_loss)
    expected=result.reconstruction_loss+0.05*result.quantization_loss+result.source_sparse_balanced_influence_loss
    assert torch.allclose(result.loss,expected,atol=1e-6,rtol=1e-6)
    result.source_sparse_balanced_influence_loss.backward()
    param=model.mira.codec.bias_head.weight
    assert param.grad is not None and torch.isfinite(param.grad).all()
    assert param.grad.abs().sum()>0
    assert model.axis.kind_head.weight.grad is None
    assert model.iris.scene.scene_queries.grad is None


def test_optional_sparse_loss_disabled_preserves_previous_default():
    model,batch,registry=_fixture()
    trainer=make_trainer(model,registry,0.)
    result=trainer.objective(batch)
    assert result.source_sparse_balanced_influence_loss is None
    assert torch.allclose(
        result.loss,result.reconstruction_loss+0.05*result.quantization_loss,
        atol=1e-6,rtol=1e-6,
    )


def test_sparse_loss_policy_is_checkpoint_and_phase_handoff_identity(tmp_path):
    model,batch,registry=_fixture()
    trainer=make_trainer(model,registry,.5)
    trainer.train_step(batch)
    checkpoint=tmp_path/"codec-sparse.pt"
    trainer.save_checkpoint(checkpoint)
    different=make_trainer(model,registry,.6)
    with pytest.raises(ValueError,match="SOURCE_CODEC_RESUME_DRIFT"):
        different.load_checkpoint(checkpoint)
    assert trainer.phase_objective_policy_sha256!=different.phase_objective_policy_sha256
    assert make_trainer(model,registry,.5).load_checkpoint(checkpoint) is None


@pytest.mark.parametrize("invalid",[-.01,float("nan"),float("inf"),True])
def test_sparse_loss_invalid_objective_weight_rejected(invalid):
    model,batch,registry=_fixture()
    with pytest.raises(ValueError,match="SOURCE_SPARSE_POLICY"):
        make_trainer(model,registry,invalid)
