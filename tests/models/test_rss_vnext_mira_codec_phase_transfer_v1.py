from __future__ import annotations

from hashlib import sha256

import pytest
import torch

from models.rss_vnext.training_engine import RSSvNextStructuralTrainer
from models.rss_vnext.training_phase_handoff_v1 import (
    import_rss_vnext_training_phase_handoff_v1,
)
from tests.models.test_rss_vnext_mira_embedded_codec_source_training_v1 import (
    _fixture, _trainer, DATASET,
)
from tests.models.test_rss_vnext_structural_trainer import make_model


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(992)
    torch.set_num_threads(1)


def _destination():
    model=make_model()
    # In a source-trained AXIS phase the pretrained MIRA codec is a teacher,
    # never an additional trainable head.
    for parameter in model.mira.codec.parameters():
        parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=3.e-4,
    )
    trainer=RSSvNextStructuralTrainer(
        model,optimizer,dataset_manifest_sha256=DATASET,
    )
    return model,trainer


def test_embedded_codec_to_structural_phase_preserves_full_model(tmp_path):
    model,batch,source_registry=_fixture()
    source=_trainer(model,source_registry)
    source.train_step(batch)
    handoff=tmp_path/"mira-codec-to-structural.pt"
    receipt=source.export_phase_handoff(handoff)
    assert receipt.source_kind=="MIRA_CODEC"
    assert receipt.target_kind=="STRUCTURAL"
    assert receipt.source_global_step==1
    assert len(receipt.model_state_sha256)==64

    downstream,trainer=_destination()
    accepted=trainer.import_phase_handoff(
        handoff,
        expected_source_kind="MIRA_CODEC",
        expected_source_objective_policy_sha256=source.phase_objective_policy_sha256,
        expected_source_optimizer_binding_sha256=source.optimizer_ownership.binding_sha256,
    )
    assert accepted==receipt
    assert trainer.global_step==0
    assert trainer.phase_handoff_source_model_sha256==receipt.model_state_sha256
    assert not trainer.optimizer.state
    for name,weight in model.state_dict().items():
        assert torch.equal(weight,downstream.state_dict()[name])
    assert all(p.grad is None for p in downstream.parameters())
    assert all(not p.requires_grad for p in downstream.mira.codec.parameters())


def test_mira_codec_handoff_never_exports_untrained_bootstrap(tmp_path):
    model,batch,registry=_fixture()
    trainer=_trainer(model,registry)
    with pytest.raises(ValueError,match="HANDOFF_SOURCE_POLICY_INVALID"):
        trainer.export_phase_handoff(tmp_path/"untrained.pt")


def test_mira_codec_handoff_rejects_source_policy_drift_and_dirty_optimizer(tmp_path):
    model,batch,registry=_fixture()
    source=_trainer(model,registry)
    source.train_step(batch)
    path=tmp_path/"trained-codec.pt"
    receipt=source.export_phase_handoff(path)

    dest,trainer=_destination()
    with pytest.raises(ValueError,match="HANDOFF_PROVENANCE_DRIFT"):
        trainer.import_phase_handoff(
            path,expected_source_kind="MIRA_CODEC",
            expected_source_objective_policy_sha256=sha256(b"wrong-policy").hexdigest(),
            expected_source_optimizer_binding_sha256=source.optimizer_ownership.binding_sha256,
        )
    assert trainer.global_step==0

    # An optimizer that has already stepped cannot borrow a codec stage.
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            p.grad=torch.ones_like(p)
    trainer.optimizer.step()
    with pytest.raises(ValueError,match="DESTINATION_OPTIMIZER_NOT_FRESH"):
        trainer.import_phase_handoff(
            path,expected_source_kind="MIRA_CODEC",
            expected_source_objective_policy_sha256=source.phase_objective_policy_sha256,
            expected_source_optimizer_binding_sha256=source.optimizer_ownership.binding_sha256,
        )
