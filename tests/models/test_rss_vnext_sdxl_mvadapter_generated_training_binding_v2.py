from __future__ import annotations

"""V1/V2 generator binder parity; no native source truth enters image encoder."""

import pytest
import torch

from tests.models.test_rss_vnext_product_sixview_trainer_parity_v1 import (
    _fixture,_trainer,
)
from models.rss_vnext.realsas_sdxl_mvadapter_construction_v2 import (
    SCHEMA,two_stage_sdxl_mv_binding_v2,
)


def _v2():
    net,batch,record=_fixture()
    batch.product_generation_abi="V2_SDXL_MVADAPTER"
    batch.product_pose_program_sha256="d"*64
    batch.product_generation_checkpoint_sha256=two_stage_sdxl_mv_binding_v2(
        stage_a_checkpoint_sha256=batch.product_canonical_checkpoint_sha256,
        stage_b_checkpoint_sha256=batch.product_multiview_checkpoint_sha256,
        source_binding_sha256=batch.product_source_binding_sha256,
        pose_program_sha256=batch.product_pose_program_sha256,
    )
    record={
        **record,
        "generator_abi":SCHEMA,
        "canonical_pose_program_sha256":batch.product_pose_program_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
    }
    return net,batch,record


def test_trainer_accepts_generated_sdxl_mv_adapter_v2_without_source_teacher():
    net,batch,record=_v2()
    trainer=_trainer(net,{batch.product_conditioning_admission_key:record})
    outcome=trainer.train_step(batch)
    assert torch.isfinite(outcome.total_loss)
    assert trainer.global_step==1


def test_v2_pose_program_is_required_and_registry_cannot_fake_authority():
    net,batch,record=_v2()
    trainer=_trainer(net,{batch.product_conditioning_admission_key:record})
    batch.product_pose_program_sha256=""
    with pytest.raises(ValueError,match="V2_POSE_PIN_REQUIRED"):
        trainer.objective(batch)
    batch.product_pose_program_sha256="d"*64
    batch.product_generation_checkpoint_sha256="a"*64
    malicious={**record,"combined_generation_checkpoint_sha256":"a"*64}
    evil=_trainer(net,{batch.product_conditioning_admission_key:malicious})
    with pytest.raises(ValueError,match="GENERATOR_BINDING_RECOMPUTE_DRIFT"):
        evil.objective(batch)


def test_versioned_v1_v2_checkpoint_bindings_do_not_alias():
    net,batch,record=_v2()
    trainer=_trainer(net,{batch.product_conditioning_admission_key:record})
    batch.product_generation_abi="V1_LATENT_ROW"
    with pytest.raises(ValueError,match="V1_POSE_PIN_UNEXPECTED"):
        trainer.objective(batch)
