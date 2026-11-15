from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from models.rss_vnext.training_axis_skin_warmup_v1 import (
    RSSvNextAxisSkinCodeWarmupV1,
    apply_rss_vnext_axis_skin_code_warmup_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from tests.models.test_rss_vnext_structural_trainer import make_model,manifest_hash


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(973)
    torch.set_num_threads(1)


def _objective(policy,step):
    rig=torch.tensor(2.0,requires_grad=True)
    skin=torch.tensor(3.0,requires_grad=True)
    value=apply_rss_vnext_axis_skin_code_warmup_v1(
        raw_axis_loss=rig.square()+skin.square(),
        raw_axis_loss_parts={"token_kind":rig.square(),"skin_code_logits":skin.square()},
        policy=policy,step=step,
    )
    value.backward()
    return float(value),float(rig.grad),float(skin.grad)


def test_cosine_skin_warmup_starts_frozen_and_ends_at_unmodified_loss():
    policy=RSSvNextAxisSkinCodeWarmupV1(start_step=2,full_step=10)
    zero=_objective(policy,0)
    at_start=_objective(policy,2)
    middle=_objective(policy,6)
    full=_objective(policy,10)
    assert zero==(4.,4.,0.)
    assert at_start==zero
    assert middle==(8.5,4.,3.)
    assert full==(13.,4.,6.)
    assert _objective(policy,20)==full


def test_warmup_disabled_is_exact_backward_compatible():
    policy=RSSvNextAxisSkinCodeWarmupV1()
    assert not policy.enabled
    assert _objective(policy,0)==(13.,4.,6.)
    rig=torch.tensor(3.,requires_grad=True)
    total=rig.square()
    output=apply_rss_vnext_axis_skin_code_warmup_v1(
        raw_axis_loss=total,
        raw_axis_loss_parts={"token_kind":total},
        policy=RSSvNextAxisSkinCodeWarmupV1(start_step=1,full_step=10),
        step=3,
    )
    assert output is total


@pytest.mark.parametrize("start,end",[(2,2),(3,1),(-1,5),(1,0),(0,-5)])
def test_invalid_warmup_policy_fails(start,end):
    with pytest.raises(ValueError,match="WARMUP_POLICY_INVALID"):
        RSSvNextAxisSkinCodeWarmupV1(start_step=start,full_step=end).validate()


def test_warmup_changes_checkpoint_identity_even_when_architecture_is_identical(tmp_path):
    policy=RSSvNextAxisSkinCodeWarmupV1(start_step=0,full_step=10)
    weights=RSSvNextTrainingWeights(axis_rig_skin_codes=1.)
    net=make_model()
    first=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),
        dataset_manifest_sha256=manifest_hash(),
        weights=weights,axis_skin_code_warmup=policy,
    )
    path=tmp_path/"warmup.pt"
    first.save_checkpoint(path,sampler_cursor=0)
    net2=make_model()
    same=RSSvNextStructuralTrainer(
        net2,torch.optim.AdamW(net2.parameters()),
        dataset_manifest_sha256=manifest_hash(),
        weights=weights,axis_skin_code_warmup=policy,
    )
    assert same.load_checkpoint(path)==0
    net3=make_model()
    changed=RSSvNextStructuralTrainer(
        net3,torch.optim.AdamW(net3.parameters()),
        dataset_manifest_sha256=manifest_hash(),
        weights=weights,
        axis_skin_code_warmup=RSSvNextAxisSkinCodeWarmupV1(start_step=0,full_step=8),
    )
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        changed.load_checkpoint(path)


def test_warmup_is_rejected_without_axis_teacher_objective():
    net=make_model()
    with pytest.raises(ValueError,match="AXIS_LOSS_REQUIRED"):
        RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters()),
            dataset_manifest_sha256=manifest_hash(),
            axis_skin_code_warmup=RSSvNextAxisSkinCodeWarmupV1(full_step=5),
        )
