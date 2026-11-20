from __future__ import annotations

"""Fail-closed pilot launcher tests. CPU fixtures are NOT GPU/quality proof."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.rss_vnext.product_real_pilot_cohort_v1 import (
    RealPilotCohortReceiptV1,
)
from models.rss_vnext.product_real_pilot_execution_v1 import (
    PilotLedgerV1,SCHEMA as LEDGER_SCHEMA,
)
from models.rss_vnext.product_real_pilot_diffusion_launcher_v1 import (
    STAGE_A,STAGE_B,RealPilotStageABPlanV1,
    build_real_pilot_stage_ab_v1,
    execute_real_pilot_stage_ab_v1,
)
from tests.models.test_rss_vnext_sdxl_mvadapter_stageab_trainer_resume_v2 import (
    _stage_a,_batch_a,_source_registry,VAE,Scheduler,
)


def _verified_cohort():
    # This is only a 300..400 asset *contract fixture*. No asset, render,
    # license or GPU receipt is forged or used to claim product capability.
    return RealPilotCohortReceiptV1(
        schema="RealSaS.RSSvNext.Real400PilotCohort.v1",
        cohort_sha256="a"*64,
        total_assets=320,train_assets=280,val_assets=20,
        unseen_test_assets=20,lineage_groups=320,
        teacher_family_coverage_train={},
        source_evidence_bytes_verified=True,
    )


def _ledger(cohort):
    return PilotLedgerV1(
        schema=LEDGER_SCHEMA,cohort_sha256=cohort.cohort_sha256,
        completed=(),
    )


def _build(cohort,ledger):
    model,conditioner,alpha=_stage_a()
    vae=VAE()
    vae._rss_vnext_vae_ancestor_sha256="d"*64
    batch=_batch_a()
    return build_real_pilot_stage_ab_v1(
        cohort=cohort,ledger=ledger,
        plan=RealPilotStageABPlanV1(
            stage=STAGE_A,optimizer_steps=2,warmup_steps=2,
        ),
        unet=model,vae=vae,noise_scheduler=Scheduler(),
        pretrained_source_sha256="b"*64,
        source_vae_sha256="d"*64,
        trusted_source_registry=_source_registry(STAGE_A,batch),
        conditioner=conditioner,canonical_alpha_head=alpha,
    )


def test_pilot_A_builds_real_fresh_production_optimizer_from_approved_cohort():
    cohort=_verified_cohort()
    phase=_build(cohort,_ledger(cohort))
    assert phase.trainer.step==0
    assert not phase.trainer.optimizer.state
    assert phase.trainer.dataset_manifest_sha256==cohort.cohort_sha256
    assert phase.source_cohort_sha256==cohort.cohort_sha256
    assert phase.pretrained_source_sha256=="b"*64
    assert phase.previous_stage_checkpoint_sha256 is None
    assert set(phase.trainer.groups)=={
        "sdxl_lora","paired_conditioner","canonical_alpha",
    }
    # Not actual trained weights or valid pretrained SDXL: this tests only
    # the outer orchestration; real GPU/weight loading is independently gated.


def test_pilot_never_admits_unverified_or_wrong_sized_source_cohort():
    cohort=_verified_cohort()
    for tampered in (
        replace(cohort,source_evidence_bytes_verified=False),
        replace(cohort,total_assets=299,train_assets=259),
        replace(cohort,unseen_test_assets=0),
        replace(cohort,train_assets=279),
    ):
        with pytest.raises(ValueError,match="COHORT_LEDGER_OR_SOURCE_UNQUALIFIED"):
            _build(tampered,_ledger(tampered))


def test_pilot_B_cannot_advance_without_real_completed_A_checkpoint():
    cohort=_verified_cohort()
    with pytest.raises(ValueError,match="COHORT_LEDGER_OR_SOURCE_UNQUALIFIED"):
        model,conditioner,head=_stage_a()
        vae=VAE()
        vae._rss_vnext_vae_ancestor_sha256="d"*64
        build_real_pilot_stage_ab_v1(
            cohort=cohort,ledger=_ledger(cohort),
            plan=RealPilotStageABPlanV1(stage=STAGE_B,optimizer_steps=2),
            unet=model,vae=vae,noise_scheduler=Scheduler(),
            pretrained_source_sha256="b"*64,
            source_vae_sha256="d"*64,
            trusted_source_registry=_source_registry(STAGE_A,_batch_a()),
            pretrained_mv_adapter_sha256="c"*64,
        )


def test_pilot_A_execution_rejects_cpu_even_for_fresh_optimizer(tmp_path):
    cohort=_verified_cohort()
    phase=_build(cohort,_ledger(cohort))
    with pytest.raises(ValueError,match="REAL_CUDA_OR_PATH_NOT_ADMITTED"):
        execute_real_pilot_stage_ab_v1(
            phase,examples=(),source_index={},cohort=cohort,
            task_checkpoint_file=tmp_path/"untrained.pt",
            gradient_audit_file=tmp_path/"fake-gradient.json",
        )
    assert not (tmp_path/"untrained.pt").exists()
    assert not (tmp_path/"fake-gradient.json").exists()


def test_pilot_policy_does_not_fabricate_gpu_or_val_acceptance():
    plan=RealPilotStageABPlanV1(
        stage=STAGE_A,optimizer_steps=300,learning_rate=5e-5,
        warmup_steps=2000,amp_dtype="BF16",
    )
    plan.validate()
    for broken in (
        replace(plan,optimizer_steps=0),
        replace(plan,learning_rate=float("nan")),
        replace(plan,amp_dtype="F32"),
        replace(plan,stage="UNSEEN_TEACHER_FREE_COMPILE"),
    ):
        with pytest.raises(ValueError,match="EXECUTION_PLAN_INVALID"):
            broken.validate()
