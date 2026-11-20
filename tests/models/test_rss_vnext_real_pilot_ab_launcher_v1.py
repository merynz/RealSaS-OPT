from __future__ import annotations

"""Synthetic ABI courts only; no release-level SDXL or MV weights used."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from models.rss_vnext.product_real_pilot_ab_launcher_v1 import (
    PilotABPhasePlanV1,
    build_real_pilot_ab_phase_v1,
    save_real_pilot_ab_resume_v1,
    load_real_pilot_ab_resume_v1,
    execute_real_pilot_ab_phase_v1,
    export_real_pilot_ab_trained_task_weights_v1,
)
from models.rss_vnext.product_real_pilot_cohort_v1 import (
    SCHEMA as COHORT_SCHEMA, RealPilotCohortReceiptV1,
)
from models.rss_vnext.product_real_pilot_execution_v1 import (
    SCHEMA as LEDGER_SCHEMA, PilotCompletedStageV1, PilotLedgerV1,
)
from models.rss_vnext.realsas_sdxl_mvadapter_stageab_trainer_v2 import (
    SourceSDXLStageABOptimizerV2,
)

COHORT="a"*64
BASE="b"*64
VAE="c"*64
ADAPTER="d"*64
REGISTRY="e"*64
OWNER="f"*64


def cohort():
    return RealPilotCohortReceiptV1(
        schema=COHORT_SCHEMA,cohort_sha256=COHORT,
        total_assets=300,train_assets=260,val_assets=20,
        unseen_test_assets=20,lineage_groups=300,
        teacher_family_coverage_train={},
        source_evidence_bytes_verified=True,
    )


def empty_ledger():
    return PilotLedgerV1(schema=LEDGER_SCHEMA,cohort_sha256=COHORT,completed=())


def synthetic_a_ledger(tmp_path):
    source=tmp_path/"trained-a.pt"
    source.write_bytes(b"synthetic task A file - never release")
    digest=sha256(source.read_bytes()).hexdigest()
    validation=tmp_path/"a-val.json"
    validation.write_text(json.dumps({
        "stage":"A_CANONICAL_SDXL",
        "cohort_sha256":COHORT,
        "evaluation_split":"VAL",
        "status":"PASS",
        "val_examples":20,
        "gradient_audit_sha256":"4"*64,
    }))
    gpu=tmp_path/"a-gpu.json"
    gpu.write_text(json.dumps({
        "device_type":"cuda",
        "device_name":"synthetic-not-a-gpu",
        "amp_dtype":"BF16",
        "peak_allocated_mib":1024.,
        "status":"PASS",
    }))
    r=PilotCompletedStageV1(
        stage="A_CANONICAL_SDXL",cohort_sha256=COHORT,
        input_weights_sha256=BASE,
        previous_stage_weights_sha256=None,
        output_checkpoint_path=str(source),
        output_checkpoint_sha256=digest,
        objective_policy_sha256="1"*64,
        optimizer_ownership_sha256=OWNER,
        validation_report_path=str(validation),
        validation_report_sha256=sha256(validation.read_bytes()).hexdigest(),
        gpu_evidence_path=str(gpu),
        gpu_evidence_sha256=sha256(gpu.read_bytes()).hexdigest(),
        audited_gradient_report_sha256="4"*64,
        trained_steps=1,trained_examples=260,val_examples=20,
        gpu_device_name="synthetic-not-a-gpu",
        amp_dtype="BF16",
    )
    ledger=PilotLedgerV1(
        schema=LEDGER_SCHEMA,cohort_sha256=COHORT,completed=(r,),
    )
    ledger.validate()
    return ledger,source


def fake_ab(stage,*,step=0,lr=5e-5,warmup=2,amp="BF16"):
    # object.__new__ deliberately bypasses pretrained constructor:
    # tests below check only the pilot gate, never native forward/backward.
    t=object.__new__(SourceSDXLStageABOptimizerV2)
    t.stage=stage
    t.dataset_manifest_sha256=COHORT
    t.pretrained_source_sha256=BASE
    t.source_vae_sha256=VAE
    t.pretrained_mv_adapter_sha256=(
        ADAPTER if stage.startswith("B_") else None
    )
    t.training_noise_schedule_sha256=(
        "5"*64 if stage.startswith("B_") else None
    )
    t.guidance_policy_sha256=(
        "6"*64 if stage.startswith("B_") else None
    )
    t.source_registry_sha256=REGISTRY
    t.optimizer_binding=OWNER
    t.base_learning_rate=lr
    t.warmup_steps=warmup
    t.amp_dtype=amp
    t.step=step
    t.optimizer=SimpleNamespace(state={})
    def save(path):
        Path(path).write_text(json.dumps({"step":t.step}))
    def load(path):
        t.step=json.loads(Path(path).read_text())["step"]
    def export(path):
        return {"saved":str(path),"step":t.step}
    t.save_resume_checkpoint=save
    t.load_resume_checkpoint=load
    t.export_trained_task_weights=export
    return t


def test_ab_plan_validation_and_a_cannot_skip_or_invent_prior():
    plan=PilotABPhasePlanV1("A_CANONICAL_SDXL",5,5e-5,warmup_steps=2)
    phase=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    assert phase.plan.stage=="A_CANONICAL_SDXL"
    for altered in (
        replace(plan,stage="UNKNOWN"),
        replace(plan,optimizer_steps=0),
        replace(plan,warmup_steps=-1),
        replace(plan,amp_dtype="UNVERIFIED"),
    ):
        with pytest.raises(ValueError,match="PLAN_INVALID"):
            altered.validate()
    with pytest.raises(ValueError,match="TRAINER_NOT_ADMITTED"):
        build_real_pilot_ab_phase_v1(
            trainer=fake_ab(plan.stage,step=2),plan=plan,
            cohort=cohort(),ledger=empty_ledger(),
        )


def test_stage_b_requires_real_a_task_file_bytes(tmp_path):
    b_plan=PilotABPhasePlanV1(
        "B_PAIRED_RGB_NORMAL_MVADAPTER",5,5e-5,warmup_steps=2,
    )
    with pytest.raises(ValueError,match="TRAINER_NOT_ADMITTED"):
        build_real_pilot_ab_phase_v1(
            trainer=fake_ab(b_plan.stage),plan=b_plan,
            cohort=cohort(),ledger=empty_ledger(),
        )
    ledger,a=synthetic_a_ledger(tmp_path)
    stage_b=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(b_plan.stage),plan=b_plan,
        cohort=cohort(),ledger=ledger,prior_a_task_weights_path=a,
    )
    assert stage_b.previous_stage_task_sha256==ledger.completed[0].output_checkpoint_sha256
    a.write_bytes(b"changed A bytes")
    with pytest.raises(RuntimeError,match="ARTIFACT_BYTES_DRIFT"):
        build_real_pilot_ab_phase_v1(
            trainer=fake_ab(b_plan.stage),plan=b_plan,
            cohort=cohort(),ledger=ledger,prior_a_task_weights_path=a,
        )


def test_ab_restart_refuses_plan_pretrained_or_checkpoint_drift(tmp_path):
    plan=PilotABPhasePlanV1("A_CANONICAL_SDXL",5,5e-5,warmup_steps=2)
    source=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage,step=0),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    source.trainer.step=3
    ckpt=tmp_path/"a-resume.pt"
    save_real_pilot_ab_resume_v1(
        source,checkpoint_path=ckpt,sampler_cursor=13,
    )
    fresh=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    restored=load_real_pilot_ab_resume_v1(
        fresh,checkpoint_path=ckpt,
    )
    assert (restored.trainer.step,restored.verified_resume_step,
            restored.verified_resume_cursor)==(3,3,13)
    independent_fresh=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    with pytest.raises(ValueError,match="PROVENANCE_DRIFT"):
        load_real_pilot_ab_resume_v1(
            replace(independent_fresh,plan=replace(plan,warmup_steps=3)),
            checkpoint_path=ckpt,
        )
    altered=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    altered.trainer.pretrained_source_sha256="9"*64
    with pytest.raises(ValueError,match="PROVENANCE_DRIFT"):
        load_real_pilot_ab_resume_v1(
            altered,checkpoint_path=ckpt,
        )
    ckpt.write_text("corrupted A optimizer")
    with pytest.raises(ValueError,match="PROVENANCE_DRIFT"):
        load_real_pilot_ab_resume_v1(
            build_real_pilot_ab_phase_v1(
                trainer=fake_ab(plan.stage),plan=plan,
                cohort=cohort(),ledger=empty_ledger(),
            ),checkpoint_path=ckpt,
        )


def test_ab_execute_exact_resume_cursor_and_release_only_at_budget(monkeypatch,tmp_path):
    import models.rss_vnext.product_real_pilot_ab_launcher_v1 as module
    plan=PilotABPhasePlanV1("A_CANONICAL_SDXL",5,5e-5,warmup_steps=2)
    phase=build_real_pilot_ab_phase_v1(
        trainer=fake_ab(plan.stage),plan=plan,
        cohort=cohort(),ledger=empty_ledger(),
    )
    phase.trainer.step=3
    ckpt=tmp_path/"a.pt"
    save_real_pilot_ab_resume_v1(
        phase,checkpoint_path=ckpt,sampler_cursor=42,
    )
    recovered=load_real_pilot_ab_resume_v1(
        build_real_pilot_ab_phase_v1(
            trainer=fake_ab(plan.stage),plan=plan,
            cohort=cohort(),ledger=empty_ledger(),
        ),checkpoint_path=ckpt,
    )
    with pytest.raises(RuntimeError,match="UNFINISHED_WEIGHTS_EXPORT"):
        export_real_pilot_ab_trained_task_weights_v1(
            recovered,output_path=tmp_path/"trained-a.pt",
        )
    with pytest.raises(ValueError,match="RESUME_REQUIRES_CHECKPOINT"):
        execute_real_pilot_ab_phase_v1(
            recovered,examples=(),source_index={},cohort=cohort(),
            resume_sampler_cursor=41,
        )
    used=[]
    def runner(*,trainer,staged_examples,source_index,cohort,max_steps):
        used.append(max_steps)
        trainer.step+=max_steps
        return [{"step":trainer.step}]
    monkeypatch.setattr(module,"run_source_qualified_pilot_updates_v1",runner)
    report=execute_real_pilot_ab_phase_v1(
        recovered,examples=(),source_index={},cohort=cohort(),
        resume_sampler_cursor=42,
    )
    assert used==[2] and report[0]["step"]==5
    assert export_real_pilot_ab_trained_task_weights_v1(
        recovered,output_path=tmp_path/"trained-a.pt",
    )["step"]==5

def test_stage_b_rechecks_prior_a_validation_and_gpu_bytes(tmp_path):
    b_plan=PilotABPhasePlanV1(
        "B_PAIRED_RGB_NORMAL_MVADAPTER",5,5e-5,warmup_steps=2,
    )
    ledger,a=synthetic_a_ledger(tmp_path)
    validation=Path(ledger.completed[0].validation_report_path)
    gpu=Path(ledger.completed[0].gpu_evidence_path)
    for target in (validation,gpu):
        original=target.read_bytes()
        target.write_bytes(original+b"\n")
        with pytest.raises(RuntimeError,match="ARTIFACT_BYTES_DRIFT"):
            build_real_pilot_ab_phase_v1(
                trainer=fake_ab(b_plan.stage),plan=b_plan,
                cohort=cohort(),ledger=ledger,
                prior_a_task_weights_path=a,
            )
        target.write_bytes(original)
