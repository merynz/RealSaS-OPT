from __future__ import annotations

"""Real 300–400-source A/B diffusion pilot launcher (never data foundry).

Use the exact pinned SDXL Stage A or pinned native MV-Adapter Stage B
modules already loaded and SHA-verified by the upstream snapshot court.
This launcher makes the A->B dependency executable, refuses user-split
leakage and performs real optimizer updates using only independently
registered SOURCE tensors. It never promotes a checkpoint to ModelRelease
or marks a pilot ledger stage PASS without a separate VAL/GPU court.

Stage B starts from its own learned pretrained MV-Adapter, NOT the Stage A
UNet; the ledger carries trained A checkpoint as a strict dependency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any,Iterable,Mapping
import math
import os
import json
import tempfile

import torch
from torch import Tensor,nn

from .product_real_pilot_cohort_v1 import (
    RealPilotAssetV1,RealPilotCohortReceiptV1,
)
from .product_real_pilot_execution_v1 import (
    PilotLedgerV1,RealPilotTrainingEnvelopeV1,
    run_source_qualified_pilot_updates_v1,_file_digest,
)
from .realsas_sdxl_mvadapter_stageab_trainer_v2 import (
    SourceSDXLStageABOptimizerV2,
)
from .realsas_sdxl_stage_a_source_training_v2 import (
    SDXLCanonicalAlphaHeadV2,
)
from .realsas_mvadapter_dual_reference_v2 import (
    MVAdapterRGBNormalTaskHeadsV2,
)
from .paired_sdxl_lora_prior_v1 import (
    RSSvNextTwoImageConditionerV1,
)

STAGE_A="A_CANONICAL_SDXL"
STAGE_B="B_PAIRED_RGB_NORMAL_MVADAPTER"
STAGES=(STAGE_A,STAGE_B)
SCHEMA="RealSaS.RSSvNext.RealPilotStageABGPUExecution.v1"


def _is_sha64(value:object)->bool:
    return type(value) is str and len(value)==64 and all(
        c in "0123456789abcdef" for c in value
    )


@dataclass(frozen=True)
class RealPilotStageABPlanV1:
    stage:str
    optimizer_steps:int
    learning_rate:float=5e-5
    warmup_steps:int=2000
    amp_dtype:str="BF16"

    def validate(self)->None:
        if (
            self.stage not in STAGES
            or type(self.optimizer_steps) is not int
            or not 1<=self.optimizer_steps<=10_000_000
            or type(self.learning_rate) not in (int,float)
            or not math.isfinite(self.learning_rate)
            or not 1e-8<=self.learning_rate<=1e-2
            or type(self.warmup_steps) is not int
            or not 0<=self.warmup_steps<=100_000
            or self.amp_dtype not in ("BF16","FP16")
        ):
            raise ValueError("RSS_VNEXT_PILOT_STAGE_AB_EXECUTION_PLAN_INVALID")


@dataclass(frozen=True)
class PreparedRealPilotStageABV1:
    plan:RealPilotStageABPlanV1
    trainer:SourceSDXLStageABOptimizerV2
    source_cohort_sha256:str
    pretrained_source_sha256:str
    pretrained_mv_adapter_sha256:str|None
    previous_stage_checkpoint_sha256:str|None
    optimizer_binding_sha256:str


@dataclass(frozen=True)
class ExecutedRealPilotStageABCandidateV1:
    """Candidate training evidence, NEVER a verified pilot ledger receipt."""
    stage:str
    source_cohort_sha256:str
    steps:int
    total_losses:tuple[float,...]
    trainable_gradient_groups:tuple[str,...]
    gradient_audit_file:str
    gradient_audit_sha256:str
    task_checkpoint_file:str
    task_checkpoint_sha256:str
    device_name:str
    measured_peak_allocated_mib:float
    schema:str=SCHEMA
    status:str="TRAINED_CANDIDATE_REQUIRES_SEPARATE_VAL_AND_GPU_ADMISSION"


def build_real_pilot_stage_ab_v1(
    *,cohort:RealPilotCohortReceiptV1,
    ledger:PilotLedgerV1,
    plan:RealPilotStageABPlanV1,
    unet:nn.Module,vae:nn.Module,
    noise_scheduler:Any,
    pretrained_source_sha256:str,
    source_vae_sha256:str,
    trusted_source_registry:Mapping[str,Mapping[str,Any]],
    pretrained_mv_adapter_sha256:str|None=None,
    conditioner:RSSvNextTwoImageConditionerV1|None=None,
    canonical_alpha_head:SDXLCanonicalAlphaHeadV2|None=None,
    mv_pipeline:Any=None,
    mv_task_heads:MVAdapterRGBNormalTaskHeadsV2|None=None,
)->PreparedRealPilotStageABV1:
    """Construct actual upstream-weighted optimizer; NEVER select sources.

    A and B have independent weight ancestors. The caller must have loaded
    the pretrained SDXL/MV snapshots via the exact SHA-verified loaders;
    the optimizer itself checks their in-memory verified provenance and VAE.
    """
    plan.validate()
    ledger.validate()
    if (
        not isinstance(cohort,RealPilotCohortReceiptV1)
        or not cohort.source_evidence_bytes_verified
        or not 300<=cohort.total_assets<=400
        or cohort.train_assets<1
        or cohort.val_assets<1
        or cohort.unseen_test_assets<1
        or cohort.train_assets+cohort.val_assets
            +cohort.unseen_test_assets!=cohort.total_assets
        or cohort.cohort_sha256!=ledger.cohort_sha256
        or ledger.next_stage!=plan.stage
        or not isinstance(trusted_source_registry,Mapping)
        or not trusted_source_registry
        or not _is_sha64(pretrained_source_sha256)
        or not _is_sha64(source_vae_sha256)
    ):
        raise ValueError("RSS_VNEXT_PILOT_STAGE_AB_COHORT_LEDGER_OR_SOURCE_UNQUALIFIED")
    prior=None
    if plan.stage==STAGE_A:
        if ledger.completed or pretrained_mv_adapter_sha256 is not None:
            raise ValueError("RSS_VNEXT_PILOT_STAGE_A_CANNOT_SKIP_TO_B")
    else:
        if (
            len(ledger.completed)!=1
            or ledger.completed[-1].stage!=STAGE_A
            or not _is_sha64(pretrained_mv_adapter_sha256)
        ):
            raise ValueError("RSS_VNEXT_PILOT_STAGE_B_PREVIOUS_TRAINED_A_REQUIRED")
        preceding=ledger.completed[-1]
        preceding.validate()
        if _file_digest(preceding.output_checkpoint_path)!=preceding.output_checkpoint_sha256:
            raise ValueError("RSS_VNEXT_PILOT_STAGE_B_A_CHECKPOINT_BYTES_DRIFT")
        # B does NOT import A's SDXL UNet as adapter weights.
        prior=preceding.output_checkpoint_sha256
    trainer=SourceSDXLStageABOptimizerV2(
        stage=plan.stage,unet=unet,vae=vae,
        noise_scheduler=noise_scheduler,
        source_manifest_sha256=cohort.cohort_sha256,
        pretrained_source_sha256=pretrained_source_sha256,
        source_vae_sha256=source_vae_sha256,
        trusted_source_registry=trusted_source_registry,
        pretrained_mv_adapter_sha256=pretrained_mv_adapter_sha256,
        conditioner=conditioner,
        task_a_alpha=canonical_alpha_head,
        mv_pipeline=mv_pipeline,
        task_b_heads=mv_task_heads,
        lr=float(plan.learning_rate),
        warmup_steps=plan.warmup_steps,
        amp_dtype=plan.amp_dtype,
    )
    if (
        trainer.step!=0
        or trainer.optimizer.state
        or trainer.dataset_manifest_sha256!=cohort.cohort_sha256
    ):
        raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_TRAINER_NOT_FRESH")
    return PreparedRealPilotStageABV1(
        plan=plan,trainer=trainer,
        source_cohort_sha256=cohort.cohort_sha256,
        pretrained_source_sha256=pretrained_source_sha256,
        pretrained_mv_adapter_sha256=pretrained_mv_adapter_sha256,
        previous_stage_checkpoint_sha256=prior,
        optimizer_binding_sha256=trainer.optimizer_binding,
    )


class _StageABGradientRecorder:
    """Transparent proxy: the same production trainer remains sole updater."""

    def __init__(self,trainer:SourceSDXLStageABOptimizerV2):
        self.trainer=trainer
        self.dataset_manifest_sha256=trainer.dataset_manifest_sha256
        self.grads:list[dict[str,float]]=[]

    def assert_batch_source_identity(
        self,envelope:RealPilotTrainingEnvelopeV1,
    )->None:
        # A/B source receipts bind one qualified asset, not a free-form
        # minibatch. Until per-subject multi-key receipts exist, B=1 is
        # mandatory. A hidden second subject must not enter a TRAIN update.
        batch=envelope.batch
        if self.trainer.stage==STAGE_A:
            first=getattr(batch,"source_tokens",None)
        else:
            first=getattr(batch,"canonical_rgba",None)
        key=getattr(batch,"source_render_qualification_key",None)
        receipt=(
            self.trainer.trusted_source_registry.get(key)
            if type(key) is str and key else None
        )
        if (
            not isinstance(first,Tensor)
            or first.ndim<2
            or first.shape[0]!=1
            or not isinstance(receipt,Mapping)
            or receipt.get("source_asset_sha256")
                !=envelope.source_asset_sha256
            or receipt.get("source_render_qualification_key")!=key
        ):
            raise ValueError(
                "RSS_VNEXT_PILOT_STAGE_AB_PER_ASSET_TEACHER_IDENTITY_DRIFT"
            )

    def train_step(self,batch:Any):
        result=self.trainer.train_step(batch)
        audited=dict(result.gradient_norms)
        if (
            not audited or any(
                not math.isfinite(x) or x<0
                for x in audited.values()
            )
        ):
            raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_GRADIENT_AUDIT_INVALID")
        self.grads.append(audited)
        return result


def _write_audit_atomic(path:Path,report:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("RSS_VNEXT_PILOT_STAGE_AB_OUTPUT_EXISTS")
    with tempfile.NamedTemporaryFile(
        mode="w",encoding="utf-8",dir=path.parent,
        prefix="rss-ab-gradient-",suffix=".json",delete=False,
    ) as stream:
        temporary=Path(stream.name)
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_real_pilot_stage_ab_v1(
    phase:PreparedRealPilotStageABV1,*,
    examples:Iterable[RealPilotTrainingEnvelopeV1],
    source_index:Mapping[str,RealPilotAssetV1],
    cohort:RealPilotCohortReceiptV1,
    task_checkpoint_file:str|Path,
    gradient_audit_file:str|Path,
)->ExecutedRealPilotStageABCandidateV1:
    """Actual CUDA AdamW updates on the user's TRAIN sources only.

    The runner writes trained *candidate* task weights and actual measured
    per-owner gradient norms. VAL quality, independent GPU receipt and ledger
    promotion are separate; no PASS file is generated or inferred here.
    """
    plan=phase.plan
    plan.validate()
    trainer=phase.trainer
    output=Path(task_checkpoint_file).expanduser().resolve()
    audit=Path(gradient_audit_file).expanduser().resolve()
    if (
        cohort.cohort_sha256!=phase.source_cohort_sha256
        or not cohort.source_evidence_bytes_verified
        or trainer.stage!=plan.stage
        or trainer.step!=0 or trainer.optimizer.state
        or trainer.optimizer_binding!=phase.optimizer_binding_sha256
        or trainer.dataset_manifest_sha256!=cohort.cohort_sha256
        or trainer.device.type!="cuda"
        or not torch.cuda.is_available()
        or output==audit or output.exists() or audit.exists()
        or Path(task_checkpoint_file).is_symlink()
        or Path(gradient_audit_file).is_symlink()
    ):
        raise ValueError("RSS_VNEXT_PILOT_STAGE_AB_REAL_CUDA_OR_PATH_NOT_ADMITTED")
    torch.cuda.reset_peak_memory_stats(trainer.device)
    spy=_StageABGradientRecorder(trainer)
    reports=run_source_qualified_pilot_updates_v1(
        trainer=spy,staged_examples=examples,
        source_index=source_index,cohort=cohort,
        max_steps=plan.optimizer_steps,
    )
    if (
        trainer.step!=plan.optimizer_steps
        or len(spy.grads)!=plan.optimizer_steps
        or set(spy.grads[0])!=set(trainer.groups)
        or any(set(g)!=set(trainer.groups) for g in spy.grads)
        or any(not any(g[name]>0. for g in spy.grads)
               for name in trainer.groups)
    ):
        raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_NO_ACTUAL_OPTIMIZER_GRADIENT")
    if any(not math.isfinite(float(item["total_loss"])) for item in reports):
        raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_NONFINITE_REPORTED_LOSS")
    task=trainer.export_trained_task_weights(output)
    digest=_file_digest(output)
    if digest!=task.checkpoint_sha256:
        raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_TASK_EXPORT_DRIFT")
    report={
        "schema":SCHEMA,
        "stage":plan.stage,
        "cohort_sha256":cohort.cohort_sha256,
        "optimizer_binding_sha256":phase.optimizer_binding_sha256,
        "pretrained_source_sha256":phase.pretrained_source_sha256,
        "pretrained_mv_adapter_sha256":phase.pretrained_mv_adapter_sha256,
        "previous_stage_checkpoint_sha256":phase.previous_stage_checkpoint_sha256,
        "step_gradient_norms":spy.grads,
        "step_losses":[float(x["total_loss"]) for x in reports],
        "trained_steps":trainer.step,
        "task_checkpoint_sha256":digest,
        "admission_status":"CANDIDATE_NOT_VAL_OR_GPU_QUALIFIED",
    }
    _write_audit_atomic(audit,report)
    peak=float(torch.cuda.max_memory_allocated(trainer.device)/1048576.)
    if peak<=0 or not math.isfinite(peak):
        raise RuntimeError("RSS_VNEXT_PILOT_STAGE_AB_GPU_MEASUREMENT_MISSING")
    return ExecutedRealPilotStageABCandidateV1(
        stage=plan.stage,source_cohort_sha256=cohort.cohort_sha256,
        steps=trainer.step,
        total_losses=tuple(float(x["total_loss"]) for x in reports),
        trainable_gradient_groups=tuple(trainer.groups),
        gradient_audit_file=str(audit),
        gradient_audit_sha256=_file_digest(audit),
        task_checkpoint_file=str(output),
        task_checkpoint_sha256=digest,
        device_name=torch.cuda.get_device_name(trainer.device),
        measured_peak_allocated_mib=peak,
    )


__all__=[
    "SCHEMA","STAGE_A","STAGE_B","STAGES",
    "RealPilotStageABPlanV1",
    "PreparedRealPilotStageABV1",
    "ExecutedRealPilotStageABCandidateV1",
    "build_real_pilot_stage_ab_v1",
    "execute_real_pilot_stage_ab_v1",
]
