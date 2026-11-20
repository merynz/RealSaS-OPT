from __future__ import annotations

"""Real SDXL/MV-Adapter A/B optimizer lifecycle and source-graph audit.

A: pretrained SDXL UNet with PEFT LoRA + BOTH-source pose conditioner + alpha.
B: actual pretrained MV-Adapter trainable '_mv'/'_ref', T2IAdapter camera
encoder, and NEW normal-modality/alpha heads, matching upstream's adapter
freeze policy. AdamW and 2000-step linear warmup for B follow official code.

This is executable PyTorch, not a synthetic-only 'plan'. The caller must
provide qualified real source tensors + pinned local pretrained checkpoints.
No data split is selected/modified and no training starts on import.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import os
import tempfile

import torch
from torch import Tensor, nn

from .paired_sdxl_lora_prior_v1 import RSSvNextTwoImageConditionerV1
from .realsas_sdxl_stage_a_source_training_v2 import (
    SDXLCanonicalAlphaHeadV2,SDXLSourceStageABatchV2,
    source_sdxl_canonical_stage_a_loss_v2,
)
from .realsas_mvadapter_dual_reference_v2 import (
    MVAdapterRGBNormalTaskHeadsV2,
    official_mv_adapter_trainable_parameters_v2,
)
from .realsas_mvadapter_source_training_v2 import (
    MVAdapterSourceStageBBatchV2,mv_adapter_stage_b_loss_v2,
)

SCHEMA="RealSaS.RSSvNext.PairedSDXLMVAdapterTrainingState.v2"


def _sha(v:object)->bool:
    return type(v) is str and len(v)==64 and all(
        c in "0123456789abcdef" for c in v
    )


def _param_grad_audit(groups:dict[str,tuple[nn.Parameter,...]])->dict[str,float]:
    """Every intended live component must receive finite differentiable data."""
    results={}
    for name,group in groups.items():
        if not group:
            raise RuntimeError("RSS_VNEXT_STAGE_AB_EMPTY_TRAINABLE_GROUP:"+name)
        live=[p.grad for p in group if p.grad is not None]
        if not live:
            raise RuntimeError("RSS_VNEXT_STAGE_AB_MISSING_TRAINABLE_GRADIENT:"+name)
        if any(
            grad.is_sparse or not bool(torch.isfinite(grad).all())
            for grad in live
        ):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_NONFINITE_GRADIENT:"+name)
        results[name]=float(torch.sqrt(sum(
            grad.float().square().sum() for grad in live
        )).item())
    return results


def _optimizer_binding(groups:dict[str,tuple[nn.Parameter,...]])->str:
    counts={name:len(params) for name,params in groups.items()}
    return sha256(json.dumps(counts,sort_keys=True,
                              separators=(",",":")).encode()).hexdigest()


@dataclass
class StageABStepV2:
    total_loss:Tensor
    rgb_epsilon_loss:Tensor
    normal_epsilon_loss:Tensor|None
    alpha_loss:Tensor
    gradient_norms:dict[str,float]
    step:int
    learning_rate:float


class SourceSDXLStageABOptimizerV2:
    """One live stage: distinct model states, no illegal simultaneous A/B fit."""

    def __init__(
        self,*,stage:str,unet:nn.Module,vae:nn.Module,
        noise_scheduler:Any,
        source_manifest_sha256:str,
        pretrained_source_sha256:str,
        source_vae_sha256:str,
        pretrained_mv_adapter_sha256:str|None=None,
        conditioner:RSSvNextTwoImageConditionerV1|None=None,
        task_a_alpha:SDXLCanonicalAlphaHeadV2|None=None,
        mv_pipeline:Any=None,
        task_b_heads:MVAdapterRGBNormalTaskHeadsV2|None=None,
        lr:float=5e-5,warmup_steps:int=2000,
        amp_dtype:str="BF16",
    )->None:
        if (
            stage not in ("A_CANONICAL_SDXL","B_PAIRED_RGB_NORMAL_MVADAPTER")
            or not _sha(source_manifest_sha256)
            or not _sha(pretrained_source_sha256)
            or not _sha(source_vae_sha256)
            or getattr(vae,"_rss_vnext_vae_ancestor_sha256",None)
                !=source_vae_sha256
            or (stage=="B_PAIRED_RGB_NORMAL_MVADAPTER"
                and not _sha(pretrained_mv_adapter_sha256))
            or (stage=="A_CANONICAL_SDXL"
                and pretrained_mv_adapter_sha256 is not None)
            or type(lr) not in (int,float) or not math.isfinite(lr)
            or not 1e-8<=lr<=1e-2
            or type(warmup_steps) is not int or not 0<=warmup_steps<=100000
            or amp_dtype not in ("BF16","FP16")
            or not isinstance(unet,nn.Module)
            or not isinstance(vae,nn.Module)
        ):
            raise ValueError("RSS_VNEXT_STAGE_AB_OPTIMIZER_CONTRACT_INVALID")
        self.stage=stage
        self.unet=unet
        self.vae=vae
        self.noise_scheduler=noise_scheduler
        self.source_manifest_sha256=source_manifest_sha256
        self.pretrained_source_sha256=pretrained_source_sha256
        self.source_vae_sha256=source_vae_sha256
        self.pretrained_mv_adapter_sha256=pretrained_mv_adapter_sha256
        self.conditioner=conditioner
        self.alpha=task_a_alpha
        self.pipeline=mv_pipeline
        self.heads=task_b_heads
        self.amp_dtype=amp_dtype
        self.step=0
        self.vae.requires_grad_(False)
        self.vae.eval()
        if stage=="A_CANONICAL_SDXL":
            if (
                type(conditioner) is not RSSvNextTwoImageConditionerV1
                or type(task_a_alpha) is not SDXLCanonicalAlphaHeadV2
                or mv_pipeline is not None or task_b_heads is not None
            ):
                raise ValueError("RSS_VNEXT_STAGE_AB_A_MODULE_DOMAIN")
            for name,p in self.unet.named_parameters():
                p.requires_grad_("lora_" in name.lower())
                p.grad=None
            self.conditioner.requires_grad_(True)
            self.alpha.requires_grad_(True)
            self.groups={
                "sdxl_lora":tuple(p for n,p in self.unet.named_parameters()
                                   if p.requires_grad),
                "paired_conditioner":tuple(self.conditioner.parameters()),
                "canonical_alpha":tuple(self.alpha.parameters()),
            }
        else:
            if (
                mv_pipeline is None or mv_pipeline.unet is not unet
                or mv_pipeline.vae is not vae
                or type(task_b_heads) is not MVAdapterRGBNormalTaskHeadsV2
                or conditioner is not None or task_a_alpha is not None
            ):
                raise ValueError("RSS_VNEXT_STAGE_AB_B_MODULE_DOMAIN")
            official_mv_adapter_trainable_parameters_v2(mv_pipeline)
            self.heads.requires_grad_(True)
            self.groups={
                "mv_adapter_row":tuple(p for n,p in self.unet.named_parameters()
                                       if "_mv" in n and p.requires_grad),
                "mv_adapter_reference":tuple(
                    p for n,p in self.unet.named_parameters()
                    if "_ref" in n and p.requires_grad),
                "native_camera_encoder":tuple(
                    self.pipeline.cond_encoder.parameters()),
                "rgb_normal_task_heads":tuple(self.heads.parameters()),
            }
        if any(not ps or any(not p.requires_grad for p in ps)
               for ps in self.groups.values()):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_OPTIMIZER_OWNER_MISSING")
        all_params=[p for group in self.groups.values() for p in group]
        if len({id(p) for p in all_params})!=len(all_params):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_OWNER_PARAMETER_DUPLICATE")
        self.optimizer=torch.optim.AdamW(
            all_params,lr=lr,betas=(.9,.999),weight_decay=.01,
        )
        self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda tick:min(
                1.,(tick+1)/warmup_steps
            ) if warmup_steps else 1.,
        )
        self.optimizer_binding=_optimizer_binding(self.groups)
        self.device=next(self.unet.parameters()).device
        self.autocast_dtype=(torch.bfloat16 if amp_dtype=="BF16"
                             else torch.float16)
        self.scaler=torch.amp.GradScaler(
            "cuda",enabled=amp_dtype=="FP16" and self.device.type=="cuda",
        )

    def train_step(
        self,batch:SDXLSourceStageABatchV2|MVAdapterSourceStageBBatchV2,
    )->StageABStepV2:
        if self.stage=="A_CANONICAL_SDXL":
            if type(batch) is not SDXLSourceStageABatchV2:
                raise TypeError("RSS_VNEXT_STAGE_AB_WRONG_A_BATCH")
            batch.validate(self.conditioner)
        else:
            if type(batch) is not MVAdapterSourceStageBBatchV2:
                raise TypeError("RSS_VNEXT_STAGE_AB_WRONG_B_BATCH")
            batch.validate()
        self.unet.train()
        if self.conditioner is not None:self.conditioner.train()
        if self.alpha is not None:self.alpha.train()
        if self.pipeline is not None:self.pipeline.cond_encoder.train()
        if self.heads is not None:self.heads.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,dtype=self.autocast_dtype,
            enabled=self.device.type=="cuda",
        ):
            if self.stage=="A_CANONICAL_SDXL":
                output=source_sdxl_canonical_stage_a_loss_v2(
                    unet=self.unet,vae=self.vae,
                    noise_scheduler=self.noise_scheduler,
                    conditioner=self.conditioner,alpha_head=self.alpha,
                    batch=batch,
                )
                normal=None
            else:
                output=mv_adapter_stage_b_loss_v2(
                    pipe=self.pipeline,task_heads=self.heads,
                    source=batch,noise_scheduler=self.noise_scheduler,
                )
                normal=output.normal_epsilon_loss
        if not bool(torch.isfinite(output.total_loss).all()):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_NONFINITE_TOTAL_LOSS")
        self.scaler.scale(output.total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        audit=_param_grad_audit(self.groups)
        gradient_norm=torch.nn.utils.clip_grad_norm_(
            [p for group in self.groups.values() for p in group],
            1.,error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_NONFINITE_GLOBAL_GRADIENT")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        self.step+=1
        return StageABStepV2(
            total_loss=output.total_loss.detach(),
            rgb_epsilon_loss=output.rgb_epsilon_loss.detach(),
            normal_epsilon_loss=None if normal is None else normal.detach(),
            alpha_loss=output.alpha_loss.detach(),
            gradient_norms=audit,step=self.step,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def export_trained_task_weights(self,path:str|Path):
        """Only after real optimizer steps; export exact v2 task ABI."""
        if self.step < 1:
            raise RuntimeError("RSS_VNEXT_STAGE_AB_NO_TRAINED_STEP_TO_EXPORT")
        if self.stage=="A_CANONICAL_SDXL":
            from .realsas_sdxl_stage_a_sampler_v2 import (
                write_stage_a_task_weights_v2,
            )
            return write_stage_a_task_weights_v2(
                path,unet=self.unet,conditioner=self.conditioner,
                alpha_head=self.alpha,trained_steps=self.step,
                pretrained_unet_sha256=self.pretrained_source_sha256,
                vae_checkpoint_sha256=self.source_vae_sha256,
                qualified_corpus_sha256=self.source_manifest_sha256,
            )
        from .realsas_mvadapter_stage_b_sampler_v2 import (
            write_stage_b_task_weights_v2,
        )
        return write_stage_b_task_weights_v2(
            path,pipe=self.pipeline,task_heads=self.heads,
            trained_steps=self.step,
            pretrained_base_unet_sha256=self.pretrained_source_sha256,
            pretrained_mv_adapter_sha256=self.pretrained_mv_adapter_sha256,
            vae_checkpoint_sha256=self.source_vae_sha256,
            qualified_corpus_sha256=self.source_manifest_sha256,
        )

    def save_resume_checkpoint(self,path:str|Path)->None:
        """Full optimizer+RNG resume; NOT task-weight release."""
        if self.step<1:
            raise RuntimeError("RSS_VNEXT_STAGE_AB_NO_TRAINED_STEP_TO_RESUME")
        payload={
            "schema":SCHEMA,"stage":self.stage,"step":self.step,
            "source_manifest_sha256":self.source_manifest_sha256,
            "pretrained_source_sha256":self.pretrained_source_sha256,
            "pretrained_mv_adapter_sha256":self.pretrained_mv_adapter_sha256,
            "source_vae_sha256":self.source_vae_sha256,
            "optimizer_binding":self.optimizer_binding,
            "unet_trainable":{n:p.detach().cpu().contiguous()
                               for n,p in self.unet.named_parameters()
                               if p.requires_grad},
            "conditioning":(
                None if self.conditioner is None
                else self.conditioner.state_dict()
            ),
            "alpha":None if self.alpha is None else self.alpha.state_dict(),
            "camera":(
                None if self.pipeline is None
                else self.pipeline.cond_encoder.state_dict()
            ),
            "modality_heads":(
                None if self.heads is None else self.heads.state_dict()
            ),
            "optimizer":self.optimizer.state_dict(),
            "lr_scheduler":self.lr_scheduler.state_dict(),
            "scaler":self.scaler.state_dict(),
            "cpu_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [],
        }
        dest=Path(path)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=dest.parent,prefix="rss-ab-resume-",
            suffix=".pt",delete=False,
        ) as file:
            tmp=Path(file.name)
        try:
            torch.save(payload,tmp)
            os.replace(tmp,dest)
        finally:tmp.unlink(missing_ok=True)

    def load_resume_checkpoint(self,path:str|Path)->None:
        data=torch.load(path,map_location="cpu",weights_only=True)
        if (
            type(data) is not dict or data.get("schema")!=SCHEMA
            or data.get("stage")!=self.stage
            or data.get("source_manifest_sha256")
                !=self.source_manifest_sha256
            or data.get("pretrained_source_sha256")
                !=self.pretrained_source_sha256
            or data.get("pretrained_mv_adapter_sha256")
                !=self.pretrained_mv_adapter_sha256
            or data.get("source_vae_sha256")!=self.source_vae_sha256
            or data.get("optimizer_binding")!=self.optimizer_binding
            or type(data.get("step")) is not int
            or data["step"]<1
        ):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_RESUME_IDENTITY_DRIFT")
        current={n:p for n,p in self.unet.named_parameters()
                 if p.requires_grad}
        tensors=data.get("unet_trainable")
        if type(tensors) is not dict or set(current)!=set(tensors):
            raise RuntimeError("RSS_VNEXT_STAGE_AB_RESUME_TRAINABLE_KEY_DRIFT")
        for name,p in current.items():
            if not isinstance(tensors[name],Tensor) or tensors[name].shape!=p.shape:
                raise RuntimeError("RSS_VNEXT_STAGE_AB_RESUME_TENSOR_DRIFT")
        self.unet.load_state_dict(tensors,strict=False)
        if self.stage=="A_CANONICAL_SDXL":
            self.conditioner.load_state_dict(data["conditioning"],strict=True)
            self.alpha.load_state_dict(data["alpha"],strict=True)
        else:
            self.pipeline.cond_encoder.load_state_dict(
                data["camera"],strict=True,
            )
            self.heads.load_state_dict(data["modality_heads"],strict=True)
        self.optimizer.load_state_dict(data["optimizer"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        self.scaler.load_state_dict(data["scaler"])
        self.step=data["step"]
        torch.set_rng_state(data["cpu_rng"])
        if torch.cuda.is_available() and data["cuda_rng"]:
            if len(data["cuda_rng"])!=torch.cuda.device_count():
                raise RuntimeError("RSS_VNEXT_STAGE_AB_RESUME_CUDA_DEVICE_DRIFT")
            torch.cuda.set_rng_state_all(data["cuda_rng"])


__all__=[
    "SCHEMA","StageABStepV2","SourceSDXLStageABOptimizerV2",
]
