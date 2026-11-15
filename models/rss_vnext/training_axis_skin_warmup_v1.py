from __future__ import annotations

"""Deterministic, optional AXIS SkinToken warmup for staged RSS.vNext training.

TokenRig freezes its skin FSQ VAE and uses a cosine skin-token warmup. Here
the separately pretrained and frozen MIRA codec remains the sole skin-code
authority; only the *AXIS causal skin-code CE* is ramped. Skeleton grammar,
joint/frame/control losses and source-qualified posed-surface losses are not
silently downweighted. A disabled policy preserves legacy exact objectives.

Progress uses the current stage's global update step, NOT wall-clock time.
The enclosing trainer pins this policy in its checkpoint objective digest.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from typing import Mapping
import torch
from torch import Tensor

SCHEMA="RealSaS.RSSvNext.AxisSkinCodeWarmup.v1"


@dataclass(frozen=True)
class RSSvNextAxisSkinCodeWarmupV1:
    start_step:int=0
    full_step:int=0

    def validate(self)->None:
        if (
            type(self.start_step) is not int or type(self.full_step) is not int
            or self.start_step<0 or self.full_step<0
            or (self.full_step==0 and self.start_step!=0)
            or (self.full_step!=0 and self.full_step<=self.start_step)
        ):
            raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_POLICY_INVALID")

    @property
    def enabled(self)->bool:
        self.validate()
        return self.full_step>0

    def rate(self,step:int)->float:
        self.validate()
        if type(step) is not int or step<0:
            raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_STEP_INVALID")
        if not self.enabled or step>=self.full_step:
            return 1.
        if step<=self.start_step:
            return 0.
        fraction=(step-self.start_step)/(self.full_step-self.start_step)
        return 0.5*(1.-math.cos(math.pi*fraction))

    def digest(self)->str:
        self.validate()
        return sha256(json.dumps(
            {"schema":SCHEMA,"policy":asdict(self)},sort_keys=True,
            separators=(",",":"),allow_nan=False,
        ).encode("utf8")).hexdigest()


def apply_rss_vnext_axis_skin_code_warmup_v1(
    *,
    raw_axis_loss:Tensor,
    raw_axis_loss_parts:Mapping[str,Tensor],
    policy:RSSvNextAxisSkinCodeWarmupV1,
    step:int,
)->Tensor:
    """Preserve rig losses while ramping ONLY independently labelled skin CE."""
    policy.validate()
    if not isinstance(raw_axis_loss,Tensor) or raw_axis_loss.numel()!=1:
        raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_RAW_LOSS_INVALID")
    if not bool(torch.isfinite(raw_axis_loss).all()):
        raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_RAW_LOSS_NONFINITE")
    if not policy.enabled:
        return raw_axis_loss
    skin=raw_axis_loss_parts.get("skin_code_logits")
    if skin is None:
        return raw_axis_loss
    if (not isinstance(skin,Tensor) or skin.numel()!=1
        or not bool(torch.isfinite(skin).all())):
        raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_SKIN_CE_INVALID")
    value=(raw_axis_loss-skin)+skin*policy.rate(step)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_RESULT_NONFINITE")
    return value


__all__=[
    "SCHEMA","RSSvNextAxisSkinCodeWarmupV1",
    "apply_rss_vnext_axis_skin_code_warmup_v1",
]
