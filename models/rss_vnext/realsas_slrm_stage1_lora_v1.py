from __future__ import annotations

"""Clean-room Stage1 attention LoRA for the published RealSaS S-LRM recipe.

the CVPR 2025 multi-view reference uses pretrained InstantNeRF, LoRA in each self/cross
attention, learned triplane position/deconv and a new semantic decoder.
The source-released RealSaS uses loratorch; this is independently written
using PyTorch parametrization and DOES NOT import proprietary NVIDIA code.

This adapter is for the existing IRIS TransformerDecoder. It is NOT
compatible with released InstantNeRF checkpoints and is NOT proof that
such a pretrained model was loaded. Do not freeze randomly initialized
IRIS and claim RealSaS parity. Install AFTER trusted 3D pretraining and
BEFORE creating the optimizer.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .unified_character_model import RSSvNextUnifiedCharacterModel


class RealSaSAttentionWeightLoRAV1(nn.Module):
    """W = W_base + alpha/rank * B@A, with zero-initialized B."""

    def __init__(self,shape:tuple[int,int],*,rank:int,alpha:float):
        super().__init__()
        rows,cols=shape
        if (
            rows<2 or cols<2 or not 1<=rank<=min(rows,cols)
            or not math.isfinite(alpha) or alpha<=0
        ):
            raise ValueError("RSS_VNEXT_REALSAS_LORA_WEIGHT_SHAPE_OR_RANK")
        self.scale=alpha/rank
        self.down=nn.Parameter(torch.empty(rank,cols))
        self.up=nn.Parameter(torch.zeros(rows,rank))
        nn.init.kaiming_uniform_(self.down,a=math.sqrt(5))

    def forward(self,original:Tensor)->Tensor:
        if original.ndim!=2:
            raise ValueError("RSS_VNEXT_REALSAS_LORA_ORIGINAL_MATRIX_DOMAIN")
        return original+(self.up@self.down)*self.scale


@dataclass(frozen=True)
class RealSaSStageOneAttentionLoRAReceiptV1:
    adapter_architecture:str
    attention_module_paths:tuple[str,...]
    adapted_parameter_names:tuple[str,...]
    rank:int
    alpha:float
    pretrained_checkpoint_loaded_and_verified:bool
    original_instantnerf_weight_compatibility_verified:bool
    stage1_trainable_parameter_count:int
    stage1_frozen_parameter_count:int
    authority:str="ATTENTION_LORA_TRAINING_RECIPE_NOT_PRETRAINED_BACKBONE_PARITY"


def install_rss_realsas_stage1_attention_lora_v1(
    model:RSSvNextUnifiedCharacterModel, *,
    rank:int=8,alpha:float=8.,
) -> RealSaSStageOneAttentionLoRAReceiptV1:
    """Attach LoRA to every self/cross attention QKV and output projection.

    Original density/Graphics/geometry fields stay frozen, while only
    Stage1 semantic, attention adapters, triplane learned queries/deconv
    are trainable. THIS IS NOT AN UPSTREAM CHECKPOINT LOADER.
    """
    if (
        not isinstance(model,RSSvNextUnifiedCharacterModel)
        or getattr(model,"realsas_stage1_lora_adapter_installed",False)
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE1_LORA_MODEL_OR_REINSTALL")
    mods=[(path,module) for path,module in
          model.iris.scene.decoder.named_modules()
          if isinstance(module,nn.MultiheadAttention)]
    if (
        not mods or len(mods)>128
        or not any("self_attn" in path for path,_ in mods)
        or not any("multihead_attn" in path or "cross_attn" in path for path,_ in mods)
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE1_LORA_SELF_CROSS_BLOCKS_REQUIRED")
    if not 1<=rank<=64 or not math.isfinite(alpha) or alpha<=0:
        raise ValueError("RSS_VNEXT_REALSAS_STAGE1_LORA_RANK_ALPHA_DOMAIN")
    targets=[]
    for path,module in mods:
        keys=(
            ("in_proj_weight","out_proj.weight") if module.in_proj_weight is not None
            else ("q_proj_weight","k_proj_weight","v_proj_weight","out_proj.weight")
        )
        for key in keys:
            owner=module.out_proj if key=="out_proj.weight" else module
            attr="weight" if key=="out_proj.weight" else key
            matrix=getattr(owner,attr,None)
            if (
                not isinstance(matrix,Tensor) or matrix.ndim!=2
                or rank>min(matrix.shape)
                or parametrize.is_parametrized(owner,attr)
            ):
                raise ValueError("RSS_VNEXT_REALSAS_STAGE1_LORA_TARGET_INCOMPATIBLE")
            targets.append((f"iris.scene.decoder.{path}.{key}",owner,attr,tuple(matrix.shape)))
    model.requires_grad_(False)
    for _,owner,attr,shape in targets:
        parametrize.register_parametrization(
            owner,attr,RealSaSAttentionWeightLoRAV1(shape,rank=rank,alpha=alpha),
        )
    model.presentation_role_head.requires_grad_(True)
    model.iris.scene.scene_queries.requires_grad_(True)
    model.iris.scene.upsample.requires_grad_(True)
    model.realsas_stage1_lora_adapter_installed=True
    trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
    frozen=sum(x.numel() for x in model.parameters() if not x.requires_grad)
    if trainable<=0 or frozen<=0:
        raise ValueError("RSS_VNEXT_REALSAS_STAGE1_LORA_NO_FROZEN_OR_TRAINABLE_WEIGHTS")
    return RealSaSStageOneAttentionLoRAReceiptV1(
        adapter_architecture="RSS_IRIS_TRANSFORMER_NOT_INSTANTNERF_COMPATIBLE",
        attention_module_paths=tuple(path for path,_ in mods),
        adapted_parameter_names=tuple(x[0] for x in targets),
        rank=rank,alpha=alpha,
        pretrained_checkpoint_loaded_and_verified=False,
        original_instantnerf_weight_compatibility_verified=False,
        stage1_trainable_parameter_count=trainable,
        stage1_frozen_parameter_count=frozen,
    )


__all__=[
    "RealSaSAttentionWeightLoRAV1","RealSaSStageOneAttentionLoRAReceiptV1",
    "install_rss_realsas_stage1_attention_lora_v1",
]
