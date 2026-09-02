from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import torch
from torch import nn

try:
    from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR
except ImportError:
    from realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR

from .conditioning_v2 import ArachneConditioningBatchV2
from .skin_field_codec_v1 import SkinFieldCodecV1


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneCandidateConfigV2:
    surface_feature_dim: int = 20
    joint_feature_dim: int = 8
    pair_geometry_dim: int = 10
    model_dim: int = 128
    surface_encoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.0
    architecture_id: str = "RealSaS.ArachneCandidate.SegmentAwareJointField.v2"

    def validate(self) -> None:
        if (self.surface_feature_dim, self.joint_feature_dim, self.pair_geometry_dim) != (20, 8, 10):
            raise ValueError("Arachne V2 conditioning width drift")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("model_dim/head mismatch")
        if min(self.surface_encoder_layers, self.attention_heads, self.feedforward_dim) <= 0:
            raise ValueError("invalid Arachne V2 architecture cardinality")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("invalid dropout")

    @property
    def config_hash(self) -> str:
        self.validate(); return _hash(asdict(self))


@dataclass
class ArachneRawOutputV2:
    joint_latent_mean: torch.Tensor
    joint_latent_log_sigma: torch.Tensor
    decoded_weights: torch.Tensor
    pair_logits: torch.Tensor
    pair_attention: torch.Tensor


class ArachneCandidateV2(nn.Module):
    """Segment-aware qualified-skeleton-conditioned skin-field latent predictor."""
    def __init__(
        self,
        codec: SkinFieldCodecV1,
        config: ArachneCandidateConfigV2 = ArachneCandidateConfigV2(),
        *,
        freeze_codec: bool = True,
    ):
        super().__init__(); config.validate(); self.config=config; self.codec=codec
        if freeze_codec:
            for parameter in self.codec.parameters(): parameter.requires_grad_(False)
        d=config.model_dim
        self.surface_in=nn.Sequential(nn.Linear(20,d),nn.GELU(),nn.Linear(d,d))
        self.joint_in=nn.Sequential(nn.Linear(8,d),nn.GELU(),nn.Linear(d,d))
        enc=nn.TransformerEncoderLayer(d,config.attention_heads,config.feedforward_dim,config.dropout,batch_first=True,norm_first=True,activation="gelu")
        self.surface_encoder=nn.TransformerEncoder(enc,config.surface_encoder_layers,norm=nn.LayerNorm(d))
        self.parent_context=nn.Linear(d,d,bias=False)
        self.root_embed=nn.Embedding(2,d)
        self.key=nn.Linear(d,d,bias=False); self.value=nn.Linear(d,d,bias=False)
        self.pair_bias=nn.Sequential(nn.Linear(10,d),nn.GELU(),nn.Linear(d,1))
        self.pair_value=nn.Sequential(nn.Linear(10,d),nn.GELU(),nn.Linear(d,d))
        self.out_norm=nn.LayerNorm(d)
        self.out_ff=nn.Sequential(nn.Linear(d,config.feedforward_dim),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(config.feedforward_dim,d))
        self.to_latent_mean=nn.Linear(d,codec.config.latent_dim)
        self.to_latent_log_sigma=nn.Linear(d,codec.config.latent_dim)

    @property
    def codec_config_hash(self) -> str:
        return self.codec.config.config_hash

    def _joint_queries(self,joint_features,parent_indices,joint_mask):
        base=self.joint_in(joint_features); safe=parent_indices.clamp_min(0); parent=torch.gather(base,1,safe[...,None].expand(-1,-1,base.shape[-1])); has=(parent_indices>=0)&joint_mask.bool(); root=(parent_indices<0).long().clamp(0,1); q=base+self.parent_context(parent)*has[...,None].to(base.dtype)+self.root_embed(root); return q*joint_mask[...,None].to(base.dtype)

    def forward(self,surface_features,joint_features,surface_mask,joint_mask,parent_indices,pair_geometry,pair_mask)->ArachneRawOutputV2:
        if surface_features.ndim!=3 or joint_features.ndim!=3 or pair_geometry.ndim!=4: raise ValueError("Arachne V2 rank mismatch")
        if surface_features.shape[-1]!=20 or joint_features.shape[-1]!=8 or pair_geometry.shape[-1]!=10: raise ValueError("Arachne V2 width mismatch")
        B,N,_=surface_features.shape; J=joint_features.shape[1]
        if surface_mask.shape!=(B,N) or joint_mask.shape!=(B,J) or parent_indices.shape!=(B,J) or pair_geometry.shape!=(B,N,J,10) or pair_mask.shape!=(B,N,J): raise ValueError("Arachne V2 shape mismatch")
        memory=self.surface_encoder(self.surface_in(surface_features),src_key_padding_mask=~surface_mask.bool())
        q=self._joint_queries(joint_features,parent_indices,joint_mask); k=self.key(memory); v=self.value(memory)
        pair=pair_geometry.permute(0,2,1,3); pm=pair_mask.permute(0,2,1).bool(); bias=self.pair_bias(pair).squeeze(-1); pv=self.pair_value(pair)
        scores=torch.einsum("bjd,bnd->bjn",q,k)/math.sqrt(self.config.model_dim)+bias; scores=scores.masked_fill(~pm,-1e4); attn=torch.softmax(scores,dim=-1)*pm.to(scores.dtype); denom=attn.sum(-1,keepdim=True).clamp_min(1e-8); attn=attn/denom
        context=torch.einsum("bjn,bjnd->bjd",attn,v[:,None,:,:]+pv); h=self.out_norm(q+context); h=h+self.out_ff(h); h=h*joint_mask[...,None].to(h.dtype)
        mean=self.to_latent_mean(h)*joint_mask[...,None].to(h.dtype); log_sigma=self.to_latent_log_sigma(h).clamp(-8.0,4.0)*joint_mask[...,None].to(h.dtype)
        weights,logits=self.codec.decode_from_latents(mean,surface_features,joint_features,surface_mask,joint_mask)
        return ArachneRawOutputV2(mean,log_sigma,weights,logits,attn)

    @torch.no_grad()
    def propose(self,conditioning:ArachneConditioningBatchV2,*,device=None)->tuple[SkinProposalIR,...]:
        device=device or next(self.parameters()).device; sf=torch.as_tensor(conditioning.surface_features,device=device,dtype=torch.float32); jf=torch.as_tensor(conditioning.joint_features,device=device,dtype=torch.float32); sm=torch.as_tensor(conditioning.surface_mask,device=device,dtype=torch.bool); jm=torch.as_tensor(conditioning.joint_mask,device=device,dtype=torch.bool); pi=torch.as_tensor(conditioning.parent_indices,device=device,dtype=torch.long); pg=torch.as_tensor(conditioning.pair_geometry,device=device,dtype=torch.float32); pm=torch.as_tensor(conditioning.pair_mask,device=device,dtype=torch.bool); self.eval(); out=self(sf,jf,sm,jm,pi,pg,pm); proposals=[]
        for b in range(sf.shape[0]):
            sids=conditioning.surface_ids[b]; jids=conditioning.joint_ids[b]; influences=[]
            for si,sid in enumerate(sids):
                row=out.decoded_weights[b,si,:len(jids)]
                for ji,jid in enumerate(jids): influences.append(SkinInfluenceProposal(sid,jid,float(row[ji].item())))
            sigma=torch.exp(out.joint_latent_log_sigma[b,:len(jids)]); proposals.append(SkinProposalIR(tuple(influences),conditioning.source_surface_hashes[b],conditioning.source_skeleton_hashes[b],model_provenance=self.config.config_hash,metadata={"conditioning_hash":conditioning.conditioning_hashes[b],"candidate_architecture":self.config.architecture_id,"codec_architecture":self.codec.config.architecture_id,"codec_config_hash":self.codec.config.config_hash,"codec_decoder_object_shared":True,"codec_frozen":all(not p.requires_grad for p in self.codec.parameters()),"latent_uncertainty_mean":float(sigma.mean().item()),"dense_proposal":True,"compiler_owns_sparsification_and_qualification":True,"full_3d_reconstruction_claim":False}))
        return tuple(proposals)
