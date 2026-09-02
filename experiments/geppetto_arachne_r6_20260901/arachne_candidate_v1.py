from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR
except ImportError:
    from realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR

from .candidate_config_v1 import ARACHNE_V1, ArachneCandidateConfigV1
from .conditioning_v1 import ArachneConditioningBatchV1
from .skin_field_codec_v1 import SkinFieldCodecV1


@dataclass
class ArachneRawOutputV1:
    joint_latents: torch.Tensor
    decoded_weights: torch.Tensor
    pair_logits: torch.Tensor


class ArachneCandidateV1(nn.Module):
    """Qualified-skeleton-conditioned joint influence-field proposer.

    The predictor emits codec latents, not canonical weights. Those latents are
    decoded by the exact SkinFieldCodecV1 decoder frozen/qualified in R6-A0.
    Compiler remains the sole authority for legal skin qualification/sparsification.
    """

    def __init__(self, codec: SkinFieldCodecV1, config: ArachneCandidateConfigV1 = ARACHNE_V1):
        super().__init__()
        config.validate()
        self.config = config
        self.codec = codec
        d = config.model_dim
        self.surface_in = nn.Sequential(nn.Linear(config.surface_feature_dim, d), nn.GELU(), nn.Linear(d, d))
        self.joint_in = nn.Sequential(nn.Linear(config.joint_feature_dim, d), nn.GELU(), nn.Linear(d, d))
        enc_layer = nn.TransformerEncoderLayer(d, config.attention_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.surface_encoder = nn.TransformerEncoder(enc_layer, config.encoder_layers, norm=nn.LayerNorm(d))
        dec_layer = nn.TransformerDecoderLayer(d, config.attention_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.cross_attention = nn.TransformerDecoder(dec_layer, config.cross_attention_layers, norm=nn.LayerNorm(d))
        self.parent_embed = nn.Embedding(3, d)
        self.parent_context = nn.Linear(d, d, bias=False)
        self.to_latent = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, codec.config.latent_dim))

    def forward(self, surface_features: torch.Tensor, joint_features: torch.Tensor, surface_mask: torch.Tensor, joint_mask: torch.Tensor, parent_indices: torch.Tensor) -> ArachneRawOutputV1:
        if parent_indices.shape != joint_mask.shape:
            raise ValueError("parent_indices shape mismatch")
        if surface_features.shape[-1] != self.config.surface_feature_dim or joint_features.shape[-1] != self.config.joint_feature_dim:
            raise ValueError("Arachne conditioner width mismatch")
        memory = self.surface_encoder(self.surface_in(surface_features), src_key_padding_mask=~surface_mask.bool())
        parent_state = torch.where(~joint_mask.bool(), torch.full_like(parent_indices, 2), torch.where(parent_indices < 0, torch.zeros_like(parent_indices), torch.ones_like(parent_indices)))
        base_q = self.joint_in(joint_features)
        safe_parent = parent_indices.clamp_min(0)
        gathered_parent = torch.gather(base_q, 1, safe_parent[..., None].expand(-1, -1, base_q.shape[-1]))
        has_parent = (parent_indices >= 0) & joint_mask.bool()
        q = base_q + self.parent_embed(parent_state.clamp(0, 2)) + self.parent_context(gathered_parent) * has_parent[..., None].to(base_q.dtype)
        q = self.cross_attention(q, memory, tgt_key_padding_mask=~joint_mask.bool(), memory_key_padding_mask=~surface_mask.bool())
        latents = self.to_latent(q) * joint_mask[..., None].to(q.dtype)
        weights, logits = self.codec.decode_from_latents(latents, surface_features, joint_features, surface_mask, joint_mask)
        return ArachneRawOutputV1(latents, weights, logits)

    @torch.no_grad()
    def propose(self, conditioning: ArachneConditioningBatchV1, *, device: torch.device | str | None = None) -> tuple[SkinProposalIR, ...]:
        device = device or next(self.parameters()).device
        sf = torch.as_tensor(conditioning.surface_features, device=device, dtype=torch.float32)
        jf = torch.as_tensor(conditioning.joint_features, device=device, dtype=torch.float32)
        sm = torch.as_tensor(conditioning.surface_mask, device=device, dtype=torch.bool)
        jm = torch.as_tensor(conditioning.joint_mask, device=device, dtype=torch.bool)
        pi = torch.as_tensor(conditioning.parent_indices, device=device, dtype=torch.long)
        self.eval()
        out = self(sf, jf, sm, jm, pi)
        proposals = []
        for b in range(sf.shape[0]):
            sids = conditioning.surface_ids[b]
            jids = conditioning.joint_ids[b]
            influences = []
            for si, sid in enumerate(sids):
                row = out.decoded_weights[b, si, :len(jids)]
                for ji, jid in enumerate(jids):
                    influences.append(SkinInfluenceProposal(sid, jid, float(row[ji].item())))
            proposals.append(SkinProposalIR(
                tuple(influences),
                conditioning.source_surface_hashes[b],
                conditioning.source_skeleton_hashes[b],
                model_provenance=self.config.config_hash,
                metadata={
                    "conditioning_hash": conditioning.conditioning_hashes[b],
                    "candidate_architecture": self.config.architecture_id,
                    "codec_architecture": self.codec.config.architecture_id,
                    "codec_config_hash": self.codec.config.config_hash,
                    "dense_proposal": True,
                    "compiler_owns_sparsification_and_qualification": True,
                    "full_3d_reconstruction_claim": False,
                },
            ))
        return tuple(proposals)
