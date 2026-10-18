from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.tessa.v1.model_v1 import TESSAConfigV1, TESSADecoderBlockV1, TESSAOutputV1
from .scene_memory_v1 import IRISTopologySceneMemoryConfigV1, IRISTopologySceneMemoryV1


class TESSADecoderOnlyV1(nn.Module):
    """The useful TESSA topology-token decoder without the legacy 17-D encoder."""

    architecture_id = "RealSaS.TESSA.DecoderOnly.v1"

    def __init__(self, config: TESSAConfigV1 | None = None) -> None:
        super().__init__()
        self.cfg = config or TESSAConfigV1()
        self.cfg.scaling_policy()
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.position_embedding = nn.Embedding(self.cfg.local_attention_window, self.cfg.d_model)
        self.decoder = nn.ModuleList([TESSADecoderBlockV1(self.cfg) for _ in range(self.cfg.decoder_layers)])
        self.final_norm = nn.LayerNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

    def _position_ids(
        self,
        length: int,
        device: torch.device,
        *,
        offset: int | torch.Tensor = 0,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        base = torch.arange(length, device=device)
        if isinstance(offset, torch.Tensor):
            value = offset.to(device=device, dtype=torch.long)
            if value.ndim != 1 or batch_size is None or value.shape != (batch_size,):
                raise ValueError("TESSA_POSITION_OFFSET_BATCH_SHAPE_INVALID")
            if value.numel() and int(value.min()) < 0:
                raise ValueError("TESSA_POSITION_OFFSET_NEGATIVE")
            return (base[None, :] + value[:, None]) % self.cfg.local_attention_window
        if int(offset) < 0:
            raise ValueError("TESSA_POSITION_OFFSET_NEGATIVE")
        return (base + int(offset)) % self.cfg.local_attention_window

    def forward(
        self,
        *,
        memory: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
    ) -> TESSAOutputV1:
        if memory.ndim != 3 or memory.shape[-1] != self.cfg.d_model:
            raise ValueError("IRIS_TESSA_MEMORY_SHAPE_INVALID")
        if input_ids.ndim != 2 or input_ids.shape[0] != memory.shape[0]:
            raise ValueError("TESSA_INPUT_IDS_SHAPE_INVALID")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.cfg.vocab_size
        ):
            raise ValueError("TESSA_INPUT_TOKEN_OUT_OF_RANGE")

        pos = self._position_ids(
            input_ids.shape[1],
            input_ids.device,
            offset=sequence_position_offset,
            batch_size=input_ids.shape[0],
        )
        token = self.token_embedding(input_ids)
        position = self.position_embedding(pos)
        if pos.ndim == 1:
            position = position[None]
        x = token + position
        for block in self.decoder:
            x = block(x, memory)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("TESSA_LABEL_SHAPE_INVALID")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.cfg.PAD,
            )
        return TESSAOutputV1(logits=logits, surface_latents=memory, loss=loss)


class IRISSceneTopologyHeadV1(nn.Module):
    """Scene-native topology head over the existing IRIS triplane."""

    architecture_id = "RealSaS.IRIS.SceneTopologyHead.v1"
    input_authority = "IRIS_SCENE_EVIDENCE_ONLY"
    output_authority = "LEARNED_TOPOLOGY_PROPOSAL_ONLY"
    legacy_gsa_17d_used = False

    def __init__(
        self,
        *,
        tessa_config: TESSAConfigV1 | None = None,
        memory_config: IRISTopologySceneMemoryConfigV1 | None = None,
    ) -> None:
        super().__init__()
        tessa = tessa_config or TESSAConfigV1()
        memory = memory_config or IRISTopologySceneMemoryConfigV1(
            topology_width=tessa.d_model,
            memory_tokens=tessa.surface_latent_count,
            attention_heads=tessa.n_heads,
        )
        memory.validate()
        if memory.topology_width != tessa.d_model:
            raise ValueError("IRIS_TESSA_TOPOLOGY_WIDTH_MISMATCH")
        if memory.memory_tokens != tessa.surface_latent_count:
            raise ValueError("IRIS_TESSA_MEMORY_TOKEN_POLICY_MISMATCH")
        if memory.attention_heads != tessa.n_heads:
            raise ValueError("IRIS_TESSA_ATTENTION_HEAD_POLICY_MISMATCH")
        self.tessa_config = tessa
        self.memory_config = memory
        self.scene_memory = IRISTopologySceneMemoryV1(memory)
        self.topology_decoder = TESSADecoderOnlyV1(tessa)

    def encode_scene_memory(self, scene_planes: torch.Tensor) -> torch.Tensor:
        return self.scene_memory(scene_planes)

    def forward(
        self,
        *,
        scene_planes: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        sequence_position_offset: int | torch.Tensor = 0,
    ) -> TESSAOutputV1:
        return self.topology_decoder(
            memory=self.encode_scene_memory(scene_planes),
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=sequence_position_offset,
        )
