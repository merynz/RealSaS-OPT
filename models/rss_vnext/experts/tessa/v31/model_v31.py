"""One allocation of the V3 tri-stream decoder with explicit V3.1 semantics.

Product token generation uses the SAME layer weights as teacher-forced CE,
but caches per-layer rolling causal K/V so long source topologies do not
recompute all historical positions at every new token.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from models.tessa.v2.model_v2 import TESSADecoderBlockV2
from models.tessa.v3.model_v3 import TESSAStructuralizerV3, TESSATriStreamMemoryV3
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from .attention_v31 import WindowedCausalSelfAttentionV31, SEMANTICS_ID


class TESSADecoderBlockV31(TESSADecoderBlockV2):
    def __init__(self, config, *, backend):
        nn.Module.__init__(self)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = WindowedCausalSelfAttentionV31(config, backend=backend)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.cross = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.feedforward_dim), nn.GELU(),
                                 nn.Linear(config.feedforward_dim, config.d_model))


@dataclass(frozen=True)
class TESSAV31IncrementalCache:
    memory: torch.Tensor
    position: int
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]


class TESSAStructuralizerV31(TESSAStructuralizerV3):
    architecture_id = "RealSaS.TESSA.Structuralizer.v3_1"
    attention_semantics = SEMANTICS_ID

    def __init__(self, *, scene_input_dim, surface_input_dim, presentation_input_dim,
                 config=TESSAStructuralizerConfigV3(), backend="SDPA"):
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        # Do not allocate then discard a second full V2 decoder/memory stack.
        self.memory = TESSATriStreamMemoryV3(scene_input_dim=scene_input_dim, surface_input_dim=surface_input_dim,
                                            presentation_input_dim=presentation_input_dim, config=config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.local_attention_window, config.d_model)
        self.decoder = nn.ModuleList([TESSADecoderBlockV31(config, backend=backend) for _ in range(config.decoder_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


    @torch.no_grad()
    def decode_next_token_v1(
        self, *, memory: torch.Tensor, token: torch.Tensor,
        cache: TESSAV31IncrementalCache | None = None,
    ) -> tuple[torch.Tensor, TESSAV31IncrementalCache]:
        """Exact unbatched-prefix V3.1 next-token logits with rolling K/V.

        At each layer the query attends to the previous window-1 entries plus
        itself, equivalent to V3.1 full-prefix SDPA. The positional embedding
        uses the same modulo-window scheme as decode_tokens(). Inference-only;
        independent model/Compiler proof still decides topology admission.
        """
        if self.training:
            raise RuntimeError("TESSA_V31_INCREMENTAL_REQUIRES_EVAL")
        if (
            token.ndim != 2 or token.shape[1] != 1
            or token.dtype != torch.long
            or memory.ndim != 3
            or memory.shape[0] != token.shape[0]
            or memory.shape[-1] != self.config.d_model
            or token.device != memory.device
            or not bool(((token >= 0) & (token < self.config.vocab_size)).all())
        ):
            raise ValueError("TESSA_V31_INCREMENTAL_TOKEN_OR_MEMORY_DOMAIN")
        if cache is None:
            position = 0
            old_keys = (None,) * len(self.decoder)
            old_values = (None,) * len(self.decoder)
        else:
            if (
                type(cache) is not TESSAV31IncrementalCache
                or cache.memory is not memory
                or len(cache.keys) != len(self.decoder)
                or len(cache.values) != len(self.decoder)
                or cache.position < 1
            ):
                raise ValueError("TESSA_V31_INCREMENTAL_FOREIGN_CACHE")
            position = cache.position
            old_keys, old_values = cache.keys, cache.values
        width = self.config.local_attention_window
        pos = torch.full_like(token, position % width)
        x = self.token_embedding(token) + self.position_embedding(pos)
        updated_k, updated_v = [], []
        for layer_id, layer in enumerate(self.decoder):
            normalized = layer.norm1(x)
            b, _, d = normalized.shape
            h = self.config.n_heads
            head_width = d // h
            qkv = layer.self_attn.qkv(normalized).reshape(
                b, 1, 3, h, head_width,
            )
            q, k, v = (
                z.transpose(1, 2)
                for z in qkv.unbind(dim=2)
            )
            if old_keys[layer_id] is not None:
                k = torch.cat((old_keys[layer_id], k), dim=2)
                v = torch.cat((old_values[layer_id], v), dim=2)
            k = k[:, :, -width:, :]
            v = v[:, :, -width:, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0., is_causal=False,
            )
            projected = layer.self_attn.out(
                attended.transpose(1, 2).contiguous().reshape(b, 1, d),
            )
            x = x + projected
            cross, _ = layer.cross(
                layer.norm2(x), memory, memory, need_weights=False,
            )
            x = x + cross
            x = x + layer.mlp(layer.norm3(x))
            updated_k.append(k)
            updated_v.append(v)
        logits = self.lm_head(self.final_norm(x))[:, 0]
        return logits, TESSAV31IncrementalCache(
            memory=memory, position=position+1,
            keys=tuple(updated_k), values=tuple(updated_v),
        )
