"""One allocation of the V3 tri-stream decoder with explicit V3.1 semantics."""
from torch import nn
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
