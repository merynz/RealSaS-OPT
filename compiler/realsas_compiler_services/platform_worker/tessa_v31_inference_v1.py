from __future__ import annotations

from dataclasses import dataclass

import torch

from compiler.realsas_compiler_services.platform_worker.model_binding import (
    MaterializedModelReleaseRef,
    require_materialized_model_release,
)
from models.tessa.v2.court_v2 import (
    TESSADecodedTopologyCourtV2,
    evaluate_decoded_topology_v2,
)
from models.tessa.v2.generation_v2 import TESSAGrammarStateV2
from models.tessa.v31.checkpoint_v1 import load_checkpoint_v1
from models.tessa.v31.incremental_v31 import IncrementalDecoderV31
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


TESSA_MODEL_FAMILY_V31 = "TESSA"
TESSA_ARCHITECTURE_ID_V31 = TESSAStructuralizerV31.architecture_id
TESSA_OUTPUT_AUTHORITY_V31 = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"


@dataclass(frozen=True)
class TESSAV31InferenceBindingV1:
    model_ref: MaterializedModelReleaseRef
    model: TESSAStructuralizerV31
    schema: str = "RealSaS.TESSAV31InferenceBinding.v1"


@dataclass(frozen=True)
class TESSAV31InferenceInputsV1:
    scene_source: torch.Tensor
    surface_source: torch.Tensor
    presentation_source: torch.Tensor
    surface_valid_mask: torch.Tensor
    presentation_valid_mask: torch.Tensor
    schema: str = "RealSaS.TESSAV31InferenceInputs.v1"

    def validate(self, model: TESSAStructuralizerV31) -> None:
        if not isinstance(model, TESSAStructuralizerV31):
            raise TypeError("TESSA_V31_INFERENCE_MODEL_REQUIRED")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("TESSA_V31_INFERENCE_MODEL_NOT_FROZEN")
        streams = (
            (self.scene_source, model.memory.scene.input_dim, "SCENE"),
            (self.surface_source, model.memory.surface.input_dim, "SURFACE"),
            (self.presentation_source, model.memory.presentation.input_dim, "PRESENTATION"),
        )
        batch = None
        for value, expected_dim, name in streams:
            if value.ndim != 3 or value.shape[-1] != int(expected_dim) or value.shape[1] <= 0:
                raise RuntimeError(f"TESSA_V31_INFERENCE_{name}_SHAPE_INVALID")
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"TESSA_V31_INFERENCE_{name}_NUMERIC_INVALID")
            batch = int(value.shape[0]) if batch is None else batch
            if int(value.shape[0]) != batch:
                raise RuntimeError("TESSA_V31_INFERENCE_BATCH_DRIFT")
        if self.surface_valid_mask.shape != self.surface_source.shape[:2]:
            raise RuntimeError("TESSA_V31_INFERENCE_SURFACE_MASK_SHAPE_INVALID")
        if self.presentation_valid_mask.shape != self.presentation_source.shape[:2]:
            raise RuntimeError("TESSA_V31_INFERENCE_PRESENTATION_MASK_SHAPE_INVALID")
        if not bool(self.surface_valid_mask.bool().any(dim=1).all()):
            raise RuntimeError("TESSA_V31_INFERENCE_SURFACE_MASK_EMPTY")
        if not bool(self.presentation_valid_mask.bool().any(dim=1).all()):
            raise RuntimeError("TESSA_V31_INFERENCE_PRESENTATION_MASK_EMPTY")


@dataclass(frozen=True)
class TESSAV31TopologyProposalV1:
    token_ids: tuple[int, ...]
    topology_court: TESSADecodedTopologyCourtV2
    completed: bool
    stop_reason: str
    model_release_id: str
    checkpoint_content_sha256: str
    schema: str = "RealSaS.TESSATopologyProposal.v3_1"
    output_authority: str = TESSA_OUTPUT_AUTHORITY_V31
    canonical_mesh_authority: bool = False


@dataclass(frozen=True)
class TESSAV31InferencePolicyV1:
    maximum_new_tokens: int = 131072

    def validate(self) -> None:
        if isinstance(self.maximum_new_tokens, bool) or int(self.maximum_new_tokens) <= 0:
            raise ValueError("TESSA_V31_INFERENCE_MAX_TOKENS_INVALID")


def load_tessa_v31_inference_v1(
    manifest: dict,
    *,
    device: torch.device | str = "cpu",
) -> TESSAV31InferenceBindingV1:
    """Load one exact Attempt-bound TESSA V3.1 ModelRelease for inference only."""

    ref = require_materialized_model_release(
        manifest,
        family=TESSA_MODEL_FAMILY_V31,
        architecture_id=TESSA_ARCHITECTURE_ID_V31,
    )
    model = load_checkpoint_v1(ref.checkpoint_path, device=device)
    if model.architecture_id != ref.architecture_id:
        raise RuntimeError("TESSA_V31_INFERENCE_ARCHITECTURE_BINDING_DRIFT")
    return TESSAV31InferenceBindingV1(model_ref=ref, model=model)


@torch.no_grad()
def encode_tessa_memory_v31(
    binding: TESSAV31InferenceBindingV1,
    inputs: TESSAV31InferenceInputsV1,
) -> torch.Tensor:
    inputs.validate(binding.model)
    memory = binding.model.encode_memory(
        scene_source=inputs.scene_source,
        surface_source=inputs.surface_source,
        presentation_source=inputs.presentation_source,
        surface_valid_mask=inputs.surface_valid_mask,
        presentation_valid_mask=inputs.presentation_valid_mask,
    )
    expected = (
        inputs.scene_source.shape[0],
        binding.model.config.total_conditioning_tokens,
        binding.model.config.d_model,
    )
    if tuple(memory.shape) != expected or not bool(torch.isfinite(memory).all()):
        raise RuntimeError("TESSA_V31_INFERENCE_MEMORY_INVALID")
    return memory


@torch.no_grad()
def generate_topology_proposal_v31(
    binding: TESSAV31InferenceBindingV1,
    inputs: TESSAV31InferenceInputsV1,
    *,
    policy: TESSAV31InferencePolicyV1 = TESSAV31InferencePolicyV1(),
) -> TESSAV31TopologyProposalV1:
    """Greedy exact rolling-window V3.1 inference; proposal evidence only."""

    policy.validate()
    memory = encode_tessa_memory_v31(binding, inputs)
    if memory.shape[0] != 1:
        raise RuntimeError("TESSA_V31_INFERENCE_REQUIRES_SINGLE_ASSET")
    model = binding.model
    cfg = model.config
    tokens = [int(cfg.BOS)]
    grammar = TESSAGrammarStateV2()
    incremental = IncrementalDecoderV31(model, memory)
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"

    for _ in range(int(policy.maximum_new_tokens)):
        logits = incremental.step(tokens[-1])[0]
        allowed = grammar.allowed_token_mask(config=cfg, device=memory.device)
        if not bool(allowed.any()):
            raise RuntimeError("TESSA_V31_INFERENCE_NO_ALLOWED_TOKEN")
        token = int(logits.float().masked_fill(~allowed, float("-inf")).argmax().item())
        tokens.append(token)
        grammar.advance(token, config=cfg)
        if grammar.complete:
            stop_reason = "EOS"
            break

    court = evaluate_decoded_topology_v2(tokens, config=cfg)
    return TESSAV31TopologyProposalV1(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(grammar.complete),
        stop_reason=stop_reason,
        model_release_id=binding.model_ref.model_release_id,
        checkpoint_content_sha256=binding.model_ref.checkpoint_content_sha256,
    )
