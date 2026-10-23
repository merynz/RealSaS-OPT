from __future__ import annotations

from dataclasses import dataclass

import torch

from compiler.realsas_compiler_services.platform_worker.model_binding import (
    MaterializedModelReleaseRef,
    require_materialized_model_release,
)
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.presentation_evidence_v1 import (
    RoleFreePresentationEvidenceV1,
)
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.iris.vnext.surface_v1 import IRISSurfaceEvidenceV1
from models.tessa.v2.court_v2 import (
    TESSADecodedTopologyCourtV2,
    evaluate_decoded_topology_v2,
)
from models.tessa.v2.generation_v2 import TESSAGrammarStateV2
from models.tessa.v2.windowed_incremental_generation_v2 import (
    TESSAWindowedIncrementalStateV2,
    incremental_decode_step_windowed_v2,
)
from models.tessa.v3.checkpoint_v1 import load_checkpoint_v1
from models.tessa.v3.model_v3 import TESSAStructuralizerV3


TESSA_MODEL_FAMILY = "TESSA"
TESSA_ARCHITECTURE_ID = TESSAStructuralizerV3.architecture_id
TESSA_OUTPUT_AUTHORITY = "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY"


class _InferenceSourceFormatterV1(IRISVNextStructuralizationBridgeV4):
    """Reuse the current bridge input contract without allocating a second TESSA."""

    def __init__(self, iris_model: IRISVNextEvidenceModelV3) -> None:
        torch.nn.Module.__init__(self)
        iris_model.config.validate()
        self.iris_config = iris_model.config


@dataclass(frozen=True)
class TESSAV3InferenceBindingV1:
    model_ref: MaterializedModelReleaseRef
    model: TESSAStructuralizerV3
    schema: str = "RealSaS.TESSAV3InferenceBinding.v1"


@dataclass(frozen=True)
class TESSAV3InferenceInputsV1:
    scene_source: torch.Tensor
    surface_source: torch.Tensor
    presentation_source: torch.Tensor
    surface_valid_mask: torch.Tensor
    presentation_valid_mask: torch.Tensor
    schema: str = "RealSaS.TESSAV3InferenceInputs.v1"

    def validate(self, model: TESSAStructuralizerV3) -> None:
        if not isinstance(model, TESSAStructuralizerV3):
            raise TypeError("TESSA_V3_INFERENCE_MODEL_REQUIRED")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("TESSA_V3_INFERENCE_MODEL_NOT_FROZEN")
        streams = (
            (self.scene_source, model.memory.scene.input_dim, "SCENE"),
            (self.surface_source, model.memory.surface.input_dim, "SURFACE"),
            (self.presentation_source, model.memory.presentation.input_dim, "PRESENTATION"),
        )
        batch = None
        for value, expected_dim, name in streams:
            if value.ndim != 3 or value.shape[-1] != int(expected_dim) or value.shape[1] <= 0:
                raise RuntimeError(f"TESSA_V3_INFERENCE_{name}_SHAPE_INVALID")
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"TESSA_V3_INFERENCE_{name}_NUMERIC_INVALID")
            batch = int(value.shape[0]) if batch is None else batch
            if int(value.shape[0]) != batch:
                raise RuntimeError("TESSA_V3_INFERENCE_BATCH_DRIFT")
        if self.surface_valid_mask.shape != self.surface_source.shape[:2]:
            raise RuntimeError("TESSA_V3_INFERENCE_SURFACE_MASK_SHAPE_INVALID")
        if self.presentation_valid_mask.shape != self.presentation_source.shape[:2]:
            raise RuntimeError("TESSA_V3_INFERENCE_PRESENTATION_MASK_SHAPE_INVALID")
        if not bool(self.surface_valid_mask.bool().any(dim=1).all()):
            raise RuntimeError("TESSA_V3_INFERENCE_SURFACE_MASK_EMPTY")
        if not bool(self.presentation_valid_mask.bool().any(dim=1).all()):
            raise RuntimeError("TESSA_V3_INFERENCE_PRESENTATION_MASK_EMPTY")


@dataclass(frozen=True)
class TESSATopologyProposalV1:
    token_ids: tuple[int, ...]
    topology_court: TESSADecodedTopologyCourtV2
    completed: bool
    stop_reason: str
    model_release_id: str
    checkpoint_content_sha256: str
    schema: str = "RealSaS.TESSATopologyProposal.v1"
    output_authority: str = TESSA_OUTPUT_AUTHORITY
    canonical_mesh_authority: bool = False


@dataclass(frozen=True)
class TESSAInferencePolicyV1:
    maximum_new_tokens: int = 131072

    def validate(self) -> None:
        if isinstance(self.maximum_new_tokens, bool) or int(self.maximum_new_tokens) <= 0:
            raise ValueError("TESSA_V3_INFERENCE_MAX_TOKENS_INVALID")


def load_tessa_v3_inference_v1(
    manifest: dict,
    *,
    device: torch.device | str = "cpu",
) -> TESSAV3InferenceBindingV1:
    """Load only the accepted Attempt-bound TESSA V3 ModelRelease."""

    ref = require_materialized_model_release(
        manifest,
        family=TESSA_MODEL_FAMILY,
        architecture_id=TESSA_ARCHITECTURE_ID,
    )
    model = load_checkpoint_v1(ref.checkpoint_path, device=device)
    if model.architecture_id != ref.architecture_id:
        raise RuntimeError("TESSA_V3_INFERENCE_ARCHITECTURE_BINDING_DRIFT")
    return TESSAV3InferenceBindingV1(model_ref=ref, model=model)


def build_inputs_from_iris_v1(
    iris_model: IRISVNextEvidenceModelV3,
    evidence: IRISSurfaceEvidenceV1,
    *,
    presentation: RoleFreePresentationEvidenceV1 | None = None,
) -> TESSAV3InferenceInputsV1:
    """Lower current IRIS evidence into the exact current TESSA bridge contract."""

    if not isinstance(iris_model, IRISVNextEvidenceModelV3):
        raise TypeError("TESSA_V3_INFERENCE_IRIS_MODEL_REQUIRED")
    if iris_model.training or any(parameter.requires_grad for parameter in iris_model.parameters()):
        raise RuntimeError("TESSA_V3_INFERENCE_IRIS_MODEL_NOT_FROZEN")
    evidence.validate(iris_model)
    formatter = _InferenceSourceFormatterV1(iris_model)
    scene = formatter.scene_source(evidence.scene_planes)
    surface = formatter.surface_source(
        witness_xyz=evidence.surface_xyz_tessa,
        witness_normal=evidence.surface_normal,
        witness_h=evidence.surface_h,
        witness_mechanical=evidence.mechanical,
        contact_persistence=evidence.mechanical["contact_persistence_probability"],
        contact_persistence_valid=evidence.contact_persistence_valid,
    )
    typed_presentation = formatter.presentation_source(
        reference=surface,
        presentation=presentation,
    )
    result = TESSAV3InferenceInputsV1(
        scene_source=scene,
        surface_source=surface,
        presentation_source=typed_presentation.values,
        surface_valid_mask=torch.ones(
            surface.shape[:2], device=surface.device, dtype=torch.bool
        ),
        presentation_valid_mask=typed_presentation.valid_mask.bool(),
    )
    return result


@torch.no_grad()
def encode_tessa_memory_v1(
    binding: TESSAV3InferenceBindingV1,
    inputs: TESSAV3InferenceInputsV1,
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
    if tuple(memory.shape) != tuple(expected) or not bool(torch.isfinite(memory).all()):
        raise RuntimeError("TESSA_V3_INFERENCE_MEMORY_INVALID")
    return memory


@torch.no_grad()
def generate_topology_proposal_v1(
    binding: TESSAV3InferenceBindingV1,
    inputs: TESSAV3InferenceInputsV1,
    *,
    policy: TESSAInferencePolicyV1 = TESSAInferencePolicyV1(),
) -> TESSATopologyProposalV1:
    """Greedy exact-windowed product inference; output is proposal evidence only."""

    policy.validate()
    memory = encode_tessa_memory_v1(binding, inputs)
    if memory.shape[0] != 1:
        raise RuntimeError("TESSA_V3_INFERENCE_REQUIRES_SINGLE_ASSET")
    model = binding.model
    cfg = model.config
    tokens = [int(cfg.BOS)]
    grammar = TESSAGrammarStateV2()
    incremental = TESSAWindowedIncrementalStateV2.for_model(model)
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"

    for _ in range(int(policy.maximum_new_tokens)):
        current = torch.tensor([tokens[-1]], dtype=torch.long, device=memory.device)
        logits = incremental_decode_step_windowed_v2(
            model,
            memory=memory,
            token_ids=current,
            state=incremental,
        )[0]
        allowed = grammar.allowed_token_mask(config=cfg, device=memory.device)
        if not bool(allowed.any()):
            raise RuntimeError("TESSA_V3_INFERENCE_NO_ALLOWED_TOKEN")
        token = int(logits.float().masked_fill(~allowed, float("-inf")).argmax().item())
        tokens.append(token)
        grammar.advance(token, config=cfg)
        if grammar.complete:
            stop_reason = "EOS"
            break

    court = evaluate_decoded_topology_v2(tokens, config=cfg)
    return TESSATopologyProposalV1(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(grammar.complete),
        stop_reason=stop_reason,
        model_release_id=binding.model_ref.model_release_id,
        checkpoint_content_sha256=binding.model_ref.checkpoint_content_sha256,
    )
