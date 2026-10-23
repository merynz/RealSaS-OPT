from __future__ import annotations

from dataclasses import dataclass
import io
import json
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from compiler.realsas_compiler_services.platform_worker.iris_inference_v1 import (
    encode_scene_v1,
    load_iris_vnext_inference_v1,
)
from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v1 import (
    SHAPE_INPUT_ARTIFACT_TYPE_V1,
    decode_shape_input_v1,
    encode_shape_input_v1,
)
from compiler.realsas_compiler_services.platform_worker.tessa_inference_v1 import (
    TESSAInferencePolicyV1,
    build_inputs_from_iris_v1,
    generate_topology_proposal_v1,
    load_tessa_v3_inference_v1,
)
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1
from models.iris.vnext.surface_v1 import IRISSurfaceEvidenceV1, extract_surface_evidence_v1


SHAPE_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductShapeInference.v1"
SHAPE_OUTPUT_ARTIFACT_TYPE_V1 = "RealSaS.ShapeProposal"
SHAPE_OUTPUT_AUTHORITY_V1 = "LEARNED_SHAPE_PROPOSAL_ONLY"


@dataclass(frozen=True)
class ShapeInferenceResultV1:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.ShapeInferenceResult.v1"


def _tensor_numpy(value: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    out = value.detach().cpu().contiguous().numpy()
    return out if dtype is None else out.astype(dtype, copy=False)


def _proposal_metadata(*, iris_binding: Any, tessa_binding: Any, proposal: Any) -> dict[str, Any]:
    court = proposal.topology_court.to_dict()
    return {
        "schema": "RealSaS.ShapeProposal.v1",
        "authority": SHAPE_OUTPUT_AUTHORITY_V1,
        "canonical_mesh_authority": False,
        "mesh_authority": False,
        "temporary_iris_mesh_serialized": False,
        "completed": bool(proposal.completed),
        "stop_reason": str(proposal.stop_reason),
        "topology_court": court,
        "models": {
            "IRIS": {
                "model_release_id": iris_binding.model_ref.model_release_id,
                "checkpoint_content_sha256": iris_binding.model_ref.checkpoint_content_sha256,
            },
            "TESSA": {
                "model_release_id": tessa_binding.model_ref.model_release_id,
                "checkpoint_content_sha256": tessa_binding.model_ref.checkpoint_content_sha256,
            },
        },
    }


def encode_shape_proposal_v1(
    *,
    iris_binding: Any,
    tessa_binding: Any,
    surface: IRISSurfaceEvidenceV1,
    proposal: Any,
) -> bytes:
    metadata = _proposal_metadata(
        iris_binding=iris_binding,
        tessa_binding=tessa_binding,
        proposal=proposal,
    )
    mechanical = surface.mechanical
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
        token_ids=np.asarray(proposal.token_ids, dtype=np.int64),
        surface_xyz_tessa=_tensor_numpy(surface.surface_xyz_tessa, np.float32),
        surface_normal=_tensor_numpy(surface.surface_normal, np.float32),
        surface_h=_tensor_numpy(surface.surface_h, np.float32),
        stretch_major=_tensor_numpy(mechanical["stretch_major"], np.float32),
        stretch_minor=_tensor_numpy(mechanical["stretch_minor"], np.float32),
        tangent_direction_2theta=_tensor_numpy(mechanical["tangent_direction_2theta"], np.float32),
        bend_demand=_tensor_numpy(mechanical["bend_demand"], np.float32),
        rigidity_probability=_tensor_numpy(mechanical["rigidity_probability"], np.float32),
        contact_persistence_probability=_tensor_numpy(
            mechanical["contact_persistence_probability"], np.float32
        ),
        contact_persistence_valid=_tensor_numpy(
            surface.contact_persistence_valid.to(torch.uint8), np.uint8
        ),
    )
    return stream.getvalue()


def decode_shape_proposal_metadata_v1(payload: bytes) -> dict[str, Any]:
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata_json"].reshape(-1)[0]))
            token_ids = np.asarray(bundle["token_ids"], dtype=np.int64)
    except Exception as exc:
        raise RuntimeError("SHAPE_PROPOSAL_DECODE_FAILED") from exc
    if metadata.get("schema") != "RealSaS.ShapeProposal.v1":
        raise RuntimeError("SHAPE_PROPOSAL_SCHEMA_DRIFT")
    if metadata.get("authority") != SHAPE_OUTPUT_AUTHORITY_V1:
        raise RuntimeError("SHAPE_PROPOSAL_AUTHORITY_DRIFT")
    if metadata.get("canonical_mesh_authority") is not False or metadata.get("mesh_authority") is not False:
        raise RuntimeError("SHAPE_PROPOSAL_MESH_AUTHORITY_FORBIDDEN")
    metadata["token_count"] = int(token_ids.size)
    return metadata


def run_shape_inference_v1(
    manifest: dict[str, Any],
    inputs: IRISVNextSceneInputsV1,
    *,
    extractor: Any,
    device: torch.device | str = "cpu",
    maximum_new_tokens: int = 131072,
) -> ShapeInferenceResultV1:
    """Run exact Attempt-bound IRIS -> Surface -> TESSA inference.

    The output is rich learned Shape evidence only. No canonical Mesh is minted
    here; the later Compiler Mesh capability remains the sole geometry authority.
    """
    if isinstance(maximum_new_tokens, bool) or int(maximum_new_tokens) <= 0 or int(maximum_new_tokens) > 131072:
        raise ValueError("SHAPE_MAXIMUM_NEW_TOKENS_INVALID")
    inputs.validate()
    timings: dict[str, float] = {}
    total_start = perf_counter()

    start = perf_counter()
    iris_binding = load_iris_vnext_inference_v1(manifest, device=device)
    timings["iris_load_seconds"] = perf_counter() - start

    start = perf_counter()
    scene = encode_scene_v1(iris_binding, inputs)
    timings["iris_scene_seconds"] = perf_counter() - start

    start = perf_counter()
    surface = extract_surface_evidence_v1(
        iris_binding.model,
        scene,
        extractor,
        resolution=int(iris_binding.model.config.extraction_training_resolution),
        witness_count=int(iris_binding.model.config.surface_witness_count),
    )
    timings["surface_seconds"] = perf_counter() - start

    start = perf_counter()
    tessa_binding = load_tessa_v3_inference_v1(manifest, device=device)
    timings["tessa_load_seconds"] = perf_counter() - start

    start = perf_counter()
    tessa_inputs = build_inputs_from_iris_v1(iris_binding.model, surface)
    timings["tessa_input_seconds"] = perf_counter() - start

    start = perf_counter()
    proposal = generate_topology_proposal_v1(
        tessa_binding,
        tessa_inputs,
        policy=TESSAInferencePolicyV1(maximum_new_tokens=int(maximum_new_tokens)),
    )
    timings["tessa_decode_seconds"] = perf_counter() - start

    if not proposal.completed:
        raise RuntimeError("SHAPE_TOPOLOGY_PROPOSAL_INCOMPLETE:" + str(proposal.stop_reason))
    court = proposal.topology_court
    if not court.grammar_valid or court.decode_failure is not None or not court.finite_geometry or not court.within_scale_policy:
        raise RuntimeError("SHAPE_TOPOLOGY_PROPOSAL_INVALID")

    payload = encode_shape_proposal_v1(
        iris_binding=iris_binding,
        tessa_binding=tessa_binding,
        surface=surface,
        proposal=proposal,
    )
    timings["total_seconds"] = perf_counter() - total_start
    diagnostics = {
        "schema": "RealSaS.ShapeInferenceDiagnostics.v1",
        "authority": SHAPE_OUTPUT_AUTHORITY_V1,
        "canonical_mesh_authority": False,
        "token_count": len(proposal.token_ids),
        "surface_witness_count": int(surface.surface_xyz_tessa.shape[1]),
        "timings": {key: float(value) for key, value in timings.items()},
    }
    return ShapeInferenceResultV1(payload=payload, diagnostics=diagnostics)


# Kept importable for small contract tests which want a lightweight fake proposal.
def _test_proposal(**kwargs: Any) -> Any:
    return SimpleNamespace(**kwargs)
