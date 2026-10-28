from __future__ import annotations

from pathlib import Path

import torch

from compiler.realsas_compiler_services.platform_worker import iris_inference_v31 as iris_runtime
from compiler.realsas_compiler_services.platform_worker import tessa_inference_v31 as tessa_runtime
from compiler.realsas_compiler_services.platform_worker.model_binding import MaterializedModelReleaseRef
from models.iris.v31.model_v31 import IRISConfigV31, IRISV31
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


def _ref(family: str, architecture_id: str) -> MaterializedModelReleaseRef:
    return MaterializedModelReleaseRef(
        family=family,
        architecture_id=architecture_id,
        model_release_id="release-test",
        behavior_identity_sha256="a" * 64,
        checkpoint_content_sha256="b" * 64,
        checkpoint_path=Path("/tmp/test-checkpoint.pt"),
    )


def _tessa() -> TESSAStructuralizerV31:
    config = TESSAStructuralizerConfigV3(
        d_model=32,
        n_heads=4,
        decoder_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=2,
        surface_memory_tokens=2,
        presentation_memory_tokens=2,
        conditioning_token_budget=6,
        memory_source_width=16,
        memory_refinement_layers=1,
        local_attention_window=16,
        query_chunk_size=4,
        gradient_checkpointing=False,
    )
    model = TESSAStructuralizerV31(
        scene_input_dim=8,
        surface_input_dim=9,
        presentation_input_dim=10,
        config=config,
        backend="SDPA",
    )
    model.eval().requires_grad_(False)
    return model


def test_iris_v31_loader_requests_exact_architecture(monkeypatch) -> None:
    model = IRISV31(
        IRISConfigV31(
            carrier_resolution=4,
            carrier_channels=8,
            mechanical_channels=("stretch_major",),
            pair_channels=("relative_depth",),
            lift_chunk=32,
        )
    )
    model.eval().requires_grad_(False)
    seen = {}

    def require(manifest, *, family, architecture_id):
        seen.update(family=family, architecture_id=architecture_id)
        return _ref(family, architecture_id)

    monkeypatch.setattr(iris_runtime, "require_materialized_model_release", require)
    monkeypatch.setattr(iris_runtime, "load_checkpoint_v31", lambda path, device="cpu": model)
    binding = iris_runtime.load_iris_v31_inference({"run_id": "test"})
    assert seen == {
        "family": "IRIS",
        "architecture_id": "RealSaS.IRIS.DenseObservationEvidence.v3_1",
    }
    assert binding.model is model


def test_tessa_v31_loader_requests_exact_architecture_and_semantics(monkeypatch) -> None:
    model = _tessa()
    seen = {}

    def require(manifest, *, family, architecture_id):
        seen.update(family=family, architecture_id=architecture_id)
        return _ref(family, architecture_id)

    monkeypatch.setattr(tessa_runtime, "require_materialized_model_release", require)
    monkeypatch.setattr(tessa_runtime, "load_checkpoint_v31", lambda path, device="cpu": model)
    binding = tessa_runtime.load_tessa_v31_inference({"run_id": "test"})
    assert seen == {
        "family": "TESSA",
        "architecture_id": "RealSaS.TESSA.Structuralizer.v3_1",
    }
    assert binding.model.attention_semantics == "ROLLING_CAUSAL_SELF_INCLUSIVE_V31"
