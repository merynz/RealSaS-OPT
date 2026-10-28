from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v31 import (
    decode_shape_input_v31,
    encode_shape_input_v31,
)
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.v31.checkpoint_v31 import load_checkpoint_v31 as load_iris_checkpoint_v31
from models.iris.v31.checkpoint_v31 import write_checkpoint_v31 as write_iris_checkpoint_v31
from models.iris.v31.inference_input_v31 import IRISV31SceneInputs
from models.iris.v31.model_v31 import IRISConfigV31, IRISV31
from models.iris.v31.surface_contract_v31 import (
    canonical_xyz_fps_v31,
    iris_to_tessa_coordinates_v31,
    tessa_to_iris_coordinates_v31,
)
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v31.checkpoint_v31 import load_checkpoint_v31 as load_tessa_checkpoint_v31
from models.tessa.v31.checkpoint_v31 import write_checkpoint_v31 as write_tessa_checkpoint_v31
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


def _camera() -> torch.Tensor:
    row = torch.tensor(
        [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0,
        ],
        dtype=torch.float32,
    )
    return row.reshape(1, 1, 13).expand(1, 8, 13).clone()


def _scene_inputs() -> IRISV31SceneInputs:
    return IRISV31SceneInputs(
        final=torch.zeros(1, 8, 37, 37, 384, dtype=torch.float32),
        layer2=torch.zeros(1, 8, 37, 37, 384, dtype=torch.float32),
        alpha=torch.zeros(1, 8, 37, 37, 196, dtype=torch.float32),
        camera_features=_camera(),
        foundation_authority_sha256=DINOv2FoundationAuthorityV2().authority_hash,
        alpha_resampling_policy_sha256="a" * 64,
    )


def test_v31_shape_input_roundtrip_binds_alpha_policy_without_selecting_it() -> None:
    source = _scene_inputs()
    payload = encode_shape_input_v31(source)
    decoded = decode_shape_input_v31(payload)
    assert torch.equal(decoded.final, source.final)
    assert torch.equal(decoded.layer2, source.layer2)
    assert torch.equal(decoded.alpha, source.alpha)
    assert torch.equal(decoded.camera_features, source.camera_features)
    assert decoded.foundation_authority_sha256 == source.foundation_authority_sha256
    assert decoded.alpha_resampling_policy_sha256 == "a" * 64

    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        assert "alpha_resampling_policy_sha256" in bundle.files
        assert "native_rgba" not in bundle.files
        assert "teacher" not in bundle.files

    bad = IRISV31SceneInputs(
        final=source.final,
        layer2=source.layer2,
        alpha=source.alpha,
        camera_features=source.camera_features,
        foundation_authority_sha256=source.foundation_authority_sha256,
        alpha_resampling_policy_sha256="UNSEALED",
    )
    with pytest.raises(ValueError, match="IRIS_V31_INPUT_ALPHA_RESAMPLING_POLICY_HASH"):
        bad.validate()


def test_v31_coordinates_are_explicit_inverse_and_xyz_fps_is_permutation_invariant() -> None:
    tessa = torch.tensor(
        [[-0.5, 0.0, 0.5], [0.25, -0.125, 0.0]], dtype=torch.float32
    )
    iris = tessa_to_iris_coordinates_v31(tessa)
    assert torch.equal(iris, tessa * 2.0)
    assert torch.equal(iris_to_tessa_coordinates_v31(iris), tessa)

    points = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    first = canonical_xyz_fps_v31(points, count=4)
    permutation = torch.tensor([6, 3, 0, 5, 2, 1, 4])
    second = canonical_xyz_fps_v31(points[permutation], count=4)
    first_set = sorted(tuple(float(x) for x in row) for row in first.tolist())
    second_set = sorted(tuple(float(x) for x in row) for row in second.tolist())
    assert first_set == second_set


def test_v31_iris_checkpoint_is_exact_architecture_and_inference_only(tmp_path: Path) -> None:
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
    path = write_iris_checkpoint_v31(tmp_path / "iris-v31.pt", model)
    loaded = load_iris_checkpoint_v31(path)
    assert loaded.architecture_id == "RealSaS.IRIS.DenseObservationEvidence.v3_1"
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.config.mechanical_channels == ("stretch_major",)
    assert loaded.config.pair_channels == ("relative_depth",)


def test_v31_tessa_checkpoint_binds_attention_semantics_and_backend(tmp_path: Path) -> None:
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
    path = write_tessa_checkpoint_v31(tmp_path / "tessa-v31.pt", model)
    loaded = load_tessa_checkpoint_v31(path)
    assert loaded.architecture_id == "RealSaS.TESSA.Structuralizer.v3_1"
    assert loaded.attention_semantics == "ROLLING_CAUSAL_SELF_INCLUSIVE_V31"
    assert loaded.decoder[0].self_attn.backend == "SDPA"
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
