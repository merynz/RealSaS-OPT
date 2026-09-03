import copy
import pytest

from compiler.realsas_compiler_services.export.runtime_deploy_bake import (
    decode_runtime_deploy_bake,
    encode_runtime_deploy_bake,
)


def _frames():
    return [
        {
            "mesh_vertices_by_id": {
                "V0/body": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                "V1/body": ((10.0, 10.0), (11.0, 10.0), (10.0, 11.0)),
            },
            "render_order_by_view": {"V0": ("V0/body",), "V1": ("V1/body",)},
            "semantic_sha256": "11" * 32,
        },
        {
            "mesh_vertices_by_id": {
                "V0/body": ((0.0, 0.0), (1.2, 0.0), (0.0, 1.1)),
                "V1/body": ((10.0, 10.0), (11.2, 10.0), (10.0, 11.1)),
            },
            "render_order_by_view": {"V0": ("V0/body",), "V1": ("V1/body",)},
            "semantic_sha256": "22" * 32,
        },
    ]


def test_runtime_deploy_bake_roundtrip_is_exact_enough_for_float32_contract():
    encoded = encode_runtime_deploy_bake(
        clip_id="idle",
        duration_seconds=1.0,
        fps=1.0,
        loop=True,
        sample_times_seconds=(0.0, 1.0),
        frames=_frames(),
        evaluator_semantic_version="RealSaS.CurrentDirectionalEvaluator.v1",
        sampling_policy="qualification_owned_exact_frames",
    )
    decoded = decode_runtime_deploy_bake(encoded)
    assert decoded["clip_id"] == "idle"
    assert decoded["loop"] is True
    assert [row["semantic_sha256"] for row in decoded["frames"]] == ["11" * 32, "22" * 32]
    assert tuple(decoded["frames"][1]["mesh_vertices_by_id"]["V0/body"][1]) == pytest.approx((1.2, 0.0), abs=1e-6)
    assert decoded["frames"][0]["render_order_by_view"]["V1"] == ["V1/body"]


def test_runtime_deploy_bake_rejects_mesh_identity_drift():
    frames = _frames()
    frames[1] = copy.deepcopy(frames[1])
    frames[1]["mesh_vertices_by_id"].pop("V1/body")
    with pytest.raises(ValueError, match="mesh identity drift"):
        encode_runtime_deploy_bake(
            clip_id="idle",
            duration_seconds=1.0,
            fps=1.0,
            loop=True,
            sample_times_seconds=(0.0, 1.0),
            frames=frames,
            evaluator_semantic_version="E",
            sampling_policy="P",
        )


def test_runtime_deploy_bake_rejects_payload_tamper():
    encoded = encode_runtime_deploy_bake(
        clip_id="idle",
        duration_seconds=1.0,
        fps=1.0,
        loop=True,
        sample_times_seconds=(0.0, 1.0),
        frames=_frames(),
        evaluator_semantic_version="E",
        sampling_policy="P",
    )
    encoded = dict(encoded)
    encoded["payload_sha256"] = "00" * 32
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        decode_runtime_deploy_bake(encoded)
