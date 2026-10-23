from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_axis_runtime_checkpoint_packs_full_inference_state(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.axis.runtime_checkpoint_v1 import (
        AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1,
        AXIS_V541_ARCHITECTURE_ID,
        build_axis_runtime_checkpoint_v1,
        load_axis_runtime_model_v1,
    )

    model = AXISBFSDiscreteXYZV541()
    full_state = model.state_dict()
    base_state = {
        key: value.detach().cpu().clone()
        for key, value in full_state.items()
        if not key.startswith("coord_token_delta.")
    }
    coord_delta = {
        key: value.detach().cpu().clone()
        for key, value in full_state.items()
        if key.startswith("coord_token_delta.")
    }

    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "model": base_state,
            "config_hash": model.config.config_hash,
            "step": 896,
            "historical_checkpoint_loaded": False,
            "teacher_feedback_used": False,
        },
        base_path,
    )
    v53_path = tmp_path / "v53.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41TreeGrammarProgress.v5_3",
            "base_checkpoint_sha256": _sha(base_path),
            "trainable_delta": {},
        },
        v53_path,
    )
    v541_path = tmp_path / "v541.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1",
            "architecture": AXIS_V541_ARCHITECTURE_ID,
            "base_checkpoint_sha256": _sha(base_path),
            "v53_progress_sha256": _sha(v53_path),
            "hard_causal_parent_authority": True,
            "compiler_tree_reselection": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
            "coord_bins_per_axis": 256,
            "diffusion_runtime_authority": False,
            "diffusion_invocation_count": 0,
            "trainable_model_delta": coord_delta,
            "terminal_eval": {"pass": True},
        },
        v541_path,
    )

    packed = build_axis_runtime_checkpoint_v1(
        base_checkpoint_path=base_path,
        v53_progress_path=v53_path,
        v541_composite_path=v541_path,
    )
    assert packed["schema"] == AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1
    assert packed["architecture_id"] == AXIS_V541_ARCHITECTURE_ID
    assert packed["runtime_rules"]["self_contained_inference_state"] is True
    assert packed["runtime_rules"]["diffusion_runtime_authority"] is False
    assert not any(key.startswith("diffusion.") for key in packed["state_dict"])

    runtime_path = tmp_path / "runtime.pt"
    torch.save(packed, runtime_path)
    loaded, payload = load_axis_runtime_model_v1(runtime_path)
    assert payload["schema"] == AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1
    assert loaded._runtime_diffusion_retired is True
    assert not any(key.startswith("diffusion.") for key in loaded.state_dict())


def test_axis_runtime_loader_rejects_historical_partial_composite(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from models.axis.runtime_checkpoint_v1 import load_axis_runtime_model_v1

    partial = tmp_path / "historical-v541-composite.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1",
            "architecture": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
            "trainable_model_delta": {},
        },
        partial,
    )
    with pytest.raises(RuntimeError, match="AXIS_RUNTIME_CHECKPOINT_SCHEMA_DRIFT"):
        load_axis_runtime_model_v1(partial)


def test_axis_runtime_packer_rejects_component_sha_drift(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.axis.runtime_checkpoint_v1 import build_axis_runtime_checkpoint_v1

    model = AXISBFSDiscreteXYZV541()
    state = model.state_dict()
    base = tmp_path / "base.pt"
    torch.save(
        {
            "model": {k: v for k, v in state.items() if not k.startswith("coord_token_delta.")},
            "config_hash": model.config.config_hash,
        },
        base,
    )
    v53 = tmp_path / "v53.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41TreeGrammarProgress.v5_3",
            "base_checkpoint_sha256": "0" * 64,
            "trainable_delta": {},
        },
        v53,
    )
    v541 = tmp_path / "v541.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1",
            "architecture": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
            "base_checkpoint_sha256": _sha(base),
            "v53_progress_sha256": _sha(v53),
            "hard_causal_parent_authority": True,
            "compiler_tree_reselection": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
            "coord_bins_per_axis": 256,
            "diffusion_runtime_authority": False,
            "diffusion_invocation_count": 0,
            "trainable_model_delta": {
                k: v for k, v in state.items() if k.startswith("coord_token_delta.")
            },
            "terminal_eval": {"pass": True},
        },
        v541,
    )
    with pytest.raises(RuntimeError, match="V53_BASE_SHA_DRIFT"):
        build_axis_runtime_checkpoint_v1(
            base_checkpoint_path=base,
            v53_progress_path=v53,
            v541_composite_path=v541,
        )
