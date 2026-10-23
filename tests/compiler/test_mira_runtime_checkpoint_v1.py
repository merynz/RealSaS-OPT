from __future__ import annotations

import pytest


def test_mira_v55_promoted_readout_has_sealed_parameter_contract():
    torch = pytest.importorskip("torch")
    from models.mira.direct_readout_v55 import (
        MIRA_V55_DECODER_PARAMETER_COUNT,
        make_mira_v55_readout,
    )

    model = make_mira_v55_readout(surface_dim=640, token_dim=512, pair_dim=10)
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == MIRA_V55_DECODER_PARAMETER_COUNT
    )
    assert len(model.state_dict()) == 34

    torch.manual_seed(7)
    surface = torch.randn(5, 640)
    geom = torch.randn(5, 7)
    pair = torch.randn(5, 3, 10)
    tokens = torch.randn(3, 4, 512)
    legal = torch.ones(5, 3, dtype=torch.bool)
    legal[0, 1] = False

    with torch.no_grad():
        logits, weights = model.decode_all(
            surface,
            geom,
            pair,
            tokens,
            legal,
            chunk=2,
        )
    assert logits.shape == (5, 3)
    assert weights.shape == (5, 3)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-7, rtol=0)
    assert weights[0, 1].item() == 0.0


def test_mira_runtime_packer_fails_closed_on_component_sha_drift(tmp_path):
    pytest.importorskip("torch")
    from models.mira.runtime_checkpoint_v1 import build_mira_runtime_checkpoint_v1

    base = tmp_path / "base.pt"
    composite = tmp_path / "composite.pt"
    base.write_bytes(b"base")
    composite.write_bytes(b"composite")

    with pytest.raises(RuntimeError, match="MIRA_RUNTIME_BASE_V6_SHA_DRIFT"):
        build_mira_runtime_checkpoint_v1(
            base_v6_checkpoint_path=base,
            v55_composite_path=composite,
            expected_base_v6_sha256="0" * 64,
            expected_v55_composite_sha256="1" * 64,
        )


def test_mira_runtime_loader_rejects_historical_partial_composite(tmp_path):
    torch = pytest.importorskip("torch")
    from models.mira.runtime_checkpoint_v1 import load_mira_runtime_model_v1

    path = tmp_path / "historical.pt"
    torch.save(
        {
            "schema": "RealSaS.MIRAMDirectComposite.v5_5",
            "decoder": {},
        },
        path,
    )
    with pytest.raises(RuntimeError, match="MIRA_RUNTIME_CHECKPOINT_SCHEMA_DRIFT"):
        load_mira_runtime_model_v1(path)
