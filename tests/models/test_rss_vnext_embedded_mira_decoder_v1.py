from __future__ import annotations

"""Actual RSS unified checkpoint -> self-contained MIRA FSQ decoder.

This court verifies token meaning and exact embedded learned decoder weights.
It intentionally makes no claim that soft expected-code training inference is
equal to hard discrete token decoding, nor that the puppet is controllable.
"""

from hashlib import sha256
from io import BytesIO

import pytest
import torch

from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_checkpoint_v1 import (
    load_inference_checkpoint_v1, save_inference_checkpoint_v1,
)
from models.rss_vnext.model_config_archive_v1 import _digest as config_sha256
from models.rss_vnext.embedded_mira_decoder_v1 import (
    SCHEMA, encode_embedded_mira_decoder_v1,
    load_embedded_mira_decoder_v1,
)


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(41)
    torch.set_num_threads(1)


def _ready(tmp_path):
    cfg = RSSvNextConfig.tiny_for_test()
    encoder = lambda: RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim)
    net = RSSvNextUnifiedCharacterModel(cfg, vision_encoder=encoder()).eval()
    path = tmp_path / "rss_unified.pt"
    sha = save_inference_checkpoint_v1(path, net)
    loaded = load_inference_checkpoint_v1(
        path, expected_sha256=sha, vision_encoder=encoder(),
        product_mode=False,
    )
    raw = encode_embedded_mira_decoder_v1(
        loaded, parent_unified_checkpoint_sha256=sha,
    )
    digest = sha256(raw).hexdigest()
    return cfg, loaded, sha, raw, digest


def _load(cfg, sha, raw, digest):
    return load_embedded_mira_decoder_v1(
        raw, expected_parent_unified_checkpoint_sha256=sha,
        expected_subcheckpoint_sha256=digest,
        expected_model_config_sha256=config_sha256(cfg),
        model_config=cfg,
    )


def test_one_unified_checkpoint_exports_actual_mira_decoder_not_fake_codec_binding(tmp_path):
    cfg, net, sha, raw, digest = _ready(tmp_path)
    decoder, binding = _load(cfg, sha, raw, digest)
    assert binding.architecture_id == MIRA_V6_CODEC_ID
    assert binding.checkpoint_hash == digest
    assert binding.checkpoint_hash != sha
    assert binding.codebook_size == cfg.axis_config().skin_vocab_size
    assert binding.tokens_per_joint == cfg.skin_tokens_per_joint
    assert not decoder.training
    assert not any(p.requires_grad for p in decoder.parameters())
    for key, truth in net.mira.state_dict().items():
        assert torch.equal(decoder.state_dict()[key], truth), key

    tokens = torch.randint(
        binding.codebook_size, (1, 2, binding.tokens_per_joint),
        dtype=torch.long,
    )
    vertex_features = torch.randn(1, 9, cfg.iris.local_feature_dim + 3)
    joints = torch.randn(1, 2, cfg.control_hidden_dim)
    expected = net.mira.codec.decode_tokens(tokens, vertex_features, joints)
    decoded = decoder.codec.decode_tokens(tokens, vertex_features, joints)
    assert torch.equal(decoded, expected)
    assert decoded.shape == (1, 9, 2)
    assert torch.allclose(decoded.sum(-1), torch.ones(1, 9), atol=1.e-6)
    assert not isinstance(decoder, RSSvNextUnifiedCharacterModel)


def test_embedding_claims_proper_source_authority_and_no_false_soft_discrete_parity(tmp_path):
    cfg, net, sha, raw, digest = _ready(tmp_path)
    from io import BytesIO
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    assert payload["schema"] == SCHEMA
    assert payload["parent_unified_checkpoint_sha256"] == sha
    assert payload["unified_model_config_sha256"] == config_sha256(cfg)
    assert payload["product_skin_authority"] is False
    assert payload["soft_to_discrete_runtime_parity_qualified"] is False
    assert payload["independent_model_release_required"] is False
    assert payload["skin_decoder_mode"] == cfg.skin_decoder_mode
    assert len(payload["decoder_state_dict"]) == len(net.mira.state_dict())


@pytest.mark.parametrize("field,value,code", [
    ("product_skin_authority", True, "PARENT_OR_AUTHORITY_DRIFT"),
    ("soft_to_discrete_runtime_parity_qualified", True, "PARENT_OR_AUTHORITY_DRIFT"),
    ("independent_model_release_required", True, "PARENT_OR_AUTHORITY_DRIFT"),
    ("codebook_size", 999999, "PINNED_CONFIG_DRIFT"),
    ("parent_unified_checkpoint_sha256", "e" * 64, "PARENT_OR_AUTHORITY_DRIFT"),
])
def test_mira_decoder_refuses_reserialized_forged_model_provenance(tmp_path, field, value, code):
    cfg, net, sha, raw, digest = _ready(tmp_path)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    payload[field] = value
    stream = BytesIO()
    torch.save(payload, stream)
    forged = stream.getvalue()
    with pytest.raises(RuntimeError, match=code):
        _load(cfg, sha, forged, sha256(forged).hexdigest())


def test_embedded_mira_decoder_requires_exact_cas_checkpoint_config_and_state(tmp_path):
    cfg, net, sha, raw, digest = _ready(tmp_path)
    with pytest.raises(RuntimeError, match="CAS_BYTES_DRIFT"):
        _load(cfg, sha, raw + b"tampered", digest)
    with pytest.raises(RuntimeError, match="PARENT_OR_AUTHORITY_DRIFT"):
        _load(cfg, "e" * 64, raw, digest)
    with pytest.raises(RuntimeError, match="RESOURCE_OR_PIN_INVALID"):
        _load(cfg, sha, raw, "not-a-sha")
    modified = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    modified["decoder_state_dict"] = dict(modified["decoder_state_dict"])
    modified["decoder_state_dict"].pop(next(iter(modified["decoder_state_dict"])))
    stream = BytesIO()
    torch.save(modified, stream)
    with pytest.raises(RuntimeError, match="STATE_KEYS_DRIFT"):
        _load(cfg, sha, stream.getvalue(), sha256(stream.getvalue()).hexdigest())


def test_export_requires_frozen_unified_model(tmp_path):
    cfg = RSSvNextConfig.tiny_for_test()
    raw = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    with pytest.raises(RuntimeError, match="FROZEN_PARENT_REQUIRED"):
        encode_embedded_mira_decoder_v1(
            raw, parent_unified_checkpoint_sha256="a" * 64,
        )
