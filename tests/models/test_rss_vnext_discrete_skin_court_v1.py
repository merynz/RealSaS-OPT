from __future__ import annotations

"""Same-unified-checkpoint discrete SkinTokens/MIRA product evidence court.

No skin, joint or drawable capability is qualified by numerical parity alone.
The actual generated FSQ IDs, joint hidden states and both dense skin outputs
must be retained so a downstream Compiler can reproduce/deform and reject.
"""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import _dynamic
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.embedded_mira_decoder_v1 import (
    encode_embedded_mira_decoder_v1, load_embedded_mira_decoder_v1,
)
from models.rss_vnext.model_config_archive_v1 import _digest as config_sha256


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(45)
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode", ["DOT_PRODUCT_BASELINE", "POINT_QUERY_CROSS_ATTENTION"])
def test_discrete_ids_replay_from_same_pinned_embedded_decoder(tmp_path, mode):
    cfg = replace(RSSvNextConfig.tiny_for_test(), skin_decoder_mode=mode)
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    net.requires_grad_(False)
    sha = "a" * 64
    raw = encode_embedded_mira_decoder_v1(
        net, parent_unified_checkpoint_sha256=sha,
    )
    decoder, binding = load_embedded_mira_decoder_v1(
        raw,
        expected_parent_unified_checkpoint_sha256=sha,
        expected_subcheckpoint_sha256=sha256(raw).hexdigest(),
        expected_model_config_sha256=config_sha256(cfg),
        model_config=cfg,
    )
    h = torch.randn(1, 7, cfg.iris.local_feature_dim)
    xyz = torch.randn(1, 7, 3).clamp(-1., 1.)
    normals = torch.nn.functional.normalize(torch.randn(1, 7, 3), dim=-1)
    joints = torch.randn(1, 3, cfg.control_hidden_dim)
    tokens = torch.randint(binding.codebook_size, (1, 3, binding.tokens_per_joint))
    with torch.no_grad():
        expected = net.mira.decode_discrete_skin_v1(
            h, xyz, joints, tokens, vertex_normals=normals,
        )
        actual = decoder.decode_discrete_skin_v1(
            h, xyz, joints, tokens, vertex_normals=normals,
        )
    assert torch.equal(actual, expected)
    assert actual.shape == (1, 7, 3)
    assert torch.allclose(actual.sum(-1), torch.ones(1, 7), atol=1.e-6)
    assert not any(p.requires_grad for p in decoder.parameters())
    if mode == "DOT_PRODUCT_BASELINE":
        # Baseline fast path uses exactly the FSQ decoder's hard-code semantics.
        packed = torch.cat((h, xyz), dim=-1)
        assert torch.equal(actual, decoder.codec.decode_tokens(tokens, packed, joints))

    with pytest.raises(ValueError, match="FSQ_TOKEN_OUT_OF_RANGE"):
        decoder.decode_discrete_skin_v1(
            h, xyz, joints, torch.full_like(tokens, binding.codebook_size),
            vertex_normals=normals,
        )
    with pytest.raises(ValueError, match="CODE_ADDRESS_DRIFT"):
        decoder.decode_discrete_skin_v1(h, xyz, joints, tokens.float(), vertex_normals=normals)
    if mode == "POINT_QUERY_CROSS_ATTENTION":
        with pytest.raises(ValueError, match="INDEXED_NORMALS_REQUIRED"):
            decoder.decode_discrete_skin_v1(h, xyz, joints, tokens)


def _dynamic_payload():
    return {
        "vertices": [[0., 0., 0.], [1., 0., 0.]],
        "weights": [[.7, .3], [.2, .8]],
        "displacement": [[0., 0., 0.], [0., 0., 0.]],
        "readiness_logits": [[0.] * 4 for _ in range(2)],
        "authority": "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL",
        "discrete_skin_weights": [[.6, .4], [.1, .9]],
        "skin_joint_hidden_states": [[0.1, 0.2], [-0.1, -0.2]],
        "soft_to_discrete_max_abs_difference": .1,
        "soft_to_discrete_mean_abs_difference": .1,
        "soft_to_discrete_parity_qualified": False,
    }


def test_compiler_accepts_measured_soft_and_discrete_evidence_without_promoting_skin():
    data = _dynamic_payload()
    _dynamic(data, 2)
    legacy = {k: v for k, v in data.items()
              if k not in {
                  "discrete_skin_weights", "skin_joint_hidden_states",
                  "soft_to_discrete_max_abs_difference",
                  "soft_to_discrete_mean_abs_difference",
                  "soft_to_discrete_parity_qualified",
              }}
    _dynamic(legacy, 2)


@pytest.mark.parametrize("field,change,reason", [
    ("discrete_skin_weights", [[1., 0.], [.1, .9]], "PARITY_METRIC_DRIFT"),
    ("skin_joint_hidden_states", [[0.1, 0.2]], "DISCRETE_SKIN_JOINT_STATES"),
    ("soft_to_discrete_max_abs_difference", 0., "PARITY_METRIC_DRIFT"),
    ("soft_to_discrete_parity_qualified", True, "AUTHORITY_OR_SIMPLEX"),
])
def test_compiler_rejects_false_discrete_skin_claims(field, change, reason):
    candidate = deepcopy(_dynamic_payload())
    candidate[field] = change
    with pytest.raises(QualificationError, match=reason):
        _dynamic(candidate, 2)
