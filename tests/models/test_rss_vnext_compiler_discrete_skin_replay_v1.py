from __future__ import annotations

"""A real pinned MIRA decoder must replay hard AXIS IDs from saved evidence."""

from dataclasses import replace

import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_discrete_skin_replay_v1 import (
    replay_rss_vnext_discrete_skin_candidate_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_model_v6 import RigTokenKindV6
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(29)
    torch.set_num_threads(1)


def _source(mode):
    cfg = replace(RSSvNextConfig.tiny_for_test(), skin_decoder_mode=mode)
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    net.requires_grad_(False)
    n, j = 6, 2
    features = torch.randn(1, n, cfg.iris.local_feature_dim)
    xyz = torch.randn(1, n, 3).clamp(-1., 1.)
    normals = torch.nn.functional.normalize(torch.randn(1, n, 3), dim=-1)
    joint_h = torch.randn(1, j, cfg.control_hidden_dim)
    ids = torch.tensor([[[0] * cfg.skin_tokens_per_joint,
                         [cfg.axis_config().skin_vocab_size-1] * cfg.skin_tokens_per_joint]])
    with torch.no_grad():
        hard = net.mira.decode_discrete_skin_v1(
            features, xyz, joint_h, ids, vertex_normals=normals,
        )[0]
    mesh = {
        "local_features": features[0].tolist(),
        "rest_xyz": xyz[0].tolist(),
        "normals_xyz": normals[0].tolist(),
    }
    soft = hard.clone()
    soft[0, :] = torch.flip(soft[0, :], dims=(-1,))
    err = (soft - hard).abs()
    dynamic = {
        "vertices": xyz[0].tolist(),
        "weights": soft.tolist(),
        "displacement": torch.zeros(1, n, 3)[0].tolist(),
        "readiness_logits": torch.zeros(1, n, 4)[0].tolist(),
        "authority": "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL",
        "discrete_skin_weights": hard.tolist(),
        "skin_joint_hidden_states": joint_h[0].tolist(),
        "soft_to_discrete_max_abs_difference": float(err.max()),
        "soft_to_discrete_mean_abs_difference": float(err.mean()),
        "soft_to_discrete_parity_qualified": False,
    }
    events = (
        [{"kind": int(RigTokenKindV6.JOINT)} for _ in range(j)] +
        [{"kind": int(RigTokenKindV6.SKIN_CODE), "skin_code_id": int(code)}
         for code in ids.flatten().tolist()]
    )
    return net, cfg, mesh, dynamic, events


def _replay(net, cfg, mesh, dynamic, events):
    return replay_rss_vnext_discrete_skin_candidate_v1(
        mesh=mesh, dynamic=dynamic, control_events=events,
        model_config=cfg, decoder=net.mira,
        checkpoint_sha256="a"*64, inference_evidence_hash="b"*64,
        embedded_decoder_sha256="c"*64,
    )


@pytest.mark.parametrize("mode", ["DOT_PRODUCT_BASELINE", "POINT_QUERY_CROSS_ATTENTION"])
def test_model_to_compiler_replay_measures_real_discrete_consequence(mode):
    net, cfg, mesh, dynamic, events = _source(mode)
    r = _replay(net, cfg, mesh, dynamic, events)
    assert r.state == "REPLAYED_UNQUALIFIED_DISCRETE_SKIN_CANDIDATE"
    assert r.vertex_count == 6 and r.joint_count == 2
    assert r.skin_code_count == cfg.skin_tokens_per_joint * 2
    assert r.dense_weight_max_abs_replay_error < 1.e-6
    assert r.soft_to_discrete_max_abs_difference >= 0
    assert not r.canonical_mesh_authority
    assert not r.canonical_rig_authority
    assert not r.canonical_skin_authority
    assert not r.deform_motion_parity_qualified


def test_compiler_rejects_reserialised_wrong_hard_weights_and_code_order():
    net, cfg, mesh, dynamic, events = _source("DOT_PRODUCT_BASELINE")
    hard = [list(x) for x in dynamic["discrete_skin_weights"]]
    hard[0] = list(reversed(hard[0]))
    candidate = dict(dynamic)
    candidate["discrete_skin_weights"] = hard
    err = (torch.tensor(candidate["weights"]) - torch.tensor(hard)).abs()
    candidate["soft_to_discrete_max_abs_difference"] = float(err.max())
    candidate["soft_to_discrete_mean_abs_difference"] = float(err.mean())
    with pytest.raises(QualificationError, match="DISCRETE_WEIGHT_MISMATCH"):
        _replay(net, cfg, mesh, candidate, events)

    wrong = [dict(x) for x in events]
    wrong[-1]["skin_code_id"] = 0
    with pytest.raises(QualificationError, match="DISCRETE_WEIGHT_MISMATCH"):
        _replay(net, cfg, mesh, dynamic, wrong)


def test_compiler_refuses_soft_only_and_missing_joint_event():
    net, cfg, mesh, dynamic, events = _source("DOT_PRODUCT_BASELINE")
    legacy = {k: v for k,v in dynamic.items() if k not in {
        "discrete_skin_weights", "skin_joint_hidden_states",
        "soft_to_discrete_max_abs_difference", "soft_to_discrete_mean_abs_difference",
        "soft_to_discrete_parity_qualified",
    }}
    with pytest.raises(QualificationError, match="SOFT_ONLY_ABSTAIN"):
        _replay(net, cfg, mesh, legacy, events)
    wrong = [event for event in events if event["kind"] != int(RigTokenKindV6.JOINT)]
    with pytest.raises(QualificationError, match="JOINT_ORDER_DRIFT"):
        _replay(net, cfg, mesh, dynamic, wrong)
