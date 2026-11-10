from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6, RigTokenKindV6,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def sample(batch: int = 1):
    images = torch.rand(batch, 2, 4, 8, 8)
    camera = torch.zeros(batch, 2, 13)
    camera[:, :, 3] = 1.0
    camera[:, :, 7] = 1.0
    camera[:, :, 11] = 1.0
    camera[:, :, 12] = 1.0
    return images, camera


def model(cfg=None):
    cfg = cfg or RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def sequence_two_joints():
    kinds = [
        RigTokenKindV6.BOS,
        RigTokenKindV6.JOINT, RigTokenKindV6.JOINT,
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.FRAME, RigTokenKindV6.FRAME,
        RigTokenKindV6.CONTROLS_START,
        RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START,
        RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,
        *([RigTokenKindV6.SKIN_CODE] * 8),
        RigTokenKindV6.EOS,
    ]
    return AXISSequenceV6.from_events(
        [RigProgramEventV6(kind) for kind in kinds], device=torch.device("cpu"),
    )


def test_two_observed_images_and_shared_memory():
    net = model()
    img, camera = sample(batch=2)
    out = net(img, camera, topology_prefix=torch.full((2, 3), net.cfg.tessa.BOS))
    assert out.state.observed_rgba.shape == (2, 2, 4, 8, 8)
    assert out.state.source_view_authority == (
        "SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK",
    )
    assert out.state.scene_triplanes.shape[:3] == (2, 3, 8)
    assert out.state.local_features.shape == (2, 8, 32)
    assert out.state.semantic_logits.shape == (2, 8, 5)
    assert out.state.affordance_logits.shape == (2, 8, 6)
    assert out.structural_logits.shape == (2, 3, net.cfg.tessa.vocab_size)
    assert out.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL"


@pytest.mark.parametrize("views", [1, 3, 8])
def test_non_two_view_inputs_fail_closed(views):
    net = model()
    img, cam = sample()
    with pytest.raises(ValueError, match="TWO_SOURCE_RGBA"):
        net(torch.rand(1, views, 4, 8, 8), cam)


def test_constructed_auxiliary_never_becomes_observed_source():
    cfg = replace(RSSvNextConfig.tiny_for_test(), aux_constructed_view_latents=6)
    net = model(cfg)
    img, camera = sample()
    out = net(img, camera)
    assert out.state.observed_image_tokens.shape[1] == 2
    assert out.state.source_view_authority == (
        "SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK",
    )
    assert out.auxiliary_view_latents.shape == (1, 6, 32)
    assert out.auxiliary_view_log_uncertainty.shape == (1, 6, 1)


def test_training_gradient_reaches_shared_scene_from_structure_and_skin():
    net = model()
    img, cam = sample()
    pair = torch.tensor([[[0, 1], [2, 3]]])
    out = net(
        img, cam,
        topology_prefix=torch.full((1, 4), net.cfg.tessa.BOS),
        rig_sequence=sequence_two_joints(),
        pair_indices=pair,
    )
    assert out.skin_weight_proposal.shape == (1, 8, 2)
    assert out.skin_joint_valid_mask.tolist() == [[True, True]]
    assert out.relation_logits.shape == (1, 2, 4)
    # Non-symmetric loss drives the joint-specific FSQ route (not constant sum=1).
    skin_loss = ((out.skin_weight_proposal[:, :, 0] - 0.8) ** 2).mean()
    loss = (
        out.structural_logits.square().mean()
        + skin_loss
        + out.structural_quality_logits.square().mean()
        + out.relation_logits.square().mean()
    )
    loss.backward()
    assert net.iris.scene.scene_queries.grad is not None
    assert torch.isfinite(net.iris.scene.scene_queries.grad).all()
    assert net.tessa.memory.scene.queries.grad is not None
    assert net.axis.skin_code_head.weight.grad is not None
    assert net.mira.codec.code_decode[0].weight.grad is not None
    assert net.vision.patch.weight.grad is not None


def test_skin_logits_do_not_use_skin_code_at_current_label_position():
    net = model()
    seq = sequence_two_joints()
    img, cam = sample()
    out = net(img, cam, rig_sequence=seq)
    first_skin = int(torch.where(seq.kind_ids[0] == int(RigTokenKindV6.SKIN_CODE))[0][0])
    # Causal target at first skin position depends only on prefix through k-1.
    from copy import deepcopy
    altered = deepcopy(seq)
    altered.skin_code_ids[0, first_skin:] = 12
    changed = net(img, cam, rig_sequence=altered)
    # First code's prediction (and its contribution to weights) cannot use itself;
    # later code targets are permitted to use earlier teacher codes.
    before = out.control_logits["skin_code_logits"][0, first_skin - 1]
    after = changed.control_logits["skin_code_logits"][0, first_skin - 1]
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)


def test_teacher_free_proposal_stays_unqualified_and_never_accepts_teacher():
    net = model().eval()
    img, cam = sample()
    out = net.propose_teacher_free(img, cam, max_topology_tokens=4)
    assert out["observed_view_count"] == 2
    assert out["qualification_status"] == "UNQUALIFIED_NEURAL_PROPOSAL"
    assert out["structural_token_proposal"].shape[1] <= 4


def test_invalid_pair_access_is_bounded():
    net = model()
    img, cam = sample()
    with pytest.raises(ValueError, match="PAIR_INDEX_RANGE"):
        net(img, cam, pair_indices=torch.tensor([[[0, 999]]]))


def test_training_resume_state_dict_reproduces_outputs():
    net = model().eval()
    img, cam = sample()
    before = net(img, cam).state.geometry["field"]
    clone = model().eval()
    clone.load_state_dict(net.state_dict(), strict=True)
    after = clone(img, cam).state.geometry["field"]
    assert torch.allclose(before, after, atol=1e-6)
