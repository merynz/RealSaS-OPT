from __future__ import annotations

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6,RigProgramEventV6,RigTokenKindV6,
)
from models.rss_vnext.contracts import RSSvNextConfig
from models.rss_vnext.dual_spatial_support_research_v1 import (
    RSSvNextDualSpatialSupportResearchV1,
)
from models.rss_vnext.dual_support_consumer_probe_v1 import (
    run_rss_vnext_dual_support_consumer_probe_v1,
)
from models.rss_vnext.unified_character_model import (
    RSSvNextUnifiedCharacterModel,RSSvNextPatchEncoderForTests,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(372)
    torch.set_num_threads(1)


def _system():
    cfg=RSSvNextConfig.tiny_for_test()
    net=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    images=torch.rand(1,2,4,8,8)
    cameras=torch.zeros((1,2,13))
    cameras[:,:,3]=cameras[:,:,7]=cameras[:,:,11]=cameras[:,:,12]=1.
    state,_,_=net.encode_observations(images,cameras)
    research=RSSvNextDualSpatialSupportResearchV1(
        cfg.iris.local_feature_dim,support_count=4,heads=4,
    ).eval()
    kinds=[
        RigTokenKindV6.BOS,
        RigTokenKindV6.JOINT,RigTokenKindV6.JOINT,
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.FRAME,RigTokenKindV6.FRAME,
        RigTokenKindV6.CONTROLS_START,RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START,
        RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,RigTokenKindV6.EOS,
    ]
    axis_prefix=AXISSequenceV6.from_events(
        [RigProgramEventV6(kind) for kind in kinds],
        device=torch.device("cpu"),
    )
    topology_prefix=torch.full((1,3),cfg.tessa.BOS,dtype=torch.long)
    return net,state,research,topology_prefix,axis_prefix


def _go(net,state,research,topology_prefix,axis_prefix,mode):
    return run_rss_vnext_dual_support_consumer_probe_v1(
        model=net,state=state,research=research,
        topology_prefix=topology_prefix,axis_prefix=axis_prefix,
        mode=mode,
    )


def test_real_product_memory_remains_exact_and_new_arms_equal_at_initialization():
    net,state,research,tp,ap=_system()
    product=_go(net,state,research,tp,ap,"PRODUCT_BASELINE")
    fixed=_go(net,state,research,tp,ap,"FIXED_VOLUME_SUPPORT")
    adaptive=_go(net,state,research,tp,ap,"LEARNED_INTERIOR_SUPPORT")
    original=net._structural_memory(state,auxiliary=None)
    assert torch.allclose(product.conditioning_memory,original,atol=0,rtol=0)
    assert fixed.conditioning_memory.shape==adaptive.conditioning_memory.shape==original.shape
    assert torch.allclose(fixed.conditioning_memory,adaptive.conditioning_memory,atol=1e-6)
    assert torch.allclose(fixed.topology_logits,adaptive.topology_logits,atol=1e-6)
    assert torch.allclose(fixed.axis_kind_logits,adaptive.axis_kind_logits,atol=1e-6)
    assert torch.allclose(
        fixed.axis_anchor_parent_logits,adaptive.axis_anchor_parent_logits,atol=1e-6,
    )
    assert product.support_candidate is None
    assert fixed.support_candidate is not None
    assert adaptive.support_candidate is not None
    assert product.scene_triplane_identity==fixed.scene_triplane_identity==adaptive.scene_triplane_identity
    assert product.topology_logits.shape==(1,3,net.cfg.tessa.vocab_size)
    assert product.axis_kind_logits.shape[0:2]==ap.kind_ids.shape
    assert adaptive.authority.startswith("RESEARCH_ONLY")


def test_position_shift_alters_actual_tessa_and_axis_consumers_without_teacher_leak():
    net,state,research,tp,ap=_system()
    fixed=_go(net,state,research,tp,ap,"FIXED_VOLUME_SUPPORT")
    with torch.no_grad():
        research.position_head[-1].bias.fill_(.22)
    adaptive=_go(net,state,research,tp,ap,"LEARNED_INTERIOR_SUPPORT")
    assert not torch.allclose(
        fixed.support_candidate.support_xyz,adaptive.support_candidate.support_xyz,
    )
    assert not torch.allclose(
        fixed.topology_logits,adaptive.topology_logits,atol=1e-8,rtol=1e-8,
    )
    assert not torch.allclose(
        fixed.axis_kind_logits,adaptive.axis_kind_logits,atol=1e-8,rtol=1e-8,
    )
    # This is sensitivity, not evidence that either result is more correct.
    for result in (fixed,adaptive):
        assert torch.isfinite(result.topology_logits).all()
        assert torch.isfinite(result.axis_kind_logits).all()
        assert torch.isfinite(result.axis_anchor_parent_logits).all()


def test_real_tessa_and_axis_losses_reach_interior_support_and_original_iris():
    net,state,research,tp,ap=_system()
    state.scene_triplanes.retain_grad()
    candidate=_go(net,state,research,tp,ap,"LEARNED_INTERIOR_SUPPORT")
    topo=torch.nn.functional.cross_entropy(
        candidate.topology_logits[:,-1],torch.tensor([9]),
    )
    axis=torch.nn.functional.cross_entropy(
        candidate.axis_kind_logits[:,-1],torch.tensor([int(RigTokenKindV6.EOS)]),
    )
    (topo+axis).backward()
    for grad in (
        state.scene_triplanes.grad,
        research.position_head[-1].weight.grad,
        research.surface_from_interior.in_proj_weight.grad,
        research.interior_from_surface.in_proj_weight.grad,
        net.tessa.memory.surface.cross.in_proj_weight.grad,
        net.axis.kind_head.weight.grad,
    ):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum()>0


@pytest.mark.parametrize("mode",[
    "TEACHER_BONE_SUPPORT","MESH_GROUND_TRUTH","AUTO_PRODUCT",
])
def test_unknown_or_privileged_modes_are_rejected(mode):
    net,state,research,tp,ap=_system()
    with pytest.raises(ValueError,match="MODE_INVALID"):
        _go(net,state,research,tp,ap,mode)


def test_research_probe_does_not_modify_product_parameters_or_state_schema():
    net,state,research,tp,ap=_system()
    before={k:v.detach().clone() for k,v in net.state_dict().items()}
    _go(net,state,research,tp,ap,"LEARNED_INTERIOR_SUPPORT")
    for key,parameter in net.state_dict().items():
        assert torch.equal(parameter,before[key])
    with pytest.raises(TypeError):
        run_rss_vnext_dual_support_consumer_probe_v1(
            model=net,state=state,research=research,topology_prefix=tp,
            axis_prefix=ap,mode="LEARNED_INTERIOR_SUPPORT",
            privileged_teacher_joints=torch.ones(1,2,3),
        )
