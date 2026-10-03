from __future__ import annotations

import torch

from models.geppetto.v2.geppetto_candidate_v2 import (
    GeppettoCandidateConfigV2,
    GeppettoCandidateV2,
)
from models.geppetto.v2.geppetto_diffusion_challenger_v1 import (
    GeppettoCandidateDiffusionLocusV1,
    GeppettoDiffusionChallengerConfigV1,
)
from models.geppetto.v2.joint_locus_diffusion_v1 import (
    ConditionalDiffusionLocusConfigV1,
)


def _configs():
    base=GeppettoCandidateConfigV2(
        model_dim=48,
        knn_k=2,
        local_layers=1,
        global_layers=1,
        decoder_layers=1,
        attention_heads=6,
        feedforward_dim=96,
        support_topk=2,
        position_modes=3,
        parent_pair_chunk=8,
        dropout=0.0,
        position_scale=1.25,
    )
    d1=GeppettoDiffusionChallengerConfigV1(
        diffusion=ConditionalDiffusionLocusConfigV1(
            condition_dim=48,
            hidden_dim=64,
            time_dim=16,
            train_steps=32,
            sample_steps=4,
            sample_count=3,
            position_scale=1.25,
        ),
        inference_seed=1234,
    )
    return base,d1


def test_d1_accepts_historical_base_checkpoint_with_only_diffusion_missing():
    torch.manual_seed(1)
    base_cfg,d1_cfg=_configs()
    base=GeppettoCandidateV2(base_cfg)
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg)
    report=d1.load_frozen_base_state_dict(base.state_dict())
    assert report["base_checkpoint_compatible"] is True
    assert report["unexpected_keys"]==()
    assert report["missing_diffusion_keys"]
    assert all(
        k.startswith("diffusion_locus_head.")
        for k in report["missing_diffusion_keys"]
    )


def test_d1_head_only_freeze_scope_is_exact():
    base_cfg,d1_cfg=_configs()
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg)
    report=d1.freeze_base_for_head_fit()
    assert report["trainable"]
    assert all(
        n.startswith("diffusion_locus_head.")
        for n in report["trainable"]
    )
    assert all(
        not p.requires_grad
        for n,p in d1.named_parameters()
        if not n.startswith("diffusion_locus_head.")
    )
    assert all(
        p.requires_grad
        for n,p in d1.named_parameters()
        if n.startswith("diffusion_locus_head.")
    )


def test_d1_changes_only_locus_dependent_outputs_at_fixed_base_state():
    torch.manual_seed(7)
    base_cfg,d1_cfg=_configs()
    base=GeppettoCandidateV2(base_cfg).eval()
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg).eval()
    d1.load_frozen_base_state_dict(base.state_dict())

    B,N=1,6
    features=torch.randn(B,N,base_cfg.surface_feature_dim)
    positions=torch.randn(B,N,3).clamp(-1.0,1.0)
    mask=torch.ones(B,N,dtype=torch.bool)

    with torch.no_grad():
        a=base(features,positions,mask,decode_steps=3)
        b=d1(features,positions,mask,decode_steps=3)

    # Latent recurrence and all heads independent of realized XYZ must be exact.
    for name in (
        "control_states",
        "existence_logits",
        "stop_logits",
        "root_logits",
        "support_presence_logits",
        "support_logits",
        "abstain_logits",
    ):
        assert torch.equal(getattr(a,name),getattr(b,name)), name

    # D1 must actually alter the locus family and therefore may alter parent
    # evidence, which explicitly consumes realized XYZ.
    assert not torch.equal(a.positions_normalized,b.positions_normalized)
    assert not torch.equal(a.parent_logits,b.parent_logits)


def test_d1_representative_is_map_real_sample_for_inherited_proposal_contract():
    torch.manual_seed(9)
    base_cfg,d1_cfg=_configs()
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg).eval()
    B,N=1,5
    features=torch.randn(B,N,base_cfg.surface_feature_dim)
    positions=torch.randn(B,N,3).clamp(-1.0,1.0)
    mask=torch.ones(B,N,dtype=torch.bool)
    with torch.no_grad():
        out=d1(features,positions,mask,decode_steps=2)
    idx=torch.argsort(
        out.position_mode_logits,dim=-1,descending=True,stable=True
    )[...,0]
    gather3=idx[...,None,None].expand(B,2,1,3)
    chosen=torch.gather(
        out.position_modes_normalized,2,gather3
    ).squeeze(2)
    assert torch.equal(chosen,out.positions_normalized)


def test_d1_forward_is_exact_seed_replay():
    torch.manual_seed(11)
    base_cfg,d1_cfg=_configs()
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg).eval()
    features=torch.randn(1,5,base_cfg.surface_feature_dim)
    positions=torch.randn(1,5,3).clamp(-1.0,1.0)
    mask=torch.ones(1,5,dtype=torch.bool)
    with torch.no_grad():
        a=d1(features,positions,mask,decode_steps=2)
        b=d1(features,positions,mask,decode_steps=2)
    assert torch.equal(a.positions_normalized,b.positions_normalized)
    assert torch.equal(a.position_modes_normalized,b.position_modes_normalized)
    assert torch.equal(a.parent_logits,b.parent_logits)
