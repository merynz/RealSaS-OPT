from __future__ import annotations

import numpy as np
import torch

from models.geppetto.v2.geppetto_candidate_v2 import (
    GeppettoCandidateConfigV2,
    GeppettoCandidateV2,
)
from models.geppetto.v2.geppetto_diffusion_challenger_v1 import (
    GeppettoCandidateDiffusionLocusV1,
    GeppettoDiffusionChallengerConfigV1,
)
from models.geppetto.v2.geppetto_diffusion_head_fit_v1 import (
    build_frozen_d0_locus_pairs_v1,
    fit_diffusion_locus_head_v1,
)
from models.geppetto.v2.joint_locus_diffusion_v1 import (
    ConditionalDiffusionLocusConfigV1,
)
from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1


def _configs():
    base=GeppettoCandidateConfigV2(
        model_dim=48,knn_k=2,local_layers=1,global_layers=1,
        decoder_layers=1,attention_heads=6,feedforward_dim=96,
        support_topk=2,position_modes=3,parent_pair_chunk=8,
        dropout=0.0,position_scale=1.25,
    )
    d1=GeppettoDiffusionChallengerConfigV1(
        diffusion=ConditionalDiffusionLocusConfigV1(
            condition_dim=48,hidden_dim=64,time_dim=16,
            train_steps=32,sample_steps=4,sample_count=3,
            position_scale=1.25,
        ),
        inference_seed=44,
    )
    return base,d1


def _base_output_and_target():
    torch.manual_seed(33)
    base_cfg,_=_configs()
    base=GeppettoCandidateV2(base_cfg).eval()
    features=torch.randn(1,6,base_cfg.surface_feature_dim)
    positions=torch.randn(1,6,3).clamp(-1.0,1.0)
    mask=torch.ones(1,6,dtype=torch.bool)
    with torch.no_grad():
        out=base(features,positions,mask,decode_steps=3)
    truth=out.positions_normalized[0,:2].detach().cpu().numpy().copy()
    truth=(truth+np.asarray([[0.08,-0.03,0.02],[-0.05,0.06,-0.01]])).clip(-1.2,1.2)
    target=GeppettoTeacherTargetV1(
        positions_normalized=truth.astype(np.float32),
        parent_indices=np.asarray([-1,0],np.int64),
        root_mask=np.asarray([1,0],bool),
        valid=True,
    )
    return base,out,target


def test_frozen_d0_pair_mapping_is_deterministic():
    _,out,target=_base_output_and_target()
    a=build_frozen_d0_locus_pairs_v1(out,[target])
    b=build_frozen_d0_locus_pairs_v1(out,[target])
    assert a.mapping_hash==b.mapping_hash
    assert a.source_rows==b.source_rows
    assert torch.equal(a.conditions,b.conditions)
    assert torch.equal(a.targets_xyz_normalized,b.targets_xyz_normalized)
    assert len(a.source_rows)==2


def test_head_only_fit_changes_only_diffusion_parameters():
    base,base_out,target=_base_output_and_target()
    base_cfg,d1_cfg=_configs()
    d1=GeppettoCandidateDiffusionLocusV1(base_cfg,d1_cfg)
    d1.load_frozen_base_state_dict(base.state_dict())
    pairs=build_frozen_d0_locus_pairs_v1(base_out,[target])

    before={
        n:p.detach().clone()
        for n,p in d1.named_parameters()
    }
    receipt=fit_diffusion_locus_head_v1(
        d1,pairs,steps=3,learning_rate=1e-3,seed=99
    )
    assert receipt["status"]=="PASS_HEAD_ONLY_FIT__AWAIT_COURT"
    assert receipt["training_row_count"]==2
    assert receipt["mapping_hash"]==pairs.mapping_hash

    changed_diffusion=0
    for n,p in d1.named_parameters():
        changed=not torch.equal(before[n],p.detach())
        if n.startswith("diffusion_locus_head."):
            changed_diffusion+=int(changed)
        else:
            assert not changed,n
    assert changed_diffusion>0


def test_pair_builder_uses_frozen_control_states_without_gradient():
    _,out,target=_base_output_and_target()
    pairs=build_frozen_d0_locus_pairs_v1(out,[target])
    assert pairs.conditions.requires_grad is False
    assert pairs.targets_xyz_normalized.requires_grad is False
