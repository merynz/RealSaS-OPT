from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.mira.skin_fsq_codec_v6 import (
    MIRASkinFSQCodecConfig,MIRASkinFSQCodecV6,
)
from models.rss_vnext.skin_pose_diagnostic_v1 import (
    evaluate_mira_skin_pose_diagnostic_v1,
)


def _inputs():
    torch.manual_seed(207)
    codec=MIRASkinFSQCodecV6(MIRASkinFSQCodecConfig(
        vertex_feature_dim=5,joint_feature_dim=7,hidden_dim=16,
        tokens_per_joint=2,levels=(3,3),
    )).eval()
    b,v,j,p=1,4,2,3
    vf=torch.randn(b,v,5)
    jf=torch.randn(b,j,7)
    ids=torch.tensor([[[0,4],[8,2]]],dtype=torch.long)
    rest=torch.tensor([[[0.,0.,0.],[.2,.5,0.],[1.,0.,0.],[1.,1.,0.]]])
    gt=torch.tensor([[[1.,0.],[.7,.3],[.3,.7],[0.,1.]]])
    matrices=torch.eye(4).expand(b,p,j,4,4).clone()
    matrices[0,1,0,0,3]=.5;matrices[0,1,1,0,3]=-.7
    matrices[0,2,0,1,3]=.8;matrices[0,2,1,1,3]=-.5
    hard=codec.decode_tokens(ids,vf,jf).detach()
    return dict(codec=codec,hard_token_ids=ids,vertex_features=vf,
        joint_features=jf,rest_xyz=rest,source_dense_weights=gt,
        training_expected_weights=hard.clone(),source_skin_matrices=matrices,
        qualified_vertex_mask=torch.ones((b,v),dtype=torch.bool),
        qualified_pose_mask=torch.ones((b,p),dtype=torch.bool),
        source_skin_protocol_sha256="a"*64,
        teacher_pose_protocol_sha256="b"*64,
        codec_checkpoint_sha256="c"*64),hard


def test_true_hard_codec_replay_matches_training_weight_in_real_lbs():
    kwargs,hard=_inputs()
    rep=evaluate_mira_skin_pose_diagnostic_v1(**kwargs)
    assert rep.authority.startswith("RESEARCH_ONLY")
    assert rep.valid_vertex_count==4 and rep.valid_pose_vertex_count==12
    assert rep.expected_to_hard_weight_max_abs < 1e-8
    assert rep.expected_to_hard_lbs_max_error < 1e-8
    assert rep.hard_to_teacher_lbs_mean_error > 0
    assert 0<=rep.hard_support_precision<=1
    assert 0<=rep.hard_support_recall<=1
    assert 0<=rep.hard_false_influence_mass<=1
    assert 0<=rep.hard_active_influence_mass_recall<=1


def test_soft_weights_are_not_a_free_discrete_pose_pass():
    kwargs,_=_inputs()
    kwargs["training_expected_weights"]=kwargs["source_dense_weights"].clone()
    rep=evaluate_mira_skin_pose_diagnostic_v1(**kwargs)
    assert rep.expected_to_teacher_lbs_mean_error < 1.e-8
    assert rep.hard_to_teacher_lbs_mean_error > 1.e-5
    assert rep.expected_to_hard_lbs_mean_error > 1.e-5


def test_unqualified_vertex_and_pose_do_not_become_zero_or_nan_truth():
    kwargs,_=_inputs()
    kwargs["qualified_vertex_mask"][0,-1]=False
    kwargs["qualified_pose_mask"][0,-1]=False
    kwargs["source_dense_weights"][0,-1]=float("nan")
    kwargs["source_skin_matrices"][0,-1]=float("nan")
    rep=evaluate_mira_skin_pose_diagnostic_v1(**kwargs)
    assert rep.valid_vertex_count==3
    assert rep.valid_pose_vertex_count==6
    assert rep.hard_to_teacher_lbs_mean_error>=0
    assert rep.hard_to_teacher_lbs_max_error>=0


@pytest.mark.parametrize("name",[
    "bad_token","wrong_shape","unknown_loss","unadmitted_codec",
    "broken_simplex","bad_receipt","no_pose"
])
def test_hard_fsq_court_fails_closed_on_domain_drift(name):
    kwargs,_=_inputs()
    if name=="bad_token":
        kwargs["hard_token_ids"][0,0,0]=999
    if name=="wrong_shape":
        kwargs["training_expected_weights"]=torch.ones(1,4,3)
    if name=="unknown_loss":
        kwargs["source_dense_weights"][0,0,0]=float("nan")
    if name=="unadmitted_codec":
        kwargs["codec"].train()
    if name=="broken_simplex":
        kwargs["training_expected_weights"][0,0]=.2
    if name=="bad_receipt":
        kwargs["codec_checkpoint_sha256"]="not-a-hash"
    if name=="no_pose":
        kwargs["qualified_pose_mask"][:]=False
    with pytest.raises((ValueError,RuntimeError)):
        evaluate_mira_skin_pose_diagnostic_v1(**kwargs)
