from __future__ import annotations

import torch

from models.geppetto.v2.geppetto_candidate_v2 import (
    GeppettoCandidateConfigV2,
    GeppettoCandidateV2,
)


def test_d0_locus_interface_preserves_historical_state_dict_keys():
    torch.manual_seed(17)
    model=GeppettoCandidateV2(GeppettoCandidateConfigV2())
    keys=set(model.state_dict().keys())
    assert "position.weight" in keys
    assert "position.bias" in keys
    assert "log_sigma.weight" in keys
    assert "log_sigma.bias" in keys
    assert "position_mode_logits.weight" in keys
    assert "position_mode_logits.bias" in keys
    assert not any(k.startswith("locus_head.") for k in keys)


def test_d0_locus_interface_is_exact_legacy_formula_on_forward_states():
    torch.manual_seed(23)
    cfg=GeppettoCandidateConfigV2()
    model=GeppettoCandidateV2(cfg).eval()
    B,N=1,7
    features=torch.randn(B,N,cfg.surface_feature_dim)
    positions=torch.randn(B,N,3).clamp(-1.0,1.0)
    mask=torch.ones(B,N,dtype=torch.bool)

    with torch.no_grad():
        out=model(features,positions,mask,decode_steps=4)

    h=out.control_states
    M=cfg.position_modes
    manual_modes=torch.tanh(model.position(h).reshape(B,4,M,3))*cfg.position_scale
    manual_ls=model.log_sigma(h.detach()).reshape(B,4,M,3).clamp(-8.0,4.0)
    manual_logits=model.position_mode_logits(h)

    idx=torch.argsort(
        manual_logits,dim=-1,descending=True,stable=True
    )[...,0]
    gather3=idx[...,None,None].expand(B,4,1,3)
    manual_pos=torch.gather(manual_modes,2,gather3).squeeze(2)
    manual_rep_ls=torch.gather(manual_ls,2,gather3).squeeze(2)

    assert torch.equal(out.position_modes_normalized,manual_modes)
    assert torch.equal(out.position_mode_log_sigma,manual_ls)
    assert torch.equal(out.position_mode_logits,manual_logits)
    assert torch.equal(out.positions_normalized,manual_pos)
    assert torch.equal(out.position_log_sigma,manual_rep_ls)


def test_d0_map_tie_policy_remains_stable_first_slot():
    modes=torch.tensor([[[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]])
    sigma=torch.zeros_like(modes)
    logits=torch.tensor([[4.,4.,3.]])
    pos,rep_sigma,idx=GeppettoCandidateV2._map_representative(
        modes,sigma,logits
    )
    assert idx.tolist()==[0]
    assert torch.equal(pos,modes[:,0])
    assert torch.equal(rep_sigma,sigma[:,0])
