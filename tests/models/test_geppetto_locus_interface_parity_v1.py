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
    # Historical _decode invoked all three heads one timestep at a time on
    # [B,D]. Preserve that exact execution shape; a single [B,K,D] GEMM may
    # legitimately choose a different CPU kernel / accumulation order.
    mode_rows=[]
    sigma_rows=[]
    logit_rows=[]
    pos_rows=[]
    rep_sigma_rows=[]
    for t in range(4):
        ht=h[:,t]
        modes_t=torch.tanh(model.position(ht).reshape(B,M,3))*cfg.position_scale
        ls_t=model.log_sigma(ht.detach()).reshape(B,M,3).clamp(-8.0,4.0)
        logits_t=model.position_mode_logits(ht)
        pos_t,rep_ls_t,_=GeppettoCandidateV2._map_representative(
            modes_t,ls_t,logits_t
        )
        mode_rows.append(modes_t)
        sigma_rows.append(ls_t)
        logit_rows.append(logits_t)
        pos_rows.append(pos_t)
        rep_sigma_rows.append(rep_ls_t)
    manual_modes=torch.stack(mode_rows,1)
    manual_ls=torch.stack(sigma_rows,1)
    manual_logits=torch.stack(logit_rows,1)
    manual_pos=torch.stack(pos_rows,1)
    manual_rep_ls=torch.stack(rep_sigma_rows,1)

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
