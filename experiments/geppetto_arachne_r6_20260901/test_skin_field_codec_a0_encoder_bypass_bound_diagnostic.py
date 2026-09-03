from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import codec_deformation_loss_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _a0_pass,
    _codec,
    _motion_and_error_ratio,
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import _bound_teacher_and_rest
from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import row_l1_error_v1
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


@torch.no_grad()
def _decoded_metrics(decoded, teacher, rest, transforms, sm, jm) -> dict[str, float]:
    rows = row_l1_error_v1(decoded, teacher, sm)
    _, ratio = _motion_and_error_ratio(rest, teacher, decoded, transforms)
    simplex = decoded.sum(-1)
    return {
        "row_l1_p95": float(torch.quantile(rows, 0.95).cpu()),
        "deformation_ratio": ratio,
        "max_simplex_residual": float((simplex[sm] - 1.0).abs().max().cpu()),
        "negative_weight_count": int((decoded[(sm[:, :, None] & jm[:, None, :])] < 0).sum().cpu()),
    }


@torch.no_grad()
def _current_metrics(codec, sf, jf, teacher, sm, jm, rest, transforms) -> dict[str, float]:
    codec.eval()
    out = codec(sf, jf, teacher, sm, jm)
    return _decoded_metrics(out.decoded_weights, teacher, rest, transforms, sm, jm)


@torch.no_grad()
def _free_metrics(codec, free_latents, sf, jf, teacher, sm, jm, rest, transforms) -> dict[str, float]:
    codec.eval()
    decoded, _ = codec.decode_from_latents(free_latents, sf, jf, sm, jm)
    return _decoded_metrics(decoded, teacher, rest, transforms, sm, jm)


def _run_current(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    surface = _surface(w); skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    codec = _codec()
    opt = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0)
    trace=[]; stable=0; pass_step=None
    metrics = _current_metrics(codec, sf, jf, teacher, sm, jm, rest, transforms)
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec,opt,sf,jf,teacher,sm,jm,rest,transforms)
        sched.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _current_metrics(codec, sf, jf, teacher, sm, jm, rest, transforms)
            ok = _a0_pass(metrics); stable = stable + 1 if ok else 0
            trace.append({"step":step,"pass":ok,"lr":float(opt.param_groups[0]["lr"]),**metrics})
            if stable >= REQUIRED_STABLE:
                pass_step=step; break
    return {"lane":"CURRENT_TEACHER_ENCODER","surface_permutation":permutation,"pass_step":pass_step,"stable_passes":stable,"final":metrics,"trace":trace}


def _run_free_latent(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    surface = _surface(w); skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    codec = _codec()

    # Fair causal start: use exactly the latent produced by the untrained teacher
    # encoder, then sever the encoder dependence. Only the latent degrees of
    # freedom are freed; the same decoder/conditioning architecture remains.
    codec.eval()
    with torch.no_grad():
        initial_latent = codec.encode_teacher_weights(sf, jf, teacher, sm, jm).detach().clone()
    free_latents = nn.Parameter(initial_latent)

    decode_params = (
        list(codec.surface_embed.parameters())
        + list(codec.joint_embed.parameters())
        + list(codec.decoder.parameters())
        + [codec.log_temperature, free_latents]
    )
    opt = torch.optim.AdamW(decode_params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0)
    trace=[]; stable=0; pass_step=None
    metrics = _free_metrics(codec, free_latents, sf, jf, teacher, sm, jm, rest, transforms)
    for step in range(1, A0_MAX_STEPS + 1):
        codec.train(); opt.zero_grad(set_to_none=True)
        decoded, _ = codec.decode_from_latents(free_latents, sf, jf, sm, jm)
        reconstruction = skin_field_codec_loss_v1(decoded, teacher, sm, jm)
        deformation = codec_deformation_loss_v1(decoded, teacher, rest, transforms, sm, jm)
        total = reconstruction["total"] + deformation["deformation_mse"]
        total.backward(); opt.step(); sched.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _free_metrics(codec, free_latents, sf, jf, teacher, sm, jm, rest, transforms)
            ok = _a0_pass(metrics); stable = stable + 1 if ok else 0
            trace.append({"step":step,"pass":ok,"lr":float(opt.param_groups[0]["lr"]),**metrics})
            if stable >= REQUIRED_STABLE:
                pass_step=step; break
    return {"lane":"FREE_PER_JOINT_LATENT_SAME_DECODER","surface_permutation":permutation,"pass_step":pass_step,"stable_passes":stable,"final":metrics,"trace":trace}


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_codec_a0_clean_bound_encoder_bypass(witness: Witness):
    current = _run_current(witness)
    free = _run_free_latent(witness)
    result={"witness":witness.name,"seed":witness.seed,"current":current,"free_latent":free}
    print("SKIN_FIELD_CODEC_A0_ENCODER_BYPASS_BOUND="+json.dumps(result,sort_keys=True))
    assert current["surface_permutation"] == free["surface_permutation"]
    assert current["lane"] == "CURRENT_TEACHER_ENCODER"
    assert free["lane"] == "FREE_PER_JOINT_LATENT_SAME_DECODER"
