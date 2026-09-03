from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from experiments.geppetto_arachne_r6_20260901.arachne_geometry_v2 import PAIR_GEOMETRY_CONTRACT_V2
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
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import _bound_teacher_and_rest
from experiments.geppetto_arachne_r6_20260901.test_skin_field_codec_a0_encoder_bypass_bound_diagnostic import _decoded_metrics
from experiments.geppetto_arachne_r6_20260901.test_skin_field_codec_a0_pair_geometry_bound_diagnostic import _decode_with_geometry
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _setup(w: Witness):
    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    return (
        cond,
        teacher,
        rest,
        permutation,
        _probe_transforms(len(w.joints)),
        torch.tensor(cond.surface_features, dtype=torch.float32),
        torch.tensor(cond.joint_features, dtype=torch.float32),
        torch.tensor(cond.pair_geometry, dtype=torch.float32),
        torch.tensor(cond.surface_mask, dtype=torch.bool),
        torch.tensor(cond.joint_mask, dtype=torch.bool),
    )


def _run_current(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cond, teacher, rest, permutation, transforms, sf, jf, geometry, sm, jm = _setup(w)
    codec = _codec()
    opt = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0)
    trace = []
    stable = 0
    pass_step = None
    metrics = None
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, opt, sf, jf, teacher, sm, jm, rest, transforms)
        sched.step()
        if step == 1 or step % CHECK_EVERY == 0:
            with torch.no_grad():
                codec.eval()
                out = codec(sf, jf, teacher, sm, jm)
                metrics = _decoded_metrics(out.decoded_weights, teacher, rest, transforms, sm, jm)
            ok = _a0_pass(metrics)
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "lr": float(opt.param_groups[0]["lr"]), **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break
    return {
        "lane": "CURRENT_NO_PAIR_CONTRACT",
        "surface_permutation": permutation,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
    }


def _run_full_contract(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cond, teacher, rest, permutation, transforms, sf, jf, geometry, sm, jm = _setup(w)
    codec = _codec()
    geo_proj = nn.Linear(len(PAIR_GEOMETRY_CONTRACT_V2), codec.config.hidden_dim, bias=False)
    nn.init.zeros_(geo_proj.weight)  # exact current function at step zero
    params = list(codec.parameters()) + list(geo_proj.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0)
    trace = []
    stable = 0
    pass_step = None
    metrics = None
    for step in range(1, A0_MAX_STEPS + 1):
        codec.train()
        geo_proj.train()
        opt.zero_grad(set_to_none=True)
        latents = codec.encode_teacher_weights(sf, jf, teacher, sm, jm)
        decoded = _decode_with_geometry(codec, geo_proj, latents, sf, jf, geometry, sm, jm)
        recon = skin_field_codec_loss_v1(decoded, teacher, sm, jm)
        deform = codec_deformation_loss_v1(decoded, teacher, rest, transforms, sm, jm)
        (recon["total"] + deform["deformation_mse"]).backward()
        opt.step()
        sched.step()
        if step == 1 or step % CHECK_EVERY == 0:
            with torch.no_grad():
                codec.eval()
                geo_proj.eval()
                latents = codec.encode_teacher_weights(sf, jf, teacher, sm, jm)
                decoded = _decode_with_geometry(codec, geo_proj, latents, sf, jf, geometry, sm, jm)
                metrics = _decoded_metrics(decoded, teacher, rest, transforms, sm, jm)
            ok = _a0_pass(metrics)
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "lr": float(opt.param_groups[0]["lr"]), **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break
    return {
        "lane": "ZERO_INIT_FULL_CANONICAL_PAIR_CONTRACT",
        "geometry_channels": list(PAIR_GEOMETRY_CONTRACT_V2),
        "surface_permutation": permutation,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
        "geo_projection_abs_max": float(geo_proj.weight.detach().abs().max().cpu()),
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_codec_a0_complete_v2_pair_contract(witness: Witness):
    current = _run_current(witness)
    full = _run_full_contract(witness)
    result = {"witness": witness.name, "seed": witness.seed, "current": current, "full_pair_contract": full}
    print("SKIN_FIELD_CODEC_A0_FULL_PAIR_CONTRACT_BOUND=" + json.dumps(result, sort_keys=True))
    assert current["surface_permutation"] == full["surface_permutation"]
    assert full["geometry_channels"] == list(PAIR_GEOMETRY_CONTRACT_V2)
