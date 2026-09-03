from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import row_l1_error_v1
from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import codec_deformation_loss_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_DEFORMATION_RATIO_MAX,
    A0_MAX_STEPS,
    A0_ROW_L1_P95_MAX,
    CHECK_EVERY,
    WITNESSES,
    _codec,
    _motion_and_error_ratio,
    _probe_transforms,
    _skeleton,
    _surface,
    _teacher_weights,
)


DIAGNOSTIC_WITNESSES = tuple(w for w in WITNESSES if w.name in {"branch_blend_4", "sharp_fork_5"})


def _conditioning(w):
    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher = _teacher_weights(w)
    rest = torch.tensor(w.points, dtype=torch.float32)[None]
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    return cond, teacher, rest, transforms, sf, jf, sm, jm


def _metrics(weights, teacher, rest, transforms, sm, *, logits=None, codec=None):
    rows = row_l1_error_v1(weights, teacher, sm)
    _, ratio = _motion_and_error_ratio(rest, teacher, weights, transforms)
    out = {
        "row_l1_p95": float(torch.quantile(rows, 0.95).detach().cpu()),
        "row_l1_max": float(rows.max().detach().cpu()),
        "deformation_ratio": float(ratio),
    }
    if logits is not None:
        valid_logits = logits[sm[:, :, None].expand_as(logits)]
        out["logit_span"] = float((valid_logits.max() - valid_logits.min()).detach().cpu())
    if codec is not None:
        temp = torch.nn.functional.softplus(codec.log_temperature) + codec.config.temperature_floor
        out["temperature"] = float(temp.detach().cpu())
    return out


def _optimize_free_latent_decoder(w):
    torch.manual_seed(w.seed)
    _, teacher, rest, transforms, sf, jf, sm, jm = _conditioning(w)
    codec = _codec()
    latent = nn.Parameter(torch.zeros((1, len(w.joints), codec.config.latent_dim), dtype=torch.float32))
    params = [latent] + list(codec.surface_embed.parameters()) + list(codec.joint_embed.parameters()) + list(codec.decoder.parameters()) + [codec.log_temperature]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    trace = []
    for step in range(1, A0_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        weights, logits = codec.decode_from_latents(latent, sf, jf, sm, jm)
        recon = skin_field_codec_loss_v1(weights, teacher, sm, jm)
        deform = codec_deformation_loss_v1(weights, teacher, rest, transforms, sm, jm)
        total = recon["total"] + deform["deformation_mse"]
        total.backward()
        optimizer.step()
        if step == 1 or step % CHECK_EVERY == 0 or step == A0_MAX_STEPS:
            with torch.no_grad():
                weights_eval, logits_eval = codec.decode_from_latents(latent, sf, jf, sm, jm)
                m = _metrics(weights_eval, teacher, rest, transforms, sm, logits=logits_eval, codec=codec)
                m.update({
                    "step": step,
                    "reconstruction": float(skin_field_codec_loss_v1(weights_eval, teacher, sm, jm)["total"].cpu()),
                })
                trace.append(m)
    final = trace[-1]
    final["passes_frozen_a0_behavior"] = bool(
        final["row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and final["deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
    )
    return {"lane": "FREE_PER_JOINT_LATENT_SAME_DECODER", "final": final, "trace": trace}


def _optimize_free_pair_logits(w):
    torch.manual_seed(w.seed)
    _, teacher, rest, transforms, _, _, sm, jm = _conditioning(w)
    logits = nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.AdamW([logits], lr=1e-3, weight_decay=1e-4)
    trace = []
    for step in range(1, A0_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        masked = logits.masked_fill(~jm[:, None, :], -1e4)
        weights = torch.softmax(masked, dim=-1) * sm[:, :, None].to(logits.dtype)
        recon = skin_field_codec_loss_v1(weights, teacher, sm, jm)
        deform = codec_deformation_loss_v1(weights, teacher, rest, transforms, sm, jm)
        total = recon["total"] + deform["deformation_mse"]
        total.backward()
        optimizer.step()
        if step == 1 or step % CHECK_EVERY == 0 or step == A0_MAX_STEPS:
            with torch.no_grad():
                masked_eval = logits.masked_fill(~jm[:, None, :], -1e4)
                weights_eval = torch.softmax(masked_eval, dim=-1) * sm[:, :, None].to(logits.dtype)
                m = _metrics(weights_eval, teacher, rest, transforms, sm, logits=masked_eval)
                m.update({"step": step, "reconstruction": float(skin_field_codec_loss_v1(weights_eval, teacher, sm, jm)["total"].cpu())})
                trace.append(m)
    final = trace[-1]
    final["passes_frozen_a0_behavior"] = bool(
        final["row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and final["deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
    )
    return {"lane": "FREE_PAIR_LOGITS_SOFTMAX_UPPER_BOUND", "final": final, "trace": trace}


@pytest.mark.parametrize("witness", DIAGNOSTIC_WITNESSES, ids=lambda w: w.name)
def test_codec_a0_capacity_decomposition_diagnostic(witness):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "free_latent_decoder": _optimize_free_latent_decoder(witness),
        "free_pair_logits": _optimize_free_pair_logits(witness),
    }
    print("ARACHNE_CODEC_A0_CAPACITY_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))

    # This is a diagnostic, not a product PASS gate. The free-pair-logit lane is
    # required only as a self-check that the frozen behavioral metrics are reachable.
    assert result["free_pair_logits"]["final"]["passes_frozen_a0_behavior"], result
