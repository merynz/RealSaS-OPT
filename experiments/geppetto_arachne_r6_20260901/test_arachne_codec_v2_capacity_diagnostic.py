from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import (
    row_l1_error_v1,
    top_fraction_row_l1_tail_v1,
)
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
    out["passes_frozen_a0_behavior"] = bool(
        out["row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and out["deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
    )
    return out


def _exact_pair_logit_oracle(w):
    _, teacher, rest, transforms, _, _, sm, jm = _conditioning(w)
    logits = torch.log(teacher.clamp_min(1e-12)).masked_fill(~jm[:, None, :], -1e4)
    weights = torch.softmax(logits, dim=-1) * sm[:, :, None].to(logits.dtype)
    return _metrics(weights, teacher, rest, transforms, sm, logits=logits)


def _optimize_free_latent_decoder(w, *, behavioral_aligned: bool):
    torch.manual_seed(w.seed)
    _, teacher, rest, transforms, sf, jf, sm, jm = _conditioning(w)
    codec = _codec()
    latent = nn.Parameter(torch.zeros((1, len(w.joints), codec.config.latent_dim), dtype=torch.float32))
    params = [latent] + list(codec.surface_embed.parameters()) + list(codec.joint_embed.parameters()) + list(codec.decoder.parameters()) + [codec.log_temperature]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    best = None
    final = None
    for step in range(1, A0_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        weights, _ = codec.decode_from_latents(latent, sf, jf, sm, jm)
        deform = codec_deformation_loss_v1(weights, teacher, rest, transforms, sm, jm)
        if behavioral_aligned:
            mean_row = row_l1_error_v1(weights, teacher, sm).mean()
            tail = top_fraction_row_l1_tail_v1(weights, teacher, sm, fraction=0.10)
            total = mean_row + tail + deform["deformation_mse"]
        else:
            recon = skin_field_codec_loss_v1(weights, teacher, sm, jm)
            total = recon["total"] + deform["deformation_mse"]
        total.backward()
        optimizer.step()
        if step == 1 or step % CHECK_EVERY == 0 or step == A0_MAX_STEPS:
            with torch.no_grad():
                weights_eval, logits_eval = codec.decode_from_latents(latent, sf, jf, sm, jm)
                m = _metrics(weights_eval, teacher, rest, transforms, sm, logits=logits_eval, codec=codec)
                m["step"] = step
                if best is None or (m["row_l1_p95"], m["deformation_ratio"]) < (best["row_l1_p95"], best["deformation_ratio"]):
                    best = dict(m)
                final = dict(m)
    return {
        "lane": "FREE_PER_JOINT_LATENT_SAME_DECODER",
        "objective": "MEAN_ROW_L1_PLUS_HARD_TAIL_PLUS_DEFORMATION" if behavioral_aligned else "CURRENT_CODEC_RECONSTRUCTION_PLUS_DEFORMATION",
        "best": best,
        "final": final,
    }


def _teacher_distribution_telemetry(w):
    _, teacher, _, _, _, _, sm, jm = _conditioning(w)
    valid = sm[:, :, None] & jm[:, None, :]
    active = (teacher >= 1e-3) & valid
    inactive_positive = (teacher > 0.0) & (~active) & valid
    inactive_mass = (teacher * inactive_positive.to(teacher.dtype)).sum(dim=-1)[sm]
    entropy = -(teacher.clamp_min(1e-12) * torch.log(teacher.clamp_min(1e-12))).sum(dim=-1)[sm]
    return {
        "max_inactive_mass_per_row": float(inactive_mass.max().cpu()),
        "mean_inactive_mass_per_row": float(inactive_mass.mean().cpu()),
        "mixed_active_inactive_row_count": int(((active.any(-1) & inactive_positive.any(-1)) & sm).sum().cpu()),
        "teacher_entropy_min": float(entropy.min().cpu()),
        "teacher_entropy_max": float(entropy.max().cpu()),
    }


@pytest.mark.parametrize("witness", DIAGNOSTIC_WITNESSES, ids=lambda w: w.name)
def test_codec_a0_capacity_decomposition_diagnostic(witness):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "teacher_distribution": _teacher_distribution_telemetry(witness),
        "exact_pair_logit_oracle": _exact_pair_logit_oracle(witness),
        "free_latent_current_objective": _optimize_free_latent_decoder(witness, behavioral_aligned=False),
        "free_latent_behavioral_aligned_objective": _optimize_free_latent_decoder(witness, behavioral_aligned=True),
    }
    print("ARACHNE_CODEC_A0_CAPACITY_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))

    # Metric/verified-LBS reachability self-check. This is exact by construction
    # and does not conflate mathematical capacity with a deliberately weak optimizer.
    assert result["exact_pair_logit_oracle"]["passes_frozen_a0_behavior"], result
