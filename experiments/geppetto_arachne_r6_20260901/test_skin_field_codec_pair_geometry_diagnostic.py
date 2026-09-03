from __future__ import annotations

import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

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
PAIR_POINT_GEOMETRY_WIDTH = 4  # dx, dy, dz, point-control distance


def _fixture(w):
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
    pg = torch.tensor(cond.pair_geometry[..., :PAIR_POINT_GEOMETRY_WIDTH], dtype=torch.float32)
    return cond, teacher, rest, transforms, sf, jf, sm, jm, pg


def _metrics(weights, teacher, rest, transforms, sm):
    rows = row_l1_error_v1(weights, teacher, sm)
    _, ratio = _motion_and_error_ratio(rest, teacher, weights, transforms)
    return {
        "row_l1_p95": float(torch.quantile(rows, 0.95).detach().cpu()),
        "row_l1_max": float(rows.max().detach().cpu()),
        "deformation_ratio": float(ratio),
        "passes_frozen_a0_behavior": bool(
            float(torch.quantile(rows, 0.95).detach().cpu()) <= A0_ROW_L1_P95_MAX
            and float(ratio) <= A0_DEFORMATION_RATIO_MAX
        ),
    }


def _run_pair_geometry_augmented_decoder(w):
    torch.manual_seed(w.seed)
    _, teacher, rest, transforms, sf, jf, sm, jm, pg = _fixture(w)
    codec = _codec()
    h = codec.config.hidden_dim
    pair_decoder = nn.Sequential(
        nn.Linear(2 * h + codec.config.latent_dim + PAIR_POINT_GEOMETRY_WIDTH, h),
        nn.GELU(),
        nn.Linear(h, 1),
    )
    params = (
        list(codec.surface_embed.parameters())
        + list(codec.joint_embed.parameters())
        + list(codec.weight_embed.parameters())
        + list(codec.encoder.parameters())
        + list(pair_decoder.parameters())
        + [codec.log_temperature]
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    best = None
    final = None
    stable = 0
    pass_step = None

    def decode():
        latents = codec.encode_teacher_weights(sf, jf, teacher, sm, jm)
        s = codec.surface_embed(sf)
        j = codec.joint_embed(jf)
        pair = torch.cat([
            s[:, :, None, :].expand(-1, -1, j.shape[1], -1),
            j[:, None, :, :].expand(-1, s.shape[1], -1, -1),
            latents[:, None, :, :].expand(-1, s.shape[1], -1, -1),
            pg,
        ], dim=-1)
        logits = pair_decoder(pair).squeeze(-1)
        temperature = F.softplus(codec.log_temperature) + codec.config.temperature_floor
        logits = (logits / temperature).masked_fill(~jm[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1) * sm[:, :, None].to(logits.dtype)
        return weights, logits, temperature

    for step in range(1, A0_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        weights, _, _ = decode()
        reconstruction = skin_field_codec_loss_v1(weights, teacher, sm, jm)
        deformation = codec_deformation_loss_v1(weights, teacher, rest, transforms, sm, jm)
        total = reconstruction["total"] + deformation["deformation_mse"]
        total.backward()
        optimizer.step()

        if step == 1 or step % CHECK_EVERY == 0 or step == A0_MAX_STEPS:
            with torch.no_grad():
                weights_eval, logits_eval, temperature = decode()
                m = _metrics(weights_eval, teacher, rest, transforms, sm)
                valid_logits = logits_eval[sm[:, :, None].expand_as(logits_eval)]
                m.update({
                    "step": step,
                    "temperature": float(temperature.detach().cpu()),
                    "logit_span": float((valid_logits.max() - valid_logits.min()).detach().cpu()),
                })
                final = dict(m)
                if best is None or (m["row_l1_p95"], m["deformation_ratio"]) < (best["row_l1_p95"], best["deformation_ratio"]):
                    best = dict(m)
                if m["passes_frozen_a0_behavior"]:
                    stable += 1
                    if stable >= 3 and pass_step is None:
                        pass_step = step
                else:
                    stable = 0

    return {
        "lane": "CURRENT_TEACHER_ENCODER_PLUS_EXPLICIT_POINT_CONTROL_GEOMETRY_DECODER",
        "pair_geometry_fields": ["dx", "dy", "dz", "distance"],
        "best": best,
        "final": final,
        "stable_passes_at_end": stable,
        "first_sustained_pass_step": pass_step,
    }


@pytest.mark.parametrize("witness", DIAGNOSTIC_WITNESSES, ids=lambda w: w.name)
def test_pair_geometry_representation_causal_diagnostic(witness):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "pair_geometry_decoder": _run_pair_geometry_augmented_decoder(witness),
    }
    print("SKIN_FIELD_CODEC_PAIR_GEOMETRY_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))
    # Diagnostic only. The frozen product panel remains the authority and is not
    # changed based on this result.
    assert result["pair_geometry_decoder"]["final"] is not None
