from __future__ import annotations

import json
import math

import pytest
import torch

from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    WITNESSES,
    Witness,
    _codec,
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import _bound_teacher_and_rest
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _center_rows(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _row_record(*, row: int, surface_id: str, rest_row: torch.Tensor, teacher_row: torch.Tensor, predicted_row: torch.Tensor, logits_row: torch.Tensor) -> dict:
    delta = predicted_row - teacher_row
    teacher_logits = torch.log(teacher_row.clamp_min(1e-12))
    pred_centered = _center_rows(logits_row[None])[0]
    teacher_centered = _center_rows(teacher_logits[None])[0]
    logit_delta = pred_centered - teacher_centered
    return {
        "row": int(row),
        "surface_id": surface_id,
        "rest_xyz": [float(v) for v in rest_row.cpu()],
        "row_l1": float(delta.abs().sum().cpu()),
        "teacher": [float(v) for v in teacher_row.cpu()],
        "predicted": [float(v) for v in predicted_row.cpu()],
        "signed_delta": [float(v) for v in delta.cpu()],
        "decoder_logits_centered": [float(v) for v in pred_centered.cpu()],
        "teacher_logprob_centered": [float(v) for v in teacher_centered.cpu()],
        "centered_logit_delta": [float(v) for v in logit_delta.cpu()],
        "largest_excess_joint": int(torch.argmax(delta).item()),
        "largest_missing_joint": int(torch.argmin(delta).item()),
        "largest_abs_logit_delta_joint": int(torch.argmax(logit_delta.abs()).item()),
    }


def _run_anatomy(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)

    codec = _codec()
    opt = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0)

    for _ in range(A0_MAX_STEPS):
        train_codec_r6_a0_step_v1(codec, opt, sf, jf, teacher, sm, jm, rest, transforms)
        scheduler.step()

    codec.eval()
    with torch.no_grad():
        out = codec(sf, jf, teacher, sm, jm)

    pred = out.decoded_weights[0, sm[0]][:, jm[0]]
    truth = teacher[0, sm[0]][:, jm[0]]
    logits = out.pair_logits[0, sm[0]][:, jm[0]]
    rest_rows = rest[0, sm[0]]
    row_l1 = (pred - truth).abs().sum(dim=-1)
    order = torch.argsort(row_l1, descending=True)

    records = [
        _row_record(
            row=int(i),
            surface_id=str(cond.surface_ids[0][int(i)]),
            rest_row=rest_rows[int(i)],
            teacher_row=truth[int(i)],
            predicted_row=pred[int(i)],
            logits_row=logits[int(i)],
        )
        for i in order
    ]

    signed_mass = (pred - truth).sum(dim=0)
    abs_mass = (pred - truth).abs().sum(dim=0)
    learned_temperature = float((torch.nn.functional.softplus(codec.log_temperature) + codec.config.temperature_floor).detach().cpu())

    n = int(row_l1.numel())
    q_index = 0.95 * float(max(0, n - 1))
    q_lo = int(math.floor(q_index))
    q_hi = int(math.ceil(q_index))
    asc = torch.sort(row_l1).values

    return {
        "witness": w.name,
        "seed": w.seed,
        "surface_permutation": permutation,
        "surface_ids": list(cond.surface_ids[0]),
        "joint_ids": list(cond.joint_ids[0]),
        "final_lr": float(opt.param_groups[0]["lr"]),
        "learned_temperature": learned_temperature,
        "row_l1_p95": float(torch.quantile(row_l1, 0.95).cpu()),
        "row_l1_max": float(row_l1.max().cpu()),
        "p95_interpolation": {
            "ascending_lower_rank": q_lo,
            "ascending_upper_rank": q_hi,
            "lower_value": float(asc[q_lo].cpu()),
            "upper_value": float(asc[q_hi].cpu()),
        },
        "joint_signed_mass_delta": [float(v) for v in signed_mass.cpu()],
        "joint_abs_mass_error": [float(v) for v in abs_mass.cpu()],
        "rows_descending_error": records,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_codec_a0_cosine_row_error_anatomy(witness: Witness):
    result = _run_anatomy(witness)
    print("SKIN_FIELD_CODEC_A0_ROW_ERROR_ANATOMY=" + json.dumps(result, sort_keys=True))

    # Diagnostic validity only. This must not encode a hoped-for PASS outcome.
    assert result["final_lr"] == pytest.approx(0.0, abs=1e-15)
    assert len(result["rows_descending_error"]) == len(witness.points)
    assert len(result["joint_signed_mass_delta"]) == len(witness.joints)
