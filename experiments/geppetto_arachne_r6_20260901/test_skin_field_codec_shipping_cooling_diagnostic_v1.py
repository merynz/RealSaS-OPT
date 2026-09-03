from __future__ import annotations

import json

import pytest
import torch

from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import (
    _bound_teacher_and_rest,
)
from experiments.geppetto_arachne_r6_20260901.test_skin_field_codec_shipping_capacity_v1 import (
    _shipping_metrics,
    _shipping_pass,
)
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _run_shipping_lane(witness: Witness, *, cosine: bool, lane: str) -> dict:
    torch.manual_seed(witness.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(witness)
    skeleton = _skeleton(witness)
    conditioning = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(witness, conditioning)
    transforms = _probe_transforms(len(witness.joints))

    sf = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    jf = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    sm = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    jm = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec = SkinFieldCodecV1()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=A0_MAX_STEPS, eta_min=0.0)
        if cosine
        else None
    )

    trace = []
    stable = 0
    pass_step = None
    first_pass_step = None
    best_raw_p95 = float("inf")
    best_qualified_p95 = float("inf")
    metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)

    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        if scheduler is not None:
            scheduler.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)
            ok = _shipping_pass(metrics, len(witness.points))
            if ok and first_pass_step is None:
                first_pass_step = step
            stable = stable + 1 if ok else 0
            best_raw_p95 = min(best_raw_p95, metrics["raw_row_l1_p95"])
            best_qualified_p95 = min(best_qualified_p95, metrics["qualified_row_l1_p95"])
            trace.append({
                "step": step,
                "pass": ok,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **metrics,
            })
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    return {
        "lane": lane,
        "scheduler": "COSINE_TO_ZERO" if cosine else "CONSTANT",
        "codec_config_hash": codec.config.config_hash,
        "surface_permutation": permutation,
        "first_pass_step": first_pass_step,
        "pass_step": pass_step,
        "stable_passes": stable,
        "best_raw_row_l1_p95": best_raw_p95,
        "best_qualified_row_l1_p95": best_qualified_p95,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "final": metrics,
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_shipping_codec_optimizer_cooling_full_panel_ab(witness: Witness):
    constant = _run_shipping_lane(witness, cosine=False, lane="SHIPPING_CONSTANT_LR")
    cosine = _run_shipping_lane(witness, cosine=True, lane="SHIPPING_COSINE_TO_ZERO")
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "constant": constant,
        "cosine": cosine,
    }
    print("SKIN_FIELD_CODEC_SHIPPING_COOLING_AB=" + json.dumps(result, sort_keys=True))

    # Diagnostic validity only. Do not encode the hoped-for outcome as an assertion.
    assert constant["codec_config_hash"] == cosine["codec_config_hash"]
    assert constant["surface_permutation"] == cosine["surface_permutation"]
    assert constant["scheduler"] == "CONSTANT"
    assert cosine["scheduler"] == "COSINE_TO_ZERO"
