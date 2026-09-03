from __future__ import annotations

import json

import pytest
import torch

from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _a0_metrics,
    _a0_pass,
    _codec,
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import _bound_teacher_and_rest
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _run_lane(w: Witness, *, cosine: bool, lane: str) -> dict:
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A0_MAX_STEPS, eta_min=0.0) if cosine else None

    trace = []
    stable = 0
    pass_step = None
    metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, opt, sf, jf, teacher, sm, jm, rest, transforms)
        if scheduler is not None:
            scheduler.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
            ok = _a0_pass(metrics)
            stable = stable + 1 if ok else 0
            trace.append({
                "step": step,
                "pass": ok,
                "lr": float(opt.param_groups[0]["lr"]),
                **metrics,
            })
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    return {
        "lane": lane,
        "scheduler": "COSINE_TO_ZERO" if cosine else "CONSTANT",
        "surface_permutation": permutation,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final_lr": float(opt.param_groups[0]["lr"]),
        "final": metrics,
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_codec_a0_optimizer_cooling_full_panel_ab(witness: Witness):
    constant = _run_lane(witness, cosine=False, lane="CURRENT_CONSTANT_LR")
    cosine = _run_lane(witness, cosine=True, lane="COSINE_COOLING")
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "constant": constant,
        "cosine": cosine,
    }
    print("SKIN_FIELD_CODEC_A0_OPTIMIZER_COOLING_AB=" + json.dumps(result, sort_keys=True))

    # Diagnostic validity only; do not bake the hoped-for result into the test.
    assert constant["surface_permutation"] == cosine["surface_permutation"]
    assert constant["scheduler"] == "CONSTANT"
    assert cosine["scheduler"] == "COSINE_TO_ZERO"
