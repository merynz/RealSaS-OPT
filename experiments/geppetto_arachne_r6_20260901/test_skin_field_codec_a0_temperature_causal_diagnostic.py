from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    _a0_metrics,
    _a0_pass,
    _codec,
    _probe_transforms,
    _skeleton,
    _surface,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import _bound_teacher_and_rest
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _temperature(codec) -> float:
    return float((F.softplus(codec.log_temperature) + codec.config.temperature_floor).detach().cpu())


def _fixture(w):
    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = __import__(
        "experiments.geppetto_arachne_r6_20260901.conditioning_v2",
        fromlist=["ArachneConditioningAdapterV2"],
    ).ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    return cond, teacher, rest, transforms, sf, jf, sm, jm, permutation


def _run_lane(w, *, optimize_temperature: bool):
    torch.manual_seed(w.seed)
    cond, teacher, rest, transforms, sf, jf, sm, jm, permutation = _fixture(w)
    codec = _codec()
    initial_temperature = _temperature(codec)
    if optimize_temperature:
        parameters = list(codec.parameters())
    else:
        parameters = [p for name, p in codec.named_parameters() if name != "log_temperature"]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)

    trace = []
    stable = 0
    pass_step = None
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        if not optimize_temperature:
            # A source-neutral diagnostic invariant: excluded global calibration
            # parameter must remain bit-exact at initialization.
            if float(codec.log_temperature.detach().cpu()) != 0.0:
                raise AssertionError("frozen temperature parameter changed")
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
            ok = _a0_pass(metrics)
            stable = stable + 1 if ok else 0
            trace.append({
                "step": step,
                "pass": ok,
                "temperature": _temperature(codec),
                **metrics,
            })
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    temperatures = [x["temperature"] for x in trace]
    return {
        "lane": "CURRENT_LEARNED_GLOBAL_TEMPERATURE" if optimize_temperature else "FROZEN_INITIAL_GLOBAL_TEMPERATURE",
        "surface_permutation": permutation,
        "initial_temperature": initial_temperature,
        "temperature_min": min(temperatures),
        "temperature_max": max(temperatures),
        "temperature_final": temperatures[-1],
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": trace[-1],
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_global_temperature_a0_stability_causal_diagnostic(witness):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "learned_temperature": _run_lane(witness, optimize_temperature=True),
        "frozen_temperature": _run_lane(witness, optimize_temperature=False),
    }
    print("SKIN_FIELD_CODEC_A0_TEMPERATURE_CAUSAL_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))
    # Diagnostic only. The Bound V2 panel remains unchanged behavioral authority.
    assert result["learned_temperature"]["final"] is not None
    assert result["frozen_temperature"]["final"] is not None
