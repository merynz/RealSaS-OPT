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
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel_bound_v2 import (
    _bound_teacher_and_rest,
)
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _run_lane(w: Witness, *, deformation_weight: float, lane: str) -> dict:
    # Reset the full stochastic state so both lanes start from the same model init
    # and differ only in the A0 objective term under test.
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

    trace = []
    stable = 0
    pass_step = None
    metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(
            codec,
            opt,
            sf,
            jf,
            teacher,
            sm,
            jm,
            rest,
            transforms,
            reconstruction_weight=1.0,
            deformation_weight=deformation_weight,
        )
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
            ok = _a0_pass(metrics)
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    return {
        "lane": lane,
        "deformation_weight": deformation_weight,
        "surface_permutation": permutation,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_codec_a0_reconstruction_only_full_horizon_ab(witness: Witness):
    current = _run_lane(witness, deformation_weight=1.0, lane="CURRENT_A0_OBJECTIVE")
    reconstruction_only = _run_lane(
        witness,
        deformation_weight=0.0,
        lane="RECONSTRUCTION_ONLY_A0_OBJECTIVE",
    )
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "current": current,
        "reconstruction_only": reconstruction_only,
    }
    print("SKIN_FIELD_CODEC_A0_RECONSTRUCTION_AB=" + json.dumps(result, sort_keys=True))

    # Diagnostic validity only. Do not encode the hoped-for scientific result as
    # an assertion before observing it; the frozen behavioral panel remains the
    # acceptance authority.
    assert current["surface_permutation"] == reconstruction_only["surface_permutation"]
    assert current["deformation_weight"] == 1.0
    assert reconstruction_only["deformation_weight"] == 0.0
