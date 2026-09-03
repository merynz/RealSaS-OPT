from __future__ import annotations

import json

import pytest
import torch

from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import top_fraction_row_l1_tail_v1
from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import codec_deformation_loss_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1
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


TAIL_FRACTION = 0.10
TAIL_WEIGHT = 1.0


def _fixture(w):
    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
    transforms = _probe_transforms(len(w.joints))
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    return cond, teacher, rest, transforms, sf, jf, sm, jm, permutation


def _run_lane(w, *, with_tail: bool):
    torch.manual_seed(w.seed)
    cond, teacher, rest, transforms, sf, jf, sm, jm, permutation = _fixture(w)
    codec = _codec()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    trace = []
    stable = 0
    pass_step = None

    for step in range(1, A0_MAX_STEPS + 1):
        if with_tail:
            codec.train()
            optimizer.zero_grad(set_to_none=True)
            output = codec(sf, jf, teacher, sm, jm)
            reconstruction = skin_field_codec_loss_v1(output.decoded_weights, teacher, sm, jm)
            deformation = codec_deformation_loss_v1(
                output.decoded_weights, teacher, rest, transforms, sm, jm
            )
            tail = top_fraction_row_l1_tail_v1(
                output.decoded_weights,
                teacher,
                sm,
                fraction=TAIL_FRACTION,
            )
            total = reconstruction["total"] + deformation["deformation_mse"] + TAIL_WEIGHT * tail
            total.backward()
            optimizer.step()
        else:
            train_codec_r6_a0_step_v1(
                codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms
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
        "lane": "CURRENT_PLUS_TOP10_ROW_L1_TAIL" if with_tail else "CURRENT_A0_OBJECTIVE",
        "tail_fraction": TAIL_FRACTION if with_tail else 0.0,
        "tail_weight": TAIL_WEIGHT if with_tail else 0.0,
        "surface_permutation": permutation,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": trace[-1],
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_id_bound_a0_tail_stability_causal_diagnostic(witness):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "current": _run_lane(witness, with_tail=False),
        "plus_tail": _run_lane(witness, with_tail=True),
    }
    print("SKIN_FIELD_CODEC_A0_TAIL_CAUSAL_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))
    # Diagnostic only. Frozen behavioral authority remains bound V2.
    assert result["current"]["final"] is not None
    assert result["plus_tail"]["final"] is not None
