from __future__ import annotations

import copy
import json

import torch

from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import codec_deformation_loss_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    CHECK_EVERY,
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


CONTINUATION_STEPS = 32
TRACE_STEPS = {1, 4, 8, 16, 32}
SHARP_WITNESS = next(w for w in WITNESSES if w.name == "sharp_fork_5")


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


def _train_until_first_scalar_pass(w):
    torch.manual_seed(w.seed)
    cond, teacher, rest, transforms, sf, jf, sm, jm, permutation = _fixture(w)
    codec = _codec()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    first_pass_step = None
    first_pass_metrics = None
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
            if _a0_pass(metrics):
                first_pass_step = step
                first_pass_metrics = metrics
                break
    if first_pass_step is None:
        raise AssertionError("clean sharp witness never reached an individual scalar PASS")
    return {
        "cond": cond,
        "teacher": teacher,
        "rest": rest,
        "transforms": transforms,
        "sf": sf,
        "jf": jf,
        "sm": sm,
        "jm": jm,
        "permutation": permutation,
        "model_state": copy.deepcopy(codec.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "first_pass_step": first_pass_step,
        "first_pass_metrics": first_pass_metrics,
    }


def _continue_from_branch(base, lane: str):
    codec = _codec()
    codec.load_state_dict(base["model_state"])
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(base["optimizer_state"]))
    if lane == "current_no_weight_decay":
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0

    trace = []
    for local_step in range(1, CONTINUATION_STEPS + 1):
        codec.train()
        optimizer.zero_grad(set_to_none=True)
        output = codec(base["sf"], base["jf"], base["teacher"], base["sm"], base["jm"])
        reconstruction = skin_field_codec_loss_v1(
            output.decoded_weights,
            base["teacher"],
            base["sm"],
            base["jm"],
        )
        deformation = codec_deformation_loss_v1(
            output.decoded_weights,
            base["teacher"],
            base["rest"],
            base["transforms"],
            base["sm"],
            base["jm"],
        )

        if lane in {"current", "current_no_weight_decay"}:
            total = reconstruction["total"] + deformation["deformation_mse"]
        elif lane == "reconstruction_only":
            total = reconstruction["total"]
        elif lane == "cross_entropy_only":
            total = reconstruction["cross_entropy"]
        elif lane == "mean_l1_only":
            total = reconstruction["l1"]
        elif lane == "deformation_only":
            total = deformation["deformation_mse"]
        else:
            raise ValueError(f"unknown diagnostic lane:{lane}")

        total.backward()
        optimizer.step()

        if local_step in TRACE_STEPS:
            metrics = _a0_metrics(
                codec,
                base["cond"],
                base["teacher"],
                base["rest"],
                base["transforms"],
            )
            trace.append({
                "continuation_step": local_step,
                "absolute_step": base["first_pass_step"] + local_step,
                "pass": _a0_pass(metrics),
                **metrics,
            })

    return {
        "lane": lane,
        "final": trace[-1],
        "trace": trace,
    }


def test_post_pass_objective_force_decomposition_diagnostic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = _train_until_first_scalar_pass(SHARP_WITNESS)
    lanes = (
        "current",
        "reconstruction_only",
        "cross_entropy_only",
        "mean_l1_only",
        "deformation_only",
        "current_no_weight_decay",
    )
    result = {
        "witness": SHARP_WITNESS.name,
        "seed": SHARP_WITNESS.seed,
        "surface_permutation": base["permutation"],
        "branch_checkpoint_step": base["first_pass_step"],
        "branch_checkpoint_metrics": base["first_pass_metrics"],
        "continuation_steps": CONTINUATION_STEPS,
        "lanes": {lane: _continue_from_branch(base, lane) for lane in lanes},
    }
    print("SKIN_FIELD_CODEC_A0_FORCE_DECOMPOSITION=" + json.dumps(result, sort_keys=True))

    assert result["branch_checkpoint_metrics"]["row_l1_p95"] <= 0.05
    assert result["branch_checkpoint_metrics"]["deformation_ratio"] <= 0.05
    for lane in lanes:
        assert result["lanes"][lane]["final"] is not None
