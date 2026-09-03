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
    _a0_metrics,
    _a0_pass,
    _codec,
    _probe_transforms,
    _skeleton,
    _surface,
    _teacher_weights,
)
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


DIAGNOSTIC_WITNESSES = tuple(w for w in WITNESSES if w.name in {"branch_blend_4", "sharp_fork_5"})


def _rebind_teacher_rows_to_conditioning_surface_ids(witness, conditioning, teacher_creation_order):
    creation_ids = tuple(f"{witness.name}:S:{i}" for i in range(len(witness.points)))
    creation_index = {sid: i for i, sid in enumerate(creation_ids)}
    target_ids = tuple(conditioning.surface_ids[0])
    if set(target_ids) != set(creation_ids):
        raise AssertionError("synthetic surface-ID set changed during conditioning")
    permutation = tuple(creation_index[sid] for sid in target_ids)
    return teacher_creation_order[:, permutation, :], creation_ids, target_ids, permutation


def _run_rebound_a0(witness):
    torch.manual_seed(witness.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(witness)
    skeleton = _skeleton(witness)
    conditioning = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher_creation = _teacher_weights(witness)
    teacher, creation_ids, target_ids, permutation = _rebind_teacher_rows_to_conditioning_surface_ids(
        witness, conditioning, teacher_creation
    )
    rest_creation = torch.tensor(witness.points, dtype=torch.float32)[None]
    rest = rest_creation[:, permutation, :]
    transforms = _probe_transforms(len(witness.joints))

    sf = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    jf = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    sm = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    jm = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec = _codec()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    stable = 0
    pass_step = None
    final = None

    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        if step == 1 or step % CHECK_EVERY == 0 or step == A0_MAX_STEPS:
            m = _a0_metrics(codec, conditioning, teacher, rest, transforms)
            m["step"] = step
            m["pass"] = _a0_pass(m)
            final = dict(m)
            if m["pass"]:
                stable += 1
                if stable >= REQUIRED_STABLE and pass_step is None:
                    pass_step = step
                    break
            else:
                stable = 0

    return {
        "creation_surface_ids": creation_ids,
        "conditioning_surface_ids": target_ids,
        "creation_to_conditioning_permutation": permutation,
        "identity_order": creation_ids == target_ids,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": final,
    }


@pytest.mark.parametrize("witness", DIAGNOSTIC_WITNESSES, ids=lambda w: w.name)
def test_canonical_surface_id_rebind_replays_failing_a0_witness(witness):
    result = {
        "witness": witness.name,
        "seed": witness.seed,
        "rebound_a0": _run_rebound_a0(witness),
    }
    print("ARACHNE_CODEC_TARGET_ROW_ALIGNMENT_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))

    # This is a causal diagnostic of the evaluator binding, not a relaxed gate.
    # N>=10 witnesses must expose the lexicographic-vs-creation-order mismatch.
    assert not result["rebound_a0"]["identity_order"], result
