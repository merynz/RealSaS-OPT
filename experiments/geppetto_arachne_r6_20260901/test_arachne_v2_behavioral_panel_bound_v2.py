from __future__ import annotations

import json

import pytest
import torch

from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import (
    ArachneCandidateConfigV2,
    ArachneCandidateV2,
)
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    A1_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _a0_metrics,
    _a0_pass,
    _a0_token,
    _a1_pass,
    _a1_shipping_metrics,
    _codec,
    _probe_transforms,
    _skeleton,
    _surface,
    _teacher_weights,
)
from experiments.geppetto_arachne_r6_20260901.train_arachne_r6_a1_v1 import train_arachne_r6_a1_step_v1
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _canonical_surface_permutation(w: Witness, conditioning) -> tuple[int, ...]:
    """Map synthetic creation-order rows onto the adapter's canonical surface-ID order."""
    creation_ids = tuple(f"{w.name}:S:{i}" for i in range(len(w.points)))
    creation_index = {sid: i for i, sid in enumerate(creation_ids)}
    target_ids = tuple(conditioning.surface_ids[0])
    if len(target_ids) != len(creation_ids) or set(target_ids) != set(creation_ids):
        raise AssertionError("synthetic surface-ID set changed during conditioning")
    return tuple(creation_index[sid] for sid in target_ids)


def _bound_teacher_and_rest(w: Witness, conditioning) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Bind teacher W/rest rows by surface ID instead of incidental Python row order."""
    permutation = _canonical_surface_permutation(w, conditioning)
    teacher_creation = _teacher_weights(w)
    rest_creation = torch.tensor(w.points, dtype=torch.float32)[None]
    teacher = teacher_creation[:, permutation, :]
    rest = rest_creation[:, permutation, :]
    return teacher, rest, permutation


def _run_bound_witness(w: Witness) -> dict:
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
    opt0 = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    a0_trace = []
    a0_stable = 0
    a0_pass_step = None
    a0_metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, opt0, sf, jf, teacher, sm, jm, rest, transforms)
        if step == 1 or step % CHECK_EVERY == 0:
            a0_metrics = _a0_metrics(codec, cond, teacher, rest, transforms)
            ok = _a0_pass(a0_metrics)
            a0_stable = a0_stable + 1 if ok else 0
            a0_trace.append({"step": step, "pass": ok, **a0_metrics})
            if a0_stable >= REQUIRED_STABLE:
                a0_pass_step = step
                break

    binding = {
        "surface_ids": tuple(cond.surface_ids[0]),
        "creation_to_conditioning_permutation": permutation,
        "identity_order": permutation == tuple(range(len(permutation))),
    }

    if a0_pass_step is None:
        return {
            "witness": w.name,
            "seed": w.seed,
            "status": "FAIL_A0",
            "binding": binding,
            "a0_pass_step": None,
            "a0_stable": a0_stable,
            "a0_final": a0_metrics,
            "a0_trace": a0_trace,
            "a1_pass_step": None,
        }

    token = _a0_token(codec, w, a0_metrics, a0_pass_step)
    token.validate_for(codec)
    model = ArachneCandidateV2(
        codec,
        ArachneCandidateConfigV2(
            model_dim=32,
            surface_encoder_layers=1,
            attention_heads=4,
            feedforward_dim=64,
        ),
        freeze_codec=True,
    )
    assert all(not p.requires_grad for p in model.codec.parameters())
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)

    a1_trace = []
    a1_stable = 0
    a1_pass_step = None
    a1_metrics = _a1_shipping_metrics(model, cond, surface, skeleton, teacher, rest, transforms)
    for step in range(1, A1_MAX_STEPS + 1):
        train_arachne_r6_a1_step_v1(
            model,
            opt1,
            cond,
            teacher,
            rest,
            transforms,
            a0_token=token,
            expected_surface_hashes=cond.source_surface_hashes,
            expected_skeleton_hashes=cond.source_skeleton_hashes,
        )
        if step == 1 or step % CHECK_EVERY == 0:
            a1_metrics = _a1_shipping_metrics(model, cond, surface, skeleton, teacher, rest, transforms)
            ok = _a1_pass(a1_metrics, len(w.points))
            a1_stable = a1_stable + 1 if ok else 0
            a1_trace.append({"step": step, "pass": ok, **a1_metrics})
            if a1_stable >= REQUIRED_STABLE:
                a1_pass_step = step
                break

    return {
        "witness": w.name,
        "seed": w.seed,
        "status": "PASS" if a1_pass_step is not None else "FAIL_A1",
        "binding": binding,
        "a0_pass_step": a0_pass_step,
        "a0_stable": a0_stable,
        "a0_final": a0_metrics,
        "a0_trace": a0_trace,
        "a1_pass_step": a1_pass_step,
        "a1_stable": a1_stable,
        "a1_final": a1_metrics,
        "a1_trace": a1_trace,
    }


def test_successor_harness_binds_n_ge_10_rows_by_surface_id():
    """Cause-level regression for the V1 lexicographic-row binding bug."""
    for w in WITNESSES:
        surface = _surface(w)
        skeleton = _skeleton(w)
        cond = ArachneConditioningAdapterV2()([surface], [skeleton])
        teacher, rest, permutation = _bound_teacher_and_rest(w, cond)
        assert teacher.shape[1] == len(cond.surface_ids[0])
        assert rest.shape[1] == len(cond.surface_ids[0])
        creation_ids = tuple(f"{w.name}:S:{i}" for i in range(len(w.points)))
        for row, sid in enumerate(cond.surface_ids[0]):
            creation_row = creation_ids.index(sid)
            torch.testing.assert_close(rest[0, row], torch.tensor(w.points[creation_row], dtype=torch.float32))
            torch.testing.assert_close(teacher[0, row], _teacher_weights(w)[0, creation_row])
        if len(w.points) < 10:
            assert permutation == tuple(range(len(permutation)))
        else:
            assert permutation != tuple(range(len(permutation)))


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_preregistered_arachne_codec_behavioral_witness_bound_v2(witness: Witness):
    result = _run_bound_witness(witness)
    print("ARACHNE_CODEC_BEHAVIORAL_PANEL_BOUND_V2=" + json.dumps(result, sort_keys=True))
    assert result["status"] == "PASS", result
    assert result["a0_stable"] >= REQUIRED_STABLE
    assert result["a1_stable"] >= REQUIRED_STABLE
