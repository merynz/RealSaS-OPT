from __future__ import annotations

import hashlib
import json

import pytest
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import (
    ArachneCandidateV2,
    ArachneCandidateConfigV2,
)
from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import row_l1_error_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    A0_ROW_L1_P95_MAX,
    A0_DEFORMATION_RATIO_MAX,
    A1_MAX_STEPS,
    A1_ROW_L1_P95_MAX,
    A1_DEFORMATION_RATIO_MAX,
    CHECK_EVERY,
    MAX_COMPILER_CORRECTION_L1,
    MAX_SIMPLEX,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _motion_and_error_ratio,
    _probe_transforms,
    _qualified_matrix,
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
from experiments.geppetto_arachne_r6_20260901.train_arachne_r6_a1_v1 import train_arachne_r6_a1_step_v1
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import (
    CodecA0QualificationTokenV1,
    canonical_metrics_hash_v1,
    train_codec_r6_a0_step_v1,
)


EXPECTED_SHIPPING_CODEC_HASH = "24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715"
EXPECTED_SHIPPING_ARACHNE_HASH = "ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1"


def _criteria_hash() -> str:
    payload = {
        "row_l1_p95_max": A0_ROW_L1_P95_MAX,
        "deformation_ratio_max": A0_DEFORMATION_RATIO_MAX,
        "max_simplex_residual": MAX_SIMPLEX,
        "negative_weight_count": 0,
        "compiler_total_correction_l1_max": MAX_COMPILER_CORRECTION_L1,
        "required_stable_passes": REQUIRED_STABLE,
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)+CosineAnnealingLR(T_max=1536,eta_min=0)",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _qualify_shipping_codec(witness: Witness, conditioning, surface, skeleton, teacher, rest, transforms):
    sf = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    jf = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    sm = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    jm = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec = SkinFieldCodecV1()
    assert codec.config.config_hash == EXPECTED_SHIPPING_CODEC_HASH
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=A0_MAX_STEPS, eta_min=0.0)

    stable = 0
    pass_step = None
    metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)
    trace = []
    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        scheduler.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)
            ok = _shipping_pass(metrics, len(witness.points))
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "lr": float(optimizer.param_groups[0]["lr"]), **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break
    if pass_step is None:
        raise AssertionError({"stage": "A0_SHIPPING_CODEC", "witness": witness.name, "trace": trace, "final": metrics})

    token = CodecA0QualificationTokenV1(
        status="PASS",
        codec_config_hash=codec.config.config_hash,
        source_gate=f"SHIPPING_CODEC_GENERIC_COSINE_V1:{witness.name}",
        optimizer_steps=pass_step,
        criteria_hash=_criteria_hash(),
        metrics_hash=canonical_metrics_hash_v1(metrics),
    )
    token.validate_for(codec)
    return codec, token, {"pass_step": pass_step, "stable_passes": stable, "final": metrics, "trace": trace}


@torch.no_grad()
def _a1_metrics(model, conditioning, surface, skeleton, teacher, rest, transforms) -> dict[str, float]:
    sf = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    jf = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    sm = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    jm = torch.tensor(conditioning.joint_mask, dtype=torch.bool)
    pi = torch.tensor(conditioning.parent_indices, dtype=torch.long)
    pg = torch.tensor(conditioning.pair_geometry, dtype=torch.float32)
    pm = torch.tensor(conditioning.pair_mask, dtype=torch.bool)

    model.eval()
    raw_output = model(sf, jf, sm, jm, pi, pg, pm)
    raw = raw_output.decoded_weights
    raw_rows = row_l1_error_v1(raw, teacher, sm)
    _, raw_deformation_ratio = _motion_and_error_ratio(rest, teacher, raw, transforms)
    raw_simplex = raw.sum(-1)
    valid_pair = sm[:, :, None] & jm[:, None, :]

    proposal = model.propose(conditioning)[0]
    qualified = qualify_skin(surface, skeleton, proposal)
    qualified_w = _qualified_matrix(qualified, conditioning.surface_ids[0], conditioning.joint_ids[0])
    qualified_rows = row_l1_error_v1(qualified_w, teacher, sm)
    _, qualified_deformation_ratio = _motion_and_error_ratio(rest, teacher, qualified_w, transforms)
    qualified_simplex = qualified_w.sum(-1)

    teacher_latents = model.codec.encode_teacher_weights(sf, jf, teacher, sm, jm)
    latent_error = (raw_output.joint_latent_mean - teacher_latents).abs()[jm]

    return {
        "raw_row_l1_p95": float(torch.quantile(raw_rows, 0.95).cpu()),
        "qualified_row_l1_p95": float(torch.quantile(qualified_rows, 0.95).cpu()),
        "raw_deformation_ratio": float(raw_deformation_ratio),
        "qualified_deformation_ratio": float(qualified_deformation_ratio),
        "raw_max_simplex_residual": float((raw_simplex[sm] - 1.0).abs().max().cpu()),
        "qualified_max_simplex_residual": float((qualified_simplex[sm] - 1.0).abs().max().cpu()),
        "raw_negative_weight_count": int((raw[valid_pair] < 0).sum().cpu()),
        "qualified_negative_weight_count": int((qualified_w[valid_pair] < 0).sum().cpu()),
        "compiler_total_correction_l1": float(qualified.qualification_report["total_correction_l1"]),
        "qualified_row_count": int(len(qualified.rows)),
        "latent_abs_p95": float(torch.quantile(latent_error.reshape(-1), 0.95).cpu()),
        "latent_uncertainty_mean": float(torch.exp(raw_output.joint_latent_log_sigma[jm]).mean().cpu()),
    }


def _a1_pass(metrics: dict[str, float], expected_rows: int) -> bool:
    return (
        metrics["qualified_row_count"] == expected_rows
        and metrics["raw_row_l1_p95"] <= A1_ROW_L1_P95_MAX
        and metrics["qualified_row_l1_p95"] <= A1_ROW_L1_P95_MAX
        and metrics["raw_deformation_ratio"] <= A1_DEFORMATION_RATIO_MAX
        and metrics["qualified_deformation_ratio"] <= A1_DEFORMATION_RATIO_MAX
        and metrics["raw_max_simplex_residual"] <= MAX_SIMPLEX
        and metrics["qualified_max_simplex_residual"] <= MAX_SIMPLEX
        and metrics["raw_negative_weight_count"] == 0
        and metrics["qualified_negative_weight_count"] == 0
        and metrics["compiler_total_correction_l1"] <= MAX_COMPILER_CORRECTION_L1
    )


def _run_shipping_a1(witness: Witness) -> dict:
    torch.manual_seed(witness.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(witness)
    skeleton = _skeleton(witness)
    conditioning = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(witness, conditioning)
    transforms = _probe_transforms(len(witness.joints))

    codec, token, a0 = _qualify_shipping_codec(
        witness, conditioning, surface, skeleton, teacher, rest, transforms
    )
    model = ArachneCandidateV2(codec, ArachneCandidateConfigV2(), freeze_codec=True)
    assert model.config.config_hash == EXPECTED_SHIPPING_ARACHNE_HASH
    assert model.config.model_dim == 128
    assert model.config.surface_encoder_layers == 2
    assert model.config.attention_heads == 4
    assert model.config.feedforward_dim == 384
    assert model.codec.config.config_hash == EXPECTED_SHIPPING_CODEC_HASH
    assert model.codec.config.hidden_dim == 192
    assert model.codec.config.latent_dim == 64
    assert all(not p.requires_grad for p in model.codec.parameters())

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    trace = []
    stable = 0
    pass_step = None
    metrics = _a1_metrics(model, conditioning, surface, skeleton, teacher, rest, transforms)

    for step in range(1, A1_MAX_STEPS + 1):
        losses = train_arachne_r6_a1_step_v1(
            model,
            optimizer,
            conditioning,
            teacher,
            rest,
            transforms,
            a0_token=token,
            expected_surface_hashes=conditioning.source_surface_hashes,
            expected_skeleton_hashes=conditioning.source_skeleton_hashes,
        )
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _a1_metrics(model, conditioning, surface, skeleton, teacher, rest, transforms)
            ok = _a1_pass(metrics, len(witness.points))
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "losses": losses, **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    return {
        "witness": witness.name,
        "seed": witness.seed,
        "status": "PASS" if pass_step is not None else "FAIL_SHIPPING_A1",
        "surface_permutation": permutation,
        "a0": a0,
        "a1": {
            "architecture_id": model.config.architecture_id,
            "config_hash": model.config.config_hash,
            "model_dim": model.config.model_dim,
            "surface_encoder_layers": model.config.surface_encoder_layers,
            "attention_heads": model.config.attention_heads,
            "feedforward_dim": model.config.feedforward_dim,
            "codec_config_hash": model.codec.config.config_hash,
            "pass_step": pass_step,
            "stable_passes": stable,
            "final": metrics,
            "trace": trace,
        },
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_shipping_arachne_to_frozen_shipping_codec_through_compiler_and_lbs(witness: Witness):
    result = _run_shipping_a1(witness)
    print("ARACHNE_SHIPPING_BOUNDARY_V1=" + json.dumps(result, sort_keys=True))
    assert result["status"] == "PASS", result
    assert result["a1"]["config_hash"] == EXPECTED_SHIPPING_ARACHNE_HASH, result
    assert result["a1"]["codec_config_hash"] == EXPECTED_SHIPPING_CODEC_HASH, result
    assert result["a1"]["stable_passes"] >= REQUIRED_STABLE, result
    assert result["a1"]["final"]["compiler_total_correction_l1"] <= MAX_COMPILER_CORRECTION_L1, result
