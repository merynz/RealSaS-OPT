from __future__ import annotations

import json

import pytest
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR
from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import row_l1_error_v1
from experiments.geppetto_arachne_r6_20260901.candidate_config_v1 import SKIN_FIELD_CODEC_V1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_DEFORMATION_RATIO_MAX,
    A0_MAX_STEPS,
    A0_ROW_L1_P95_MAX,
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
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import train_codec_r6_a0_step_v1


def _shipping_proposal(decoded: torch.Tensor, conditioning) -> SkinProposalIR:
    surface_ids = tuple(conditioning.surface_ids[0])
    joint_ids = tuple(conditioning.joint_ids[0])
    influences = []
    for si, surface_id in enumerate(surface_ids):
        for ji, joint_id in enumerate(joint_ids):
            influences.append(
                SkinInfluenceProposal(surface_id, joint_id, float(decoded[0, si, ji].item()))
            )
    return SkinProposalIR(
        tuple(influences),
        conditioning.source_surface_hashes[0],
        conditioning.source_skeleton_hashes[0],
        model_provenance=SKIN_FIELD_CODEC_V1.config_hash,
        metadata={
            "gate": "SHIPPING_CODEC_CAPACITY_V1",
            "codec_architecture": SKIN_FIELD_CODEC_V1.architecture_id,
            "codec_config_hash": SKIN_FIELD_CODEC_V1.config_hash,
            "teacher_encoder_lane": True,
            "compiler_owns_qualification": True,
        },
    )


@torch.no_grad()
def _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms) -> dict[str, float]:
    surface_features = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    joint_features = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    surface_mask = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    joint_mask = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec.eval()
    output = codec(surface_features, joint_features, teacher, surface_mask, joint_mask)
    raw = output.decoded_weights
    proposal = _shipping_proposal(raw, conditioning)
    qualified = qualify_skin(surface, skeleton, proposal)
    qualified_w = _qualified_matrix(qualified, conditioning.surface_ids[0], conditioning.joint_ids[0])

    raw_rows = row_l1_error_v1(raw, teacher, surface_mask)
    qualified_rows = row_l1_error_v1(qualified_w, teacher, surface_mask)
    _, raw_deformation_ratio = _motion_and_error_ratio(rest, teacher, raw, transforms)
    _, qualified_deformation_ratio = _motion_and_error_ratio(rest, teacher, qualified_w, transforms)

    raw_simplex = raw.sum(-1)
    qualified_simplex = qualified_w.sum(-1)
    valid_pair = surface_mask[:, :, None] & joint_mask[:, None, :]

    return {
        "raw_row_l1_p95": float(torch.quantile(raw_rows, 0.95).cpu()),
        "qualified_row_l1_p95": float(torch.quantile(qualified_rows, 0.95).cpu()),
        "raw_deformation_ratio": float(raw_deformation_ratio),
        "qualified_deformation_ratio": float(qualified_deformation_ratio),
        "raw_max_simplex_residual": float((raw_simplex[surface_mask] - 1.0).abs().max().cpu()),
        "qualified_max_simplex_residual": float((qualified_simplex[surface_mask] - 1.0).abs().max().cpu()),
        "raw_negative_weight_count": int((raw[valid_pair] < 0).sum().cpu()),
        "qualified_negative_weight_count": int((qualified_w[valid_pair] < 0).sum().cpu()),
        "compiler_total_correction_l1": float(qualified.qualification_report["total_correction_l1"]),
        "qualified_row_count": int(len(qualified.rows)),
    }


def _shipping_pass(metrics: dict[str, float], expected_rows: int) -> bool:
    return (
        metrics["qualified_row_count"] == expected_rows
        and metrics["raw_row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and metrics["qualified_row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and metrics["raw_deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
        and metrics["qualified_deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
        and metrics["raw_max_simplex_residual"] <= MAX_SIMPLEX
        and metrics["qualified_max_simplex_residual"] <= MAX_SIMPLEX
        and metrics["raw_negative_weight_count"] == 0
        and metrics["qualified_negative_weight_count"] == 0
        and metrics["compiler_total_correction_l1"] <= MAX_COMPILER_CORRECTION_L1
    )


def _run_shipping_witness(witness: Witness) -> dict:
    torch.manual_seed(witness.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(witness)
    skeleton = _skeleton(witness)
    conditioning = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest, permutation = _bound_teacher_and_rest(witness, conditioning)
    transforms = _probe_transforms(len(witness.joints))

    surface_features = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    joint_features = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    surface_mask = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    joint_mask = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec = SkinFieldCodecV1()
    assert codec.config == SKIN_FIELD_CODEC_V1
    assert codec.config.hidden_dim == 192
    assert codec.config.latent_dim == 64
    assert codec.config.encoder_layers == 3
    assert codec.config.decoder_layers == 3

    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    trace = []
    stable = 0
    pass_step = None
    metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)

    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(
            codec,
            optimizer,
            surface_features,
            joint_features,
            teacher,
            surface_mask,
            joint_mask,
            rest,
            transforms,
        )
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)
            ok = _shipping_pass(metrics, len(witness.points))
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    return {
        "witness": witness.name,
        "seed": witness.seed,
        "status": "PASS" if pass_step is not None else "FAIL_SHIPPING_CODEC_CAPACITY",
        "codec": {
            "architecture_id": codec.config.architecture_id,
            "config_hash": codec.config.config_hash,
            "hidden_dim": codec.config.hidden_dim,
            "latent_dim": codec.config.latent_dim,
            "encoder_layers": codec.config.encoder_layers,
            "decoder_layers": codec.config.decoder_layers,
            "dropout": codec.config.dropout,
        },
        "binding": {
            "creation_to_conditioning_permutation": permutation,
            "identity_order": permutation == tuple(range(len(permutation))),
        },
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_shipping_skin_field_codec_capacity_through_compiler_and_lbs(witness: Witness):
    result = _run_shipping_witness(witness)
    print("SKIN_FIELD_CODEC_SHIPPING_CAPACITY_V1=" + json.dumps(result, sort_keys=True))
    assert result["status"] == "PASS", result
    assert result["stable_passes"] >= REQUIRED_STABLE
