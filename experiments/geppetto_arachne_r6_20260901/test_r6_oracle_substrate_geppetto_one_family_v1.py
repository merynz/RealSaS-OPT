from __future__ import annotations

import json
import math

import numpy as np
import torch

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2
from experiments.geppetto_arachne_r6_20260901.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.geppetto_eval_v2 import geppetto_metrics_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_train_v2 import geppetto_train_step_v2
from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import (
    PERSISTENCE_TOLERANCE,
    build_u0_surface,
    build_u1_surface,
    substrate_telemetry,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import WITNESSES
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1


LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_STEPS = 2048
CHECK_EVERY = 32
REQUIRED_STABLE = 3
ONE_FAMILY = "branch_blend_4"


def _witness():
    return next(w for w in WITNESSES if w.name == ONE_FAMILY)


def _target(witness, conditioning) -> GeppettoTeacherTargetV1:
    physical = np.asarray(witness.joints, np.float32)
    normalized = conditioning.normalizations[0].normalize(physical)
    parents = np.asarray(witness.parents, np.int64)
    roots = parents < 0
    return GeppettoTeacherTargetV1(normalized, parents, roots, True)


def _shipping(model: GeppettoCandidateV2, conditioning, target: GeppettoTeacherTargetV1) -> dict[str, object]:
    device = next(model.parameters()).device
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out, counts = model.generate(f, p, m, resource_step_limit=int(m[0].sum().item()))
    k = int(counts[0])
    result: dict[str, object] = {"generated_count": k}
    if k > 0:
        result.update(geppetto_metrics_v2(out.positions_normalized[0, :k], target.positions_normalized))
        result["map_mode_indices"] = torch.argmax(out.position_mode_logits[0, :k], dim=-1).cpu().tolist()
    else:
        result.update({"matched_mae": float("inf"), "matched_p95": float("inf")})
        result["map_mode_indices"] = []
    return result


def _qualified_mechanical(model, surface, conditioning, target) -> tuple[bool, str]:
    try:
        proposal = model.propose(conditioning, resource_step_limit=len(surface.surface_nodes))[0]
        if len(proposal.joints) != len(target.positions_normalized):
            return False, f"PROPOSAL_COUNT_{len(proposal.joints)}"
        qualified = qualify_skeleton_v2(surface, proposal)
        if len(qualified.joints) != len(target.positions_normalized):
            return False, f"QUALIFIED_COUNT_{len(qualified.joints)}"
        ids = {j.canonical_joint_id for j in qualified.joints}
        illegal_parent_count = sum(
            j.parent_canonical_id is not None and j.parent_canonical_id not in ids
            for j in qualified.joints
        )
        unsupported_joint_count = sum(not j.support_surface_ids for j in qualified.joints)
        ok = (
            len(qualified.deform_root_ids) > 0
            and illegal_parent_count == 0
            and unsupported_joint_count == 0
        )
        return bool(ok), (
            f"QUALIFIED_MECHANICAL:roots={len(qualified.deform_root_ids)}:"
            f"illegal_parents={illegal_parent_count}:unsupported={unsupported_joint_count}"
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def _fit_arm(arm: str, surface, witness, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    conditioning = GeppettoConditioningAdapterV2()([surface])
    target = _target(witness, conditioning)
    teacher = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    separation = torch.cdist(teacher, teacher)
    separation = separation.masked_fill(torch.eye(len(teacher), dtype=torch.bool), float("inf"))
    unique_radius = 0.5 * float(separation.min())

    config = GeppettoCandidateConfigV2()
    assert config.model_dim == 192
    assert config.local_layers == 2 and config.global_layers == 2 and config.decoder_layers == 2
    assert config.attention_heads == 6 and config.feedforward_dim == 576
    model = GeppettoCandidateV2(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    stable = 0
    pass_step = None
    trace = []
    for step_index in range(1, MAX_STEPS + 1):
        losses = geppetto_train_step_v2(model, optimizer, conditioning, [target])
        scalar_losses = {k: float(v) for k, v in losses.items() if k != "matched_joint_count"}
        if not all(math.isfinite(v) for v in scalar_losses.values()):
            raise AssertionError({"arm": arm, "step": step_index, "nonfinite_loss": scalar_losses})
        if step_index != 1 and step_index % CHECK_EVERY:
            continue

        shipping = _shipping(model, conditioning, target)
        geometry_pass = (
            int(shipping["generated_count"]) == len(target.positions_normalized)
            and float(shipping["matched_p95"]) < unique_radius
        )
        mechanical_pass = False
        mechanical_status = "NOT_EVALUATED"
        if geometry_pass:
            mechanical_pass, mechanical_status = _qualified_mechanical(model, surface, conditioning, target)
        product_pass = bool(geometry_pass and mechanical_pass)
        stable = stable + 1 if product_pass else 0
        trace.append({
            "step": step_index,
            "shipping": shipping,
            "geometry_pass": geometry_pass,
            "mechanical_pass": mechanical_pass,
            "mechanical_status": mechanical_status,
            "stable_passes": stable,
            "loss": {k: round(v, 6) for k, v in scalar_losses.items()},
        })
        if stable >= REQUIRED_STABLE:
            pass_step = step_index
            break

    return {
        "arm": arm,
        "config_hash": config.config_hash,
        "surface_node_count": len(surface.surface_nodes),
        "unique_radius": unique_radius,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": _shipping(model, conditioning, target),
        "trace": trace,
    }


def test_r6_geppetto_one_family_u0_then_u1_observation_oracle() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    witness = _witness()
    u0, samples, cameras, visibility = build_u0_surface(witness)
    u1, evidence = build_u1_surface(witness, samples, cameras, visibility)
    telemetry = substrate_telemetry(u0, u1, visibility)

    assert len(u0.surface_nodes) > 0
    assert 0 < len(u1.surface_nodes) <= len(u0.surface_nodes)
    assert telemetry["visible_fraction"] < 1.0, "fixture must contain genuinely non-observable full-surface samples"
    assert evidence.metadata["hidden_surface_completion"] is False
    assert evidence.metadata["source_mesh_consumer_input"] is False
    assert evidence.metadata["oracle_arm"] == "U1_OBSERVATION_ORACLE_SUBSTRATE"
    assert PERSISTENCE_TOLERANCE == 1e-6
    assert all("reference_full_surface" not in n.metadata for n in u1.surface_nodes)
    assert all(
        n.derived_normal is None
        or n.metadata.get("derived_normal_operator_id") == "RealSaS.DTB-ND1.RobustLocalPlane.v1"
        for n in u1.surface_nodes
    )

    # The same model initialization seed is used for both arms; only substrate
    # information differs. U0 must pass before U1 is scientifically interpreted.
    u0_result = _fit_arm("U0_REFERENCE_FULL_SURFACE", u0, witness, witness.seed)
    print("R6_G_SYN_U0_RESULT=" + json.dumps({"telemetry": telemetry, "result": u0_result}, sort_keys=True))
    assert u0_result["pass_step"] is not None, {"telemetry": telemetry, "u0": u0_result}
    assert u0_result["stable_passes"] >= REQUIRED_STABLE, {"telemetry": telemetry, "u0": u0_result}

    u1_result = _fit_arm("U1_OBSERVATION_ORACLE_SUBSTRATE", u1, witness, witness.seed)
    print("R6_G_SYN_U1_RESULT=" + json.dumps({"telemetry": telemetry, "result": u1_result}, sort_keys=True))
    assert u1_result["pass_step"] is not None, {"telemetry": telemetry, "u0": u0_result, "u1": u1_result}
    assert u1_result["stable_passes"] >= REQUIRED_STABLE, {"telemetry": telemetry, "u0": u0_result, "u1": u1_result}
