from __future__ import annotations

import json
import numpy as np
import torch

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.types import (
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)
from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import (
    ArachneCandidateConfigV2,
    ArachneCandidateV2,
)
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.oracle_substrate_synthetic_v1 import (
    build_u0_surface,
    build_u1_surface,
    substrate_telemetry,
)
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.test_arachne_shipping_boundary_v1 import (
    EXPECTED_SHIPPING_ARACHNE_HASH,
    EXPECTED_SHIPPING_CODEC_HASH,
    _a1_metrics,
    _a1_pass,
    _criteria_hash,
)
from experiments.geppetto_arachne_r6_20260901.test_arachne_v2_behavioral_panel import (
    A0_MAX_STEPS,
    A1_MAX_STEPS,
    CHECK_EVERY,
    REQUIRED_STABLE,
    WITNESSES,
    Witness,
    _probe_transforms,
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


WITNESS = next(w for w in WITNESSES if w.name == "branch_blend_4")


def _nearest_surface_id(surface, position) -> str:
    nodes = tuple(surface.surface_nodes)
    P = np.asarray([n.P for n in nodes], np.float64)
    q = np.asarray(position, np.float64)
    return nodes[int(np.argmin(np.linalg.norm(P - q[None], axis=1)))].surface_id


def _oracle_qualified_skeleton(surface, witness: Witness):
    joints = []
    for i, (position, parent) in enumerate(zip(witness.joints, witness.parents)):
        joints.append(SkeletonProposalJoint(
            proposal_id=f"R6A1:{witness.name}:J:{i}",
            position=position,
            root_score=1.0 if parent < 0 else 0.0,
            confidence=1.0,
            support_surface_ids=(_nearest_surface_id(surface, position),),
            metadata={"oracle_mechanical_g": True, "teacher_identity_used": False},
        ))
    edges = []
    for child, parent in enumerate(witness.parents):
        if parent < 0:
            continue
        edges.append(SkeletonProposalEdge(
            edge_id=f"R6A1:{witness.name}:E:{parent}:{child}",
            parent_proposal_id=joints[parent].proposal_id,
            child_proposal_id=joints[child].proposal_id,
            score=1.0,
            confidence=1.0,
            hard_required=True,
            reason="R6_A1_ORACLE_MECHANICAL_G",
            metadata={"oracle_mechanical_g": True, "source_rig_identity_used": False},
        ))
    proposal = SkeletonProposalIR(
        tuple(joints),
        tuple(edges),
        surface.geometry_lineage_hash,
        model_provenance="R6_A1_ORACLE_MECHANICAL_G_V1",
        metadata={
            "gate": "R6_A1_ORACLE_SUBSTRATE_V1",
            "oracle_mechanical_g": True,
            "source_rig_identity_used": False,
            "compiler_owns_canonical_ids": True,
        },
    )
    qualified = qualify_skeleton_v2(surface, proposal)
    assert len(qualified.joints) == len(witness.joints)
    assert len(qualified.deform_root_ids) >= 1
    return qualified


def _teacher_and_rest(surface, skeleton, conditioning, sigma: float):
    surface_by_id = {n.surface_id: np.asarray(n.P, np.float32) for n in surface.surface_nodes}
    joint_by_id = {j.canonical_joint_id: np.asarray(j.position, np.float32) for j in skeleton.joints}
    p = torch.tensor(np.stack([surface_by_id[sid] for sid in conditioning.surface_ids[0]], axis=0), dtype=torch.float32)
    j = torch.tensor(np.stack([joint_by_id[jid] for jid in conditioning.joint_ids[0]], axis=0), dtype=torch.float32)
    d2 = (p[:, None, :] - j[None, :, :]).square().sum(-1)
    affinity = torch.exp(-d2 / (2.0 * float(sigma) ** 2)).clamp_min(1e-12)
    teacher = (affinity / affinity.sum(-1, keepdim=True))[None]
    rest = p[None]
    return teacher, rest


def _qualify_codec(witness: Witness, arm: str, conditioning, surface, skeleton, teacher, rest, transforms):
    sf = torch.tensor(conditioning.surface_features, dtype=torch.float32)
    jf = torch.tensor(conditioning.joint_features, dtype=torch.float32)
    sm = torch.tensor(conditioning.surface_mask, dtype=torch.bool)
    jm = torch.tensor(conditioning.joint_mask, dtype=torch.bool)

    codec = SkinFieldCodecV1()
    assert codec.config.config_hash == EXPECTED_SHIPPING_CODEC_HASH
    assert codec.config.hidden_dim == 192
    assert codec.config.latent_dim == 64
    assert codec.config.encoder_layers == 3
    assert codec.config.decoder_layers == 3

    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=A0_MAX_STEPS, eta_min=0.0)
    expected_rows = len(conditioning.surface_ids[0])
    stable = 0
    pass_step = None
    trace = []
    metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)

    for step in range(1, A0_MAX_STEPS + 1):
        train_codec_r6_a0_step_v1(codec, optimizer, sf, jf, teacher, sm, jm, rest, transforms)
        scheduler.step()
        if step == 1 or step % CHECK_EVERY == 0:
            metrics = _shipping_metrics(codec, conditioning, surface, skeleton, teacher, rest, transforms)
            ok = _shipping_pass(metrics, expected_rows)
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "lr": float(optimizer.param_groups[0]["lr"]), **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    if pass_step is None:
        return codec, None, {
            "status": "FAIL_A0",
            "pass_step": None,
            "stable_passes": stable,
            "final": metrics,
            "trace": trace,
        }

    token = CodecA0QualificationTokenV1(
        status="PASS",
        codec_config_hash=codec.config.config_hash,
        source_gate=f"R6_A1_ORACLE_SUBSTRATE_V1:{arm}:{witness.name}",
        optimizer_steps=pass_step,
        criteria_hash=_criteria_hash(),
        metrics_hash=canonical_metrics_hash_v1(metrics),
    )
    token.validate_for(codec)
    return codec, token, {
        "status": "PASS",
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
    }


def _run_arm(witness: Witness, arm: str, surface, telemetry: dict) -> dict:
    torch.manual_seed(witness.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    skeleton = _oracle_qualified_skeleton(surface, witness)
    conditioning = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher, rest = _teacher_and_rest(surface, skeleton, conditioning, witness.sigma)
    transforms = _probe_transforms(len(conditioning.joint_ids[0]))

    codec, token, a0 = _qualify_codec(
        witness, arm, conditioning, surface, skeleton, teacher, rest, transforms
    )
    result = {
        "arm": arm,
        "witness": witness.name,
        "seed": witness.seed,
        "telemetry": telemetry,
        "surface_lineage": surface.geometry_lineage_hash,
        "surface_rows": len(conditioning.surface_ids[0]),
        "qualified_skeleton": {
            "joint_count": len(skeleton.joints),
            "deform_root_count": len(skeleton.deform_root_ids),
            "skeleton_lineage": skeleton.skeleton_lineage_hash,
        },
        "codec_config_hash": codec.config.config_hash,
        "a0": a0,
        "a1": None,
    }
    if token is None:
        result["status"] = "FAIL_A0"
        return result

    model = ArachneCandidateV2(codec, ArachneCandidateConfigV2(), freeze_codec=True)
    assert model.config.config_hash == EXPECTED_SHIPPING_ARACHNE_HASH
    assert model.config.model_dim == 128
    assert model.config.surface_encoder_layers == 2
    assert model.config.attention_heads == 4
    assert model.config.feedforward_dim == 384
    assert model.codec.config.config_hash == EXPECTED_SHIPPING_CODEC_HASH
    assert all(not p.requires_grad for p in model.codec.parameters())

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    expected_rows = len(conditioning.surface_ids[0])
    stable = 0
    pass_step = None
    trace = []
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
            ok = _a1_pass(metrics, expected_rows)
            stable = stable + 1 if ok else 0
            trace.append({"step": step, "pass": ok, "losses": losses, **metrics})
            if stable >= REQUIRED_STABLE:
                pass_step = step
                break

    a1 = {
        "status": "PASS" if pass_step is not None else "FAIL_A1",
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "codec_config_hash": model.codec.config.config_hash,
        "pass_step": pass_step,
        "stable_passes": stable,
        "final": metrics,
        "trace": trace,
    }
    result["a1"] = a1
    result["status"] = "PASS" if pass_step is not None else "FAIL_A1"
    return result


def test_r6_arachne_one_family_u0_then_u1_oracle_substrate():
    witness = WITNESS
    u0, samples, cameras, visibility = build_u0_surface(witness)
    u1, _ = build_u1_surface(witness, samples, cameras, visibility)
    telemetry = substrate_telemetry(u0, u1, visibility)

    u0_result = _run_arm(witness, "U0_REFERENCE_FULL_SURFACE", u0, telemetry)
    print("R6_A1_U0_RESULT=" + json.dumps(u0_result, sort_keys=True))
    assert u0_result["status"] == "PASS", u0_result
    assert u0_result["a0"]["stable_passes"] >= REQUIRED_STABLE, u0_result
    assert u0_result["a1"]["stable_passes"] >= REQUIRED_STABLE, u0_result

    u1_result = _run_arm(witness, "U1_OBSERVATION_ORACLE_SUBSTRATE", u1, telemetry)
    print("R6_A1_U1_RESULT=" + json.dumps(u1_result, sort_keys=True))
    assert u1_result["status"] == "PASS", u1_result
    assert u1_result["a0"]["stable_passes"] >= REQUIRED_STABLE, u1_result
    assert u1_result["a1"]["stable_passes"] >= REQUIRED_STABLE, u1_result
