from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import QualifiedJoint, RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import (
    ArachneCandidateConfigV2,
    ArachneCandidateV2,
)
from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import row_l1_error_v1
from experiments.geppetto_arachne_r6_20260901.candidate_config_v1 import SkinFieldCodecConfigV1
from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import torch_verified_lbs_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.train_arachne_r6_a1_v1 import train_arachne_r6_a1_step_v1
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import (
    CodecA0QualificationTokenV1,
    canonical_metrics_hash_v1,
    train_codec_r6_a0_step_v1,
)


OP_HASH = "arachne-behavioral-panel-dtb-nd1"
A0_MAX_STEPS = 1536
A1_MAX_STEPS = 2048
CHECK_EVERY = 32
REQUIRED_STABLE = 3
A0_ROW_L1_P95_MAX = 0.05
A0_DEFORMATION_RATIO_MAX = 0.05
A1_ROW_L1_P95_MAX = 0.10
A1_DEFORMATION_RATIO_MAX = 0.10
MAX_SIMPLEX = 1e-6
MAX_COMPILER_CORRECTION_L1 = 1e-5


@dataclass(frozen=True)
class Witness:
    name: str
    seed: int
    points: tuple[tuple[float, float, float], ...]
    joints: tuple[tuple[float, float, float], ...]
    parents: tuple[int, ...]
    sigma: float


def _chain_points() -> tuple[tuple[float, float, float], ...]:
    out = []
    for i in range(9):
        x = -0.8 + 0.2 * i
        out.append((x, 0.10 * math.sin(0.7 * i), 0.05 * math.cos(0.5 * i)))
    return tuple(out)


def _branch_points() -> tuple[tuple[float, float, float], ...]:
    return (
        (0.00, -0.55, 0.00), (0.00, -0.30, 0.02), (0.00, -0.05, 0.00),
        (-0.15, 0.12, 0.01), (-0.32, 0.28, 0.03), (-0.52, 0.46, 0.04),
        (0.00, 0.16, 0.00), (0.00, 0.38, 0.05), (0.00, 0.60, 0.03),
        (0.16, 0.12, -0.01), (0.34, 0.30, -0.03), (0.54, 0.48, -0.04),
    )


def _fork_points() -> tuple[tuple[float, float, float], ...]:
    return (
        (0.00, -0.72, 0.00), (0.00, -0.52, 0.01), (0.00, -0.32, 0.00),
        (0.00, -0.12, -0.01), (0.00, 0.06, 0.00),
        (-0.10, 0.18, 0.01), (-0.22, 0.30, 0.02), (-0.36, 0.43, 0.04),
        (-0.52, 0.57, 0.05), (-0.68, 0.70, 0.04),
        (0.10, 0.18, -0.01), (0.22, 0.30, -0.02), (0.36, 0.43, -0.04),
        (0.52, 0.57, -0.05), (0.68, 0.70, -0.04),
    )


WITNESSES = (
    Witness(
        "chain_blend_3", 20260921, _chain_points(),
        ((-0.62, 0.02, 0.03), (0.00, 0.02, 0.00), (0.62, -0.04, -0.02)),
        (-1, 0, 1), 0.32,
    ),
    Witness(
        "branch_blend_4", 20260922, _branch_points(),
        ((0.00, -0.38, 0.00), (-0.42, 0.38, 0.03), (0.00, 0.44, 0.04), (0.44, 0.40, -0.03)),
        (-1, 0, 0, 0), 0.28,
    ),
    Witness(
        "sharp_fork_5", 20260923, _fork_points(),
        ((0.00, -0.56, 0.00), (0.00, -0.02, 0.00), (0.00, 0.10, 0.00), (-0.50, 0.56, 0.04), (0.50, 0.56, -0.04)),
        (-1, 0, 0, 1, 2), 0.18,
    ),
)


def _surface(w: Witness) -> RiggingSurfaceIR:
    nodes = []
    for i, p in enumerate(w.points):
        nodes.append(SurfaceNode(
            f"{w.name}:S:{i}", p,
            (0, 1, 2, 3, 4, 5, 6, 7),
            (f"{w.name}:obs:{i}",),
            (f"{w.name}:obs:{i}",),
            raster_bindings=((0, (float(p[0]) * 0.7, float(p[1]) * 0.7)),),
            derived_normal=(0.0, 0.0, 1.0),
        ))
    relations = tuple(
        SurfaceRelation(f"{w.name}:R:{i}", nodes[i].surface_id, nodes[i + 1].surface_id, "LOCAL", 1.0)
        for i in range(len(nodes) - 1)
    )
    return RiggingSurfaceIR(
        tuple(nodes), relations,
        geometry_lineage_hash=f"surface:{w.name}",
        metadata={"Nd_operator_sha256": OP_HASH, "raster_coordinate_system": "GRID_XY", "resolution": 1024},
    )


def _nearest_surface_index(points, joint) -> int:
    p = np.asarray(points, np.float64)
    j = np.asarray(joint, np.float64)
    return int(np.argmin(np.linalg.norm(p - j[None], axis=1)))


def _skeleton(w: Witness) -> QualifiedSkeletonIRV2:
    joints = []
    for i, (p, parent) in enumerate(zip(w.joints, w.parents)):
        support_i = _nearest_surface_index(w.points, p)
        joints.append(QualifiedJoint(
            f"{w.name}:J:{i}", p,
            None if parent < 0 else f"{w.name}:J:{parent}",
            (f"{w.name}:S:{support_i}",),
            f"{w.name}:proposal:{i}",
        ))
    roots = tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    return QualifiedSkeletonIRV2(
        tuple(joints), roots, {"deforming": False}, {"status": "PASS"}, f"skeleton:{w.name}"
    )


def _teacher_weights(w: Witness) -> torch.Tensor:
    p = torch.tensor(w.points, dtype=torch.float32)
    j = torch.tensor(w.joints, dtype=torch.float32)
    d2 = (p[:, None, :] - j[None, :, :]).square().sum(-1)
    a = torch.exp(-d2 / (2.0 * float(w.sigma) ** 2)).clamp_min(1e-12)
    return (a / a.sum(-1, keepdim=True))[None]


def _probe_transforms(joint_count: int) -> torch.Tensor:
    poses = 4
    t = torch.eye(4, dtype=torch.float32)[None, None].repeat(1, poses, joint_count, 1, 1)
    for ji in range(joint_count):
        u = float(ji + 1) / float(joint_count)
        t[0, 1, ji, 0, 3] = 0.10 * u
        t[0, 1, ji, 1, 3] = 0.035 * (-1.0 if ji % 2 else 1.0)
        t[0, 2, ji, 1, 3] = 0.085 * u
        t[0, 2, ji, 2, 3] = 0.030 * (ji - (joint_count - 1) / 2.0)
        t[0, 3, ji, 0, 3] = -0.055 * (ji - (joint_count - 1) / 2.0)
        t[0, 3, ji, 2, 3] = 0.070 * u
    return t


def _codec() -> SkinFieldCodecV1:
    return SkinFieldCodecV1(
        SkinFieldCodecConfigV1(hidden_dim=32, latent_dim=8, encoder_layers=2, decoder_layers=2)
    )


def _motion_and_error_ratio(rest, teacher_w, predicted_w, transforms) -> tuple[float, float]:
    teacher_def = torch_verified_lbs_v1(rest, teacher_w, transforms)
    pred_def = torch_verified_lbs_v1(rest, predicted_w, transforms)
    rest_pose = rest[:, None, :, :].expand_as(teacher_def)
    motion_rms = torch.sqrt((teacher_def - rest_pose).square().mean().clamp_min(1e-12))
    error_rms = torch.sqrt((pred_def - teacher_def).square().mean().clamp_min(1e-12))
    ratio = error_rms / motion_rms.clamp_min(1e-6)
    return float(error_rms.cpu()), float(ratio.cpu())


@torch.no_grad()
def _a0_metrics(codec, cond, teacher, rest, transforms) -> dict[str, float]:
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    codec.eval()
    out = codec(sf, jf, teacher, sm, jm)
    rows = row_l1_error_v1(out.decoded_weights, teacher, sm)
    _, ratio = _motion_and_error_ratio(rest, teacher, out.decoded_weights, transforms)
    simplex = out.decoded_weights.sum(-1)
    return {
        "row_l1_p95": float(torch.quantile(rows, 0.95).cpu()),
        "deformation_ratio": ratio,
        "max_simplex_residual": float((simplex[sm] - 1.0).abs().max().cpu()),
        "negative_weight_count": int((out.decoded_weights[(sm[:, :, None] & jm[:, None, :])] < 0).sum().cpu()),
    }


def _a0_pass(m: dict[str, float]) -> bool:
    return (
        m["row_l1_p95"] <= A0_ROW_L1_P95_MAX
        and m["deformation_ratio"] <= A0_DEFORMATION_RATIO_MAX
        and m["max_simplex_residual"] <= MAX_SIMPLEX
        and m["negative_weight_count"] == 0
    )


def _a0_token(codec, witness: Witness, metrics: dict[str, float], steps: int) -> CodecA0QualificationTokenV1:
    criteria = {
        "row_l1_p95_max": A0_ROW_L1_P95_MAX,
        "deformation_ratio_max": A0_DEFORMATION_RATIO_MAX,
        "max_simplex_residual": MAX_SIMPLEX,
        "negative_weight_count": 0,
        "required_stable_passes": REQUIRED_STABLE,
    }
    criteria_hash = hashlib.sha256(json.dumps(criteria, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CodecA0QualificationTokenV1(
        "PASS", codec.config.config_hash, f"SYNTHETIC_BEHAVIORAL_PANEL:{witness.name}",
        int(steps), criteria_hash, canonical_metrics_hash_v1(metrics),
    )


def _qualified_matrix(qualified, surface_ids, joint_ids) -> torch.Tensor:
    s_index = {sid: i for i, sid in enumerate(surface_ids)}
    j_index = {jid: j for j, jid in enumerate(joint_ids)}
    out = torch.zeros((1, len(surface_ids), len(joint_ids)), dtype=torch.float32)
    if len(qualified.rows) != len(surface_ids):
        raise AssertionError("qualified skin row count drift")
    for row in qualified.rows:
        if row.surface_id not in s_index:
            raise AssertionError("unknown qualified surface row")
        for jid, weight in row.influences:
            if jid not in j_index:
                raise AssertionError("unknown qualified joint influence")
            out[0, s_index[row.surface_id], j_index[jid]] = float(weight)
    return out


@torch.no_grad()
def _a1_shipping_metrics(model, cond, surface, skeleton, teacher, rest, transforms) -> dict[str, float]:
    proposal = model.propose(cond)[0]
    qualified = qualify_skin(surface, skeleton, proposal)
    w = _qualified_matrix(qualified, cond.surface_ids[0], cond.joint_ids[0])
    rows = (w - teacher).abs().sum(-1).reshape(-1)
    _, ratio = _motion_and_error_ratio(rest, teacher, w, transforms)
    simplex = w.sum(-1).reshape(-1)
    return {
        "row_l1_p95": float(torch.quantile(rows, 0.95).cpu()),
        "deformation_ratio": ratio,
        "max_simplex_residual": float((simplex - 1.0).abs().max().cpu()),
        "negative_weight_count": int((w < 0).sum().cpu()),
        "compiler_total_correction_l1": float(qualified.qualification_report["total_correction_l1"]),
        "qualified_row_count": int(len(qualified.rows)),
    }


def _a1_pass(m: dict[str, float], expected_rows: int) -> bool:
    return (
        m["qualified_row_count"] == expected_rows
        and m["row_l1_p95"] <= A1_ROW_L1_P95_MAX
        and m["deformation_ratio"] <= A1_DEFORMATION_RATIO_MAX
        and m["max_simplex_residual"] <= MAX_SIMPLEX
        and m["negative_weight_count"] == 0
        and m["compiler_total_correction_l1"] <= MAX_COMPILER_CORRECTION_L1
    )


def _run_witness(w: Witness) -> dict:
    torch.manual_seed(w.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    surface = _surface(w)
    skeleton = _skeleton(w)
    cond = ArachneConditioningAdapterV2()([surface], [skeleton])
    teacher = _teacher_weights(w)
    rest = torch.tensor(w.points, dtype=torch.float32)[None]
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
    if a0_pass_step is None:
        return {
            "witness": w.name, "seed": w.seed, "status": "FAIL_A0",
            "a0_pass_step": None, "a0_stable": a0_stable, "a0_final": a0_metrics,
            "a0_trace": a0_trace, "a1_pass_step": None,
        }

    token = _a0_token(codec, w, a0_metrics, a0_pass_step)
    token.validate_for(codec)
    model = ArachneCandidateV2(
        codec,
        ArachneCandidateConfigV2(model_dim=32, surface_encoder_layers=1, attention_heads=4, feedforward_dim=64),
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
            model, opt1, cond, teacher, rest, transforms,
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
        "a0_pass_step": a0_pass_step,
        "a0_stable": a0_stable,
        "a0_final": a0_metrics,
        "a0_trace": a0_trace,
        "a1_pass_step": a1_pass_step,
        "a1_stable": a1_stable,
        "a1_final": a1_metrics,
        "a1_trace": a1_trace,
    }


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_preregistered_arachne_codec_behavioral_witness(witness: Witness):
    result = _run_witness(witness)
    print("ARACHNE_CODEC_BEHAVIORAL_PANEL_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))
    assert result["status"] == "PASS", result
    assert result["a0_stable"] >= REQUIRED_STABLE
    assert result["a1_stable"] >= REQUIRED_STABLE
