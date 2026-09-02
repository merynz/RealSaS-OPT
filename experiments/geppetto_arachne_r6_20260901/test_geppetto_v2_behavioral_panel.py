from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2
from experiments.geppetto_arachne_r6_20260901.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.geppetto_eval_v2 import geppetto_metrics_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_train_v2 import geppetto_train_step_v2
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1


# Reuse the optimizer/checkpoint protocol frozen before the first real-family
# failure. The witnesses below were preregistered before observing their result.
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_STEPS = 2048
CHECK_EVERY = 32
REQUIRED_STABLE_PASSES = 3


@dataclass(frozen=True)
class WitnessSpec:
    name: str
    surface_count: int
    control_indices: tuple[int, ...]
    parents: tuple[int, ...]
    phase: float
    amp_y: float
    amp_z: float
    seed: int


# Deliberately heterogeneous, family-independent witnesses: different surface
# geometry, control cardinality and legal rooted topology. These constants are
# test data, not product thresholds or real-family-derived hyperparameters.
WITNESSES = (
    WitnessSpec(
        name="wave_chain_4",
        surface_count=15,
        control_indices=(2, 5, 9, 13),
        parents=(-1, 0, 1, 2),
        phase=0.31,
        amp_y=0.19,
        amp_z=0.13,
        seed=20260911,
    ),
    WitnessSpec(
        name="offset_star_4",
        surface_count=16,
        control_indices=(1, 6, 10, 14),
        parents=(-1, 0, 0, 0),
        phase=0.67,
        amp_y=0.24,
        amp_z=0.09,
        seed=20260912,
    ),
    WitnessSpec(
        name="fork_5",
        surface_count=18,
        control_indices=(1, 5, 8, 12, 16),
        parents=(-1, 0, 0, 1, 2),
        phase=1.03,
        amp_y=0.27,
        amp_z=0.07,
        seed=20260913,
    ),
)


def _surface(spec: WitnessSpec) -> RiggingSurfaceIR:
    nodes = []
    for i in range(spec.surface_count):
        u = float(i) / float(spec.surface_count - 1)
        x = -0.76 + 1.52 * u
        y = spec.amp_y * np.sin(2.0 * np.pi * u + spec.phase)
        z = spec.amp_z * np.cos(np.pi * u + 0.5 * spec.phase)
        nodes.append(SurfaceNode(
            surface_id=f"BI:{spec.name}:S:{i:03d}",
            P=(float(x), float(y), float(z)),
            support_views=(0, 1, 2, 3, 4, 5, 6, 7) if i % 2 == 0 else (0, 2, 4, 6),
            provenance_refs=(f"synthetic:{spec.name}:{i}",),
            source_observation_ids=(f"synthetic:{spec.name}:{i}",),
            raster_bindings=((0, (-0.88 + 1.76 * u, -0.62 + 1.24 * u)),),
            derived_normal=(0.0, 0.0, 1.0),
        ))
    relations = tuple(
        SurfaceRelation(
            f"BI:{spec.name}:R:{i:03d}",
            nodes[i].surface_id,
            nodes[i + 1].surface_id,
            "LOCAL",
            0.9,
        )
        for i in range(spec.surface_count - 1)
    )
    return RiggingSurfaceIR(
        tuple(nodes),
        relations,
        geometry_lineage_hash=f"behavioral-panel-{spec.name}-surface-v1",
        metadata={
            "raster_coordinate_system": "GRID_XY",
            "resolution": 1024,
            "Nd_operator_sha256": "behavioral-panel-local-geometry-v1",
        },
    )


def _config() -> GeppettoCandidateConfigV2:
    return GeppettoCandidateConfigV2(
        model_dim=24,
        knn_k=4,
        local_layers=1,
        global_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=48,
        support_topk=4,
        position_modes=3,
        parent_pair_chunk=4,
        dropout=0.0,
    )


def _teacher(spec: WitnessSpec, surface: RiggingSurfaceIR, conditioning) -> GeppettoTeacherTargetV1:
    assert len(spec.control_indices) == len(spec.parents)
    assert spec.parents[0] == -1
    physical = np.asarray([surface.surface_nodes[i].P for i in spec.control_indices], np.float32)
    normalized = conditioning.normalizations[0].normalize(physical)
    return GeppettoTeacherTargetV1(
        normalized,
        np.asarray(spec.parents, np.int64),
        np.asarray([p < 0 for p in spec.parents], bool),
        True,
    )


def _shipping(model: GeppettoCandidateV2, conditioning, target: GeppettoTeacherTargetV1) -> dict[str, object]:
    device = next(model.parameters()).device
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out, counts = model.generate(f, p, m, resource_step_limit=int(m[0].sum().item()))
    k = int(counts[0])
    result: dict[str, object] = dict(geppetto_metrics_v2(out.positions_normalized[0, :k], target.positions_normalized))
    result["generated_count"] = k
    result["map_mode_indices"] = torch.argmax(out.position_mode_logits[0, :k], dim=-1).cpu().tolist()
    return result


def _qualified_topology_exact(qualified, conditioning, target: GeppettoTeacherTargetV1) -> bool:
    joints = tuple(qualified.joints)
    if len(joints) != len(target.positions_normalized):
        return False
    qpos = conditioning.normalizations[0].normalize(np.asarray([j.position for j in joints], np.float32))
    tpos = np.asarray(target.positions_normalized, np.float32)
    qi, ti = linear_sum_assignment(np.linalg.norm(qpos[:, None, :] - tpos[None, :, :], axis=-1))
    q_to_t = {int(q): int(t) for q, t in zip(qi.tolist(), ti.tolist())}
    id_to_q = {j.canonical_joint_id: i for i, j in enumerate(joints)}
    teacher_parent = np.asarray(target.parent_indices, np.int64)
    for q, t in q_to_t.items():
        parent_id = joints[q].parent_canonical_id
        predicted_parent_t = -1 if parent_id is None else q_to_t[id_to_q[parent_id]]
        if int(predicted_parent_t) != int(teacher_parent[t]):
            return False
    return True


def _qualified_g_pass(model, surface, conditioning, target) -> tuple[bool, str]:
    try:
        proposal = model.propose(conditioning, resource_step_limit=len(surface.surface_nodes))[0]
        if len(proposal.joints) != len(target.positions_normalized):
            return False, f"PROPOSAL_COUNT_{len(proposal.joints)}"
        qualified = qualify_skeleton_v2(surface, proposal)
        if len(qualified.joints) != len(target.positions_normalized):
            return False, f"QUALIFIED_COUNT_{len(qualified.joints)}"
        return _qualified_topology_exact(qualified, conditioning, target), "QUALIFIED"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


@pytest.mark.parametrize("spec", WITNESSES, ids=lambda s: s.name)
def test_preregistered_generic_behavioral_panel(spec: WitnessSpec) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    surface = _surface(spec)
    conditioning = GeppettoConditioningAdapterV2()([surface])
    target = _teacher(spec, surface, conditioning)
    teacher = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    separation = torch.cdist(teacher, teacher)
    separation = separation.masked_fill(torch.eye(len(teacher), dtype=torch.bool), float("inf"))
    unique_radius = 0.5 * float(separation.min())

    model = GeppettoCandidateV2(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    initial = _shipping(model, conditioning, target)

    stable = 0
    pass_step = None
    trace = []
    for step_index in range(1, MAX_STEPS + 1):
        step = geppetto_train_step_v2(model, optimizer, conditioning, [target])
        losses = {k: float(v) for k, v in step.items() if k != "matched_joint_count"}
        assert all(math.isfinite(v) for v in losses.values())
        if step_index != 1 and step_index % CHECK_EVERY:
            continue

        shipping = _shipping(model, conditioning, target)
        geometry_pass = (
            int(shipping["generated_count"]) == len(target.positions_normalized)
            and float(shipping["matched_p95"]) < unique_radius
        )
        topology_pass = False
        topology_status = "NOT_EVALUATED"
        if geometry_pass:
            topology_pass, topology_status = _qualified_g_pass(model, surface, conditioning, target)
        product_pass = bool(geometry_pass and topology_pass)
        stable = stable + 1 if product_pass else 0
        trace.append({
            "step": step_index,
            "shipping": shipping,
            "geometry_pass": geometry_pass,
            "qualified_g_pass": topology_pass,
            "qualified_g_status": topology_status,
            "stable_passes": stable,
        })
        if stable >= REQUIRED_STABLE_PASSES:
            pass_step = step_index
            break

    diagnostic = {
        "witness": spec.name,
        "seed": spec.seed,
        "target_count": len(target.positions_normalized),
        "parents": spec.parents,
        "initial": initial,
        "final": _shipping(model, conditioning, target),
        "unique_radius": unique_radius,
        "pass_step": pass_step,
        "stable_passes": stable,
        "trace": trace,
    }
    print("GEPPETTO_BEHAVIORAL_PANEL_DIAGNOSTIC=" + json.dumps(diagnostic, sort_keys=True))
    assert pass_step is not None, diagnostic
    assert stable >= REQUIRED_STABLE_PASSES, diagnostic
