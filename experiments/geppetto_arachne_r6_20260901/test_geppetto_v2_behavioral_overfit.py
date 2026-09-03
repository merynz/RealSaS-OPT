from __future__ import annotations

import json
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2
from experiments.geppetto_arachne_r6_20260901.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.geppetto_eval_v2 import geppetto_metrics_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_train_v2 import geppetto_train_step_v2
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1


OP_HASH = "behavioral-integrity-synthetic-local-geometry-v1"

# These optimizer/horizon semantics were already frozen in the first-family fit
# harness before its first optimizer run. They are reused here as generic fitting
# protocol, not tuned from any observed family outcome.
FROZEN_HARNESS_LR = 3e-4
FROZEN_HARNESS_WEIGHT_DECAY = 1e-4
FROZEN_HARNESS_MAX_STEPS = 2048
FROZEN_HARNESS_CHECK_EVERY = 32
FROZEN_HARNESS_REQUIRED_STABLE_PASSES = 3


def _surface(count: int = 12) -> RiggingSurfaceIR:
    nodes = []
    for i in range(count):
        u = float(i) / float(count - 1)
        x = -0.72 + 1.44 * u
        y = 0.22 * np.sin(2.0 * np.pi * u)
        z = 0.11 * np.cos(np.pi * u)
        nodes.append(SurfaceNode(
            surface_id=f"BI:S:{i:03d}",
            P=(float(x), float(y), float(z)),
            support_views=(0, 1, 2, 3, 4, 5, 6, 7),
            provenance_refs=(f"synthetic:{i}",),
            source_observation_ids=(f"synthetic:{i}",),
            raster_bindings=((0, (-0.85 + 1.7 * u, -0.55 + 1.1 * u)),),
            derived_normal=(0.0, 0.0, 1.0),
        ))
    relations = tuple(
        SurfaceRelation(f"BI:R:{i:03d}", nodes[i].surface_id, nodes[i + 1].surface_id, "LOCAL", 0.9)
        for i in range(count - 1)
    )
    return RiggingSurfaceIR(
        tuple(nodes),
        relations,
        geometry_lineage_hash="behavioral-integrity-synthetic-surface-v1",
        metadata={
            "raster_coordinate_system": "GRID_XY",
            "resolution": 1024,
            "Nd_operator_sha256": OP_HASH,
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


def _teacher(surface: RiggingSurfaceIR, conditioning) -> GeppettoTeacherTargetV1:
    physical = np.asarray([
        surface.surface_nodes[2].P,
        surface.surface_nodes[6].P,
        surface.surface_nodes[10].P,
    ], np.float32)
    normalized = conditioning.normalizations[0].normalize(physical)
    return GeppettoTeacherTargetV1(
        normalized,
        np.asarray([-1, 0, 1], np.int64),
        np.asarray([True, False, False]),
        True,
    )


def _shipping_metrics(model: GeppettoCandidateV2, conditioning, target: GeppettoTeacherTargetV1) -> dict[str, object]:
    device = next(model.parameters()).device
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out, counts = model.generate(f, p, m, resource_step_limit=int(m[0].sum().item()))
    k = int(counts[0])
    metrics: dict[str, object] = dict(geppetto_metrics_v2(out.positions_normalized[0, :k], target.positions_normalized))
    metrics["generated_count"] = k
    metrics["map_mode_indices"] = torch.argmax(out.position_mode_logits[0, :k], dim=-1).cpu().tolist()
    metrics["stop_probabilities"] = torch.sigmoid(out.stop_logits[0, :min(k + 2, out.stop_logits.shape[1])]).cpu().tolist()
    return metrics


def _teacher_topology_exact_diagnostic(qualified, conditioning, target: GeppettoTeacherTargetV1) -> bool:
    joints = tuple(qualified.joints)
    if len(joints) != len(target.positions_normalized):
        return False
    qpos_physical = np.asarray([j.position for j in joints], np.float32)
    qpos = conditioning.normalizations[0].normalize(qpos_physical)
    tpos = np.asarray(target.positions_normalized, np.float32)
    cost = np.linalg.norm(qpos[:, None, :] - tpos[None, :, :], axis=-1)
    qi, ti = linear_sum_assignment(cost)
    q_to_t = {int(q): int(t) for q, t in zip(qi.tolist(), ti.tolist())}
    id_to_q = {j.canonical_joint_id: i for i, j in enumerate(joints)}
    teacher_parent = np.asarray(target.parent_indices, np.int64)
    for q, t in q_to_t.items():
        parent_id = joints[q].parent_canonical_id
        predicted_parent_t = -1 if parent_id is None else q_to_t[id_to_q[parent_id]]
        if int(predicted_parent_t) != int(teacher_parent[t]):
            return False
    return True


def _qualified_mechanical_status(model, surface, conditioning, target) -> tuple[bool, str, bool]:
    """Apply the canonical MECHANICAL_STRUCTURE acceptance semantics to actual G.

    Teacher graph equality is intentionally retained only as a diagnostic. The
    frozen first-family contract explicitly says it is not the product objective.
    """
    try:
        proposal = model.propose(conditioning, resource_step_limit=len(surface.surface_nodes))[0]
        if len(proposal.joints) != len(target.positions_normalized):
            return False, f"PROPOSAL_COUNT_{len(proposal.joints)}", False
        qualified = qualify_skeleton_v2(surface, proposal)
        if len(qualified.joints) != len(target.positions_normalized):
            return False, f"QUALIFIED_COUNT_{len(qualified.joints)}", False
        ids = {j.canonical_joint_id for j in qualified.joints}
        illegal_parent_count = sum(
            j.parent_canonical_id is not None and j.parent_canonical_id not in ids
            for j in qualified.joints
        )
        deform_root_count = len(qualified.deform_root_ids)
        mechanical_pass = (
            len(qualified.joints) > 0
            and deform_root_count > 0
            and illegal_parent_count == 0
        )
        teacher_exact = _teacher_topology_exact_diagnostic(qualified, conditioning, target)
        status = (
            f"QUALIFIED_MECHANICAL:roots={deform_root_count}:illegal_parents={int(illegal_parent_count)}"
        )
        return bool(mechanical_pass), status, bool(teacher_exact)
    except Exception as exc:  # failure is evidence; preserve type/message in trace
        return False, f"{type(exc).__name__}:{exc}", False


def test_small_generic_witness_optimizes_shipping_decode_and_compiler_qualified_g() -> None:
    # Scientific gate: repeated CI executions of one commit must have one result.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260903)
    np.random.seed(20260903)

    surface = _surface()
    conditioning = GeppettoConditioningAdapterV2()([surface])
    target = _teacher(surface, conditioning)
    teacher = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    sep = torch.cdist(teacher, teacher)
    sep = sep.masked_fill(torch.eye(len(teacher), dtype=torch.bool), float("inf"))
    nearest_teacher_separation = float(sep.min())
    unique_radius = 0.5 * nearest_teacher_separation

    model = GeppettoCandidateV2(_config())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_HARNESS_LR,
        weight_decay=FROZEN_HARNESS_WEIGHT_DECAY,
    )

    initial = _shipping_metrics(model, conditioning, target)
    trace = []
    stable = 0
    pass_step = None
    last_step_loss: dict[str, float] | None = None

    for step_index in range(1, FROZEN_HARNESS_MAX_STEPS + 1):
        step = geppetto_train_step_v2(model, optimizer, conditioning, [target])
        last_step_loss = {k: float(v) for k, v in step.items() if k != "matched_joint_count"}
        assert all(math.isfinite(v) for v in last_step_loss.values())

        if step_index != 1 and step_index % FROZEN_HARNESS_CHECK_EVERY:
            continue

        shipping = _shipping_metrics(model, conditioning, target)
        geometry_pass = (
            int(shipping["generated_count"]) == len(target.positions_normalized)
            and float(shipping["matched_p95"]) < unique_radius
        )
        mechanical_pass = False
        mechanical_status = "NOT_EVALUATED"
        teacher_topology_exact = False
        if geometry_pass:
            mechanical_pass, mechanical_status, teacher_topology_exact = _qualified_mechanical_status(
                model, surface, conditioning, target
            )

        product_pass = bool(geometry_pass and mechanical_pass)
        stable = stable + 1 if product_pass else 0
        trace.append({
            "step": step_index,
            "loss": {k: round(v, 6) for k, v in last_step_loss.items()},
            "shipping": shipping,
            "geometry_pass": geometry_pass,
            "qualified_mechanical_pass": mechanical_pass,
            "qualified_mechanical_status": mechanical_status,
            "teacher_topology_exact_diagnostic": teacher_topology_exact,
            "stable_passes": stable,
        })
        if stable >= FROZEN_HARNESS_REQUIRED_STABLE_PASSES:
            pass_step = step_index
            break

    final = _shipping_metrics(model, conditioning, target)
    diagnostic = {
        "initial": initial,
        "final": final,
        "nearest_teacher_separation": nearest_teacher_separation,
        "unique_radius": unique_radius,
        "pass_step": pass_step,
        "stable_passes": stable,
        "protocol": {
            "lr": FROZEN_HARNESS_LR,
            "weight_decay": FROZEN_HARNESS_WEIGHT_DECAY,
            "max_steps": FROZEN_HARNESS_MAX_STEPS,
            "check_every": FROZEN_HARNESS_CHECK_EVERY,
            "required_stable_passes": FROZEN_HARNESS_REQUIRED_STABLE_PASSES,
        },
        "trace": trace,
    }
    print("GEPPETTO_BEHAVIORAL_OVERFIT_DIAGNOSTIC=" + json.dumps(diagnostic, sort_keys=True))

    # Loss and teacher-topology equality are diagnostic only. PASS authority is
    # the actual shipping decode plus Compiler-qualified G under the canonical
    # MECHANICAL_STRUCTURE semantics, sustained for the preregistered checks.
    assert pass_step is not None, diagnostic
    assert stable >= FROZEN_HARNESS_REQUIRED_STABLE_PASSES, diagnostic
