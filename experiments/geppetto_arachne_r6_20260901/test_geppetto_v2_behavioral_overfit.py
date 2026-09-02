from __future__ import annotations

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
    # Three controls are deliberately selected from geometry only. Their count,
    # locations and topology are unrelated to any real-family witness.
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


def _shipping_metrics(model: GeppettoCandidateV2, conditioning, target: GeppettoTeacherTargetV1) -> dict[str, float]:
    device = next(model.parameters()).device
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out, counts = model.generate(f, p, m, resource_step_limit=int(m[0].sum().item()))
    k = int(counts[0])
    metrics = geppetto_metrics_v2(out.positions_normalized[0, :k], target.positions_normalized)
    metrics["generated_count"] = k
    return metrics


def _qualified_topology_is_exact(qualified, conditioning, target: GeppettoTeacherTargetV1) -> bool:
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


def test_small_generic_witness_optimizes_shipping_decode_and_compiler_qualified_g() -> None:
    torch.manual_seed(20260903)
    np.random.seed(20260903)

    surface = _surface()
    conditioning = GeppettoConditioningAdapterV2()([surface])
    target = _teacher(surface, conditioning)
    model = GeppettoCandidateV2(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    initial = _shipping_metrics(model, conditioning, target)
    first_loss = None
    last_loss = None
    for _ in range(192):
        step = geppetto_train_step_v2(model, optimizer, conditioning, [target])
        if first_loss is None:
            first_loss = float(step["total"])
        last_loss = float(step["total"])

    final = _shipping_metrics(model, conditioning, target)

    # Behavioral-integrity assertions: optimizer progress must be visible in the
    # exact object shipped by generate(), not only in an auxiliary latent mode.
    assert last_loss is not None and first_loss is not None and last_loss < first_loss
    assert final["generated_count"] == len(target.positions_normalized)
    assert final["matched_p95"] < initial["matched_p95"]

    teacher = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    sep = torch.cdist(teacher, teacher)
    sep = sep.masked_fill(torch.eye(len(teacher), dtype=torch.bool), float("inf"))
    nearest_teacher_separation = float(sep.min())
    # Below half the nearest teacher separation, each predicted locus lies inside
    # a unique teacher Voronoi neighborhood; this is geometry-derived, not a
    # real-family or product-tuned tolerance.
    assert final["matched_p95"] < 0.5 * nearest_teacher_separation

    proposal = model.propose(conditioning, resource_step_limit=len(surface.surface_nodes))[0]
    assert len(proposal.joints) == len(target.positions_normalized)
    qualified = qualify_skeleton_v2(surface, proposal)
    assert len(qualified.joints) == len(target.positions_normalized)
    assert _qualified_topology_is_exact(qualified, conditioning, target)
