from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v1 import ArachneCandidateV1
from experiments.geppetto_arachne_r6_20260901.candidate_config_v1 import (
    ArachneCandidateConfigV1,
    GeppettoCandidateConfigV1,
    SkinFieldCodecConfigV1,
)
from experiments.geppetto_arachne_r6_20260901.conditioning_v1 import (
    ArachneConditioningAdapter,
    GeppettoConditioningAdapter,
)
from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v1 import GeppettoCandidateV1
from experiments.geppetto_arachne_r6_20260901.geppetto_loss_v1 import GeppettoLossV1
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import (
    SkinFieldCodecV1,
    skin_field_codec_loss_v1,
)
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1


def _surface(prefix: str, count: int) -> RiggingSurfaceIR:
    nodes = []
    for i in range(count):
        nodes.append(SurfaceNode(
            surface_id=f"{prefix}:S:{i}",
            P=(float(i) * 0.17, float(i % 2) * 0.23, float(i % 3) * 0.11),
            support_views=(0, 1, 2, 3, 4, 5, 6, 7) if i % 2 == 0 else (0, 2, 4, 6),
            provenance_refs=(f"obs:{i}",),
            source_observation_ids=(f"obs:{i}",),
            raster_bindings=((0, (10.0 + i, 20.0 + i)),),
            derived_normal=(0.0, 0.0, 1.0),
        ))
    rel = tuple(
        SurfaceRelation(f"R:{i}", nodes[i].surface_id, nodes[i + 1].surface_id, "LOCAL", 0.8)
        for i in range(max(0, count - 1))
    )
    return RiggingSurfaceIR(tuple(nodes), rel, geometry_lineage_hash=f"surface-hash-{prefix}")


def _skeleton(prefix: str, joint_count: int) -> QualifiedSkeletonIRV2:
    joints = []
    for i in range(joint_count):
        joints.append(QualifiedJoint(
            canonical_joint_id=f"J:{prefix}:{i}",
            position=(float(i) * 0.13, float(i) * 0.07, 0.0),
            parent_canonical_id=None if i == 0 else f"J:{prefix}:{i-1}",
            support_surface_ids=(f"{prefix}:S:{min(i, 2)}",),
            source_proposal_id=f"P:{i}",
        ))
    return QualifiedSkeletonIRV2(
        tuple(joints),
        deform_root_ids=(joints[0].canonical_joint_id,),
        assembly_root_binding={"deforming": False},
        qualification_report={"status": "PASS"},
        skeleton_lineage_hash=f"skeleton-hash-{prefix}-{joint_count}",
    )


def test_conditioning_is_deterministic_permutation_safe_and_teacher_free():
    surface = _surface("A", 4)
    reversed_surface = RiggingSurfaceIR(
        tuple(reversed(surface.surface_nodes)),
        tuple(reversed(surface.local_relations)),
        geometry_lineage_hash=surface.geometry_lineage_hash,
    )
    adapter = GeppettoConditioningAdapter()
    a = adapter([surface])
    b = adapter([reversed_surface])
    assert a.surface_ids == b.surface_ids
    np.testing.assert_allclose(a.features, b.features)
    assert a.conditioning_hashes == b.conditioning_hashes
    params = set(inspect.signature(GeppettoConditioningAdapter.__call__).parameters)
    assert params == {"self", "surfaces"}
    assert "teacher" not in vars(a) and "source_bone" not in repr(vars(a))


def test_geppetto_shapes_soft_proposals_and_hungarian_loss_backward():
    surface = _surface("A", 5)
    cond = GeppettoConditioningAdapter()([surface])
    cfg = GeppettoCandidateConfigV1(model_dim=32, encoder_layers=1, decoder_layers=1, attention_heads=4, feedforward_dim=64, max_joint_queries=4, support_topk=2)
    model = GeppettoCandidateV1(cfg)
    f = torch.tensor(cond.features, dtype=torch.float32)
    p = torch.tensor(cond.positions_normalized, dtype=torch.float32)
    m = torch.tensor(cond.valid_mask, dtype=torch.bool)
    out = model(f, p, m)
    assert out.existence_logits.shape == (1, 4)
    assert out.parent_logits.shape == (1, 4, 4)
    assert out.support_logits.shape == (1, 4, 5)

    target = GeppettoTeacherTargetV1(
        positions_normalized=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]], np.float32),
        parent_indices=np.asarray([-1, 0], np.int64),
        root_mask=np.asarray([True, False]),
    )
    losses = GeppettoLossV1()(out, [target])
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        model.count[1].weight.zero_()
        model.count[1].bias.zero_()
        model.count[1].bias[2] = 10.0
        model.existence.weight.zero_()
        model.existence.bias.fill_(8.0)
    proposal = model.propose(cond)[0]
    assert len(proposal.joints) == 2
    assert all(j.proposal_id.startswith("P:JQ:") and not j.proposal_id.startswith("J:") for j in proposal.joints)
    assert proposal.edges and all(not e.hard_required and not e.hard_forbidden for e in proposal.edges)
    assert proposal.metadata["query_indices_are_not_identity"] is True
    assert proposal.metadata["full_3d_reconstruction_claim"] is False


def _teacher_batch(cond):
    b, n, _ = cond.surface_features.shape
    j = cond.joint_features.shape[1]
    w = torch.zeros((b, n, j), dtype=torch.float32)
    for bi in range(b):
        valid_n = len(cond.surface_ids[bi])
        valid_j = len(cond.joint_ids[bi])
        raw = torch.arange(1, valid_n * valid_j + 1, dtype=torch.float32).reshape(valid_n, valid_j)
        raw = raw / raw.sum(dim=1, keepdim=True)
        w[bi, :valid_n, :valid_j] = raw
    return w


def test_codec_and_arachne_variable_cardinality_lineage_and_simplex():
    surfaces = [_surface("A", 5), _surface("B", 3)]
    skeletons = [_skeleton("A", 3), _skeleton("B", 2)]
    cond = ArachneConditioningAdapter()(surfaces, skeletons)
    codec_cfg = SkinFieldCodecConfigV1(hidden_dim=32, latent_dim=8, encoder_layers=2, decoder_layers=2)
    codec = SkinFieldCodecV1(codec_cfg)
    sf = torch.tensor(cond.surface_features, dtype=torch.float32)
    jf = torch.tensor(cond.joint_features, dtype=torch.float32)
    sm = torch.tensor(cond.surface_mask, dtype=torch.bool)
    jm = torch.tensor(cond.joint_mask, dtype=torch.bool)
    teacher = _teacher_batch(cond)
    codec_out = codec(sf, jf, teacher, sm, jm)
    assert codec_out.latents.shape == (2, 3, 8)
    for bi in range(2):
        vn, vj = len(cond.surface_ids[bi]), len(cond.joint_ids[bi])
        sums = codec_out.decoded_weights[bi, :vn, :vj].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)
        assert (codec_out.decoded_weights[bi, :vn, :vj] >= 0).all()
    codec_losses = skin_field_codec_loss_v1(codec_out.decoded_weights, teacher, sm, jm)
    codec_losses["total"].backward()
    assert torch.isfinite(codec_losses["total"])

    arachne_cfg = ArachneCandidateConfigV1(model_dim=32, encoder_layers=1, cross_attention_layers=1, attention_heads=4, feedforward_dim=64)
    model = ArachneCandidateV1(codec, arachne_cfg)
    raw = model(sf, jf, sm, jm, torch.tensor(cond.parent_indices, dtype=torch.long))
    assert raw.decoded_weights.shape == teacher.shape
    proposals = model.propose(cond)
    for bi, proposal in enumerate(proposals):
        assert proposal.surface_binding_hash == cond.source_surface_hashes[bi]
        assert proposal.skeleton_binding_hash == cond.source_skeleton_hashes[bi]
        assert len(proposal.influences) == len(cond.surface_ids[bi]) * len(cond.joint_ids[bi])
        rows = {}
        for inf in proposal.influences:
            rows.setdefault(inf.surface_id, 0.0)
            rows[inf.surface_id] += inf.weight
        assert set(rows) == set(cond.surface_ids[bi])
        assert all(abs(v - 1.0) < 1e-5 for v in rows.values())
        assert proposal.metadata["compiler_owns_sparsification_and_qualification"] is True
        assert proposal.metadata["full_3d_reconstruction_claim"] is False


def test_arachne_conditioning_hash_binds_qualified_skeleton_lineage_and_topology():
    surface = _surface("A", 4)
    skel2 = _skeleton("A", 2)
    skel3 = _skeleton("A", 3)
    adapter = ArachneConditioningAdapter()
    a = adapter([surface], [skel2])
    b = adapter([surface], [skel3])
    assert a.conditioning_hashes[0] != b.conditioning_hashes[0]
    assert a.source_skeleton_hashes[0] == skel2.skeleton_lineage_hash
    assert b.source_skeleton_hashes[0] == skel3.skeleton_lineage_hash
