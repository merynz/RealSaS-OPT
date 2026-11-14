from __future__ import annotations

import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch, RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights, axis_mesh_teacher_address_digest_v1,
    source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_early_semantics import (
    make, source, _source_qualified_axis_teacher_fixture,
)
from tests.models.test_rss_vnext_structural_trainer import make_sample


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(51)
    torch.set_num_threads(1)


def fixture():
    net = make()
    net.mira.codec.requires_grad_(False)
    rgba, camera = source()
    topo = make_sample(net)
    teacher, receipt = _source_qualified_axis_teacher_fixture(net)
    state, _, _ = net.encode_observations(rgba, camera)
    mesh = net.build_mesh_candidate(
        state, topo.topology_teacher_tokens[0],
        source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    receipt = {
        **receipt,
        "mesh_teacher_semantics":
            "SOURCE_NATIVE_TESSA_TEACHER_TOPOLOGY_FOR_AXIS_PRODUCT",
        "mesh_teacher_source_reindex_proven": True,
        "mesh_teacher_tokens_sha256":
            source_surface_tensor_sha256(topo.topology_teacher_tokens),
        "mesh_teacher_valid_mask_sha256":
            source_surface_tensor_sha256(topo.topology_valid_mask),
        "observed_rgba_sha256": source_surface_tensor_sha256(rgba),
        "observed_camera_sha256": source_surface_tensor_sha256(camera),
        "mesh_teacher_topology_digest": axis_mesh_teacher_address_digest_v1(mesh),
    }
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=camera,
        topology_teacher_tokens=topo.topology_teacher_tokens,
        topology_valid_mask=topo.topology_valid_mask,
        axis_teacher_sequence=teacher, axis_teacher_admission_key="axis",
        source_qualification_key="synthetic-unit-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    weights = RSSvNextTrainingWeights(
        topology=0., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0., axis_rig_skin_codes=1.,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1.e-3),
        dataset_manifest_sha256="3"*64, weights=weights,
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        trusted_axis_teacher_registry={"axis": receipt},
    )
    return net, batch, trainer, receipt


def test_axis_uses_same_tessa_mesh_h_as_unified_product_inference(monkeypatch):
    net, batch, trainer, _ = fixture()
    original = net._control_memory
    calls = []

    def capture(state, structural, mesh=None):
        calls.append(mesh)
        return original(state, structural, mesh)

    monkeypatch.setattr(net, "_control_memory", capture)
    report = trainer.train_step(batch)
    assert report.axis_training_conditioning == "PRODUCT_MESH_MATCHED_TESSA_H"
    assert report.eligible_axis_sequences == 1
    assert report.axis_rig_skin_loss is not None
    assert torch.isfinite(report.total_loss)
    assert len(calls) == 1 and calls[0] is not None
    assert calls[0].source_authority == "TEACHER_FORCED_TRAINING_ONLY"
    assert calls[0].faces.shape[0] >= 1
    assert net.control_cross.in_proj_weight.grad is not None
    assert net.control_cross.in_proj_weight.grad.abs().sum() > 0


def test_mesh_receipt_and_exact_index_domain_cannot_be_forged():
    from dataclasses import replace
    net, batch, trainer, _ = fixture()
    altered = batch.topology_teacher_tokens.clone()
    # A token replacement must not silently be treated as source face truth.
    altered[0, -2] ^= 1
    with pytest.raises(ValueError, match="TESSA_SOURCE_BINDING_DRIFT"):
        trainer.objective(replace(batch, topology_teacher_tokens=altered))
    with pytest.raises(ValueError, match="MATCHED_TESSA_MESH_REQUIRED"):
        trainer.objective(replace(batch, topology_teacher_tokens=None))
    with pytest.raises(ValueError, match="MATCHED_TESSA_MESH_REQUIRED"):
        trainer.objective(replace(batch, topology_valid_mask=torch.tensor([False])))


def test_rig_only_pretrain_never_claims_product_axis_mesh_parity(tmp_path):
    net, batch, trainer, receipt = fixture()
    research = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1.e-3),
        dataset_manifest_sha256="3"*64,
        weights=trainer.weights,
        axis_training_policy="RIG_ONLY_PRETRAIN",
        trusted_axis_teacher_registry={"axis": receipt},
    )
    report = research.objective(batch)
    assert report.axis_training_conditioning == "RIG_ONLY_PRETRAIN_NOT_PRODUCT_PARITY"
    trainer.save_checkpoint(tmp_path/"product-axis.pt",sampler_cursor=3)
    with pytest.raises(ValueError, match="CHECKPOINT_OBJECTIVE_POLICY_DRIFT"):
        research.load_checkpoint(tmp_path/"product-axis.pt")
    with pytest.raises(ValueError, match="PRODUCT_AXIS_MESH_MATCHED_POLICY_REQUIRED"):
        RSSvNextStructuralTrainer(
            net, torch.optim.AdamW(net.parameters()),
            dataset_manifest_sha256="3"*64,
            weights=trainer.weights,
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            axis_training_policy="RIG_ONLY_PRETRAIN",
        )
