from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesEvidenceBridge,
    RSSvNextFlexiCubesSurfaceEvidence,
)
from models.rss_vnext.semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec, realsas_style_generic_schema,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,
    RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,
)


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(37)
    torch.set_num_threads(1)


def _root():
    root = os.environ.get("REALSAS_FLEXICUBES_ROOT")
    if not root:
        pytest.skip("Pinned FlexiCubes source only installed in full source CI")
    return Path(root)


class AnalyticLearnableScene(nn.Module):
    """Independent analytic SDF smoke fixture. NOT student/production inference.

    It tests real NVIDIA extraction and autograd, without pretending a random
    untrained image backbone already learned a nonempty signed field.
    """

    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.bias = nn.Parameter(torch.tensor(.03))
        self.feature_gain = nn.Parameter(torch.tensor(1.))
        self.cube_extraction = nn.Sequential(
            nn.Flatten(-2), nn.Linear(width * 8, 21),
        )

    def query(self, triplanes, xyz):
        b, n, _ = xyz.shape
        extra = triplanes.mean() * .05
        field = xyz.norm(dim=-1) - .65 + self.bias + extra
        base = torch.cat(
            [xyz, xyz.square(), xyz[..., :2]], dim=-1,
        ) * self.feature_gain + extra
        assert base.shape == (b, n, self.width)
        return {
            "H": base,
            "field": field,
            "extraction_delta": xyz.new_zeros((b, n, 3)),
        }


def test_exact_pinned_flexicubes_backprop_through_signed_field_semantic_and_cube_weights():
    core = AnalyticLearnableScene()
    decoder = nn.Linear(8, 4)
    # Deterministic *source-style* semantic halfspace x<0.1, intersected
    # with an analytic sphere. One portion of the extracted surface is driven
    # by SDF, another by semantic class boundary, so both must get gradients.
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.zero_()
        decoder.weight[0, 0] = 2.0
        decoder.bias[0] = -0.2
    bridge = RSSvNextFlexiCubesEvidenceBridge(core, resolution=5)
    source = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    extractor = PinnedFlexiCubesV1(_root(), device="cpu")
    evidence = bridge(
        source, extractor=extractor, role_head=decoder,
        schema=realsas_style_generic_schema(),
        layer=RSSvNextSemanticLayerSpec(
            "without_secondary", ("focal_detail", "primary_form", "outer_covering"),
        ),
        field_policy="SIGNED_HYPOTHESIS_ONLY",
    )
    assert evidence.xyz.shape[1] > 4
    assert evidence.faces.shape[1] == 3
    assert evidence.cube_parameters.shape == (5**3, 21)
    assert evidence.evidence_authority == "UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS"
    # Explicitly optimize extracted surface, not merely a disconnected
    # regularization head. Expect gradients through real FC to field and cube.
    loss = evidence.xyz.square().mean() + .1 * evidence.regularizer.mean()
    loss.backward()
    assert core.bias.grad is not None and torch.isfinite(core.bias.grad)
    assert core.bias.grad.abs() > 1e-8
    assert core.cube_extraction[-1].weight.grad is not None
    assert core.cube_extraction[-1].weight.grad.abs().sum() > 1e-8
    assert decoder.weight.grad is not None
    assert decoder.weight.grad.abs().sum() > 1e-8
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_open_or_unknown_signed_domain_abstains_not_faked():
    core = AnalyticLearnableScene()
    role = nn.Linear(8, 4)
    bridge = RSSvNextFlexiCubesEvidenceBridge(core, resolution=3)
    extractor = PinnedFlexiCubesV1(_root(), device="cpu")
    planes = torch.zeros(1, 3, 2, 2, 2)
    layer = RSSvNextSemanticLayerSpec("all", realsas_style_generic_schema().roles)
    with pytest.raises(ValueError, match="SIGNED_DOMAIN_NOT_PERMITTED"):
        bridge(
            planes, extractor=extractor, role_head=role,
            schema=realsas_style_generic_schema(), layer=layer,
            field_policy="OPEN_OR_UNKNOWN_UDF",
        )


def test_structural_transformer_reads_witness_surface_features_without_adopting_faces():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = torch.zeros(1, 2, 13)
    cameras[..., 3] = cameras[..., 7] = cameras[..., 11] = cameras[..., 12] = 1
    state, auxiliary, _ = net.encode_observations(rgba, cameras)
    k = cfg.semantic_role_schema().count
    xyz = state.query_xyz[:, :4].clone()
    witness = RSSvNextFlexiCubesSurfaceEvidence(
        xyz=xyz,
        faces=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
        local_features=state.local_features[:, :4] * 1.1,
        presentation_role_logits=net.presentation_role_head(state.local_features[:, :4]),
        regularizer=state.local_features.new_zeros(1),
        sample_grid_xyz=state.query_xyz,
        signed_field_grid=state.geometry["field"],
        cube_parameters=state.local_features.new_zeros(1, 21),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="all",
    )
    witness.validate(carrier_dim=cfg.iris.local_feature_dim, role_count=k)
    normal_memory = net._structural_memory(state, auxiliary)
    witnessed_memory = net._structural_memory(state, auxiliary, flexicubes=witness)
    assert normal_memory.shape == witnessed_memory.shape
    assert not torch.allclose(normal_memory, witnessed_memory)
    logits = net.tessa.decode_tokens(
        memory=witnessed_memory,
        input_ids=torch.full((1, 4), cfg.tessa.BOS, dtype=torch.long),
    ).logits
    logits.square().mean().backward()
    assert net.fc_surface_position.weight.grad is not None
    assert net.fc_surface_position.weight.grad.abs().sum() > 0
    assert net.vision.patch.weight.grad is not None
    assert net.iris.scene.scene_queries.grad is not None
    # TESSA remains responsible for new topology tokens; FC faces are just a witness.



def test_exact_fc_and_tessa_share_one_two_view_training_optimizer():
    """An admitted source surface supervises the real pinned extractor.

    Synthetic sphere labels and analytical signed field are ONLY for this
    mechanism test. This does not establish real corpus eligibility.
    """
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    original = net.iris.query

    def analytic_field_with_learned_carrier(planes, xyz):
        out = original(planes, xyz)
        # Ensure a bona-fide signed field while retaining gradients to the
        # learned geometry field and shared source triplanes.
        out["field"] = xyz.norm(dim=-1) - .65 + .03 * out["field"]
        return out

    net.iris.query = analytic_field_with_learned_carrier
    inp = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[..., 3] = cam[..., 7] = cam[..., 11] = cam[..., 12] = 1
    teacher_surface = torch.tensor([[
        [.65, 0, 0], [-.65, 0, 0], [0, .65, 0],
        [0, -.65, 0], [0, 0, .65], [0, 0, -.65],
    ]], dtype=torch.float32)
    batch = RSSvNextStructuralTrainingBatch(
        rgba=inp, camera_features=cam,
        flexicubes_extractor=PinnedFlexiCubesV1(_root(), device="cpu"),
        flexicubes_layer=RSSvNextSemanticLayerSpec(
            "all", cfg.semantic_role_schema().roles,
        ),
        flexicubes_reference_surface_xyz=teacher_surface,
        flexicubes_signed_source_admission_key="synthetic-unit-only-watertight-sphere",
        source_qualification_key="synthetic-unit-not-real-admission",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    from models.rss_vnext.training_engine import source_surface_tensor_sha256
    # Fixture registry stands in for an *external* scientific source court.
    # This test proves admission wiring only, not real source qualification.
    signed_receipt = {
        "admission_status": "QUALIFIED_SOURCE_SIGNED_SOLID",
        "source_component_domain": "SINGLE_QUALIFIED_CLOSED_COMPONENT",
        "dataset_manifest_sha256": "a"*64,
        "source_qualification_key": batch.source_qualification_key,
        "role_schema_id": cfg.semantic_role_schema_id,
        "layer_name": batch.flexicubes_layer.name,
        "surface_target_sha256": source_surface_tensor_sha256(teacher_surface),
        "animation_source_asset_sha256": None,
        "animation_source_rig_sha256": None,
        "animation_source_poses_sha256": None,
    }
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-4),
        dataset_manifest_sha256="a"*64,
        trusted_signed_source_registry={
            batch.flexicubes_signed_source_admission_key: signed_receipt,
        },
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=0,
            flexicubes_surface=1,
        ),
    )
    step = trainer.train_step(batch)
    assert step.flexicubes_surface_loss is not None
    assert torch.isfinite(step.flexicubes_surface_loss)
    assert trainer.global_step == 1
    assert net.iris.field.field_head.weight.grad is not None
    assert net.iris.cube_extraction.net[-1].weight.grad is not None
    assert net.iris.cube_extraction.net[-1].weight.grad.abs().sum() > 0
    assert net.vision.patch.weight.grad is not None

    batch.flexicubes_signed_source_admission_key = ""
    with pytest.raises(ValueError, match="SIGNED_SOURCE_PROOF_REQUIRED"):
        trainer.objective(batch)
    batch.flexicubes_signed_source_admission_key = "forged-source"
    with pytest.raises(ValueError, match="FC_SOURCE_SIGNED_DOMAIN_RECEIPT_DRIFT"):
        trainer.objective(batch)
    batch.flexicubes_signed_source_admission_key = "synthetic-unit-only-watertight-sphere"
    batch.flexicubes_reference_surface_xyz = teacher_surface + .1
    with pytest.raises(ValueError, match="FC_SOURCE_SIGNED_DOMAIN_RECEIPT_DRIFT"):
        trainer.objective(batch)


def test_signed_source_teacher_registry_is_pinned_to_checkpoint(tmp_path):
    """A new scientific foundry receipt cannot silently reuse old gradients."""
    cfg = RSSvNextConfig.tiny_for_test()
    a = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_SIGNED_SOLID",
        "source_component_domain": "SINGLE_QUALIFIED_CLOSED_COMPONENT",
        "dataset_manifest_sha256": "a" * 64,
        "source_qualification_key": "synthetic-only",
        "role_schema_id": cfg.semantic_role_schema_id,
        "layer_name": "all",
        "surface_target_sha256": "b" * 64,
        "animation_source_asset_sha256": None,
        "animation_source_rig_sha256": None,
        "animation_source_poses_sha256": None,
    }
    first = RSSvNextStructuralTrainer(
        a, torch.optim.AdamW(a.parameters()),
        dataset_manifest_sha256="a" * 64,
        trusted_signed_source_registry={"source": receipt},
    )
    path = tmp_path / "signed_source_training.pt"
    first.save_checkpoint(path, sampler_cursor=2)
    b = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    changed = {**receipt, "source_component_domain": "OPEN_SURFACE"}
    other = RSSvNextStructuralTrainer(
        b, torch.optim.AdamW(b.parameters()),
        dataset_manifest_sha256="a" * 64,
        trusted_signed_source_registry={"source": changed},
    )
    with pytest.raises(ValueError, match="CHECKPOINT_SIGNED_SOURCE_REGISTRY_DRIFT"):
        other.load_checkpoint(path)
