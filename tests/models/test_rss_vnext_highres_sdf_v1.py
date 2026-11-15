from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.flexicubes_surface_evidence import RSSvNextFlexiCubesEvidenceBridge
from models.rss_vnext.semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec, realsas_style_generic_schema,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights, source_surface_tensor_sha256,
)


@pytest.fixture(autouse=True)
def seed_cpu():
    torch.manual_seed(61)
    torch.set_num_threads(1)


def _direct_grid(res):
    # Direct algorithm exercised even when pinned source is absent.
    adapter = PinnedFlexiCubesV1.__new__(PinnedFlexiCubesV1)
    adapter.device = torch.device("cpu")
    return adapter.construct_iris_grid(res)


def test_product_irIS_high_res_but_training_remains_small():
    cfg = RSSvNextConfig()
    assert cfg.iris.extraction_training_resolution == 24
    assert cfg.iris.extraction_inference_resolution == (100, 100, 150)
    assert cfg.iris.extraction_field_query_chunk <= 16384
    assert cfg.iris.extraction_cube_query_chunk <= 2048
    assert cfg.iris.extraction_surface_query_chunk <= 16384
    cfg.validate()
    assert RSSvNextConfig.tiny_for_test().iris.extraction_inference_resolution == (4, 4, 4)
    with pytest.raises(ValueError, match="INFERENCE_RESOLUTION_BUDGET"):
        replace(
            cfg.iris, extraction_inference_resolution=(256, 256, 256),
        ).validate()


def test_indexed_anisotropic_grid_has_no_seams_missing_cells_or_invalid_corner_ids():
    nx, ny, nz = 3, 4, 5
    xyz, cubes = _direct_grid((nx, ny, nz))
    assert xyz.shape == ((nx+1)*(ny+1)*(nz+1), 3)
    assert cubes.shape == (nx*ny*nz, 8)
    assert cubes.dtype == torch.long
    assert cubes.min() == 0
    assert int(cubes.max()) < len(xyz)
    assert torch.unique(xyz, dim=0).shape == xyz.shape
    assert torch.allclose(
        xyz[cubes[0]],
        torch.tensor([
            [-1., -1., -1.],
            [-1. + 2./nx, -1., -1.],
            [-1., -1.+2./ny, -1.],
            [-1.+2./nx, -1.+2./ny, -1.],
            [-1., -1., -1.+2./nz],
            [-1.+2./nx, -1., -1.+2./nz],
            [-1., -1.+2./ny, -1.+2./nz],
            [-1.+2./nx, -1.+2./ny, -1.+2./nz],
        ]),
        atol=1.e-6, rtol=0,
    )
    # Cubes sharing a face must share identical vertex IDs, not just xyz.
    assert torch.equal(torch.sort(cubes[0, [1,3,5,7]]).values,
                       torch.sort(cubes[ny*nz, [0,2,4,6]]).values)


class _AnalyticIRIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.cube_extraction = nn.Sequential(nn.Flatten(-2), nn.Linear(8*8, 21))
        self.max_query = 0
    def query(self, planes, points):
        self.max_query = max(self.max_query, int(points.shape[1]))
        xyz = points
        x = torch.cat((xyz, xyz.square(), xyz[..., :2]), dim=-1)
        return {
            "H": x,
            "field": xyz.norm(dim=-1) - 0.65,
            "extraction_delta": torch.zeros_like(xyz),
        }


class _FakeExtractor:
    def __init__(self):
        self.n_cells = 0
        self.active_params = 0
        self.resolution_seen = None
    def construct_iris_grid(self, resolution):
        return _direct_grid(resolution)
    def extract(self, *, vertices_iris, field, cubes, raw_parameters,
                resolution, training):
        self.n_cells = int(cubes.shape[0])
        self.active_params = int((raw_parameters.abs().sum(dim=-1) > 0).sum())
        self.resolution_seen = resolution
        self.training_seen = bool(training)
        # Fake fixed connectivity proves bounded dataflow, not FC geometry.
        verts = vertices_iris[:4]
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
        return verts, faces, vertices_iris.new_zeros(1)


def test_product_surface_queries_bounded_chunks_and_only_active_cubes():
    iris = _AnalyticIRIS().eval()
    layer = RSSvNextSemanticLayerSpec("all", realsas_style_generic_schema().roles)
    role = nn.Linear(8, 4).eval()
    adapter = _FakeExtractor()
    bridge = RSSvNextFlexiCubesEvidenceBridge(
        iris, resolution=3,
        inference_resolution=(6, 5, 7),
        field_query_chunk=17, cube_query_chunk=3, surface_query_chunk=2,
        max_cubes=512,
    ).eval()
    with torch.no_grad():
        evidence = bridge(
            torch.ones(1, 3, 2, 2, 2),
            extractor=adapter, role_head=role,
            schema=realsas_style_generic_schema(), layer=layer,
            field_policy="SIGNED_HYPOTHESIS_ONLY",
        )
    assert adapter.resolution_seen == (6,5,7)
    assert adapter.training_seen is False
    assert adapter.n_cells == 6*5*7
    assert 0 < adapter.active_params < adapter.n_cells
    assert iris.max_query <= max(17, 3*8, 2)
    assert evidence.sample_grid_xyz.shape == (1, (6+1)*(5+1)*(7+1), 3)
    assert evidence.cube_parameters.shape == (6*5*7, 21)
    assert evidence.xyz.shape == (1, 4, 3)


def _tiny_sdf_batch_and_receipt(model, *, domain):
    rgba = torch.rand(1,2,4,8,8)
    cameras = torch.zeros(1,2,13)
    cameras[..., 3] = cameras[..., 7] = cameras[..., 11] = cameras[..., 12] = 1.
    state, _, _ = model.encode_observations(rgba, cameras)
    xyz = state.query_xyz
    sdf = xyz.norm(dim=-1) - .65
    mask = torch.ones_like(sdf, dtype=torch.bool)
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_QUERY_DISTANCE_FIELDS",
        "dataset_manifest_sha256": "a"*64,
        "source_qualification_key": "synthetic-exact-sphere-only",
        "query_xyz_sha256": source_surface_tensor_sha256(xyz),
        "source_coordinate_system": "IRIS_NORMALIZED_SOURCE_XYZ",
        "teacher_semantics": "SOURCE_EXACT_SIGNED_SOLID_BOUNDARY_DISTANCE_V1",
        "signed_domain": domain,
        "udf_target_sha256": None, "udf_mask_sha256": None,
        "inside_target_sha256": None, "inside_mask_sha256": None,
        "sdf_target_sha256": source_surface_tensor_sha256(sdf),
        "sdf_mask_sha256": source_surface_tensor_sha256(mask),
        "sdf_distance_scope": "EXACT_BOUNDARY_OF_THE_SAME_QUALIFIED_SOLID",
        "sdf_sign_scope": "GLOBAL_NEGATIVE_INSIDE_POSITIVE_OUTSIDE",
        "assembly_boolean_operations_qualified": True,
    }
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=cameras,
        source_signed_sdf_target=sdf, source_signed_sdf_mask=mask,
        source_field_admission_key="exact",
        source_qualification_key="synthetic-exact-sphere-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    return batch, receipt


def _trainer_and_model():
    cfg = RSSvNextConfig.tiny_for_test()
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    return model


def test_exact_signed_sdf_trains_actual_flexicubes_field_with_source_receipt():
    model = _trainer_and_model()
    batch, receipt = _tiny_sdf_batch_and_receipt(
        model, domain="SINGLE_QUALIFIED_CLOSED_COMPONENT",
    )
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1.e-3),
        dataset_manifest_sha256="a"*64,
        trusted_source_field_registry={"exact": receipt},
        weights=RSSvNextTrainingWeights(
            topology=0., motion_preference=0., source_semantics=0.,
            flexicubes_surface=0., source_signed_sdf=1.,
        ),
    )
    result = trainer.train_step(batch)
    assert result.source_signed_sdf_loss is not None
    assert result.eligible_source_signed_sdf_queries == batch.source_signed_sdf_mask.numel()
    assert torch.isfinite(result.total_loss)
    assert model.iris.field.field_head.weight.grad is not None
    assert model.iris.field.field_head.weight.grad.abs().sum() > 0
    assert model.iris.scene.scene_queries.grad is not None


def test_modular_union_sdf_requires_real_assembly_proof_not_component_parity():
    model = _trainer_and_model()
    batch, receipt = _tiny_sdf_batch_and_receipt(
        model, domain="EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
    )
    weights = RSSvNextTrainingWeights(
        topology=0., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0., source_signed_sdf=1.,
    )
    valid = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="a"*64,
        trusted_source_field_registry={"exact": receipt}, weights=weights,
    )
    assert valid.objective(batch).source_signed_sdf_loss is not None
    bad = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="a"*64,
        trusted_source_field_registry={
            "exact": {**receipt, "assembly_boolean_operations_qualified": False},
        }, weights=weights,
    )
    with pytest.raises(ValueError, match="SOURCE_SDF_EXACT_SOLID_PROOF_REQUIRED"):
        bad.objective(batch)
    no_solid = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="a"*64,
        trusted_source_field_registry={
            "exact": {**receipt, "signed_domain": "OPEN_OR_UNKNOWN_UNSIGNED_ONLY"},
        }, weights=weights,
    )
    with pytest.raises(ValueError, match="SOURCE_SDF_EXACT_SOLID_PROOF_REQUIRED"):
        no_solid.objective(batch)


def test_explicit_highres_training_recomputes_chunks_and_backprops_to_cube_head():
    iris = _AnalyticIRIS().train()
    role = nn.Linear(8, 4).train()
    bridge = RSSvNextFlexiCubesEvidenceBridge(
        iris, resolution=3,
        inference_resolution=(6, 5, 7),
        field_query_chunk=17, cube_query_chunk=3, surface_query_chunk=2,
        max_cubes=512,
    ).train()
    fake = _FakeExtractor()
    layer = RSSvNextSemanticLayerSpec("all", realsas_style_generic_schema().roles)
    evidence = bridge(
        torch.ones(1, 3, 2, 2, 2),
        extractor=fake, role_head=role,
        schema=realsas_style_generic_schema(), layer=layer,
        field_policy="SIGNED_HYPOTHESIS_ONLY",
        high_resolution_training=True,
    )
    assert fake.training_seen is True
    loss = (
        evidence.cube_parameters.square().sum()
        + evidence.presentation_role_logits.square().mean()
    )
    loss.backward()
    assert iris.max_query <= max(17, 3*8, 2)
    assert iris.cube_extraction[-1].weight.grad is not None
    assert iris.cube_extraction[-1].weight.grad.abs().sum() > 0
    assert role.weight.grad is not None
    assert role.weight.grad.abs().sum() > 0
    bridge.eval()
    with pytest.raises(ValueError, match="HIGHRES_TRAIN_PROFILE_INVALID"):
        bridge(
            torch.ones(1, 3, 2, 2, 2),
            extractor=fake, role_head=role,
            schema=realsas_style_generic_schema(), layer=layer,
            field_policy="SIGNED_HYPOTHESIS_ONLY",
            high_resolution_training=True,
        )


def test_highres_training_requires_explicit_source_surface_receipt():
    model = _trainer_and_model()
    batch, receipt = _tiny_sdf_batch_and_receipt(
        model, domain="SINGLE_QUALIFIED_CLOSED_COMPONENT",
    )
    batch.flexicubes_high_resolution_training = True
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="a"*64,
        trusted_source_field_registry={"exact": receipt},
        weights=RSSvNextTrainingWeights(
            topology=0., motion_preference=0., source_semantics=0.,
            flexicubes_surface=0., source_signed_sdf=1.,
        ),
    )
    with pytest.raises(ValueError, match="TRAIN_HIGHRES_SOURCE_AND_SURFACE_REQUIRED"):
        trainer.objective(batch)



def test_multi_field_teacher_cannot_supply_individually_hashed_but_conflicting_solids():
    """Checks cross-task truth consistency before a shared IRIS update."""
    model = _trainer_and_model()
    batch, receipt = _tiny_sdf_batch_and_receipt(
        model, domain="SINGLE_QUALIFIED_CLOSED_COMPONENT",
    )
    sdf = batch.source_signed_sdf_target
    assert sdf is not None
    udf = sdf.abs()
    inside = (sdf < 0).float()
    mask = batch.source_signed_sdf_mask
    batch.source_udf_target = udf
    batch.source_udf_mask = mask
    batch.source_inside_sign_target = inside
    batch.source_inside_sign_mask = mask
    receipt.update({
        "udf_target_sha256": source_surface_tensor_sha256(udf),
        "udf_mask_sha256": source_surface_tensor_sha256(mask),
        "inside_target_sha256": source_surface_tensor_sha256(inside),
        "inside_mask_sha256": source_surface_tensor_sha256(mask),
    })
    weights = RSSvNextTrainingWeights(
        topology=0., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0., source_signed_sdf=1.,
        source_udf=1., source_inside_sign=1.,
    )
    def trainer(row):
        return RSSvNextStructuralTrainer(
            model, torch.optim.AdamW(model.parameters()),
            dataset_manifest_sha256="a"*64,
            trusted_source_field_registry={"exact": row},
            weights=weights,
        )
    good = trainer(receipt).objective(batch)
    assert good.source_udf_loss is not None
    assert good.source_inside_sign_loss is not None
    assert good.source_signed_sdf_loss is not None

    from dataclasses import replace as dc_replace
    incorrect_udf = udf + .2
    wrong_udf = {
        **receipt,
        "udf_target_sha256": source_surface_tensor_sha256(incorrect_udf),
    }
    with pytest.raises(ValueError, match="SOURCE_SDF_UDF_TEACHER_CONFLICT"):
        trainer(wrong_udf).objective(
            dc_replace(batch, source_udf_target=incorrect_udf),
        )
    inverted = 1. - inside
    wrong_sign = {
        **receipt,
        "inside_target_sha256": source_surface_tensor_sha256(inverted),
    }
    with pytest.raises(ValueError, match="SOURCE_SDF_SIGN_TEACHER_CONFLICT"):
        trainer(wrong_sign).objective(
            dc_replace(batch, source_inside_sign_target=inverted),
        )
