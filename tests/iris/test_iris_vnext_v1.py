from __future__ import annotations

import numpy as np
import pytest
import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.flexicubes_v1 import (
    FLEXICUBES_PARAMETER_SEMANTICS,
    split_flexicubes_parameters_v1,
)
from models.iris.vnext.model_v1 import IRISVNextV1
from models.iris.vnext.teacher_truth_v1 import build_mechanical_demand_truth_v1
from models.iris.vnext.training_v1 import assert_batched_fit8_v1


def _cfg():
    return IRISVNextConfigV1(
        image_token_dim=16,
        camera_feature_dim=5,
        scene_dim=32,
        camera_embed_dim=16,
        plane_size=2,
        plane_channels=8,
        field_hidden_dim=32,
        field_residual_blocks=2,
        local_feature_dim=32,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        topology_width=32,
        topology_heads=4,
        topology_decoder_layers=2,
        topology_memory_tokens=64,
        topology_refinement_layers=1,
        topology_source_width=16,
        appearance_descriptor_dim=8,
        relation_embedding_dim=8,
        maximum_extraction_delta=0.05,
        extraction_training_resolution=3,
        surface_witness_count=12,
    )


def test_vnext_single_scene_geometry_mechanical_structure_contract():
    torch.manual_seed(0)
    cfg = _cfg()
    model = IRISVNextV1(cfg)
    b, v, t = 2, 8, 4
    image = torch.randn(b, v, t, 16)
    xy = torch.randn(b, v, t, 2).clamp(-1, 1)
    camera = torch.randn(b, v, 5)
    planes = model.encode_scene(image, xy, camera)
    assert planes.shape == (b, 3, 8, 4, 4)
    points_iris = torch.rand(b, 7, 3) * 2 - 1
    evidence = model.query_evidence(planes, points_iris)
    assert evidence["H"].shape == (b, 7, 32)
    assert evidence["mechanical_demand"]["tangent_direction_2theta"].shape == (b, 7, 2)
    assert model.semantic_taxonomy_used is False
    assert model.independent_visual_mesh_allowed is False

    # StructureMemory is explicitly in the established TESSA [-0.5,0.5] domain.
    xyz_tessa = points_iris[:, :5] * 0.5
    normal = torch.nn.functional.normalize(torch.randn(b, 5, 3), dim=-1)
    witness = model.query_evidence(planes, points_iris[:, :5])
    ids = torch.full((b, 6), cfg.tessa_decoder_config().BOS, dtype=torch.long)
    output = model.decode_topology(
        scene_planes=planes,
        witness_xyz=xyz_tessa,
        witness_normal=normal,
        witness_h=witness["H"],
        witness_log_uncertainty=witness["log_uncertainty"],
        witness_mechanical=witness["mechanical_demand"],
        input_ids=ids,
    )
    assert output.logits.shape[:2] == ids.shape
    assert output.surface_latents.shape == (b, 64, 32)

    with pytest.raises(ValueError, match="IRIS_VNEXT_WITNESS_COORDINATE_DOMAIN_DRIFT"):
        model.decode_topology(
            scene_planes=planes,
            witness_xyz=points_iris[:, :5],
            witness_normal=normal,
            witness_h=witness["H"],
            witness_log_uncertainty=witness["log_uncertainty"],
            witness_mechanical=witness["mechanical_demand"],
            input_ids=ids,
        )


def test_flexicubes_21d_semantics_are_12_plus_8_plus_1_not_teacher_labels():
    cfg = _cfg()
    model = IRISVNextV1(cfg)
    h = torch.randn(2, 3, 8, cfg.local_feature_dim)
    raw = model.scene_core.cube_extraction(h)
    assert raw.shape == (2, 3, 21)
    assert model.scene_core.cube_extraction.parameter_semantics == FLEXICUBES_PARAMETER_SEMANTICS
    interpreted = split_flexicubes_parameters_v1(
        raw, weight_scale=cfg.flexicubes_weight_scale
    )
    assert interpreted.beta_raw.shape == (2, 3, 12)
    assert interpreted.alpha_raw.shape == (2, 3, 8)
    assert interpreted.gamma_raw.shape == (2, 3)
    assert torch.all(interpreted.beta > 0)
    assert torch.all(interpreted.alpha > 0)
    assert torch.all((interpreted.gamma > 0) & (interpreted.gamma < 1))


def test_vnext_parameter_groups_are_disjoint():
    model = IRISVNextV1(_cfg())
    groups = model.parameter_groups()
    sets = [{id(parameter) for parameter in value} for value in groups.values()]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set().union(*sets) == {id(parameter) for parameter in model.parameters()}


def test_mechanical_truth_is_training_only_motion_supervision():
    rest = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    posed = np.stack((rest, np.array([[0.0, 0, 0], [1.25, 0, 0], [0, 1, 0]])))
    truth = build_mechanical_demand_truth_v1(rest, faces, posed)
    assert truth.stretch_major[0] > 0
    assert truth.edge_strain_p95[0] > 0
    assert truth.metadata["teacher_only"] is True
    assert truth.metadata["rig_or_skin_inference_input"] is False


def test_fit8_kernel_rejects_subject_serial_shape():
    one = {
        "image_tokens": torch.zeros(1, 8, 2, 16),
        "token_xy": torch.zeros(1, 8, 2, 2),
        "camera_features": torch.zeros(1, 8, 5),
    }
    with pytest.raises(ValueError, match="IRIS_VNEXT_REQUIRES_BATCHED_COHORT"):
        assert_batched_fit8_v1(one, expected=8)
    eight = {
        "image_tokens": torch.zeros(8, 8, 2, 16),
        "token_xy": torch.zeros(8, 8, 2, 2),
        "camera_features": torch.zeros(8, 8, 5),
    }
    assert_batched_fit8_v1(eight, expected=8)
