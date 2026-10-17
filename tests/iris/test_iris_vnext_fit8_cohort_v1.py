from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from models.iris.vnext.cohort_v1 import SUBJECT_IDS, VNextFit8PolicyV1, VNextStageV1
from models.iris.vnext.cohort_v2 import IRISVNextFit8TrainerV2
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cfg():
    return IRISVNextConfigV1(
        image_token_dim=16,
        camera_feature_dim=5,
        scene_dim=32,
        camera_embed_dim=16,
        plane_size=2,
        plane_channels=8,
        field_hidden_dim=32,
        field_residual_blocks=1,
        local_feature_dim=32,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        topology_width=32,
        topology_heads=4,
        topology_decoder_layers=1,
        topology_memory_tokens=64,
        topology_refinement_layers=1,
        topology_source_width=16,
        appearance_descriptor_dim=8,
        relation_embedding_dim=8,
        extraction_training_resolution=3,
        surface_witness_count=12,
    )


def _payload(subject: str, seed: int, subject_index: int):
    generator = torch.Generator().manual_seed(seed)
    source_generator = torch.Generator().manual_seed(314159)
    relation_shape = (8, 5)
    appearance_shape = (8, 6)
    theta = torch.linspace(0.0, 2.0 * torch.pi, 12)
    teacher_surface = torch.stack(
        (0.45 * torch.cos(theta), 0.45 * torch.sin(theta), torch.zeros_like(theta)),
        dim=-1,
    )
    pad = _cfg().tessa_decoder_config().PAD
    topology_input = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    topology_labels = torch.tensor([[2, 3, 4, 1], [3, 2, 1, 4]], dtype=torch.long)
    count = 1 if subject_index % 3 == 0 else 2
    if count == 1:
        topology_input[1].fill_(pad)
        topology_labels[1].fill_(pad)
    return {
        "subject_id": subject,
        "teacher_inputs_in_inference_forward": False,
        "image_tokens": torch.randn(8, 2, 16, generator=source_generator),
        "token_xy": torch.rand(8, 2, 2, generator=source_generator) * 2 - 1,
        "camera_features": torch.randn(8, 5, generator=source_generator),
        "geometry_points": torch.rand(7, 3, generator=generator) * 2 - 1,
        "geometry_targets": torch.zeros(7),
        "teacher_surface_points_iris": teacher_surface,
        "mechanical_points": torch.rand(6, 3, generator=generator) * 2 - 1,
        "mechanical_normals": F.normalize(torch.randn(6, 3, generator=generator), dim=-1),
        "mechanical_stretch_major": torch.full((6,), 0.1),
        "mechanical_stretch_minor": torch.full((6,), 0.05),
        "mechanical_tangent_direction_2theta": torch.tensor([[1.0, 0.0]]).repeat(6, 1),
        "mechanical_bend_demand": torch.full((6,), 0.02),
        "mechanical_rigidity_probability": torch.full((6,), 0.8),
        "appearance_points": torch.rand(*appearance_shape, 3, generator=generator) * 2 - 1,
        "appearance_rgba": torch.rand(*appearance_shape, 4, generator=generator),
        "appearance_alpha_boundary": torch.randint(0, 2, appearance_shape, generator=generator).float(),
        "appearance_boundary": torch.randint(0, 2, appearance_shape, generator=generator).float(),
        "relation_point_a": torch.rand(*relation_shape, 3, generator=generator) * 2 - 1,
        "relation_point_b": torch.rand(*relation_shape, 3, generator=generator) * 2 - 1,
        "relation_same_drawable_affinity": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_cut_probability": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_attachment_affinity": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_alpha_boundary_probability": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_appearance_boundary_probability": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_front_probability": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_occlusion_probability": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_visibility_probability_a": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "relation_visibility_probability_b": torch.randint(0, 2, relation_shape, generator=generator).float(),
        "topology_input_ids": topology_input,
        "topology_labels": topology_labels,
        "topology_window_offsets": torch.tensor([0, 19 + subject_index], dtype=torch.long),
        "topology_window_count": torch.tensor(count, dtype=torch.long),
    }


def _center_test_field_on_zero(trainer: IRISVNextFit8TrainerV2) -> None:
    with torch.no_grad():
        source = trainer.cohort.tensors
        planes = trainer.model.encode_scene(
            source["image_tokens"], source["token_xy"], source["camera_features"]
        )
        grid, _ = trainer.extractor.construct_iris_grid(
            trainer.model.config.extraction_training_resolution
        )
        grid = grid.to(trainer.device).unsqueeze(0).expand(8, -1, -1)
        field = trainer.model.query_evidence(planes, grid)["field"]
        low = field[0].min()
        high = field[0].max()
        assert float(high - low) > 1.0e-6
        trainer.model.scene_core.field.field_head.bias.add_(-(low + high) * 0.5)


def test_vnext_fit8_trainer_executes_complete_curriculum_as_real_batch8(tmp_path):
    manifest_rows = []
    for index, subject in enumerate(SUBJECT_IDS):
        payload = _payload(subject, 100 + index, index)
        path = tmp_path / f"{index}.pt"
        torch.save(payload, path)
        manifest_rows.append(
            {"subject_id": subject, "shard_file": path.name, "shard_sha256": _sha(path)}
        )
    manifest = {
        "schema": "RealSaS.IRISVNextFit8Manifest.v3",
        "teacher_inputs_in_inference_forward": False,
        "subjects": manifest_rows,
    }
    stages = (
        VNextStageV1("G1_TEST", 1, field_weight=1.0),
        VNextStageV1("G2_TEST", 1, extraction_weight=1.0),
        VNextStageV1("M1_TEST", 1, mechanical_weight=1.0),
        VNextStageV1("A1_TEST", 1, appearance_weight=1.0),
        VNextStageV1("R1_TEST", 1, relation_weight=1.0),
        VNextStageV1("T1_TEST", 1, structure_lr_scale=1.0, topology_weight=1.0),
        VNextStageV1(
            "J1_TEST",
            1,
            structure_lr_scale=0.5,
            field_weight=0.25,
            extraction_weight=0.25,
            mechanical_weight=0.25,
            appearance_weight=0.25,
            relation_weight=0.25,
            topology_weight=0.25,
        ),
    )
    policy = VNextFit8PolicyV1(
        geometry_samples_per_subject=3,
        mechanical_samples_per_subject=2,
        appearance_samples_per_subject=3,
        relation_pairs_per_subject=3,
        checkpoint_interval=100,
        eval_interval=100,
        stages=stages,
    )
    trainer = IRISVNextFit8TrainerV2(
        data_root=tmp_path,
        manifest=manifest,
        checkpoint_root=tmp_path / "cp",
        device="cpu",
        model_config=_cfg(),
        policy=policy,
    )

    rows = []
    for stage in stages:
        if stage.extraction_weight > 0 or stage.topology_weight > 0:
            _center_test_field_on_zero(trainer)
        rows.append(trainer.one_step())

    assert [row["stage"] for row in rows] == [stage.stage_id for stage in stages]
    assert all(row["batch_subjects"] == 8 for row in rows)
    assert all(row["shared_scene_encodes"] == 1 for row in rows)
    assert all(row["subject_serial_model_forward"] is False for row in rows)
    assert all(row["optimizer_steps"] == 1 for row in rows)
    assert rows[-1]["global_step"] == len(stages)
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in rows)
    assert rows[1]["extraction_loss"] > 0
    assert rows[1]["extraction_chamfer_loss"] > 0
    assert all(count > 0 for count in rows[1]["extracted_vertex_counts"])
    assert rows[3]["appearance_loss"] > 0
    assert rows[4]["relations_loss"] > 0
    assert rows[5]["topology_loss"] > 0
    assert len(rows[5]["topology_position_offsets"]) == 8
    assert len(set(rows[5]["topology_position_offsets"])) > 1
