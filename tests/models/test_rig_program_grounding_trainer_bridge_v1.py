from __future__ import annotations

"""Synthetic diagnostic of the real unified TESSA->AXIS->hard-MIRA training path.

This test does NOT certify corpus-source truth. Production admission is through
the trusted launcher registry, and source receipts must be independently minted.
"""

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from models.rss_vnext.training_rig_program_grounding_v1 import (
    GROUNDING_SCHEMA_V1, source_rig_grounding_digest_v1,
)
from tests.models.test_rss_vnext_axis_mesh_matched_product_training_v1 import fixture


def _synthetic_math_fixture():
    net, batch, _, axis_receipt = fixture()
    state, _, _ = net.encode_observations(batch.rgba, batch.camera_features)
    mesh = net.build_mesh_candidate(
        state, batch.topology_teacher_tokens[0],
        source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    v = mesh.rest_vertices_iris.shape[1]
    source_joint = torch.tensor([[[-0.50, 0.125, -0.125]]])
    source_parent = torch.tensor([-1], dtype=torch.long)
    source_skin = torch.ones(1, v, 1)
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_RIG_PROGRAM_MATH_V1",
        "schema": GROUNDING_SCHEMA_V1,
        "dataset_manifest_sha256": "3" * 64,
        "source_qualification_key": "synthetic-unit-only",
        "axis_teacher_sequence_sha256": axis_receipt["teacher_sequence_sha256"],
        "mesh_teacher_topology_digest": axis_receipt["mesh_teacher_topology_digest"],
        "source_tensors_sha256": source_rig_grounding_digest_v1(
            joint_xyz=source_joint, parents=source_parent,
            skin_weights=source_skin, frame_basis=None,
            dof_known_mask=None, dof_allowed=None,
            control_rest_xyz=None, anchor_local_xyz=None,
            rest_vertices=None, pose_matrices=None,
            posed_vertices=None, pose_mask=None,
        ),
        "joint_index_and_mesh_vertex_reindex_proven": True,
        "source_semantics": "NATIVE_RIG_SKIN_NOT_PSEUDOLABEL",
    }
    batch = replace(
        batch,
        axis_math_joint_xyz=source_joint,
        axis_math_parent_indices=source_parent,
        axis_math_skin_weights=source_skin,
        axis_math_admission_key="math",
    )
    weights = RSSvNextTrainingWeights(
        topology=0., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0., axis_rig_skin_codes=1.,
        axis_rig_grounding=0.5,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=weights,
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        trusted_axis_teacher_registry={"axis": axis_receipt},
        trusted_axis_math_registry={"math": receipt},
    )
    return net, batch, trainer, receipt


def test_unified_grounding_reaches_causal_axis_and_keeps_checkpoint_security(tmp_path):
    net, batch, trainer, receipt = _synthetic_math_fixture()
    result = trainer.train_step(batch)
    assert result.axis_rig_skin_loss is not None
    assert result.axis_rig_grounding_loss is not None
    assert torch.isfinite(result.axis_rig_grounding_loss)
    assert result.eligible_axis_sequences == 1
    assert net.axis.coord_head.weight.grad is not None
    assert net.axis.coord_head.weight.grad.abs().sum() > 0

    trainer.save_checkpoint(tmp_path / "grounding.pt", sampler_cursor=11)
    assert trainer.load_checkpoint(tmp_path / "grounding.pt") == 11

    altered_receipt = {
        **receipt, "source_tensors_sha256": "0" * 64,
    }
    _, _, foreign, _ = _synthetic_math_fixture()
    # Freeze source state under same trainer for a meaningful registry-drift check.
    swapped = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=trainer.weights,
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        trusted_axis_teacher_registry={
            "axis": trainer.trusted_axis_teacher_registry["axis"],
        },
        trusted_axis_math_registry={"math": altered_receipt},
    )
    with pytest.raises(ValueError, match="CHECKPOINT_AXIS_MATH_REGISTRY_DRIFT"):
        swapped.load_checkpoint(tmp_path / "grounding.pt")


def test_source_missing_truth_is_rejected_without_silent_zero_labels():
    _, batch, trainer, _ = _synthetic_math_fixture()
    with pytest.raises(ValueError, match="AXIS_MATH_SUPERVISION_INCOMPLETE"):
        trainer.objective(replace(batch, axis_math_skin_weights=None))
    with pytest.raises(ValueError, match="RIG_MATH_SOURCE_RECEIPT_UNQUALIFIED"):
        altered = batch.axis_math_joint_xyz.clone()
        altered[..., 0] += 0.25
        trainer.objective(replace(batch, axis_math_joint_xyz=altered))
