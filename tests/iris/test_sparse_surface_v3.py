from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.vnext.sparse_surface_v3 import (
    VNextScalarFieldQueryV3,
    VNextSparseSurfaceQueryPolicyV3,
    decode_vnext_sparse_surface_v3,
    direct_surface_candidate_binding_hash_v1,
    write_direct_iris_surface_bundle_v1,
)


class SphereEvidenceModel(nn.Module):
    def query_local(self, scene_planes: torch.Tensor, points_normalized: torch.Tensor):
        assert scene_planes.shape[0] == 1
        radius = torch.linalg.vector_norm(points_normalized, dim=-1)
        field = radius - 0.55
        return {
            "field": field,
            "H": torch.zeros((*field.shape, 4), device=field.device, dtype=field.dtype),
            "extraction_delta": torch.zeros((*field.shape, 3), device=field.device, dtype=field.dtype),
        }


def _scene() -> torch.Tensor:
    return torch.zeros((1, 3, 4, 2, 2), dtype=torch.float32)


def _small_sparse_policy() -> SparseRegularTetraPolicyV5:
    return SparseRegularTetraPolicyV5(
        base_cells=8,
        fine_cells=16,
        query_chunk=4096,
        cell_chunk=64,
        refine_slab_cells=4,
        fine_gid_cell_chunk=64,
        normal_chunk=4096,
    )


def test_vnext_query_adapter_preserves_coordinates_and_field_sign():
    model = SphereEvidenceModel().eval()
    adapter = VNextScalarFieldQueryV3(
        model,
        _scene(),
        policy=VNextSparseSurfaceQueryPolicyV3(model_query_chunk=2),
    )
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = adapter(points)
    np.testing.assert_allclose(values, np.asarray([-0.55, 0.0, 0.25], dtype=np.float32), atol=1e-6)
    assert adapter.coordinate_transform == "IDENTITY_NORMALIZED_MINUS1_PLUS1"
    assert adapter.scalar_transform == "IDENTITY_LEARNED_FIELD"
    assert adapter.teacher_truth_used is False


def test_vnext_query_adapter_fails_closed_on_training_mode_and_out_of_domain():
    with pytest.raises(ValueError, match="MODEL_MUST_BE_EVAL"):
        VNextScalarFieldQueryV3(SphereEvidenceModel(), _scene())

    model = SphereEvidenceModel().eval()
    adapter = VNextScalarFieldQueryV3(model, _scene())
    with pytest.raises(ValueError, match="OUT_OF_DOMAIN"):
        adapter(np.asarray([[1.1, 0.0, 0.0]], dtype=np.float32))


def test_vnext_sparse_surface_reuses_selected_tetra_decoder_without_hidden_deformation(tmp_path: Path):
    model = SphereEvidenceModel().eval()
    mesh = decode_vnext_sparse_surface_v3(
        model,
        _scene(),
        sparse_policy=_small_sparse_policy(),
        query_policy=VNextSparseSurfaceQueryPolicyV3(model_query_chunk=128),
        work_dir=tmp_path / "decoder",
    )
    radius = np.linalg.norm(mesh.vertices_normalized, axis=1)
    assert len(mesh.vertices_normalized) > 100
    assert len(mesh.faces) > 100
    assert float(np.max(np.abs(radius - 0.55))) < 0.025
    assert mesh.diagnostics["base_cells"] == 8
    assert mesh.diagnostics["fine_cells"] == 16
    assert mesh.diagnostics["deformation"] is False
    assert mesh.diagnostics["teacher_truth_used"] is False
    assert mesh.diagnostics["compiler_qualified"] is False
    assert mesh.diagnostics["scientific_pass_claimed"] is False
    assert mesh.diagnostics["field_scalar_transform"] == "IDENTITY_LEARNED_FIELD"


def test_direct_surface_bundle_is_hash_bound_but_never_self_qualifies(tmp_path: Path):
    model = SphereEvidenceModel().eval()
    sparse_policy = _small_sparse_policy()
    mesh = decode_vnext_sparse_surface_v3(
        model,
        _scene(),
        sparse_policy=sparse_policy,
        query_policy=VNextSparseSurfaceQueryPolicyV3(model_query_chunk=128),
        work_dir=tmp_path / "decoder",
    )
    kwargs = {
        "subject_id": "Snake",
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "scene_binding_sha256": "c" * 64,
        "sparse_policy": sparse_policy,
    }
    expected_candidate = direct_surface_candidate_binding_hash_v1(mesh, **kwargs)
    receipt = write_direct_iris_surface_bundle_v1(mesh, tmp_path / "bundle", **kwargs)
    assert receipt.candidate_binding_hash == expected_candidate
    assert receipt.vertex_count == len(mesh.vertices_normalized)
    assert receipt.face_count == len(mesh.faces)
    assert receipt.compiler_qualified is False
    assert receipt.training_authorized is False
    assert receipt.scientific_pass_claimed is False
    assert receipt.status == "DIRECT_IRIS_SURFACE_MATERIALIZED__COMPILER_NOT_RUN"

    payload = json.loads((tmp_path / "bundle" / "DIRECT_IRIS_SURFACE_RECEIPT.json").read_text())
    assert payload["candidate_binding_hash"] == expected_candidate
    assert payload["teacher_truth_used"] is False
    assert payload["compiler_qualified"] is False
    assert payload["training_authorized"] is False
    assert payload["scientific_pass_claimed"] is False
    assert (tmp_path / "bundle" / "DIRECT_IRIS_SURFACE.stage12.npz").is_file()
