from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
    V8_SOURCE_COMMIT,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    assert_full_sparse_policy,
    load_subject_source,
    parser,
    scene_binding_sha256,
    sha256_file,
)


def test_materializer_is_pinned_to_exact_completed_v8_authority():
    assert V8_SOURCE_COMMIT == "5083c9915533ec8ec6530c7d229c635887abfea7"
    assert V8_CHECKPOINT_SHA256 == "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    assert V8_BINDING_SHA256 == "735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2"
    assert V8_GLOBAL_STEP == 12000


def test_scientific_materializer_accepts_only_reference_t512_t1024_policy():
    assert_full_sparse_policy(SparseRegularTetraPolicyV5())
    with pytest.raises(ValueError, match="FULL_T512_T1024_POLICY_REQUIRED"):
        assert_full_sparse_policy(SparseRegularTetraPolicyV5(base_cells=8, fine_cells=16))
    with pytest.raises(ValueError, match="REFERENCE_SPARSE_POLICY_DRIFT"):
        assert_full_sparse_policy(SparseRegularTetraPolicyV5(query_chunk=131072))


def test_cli_has_no_grid_resolution_override():
    destinations = {action.dest for action in parser()._actions}
    assert "base_cells" not in destinations
    assert "fine_cells" not in destinations
    assert "resolution" not in destinations
    assert "model_query_chunk" in destinations  # execution-memory only


def test_scene_binding_is_deterministic_and_binds_scene_bytes():
    scene = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    kwargs = {
        "subject_id": "Snake",
        "prepared_shard_sha256": "a" * 64,
        "execution_source_sha": "b" * 40,
    }
    a = scene_binding_sha256(scene_planes=scene, **kwargs)
    b = scene_binding_sha256(scene_planes=scene.clone(), **kwargs)
    changed = scene.clone()
    changed[0, 0, 0, 0, 0] += 1.0
    c = scene_binding_sha256(scene_planes=changed, **kwargs)
    assert a == b
    assert a != c
    assert len(a) == 64


def test_prepared_subject_loader_preserves_batch_one_and_forces_fp32(tmp_path: Path):
    path = tmp_path / "fit8_snake_v1.pt"
    payload = {
        "image_tokens": torch.ones((8, 3, 5), dtype=torch.float16),
        "token_xy": torch.ones((8, 3, 2), dtype=torch.float16),
        "camera_features": torch.ones((8, 13), dtype=torch.float16),
    }
    torch.save(payload, path)
    source = load_subject_source(path, sha256_file(path), torch.device("cpu"))
    assert source["image_tokens"].shape == (1, 8, 3, 5)
    assert source["token_xy"].shape == (1, 8, 3, 2)
    assert source["camera_features"].shape == (1, 8, 13)
    assert all(value.dtype == torch.float32 for value in source.values())


def test_prepared_subject_loader_fails_closed_on_hash_drift(tmp_path: Path):
    path = tmp_path / "fit8_snake_v1.pt"
    torch.save(
        {
            "image_tokens": torch.ones((8, 3, 5)),
            "token_xy": torch.ones((8, 3, 2)),
            "camera_features": torch.ones((8, 13)),
        },
        path,
    )
    with pytest.raises(ValueError, match="SHARD_SHA_DRIFT"):
        load_subject_source(path, "0" * 64, torch.device("cpu"))
