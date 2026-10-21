from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext import materialize_fit8_native_v8_matched_carrier_v1 as carrier


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_native_v8_matched_carrier_is_lossless_and_authority_bounded(tmp_path, monkeypatch):
    source_sha = "1" * 40
    monkeypatch.setattr(carrier, "git_head", lambda _: source_sha)

    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    prepared_rows = []
    for subject_id in SUBJECT_IDS:
        shard = prepared_root / (subject_id.lower().replace("-", "").replace(" ", "") + ".vnext.pt")
        shard.write_bytes(subject_id.encode("utf-8"))
        prepared_rows.append(
            {
                "subject_id": subject_id,
                "shard_file": shard.name,
                "shard_sha256": _sha(shard),
            }
        )
    prepared_manifest = prepared_root / "COHORT_MANIFEST.json"
    _write_json(
        prepared_manifest,
        {
            "schema": "RealSaS.IRISVNextFit8Manifest.v4",
            "subjects": prepared_rows,
        },
    )

    native_root = tmp_path / "native"
    native_rows = []
    v8_authority = {
        "source_commit": "2" * 40,
        "checkpoint_sha256": "3" * 64,
        "binding_sha256": "4" * 64,
        "global_step": 12000,
    }
    for index, subject_id in enumerate(SUBJECT_IDS):
        subject_root = native_root / "subjects" / subject_id
        subject_root.mkdir(parents=True)
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [0.25 + index * 1e-4, 0.0, 0.0], [0.0, 0.25, 0.0]],
            dtype=np.float32,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        normals = np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        witness = subject_root / "V8_NATIVE_FLEXICUBES_R24.npz"
        np.savez_compressed(
            witness,
            vertices_normalized=vertices,
            faces=faces,
            vertex_normals=normals,
        )
        receipt = {
            "schema": "RealSaS.V8NativeFlexiCubesWitness.v1",
            "subject_id": subject_id,
            "execution_source_sha": source_sha,
            "resolution": 24,
            "surface_semantics": "EXACT_V8_TRAINED_FLEXICUBES_PATH_AT_NATIVE_TRAINING_RESOLUTION",
            "teacher_truth_used_in_inference_forward": False,
            "high_resolution_capability_claimed": False,
            "witness_sha256": _sha(witness),
            "scene_binding_sha256": f"scene-{index}",
            "scene_planes_sha256": f"planes-{index}",
            "v8_authority": v8_authority,
        }
        receipt_path = subject_root / "V8_NATIVE_FLEXICUBES_R24_RECEIPT.json"
        _write_json(receipt_path, receipt)
        native_rows.append(
            {
                "subject_id": subject_id,
                "witness_sha256": _sha(witness),
                "receipt_sha256": _sha(receipt_path),
            }
        )

    native_batch = {
        "schema": "RealSaS.Fit8V8NativeFlexiCubesMaterialization.v1",
        "status": "MATERIALIZED_NATIVE_V8_FLEXICUBES_WITNESSES__MEASUREMENT_ONLY",
        "execution_source_sha": source_sha,
        "prepared_manifest_sha256": _sha(prepared_manifest),
        "subject_order": list(SUBJECT_IDS),
        "subjects": native_rows,
        "high_resolution_capability_claimed": False,
        "tessa_invoked": False,
        "optimizer_steps_executed": 0,
    }
    _write_json(native_root / "FIT8_V8_NATIVE_FLEXICUBES_BATCH_RECEIPT.json", native_batch)

    output_root = tmp_path / "carrier"
    result = carrier.materialize(
        argparse.Namespace(
            repo_root=tmp_path,
            expected_source_sha=source_sha,
            native_root=native_root,
            prepared_manifest=prepared_manifest,
            output_dir=output_root,
        )
    )

    assert result["carrier_authority"] == "INFERENCE_ADMISSIBLE_FIXED_SURFACE_CARRIER"
    assert result["surface_carrier_policy"]["same_carrier_for_a_and_b"] is True
    assert result["surface_carrier_policy"]["raw_field_sparse_t512_t1024_invoked"] is False
    assert result["high_resolution_capability_claimed"] is False
    assert result["product_surface_claimed"] is False
    assert result["geometry_mutated"] is False
    assert result["topology_mutated"] is False

    for subject_id in SUBJECT_IDS:
        native_path = native_root / "subjects" / subject_id / "V8_NATIVE_FLEXICUBES_R24.npz"
        projected_path = output_root / "subjects" / subject_id / "DIRECT_IRIS_SURFACE.stage12.npz"
        with np.load(native_path, allow_pickle=False) as native, np.load(projected_path, allow_pickle=False) as projected:
            np.testing.assert_array_equal(projected["vertices_normalized"], native["vertices_normalized"])
            np.testing.assert_array_equal(projected["faces"], native["faces"])
            np.testing.assert_array_equal(projected["implicit_normals"], native["vertex_normals"])
