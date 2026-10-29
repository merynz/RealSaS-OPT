from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.corpus.build_truth139_geometry_teacher_v1 import build
from tools.corpus.validate_truth139_geometry_teacher_v1 import validate


def test_geometry_teacher_rebinds_exact_source_mesh_and_transform(tmp_path: Path):
    geometry = tmp_path / "geometry.npz"
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    np.savez_compressed(geometry, rest_vertices_source_world=vertices, faces=faces)

    snapshot = tmp_path / "snapshot.json"
    source_to_canonical = [
        [2.0, 0.0, 0.0, 0.25],
        [0.0, 1.5, 0.0, -0.5],
        [0.0, 0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    snapshot.write_text(json.dumps({
        "schema": "RealSaS.ImportedSourceSceneSnapshot.v2",
        "source_sha256": "a" * 64,
        "source_world_to_canonical": source_to_canonical,
    }, sort_keys=True) + "\n")

    manifest = build(
        subject_id="fixture-animal",
        geometry_npz=geometry,
        source_scene_snapshot=snapshot,
        output_dir=tmp_path / "teacher",
        maximum_surface_samples=32,
    )
    court = validate(manifest)
    assert court["status"] == "PASS_GEOMETRY_TEACHER_REBINDS_SOURCE_MESH_AND_TRANSFORM"
    assert court["validated_counts"]["vertex_count"] == 4
    assert court["validated_counts"]["face_count"] == 4
    assert court["validated_counts"]["surface_sample_count"] == 32
    assert court["policy"]["global_watertight_sdf_sign_claimed"] is False


def test_geometry_court_rejects_mutated_teacher_surface(tmp_path: Path):
    geometry = tmp_path / "geometry.npz"
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    np.savez_compressed(geometry, rest_vertices_source_world=vertices, faces=faces)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({
        "schema": "RealSaS.ImportedSourceSceneSnapshot.v2",
        "source_sha256": "b" * 64,
        "source_world_to_canonical": np.eye(4).tolist(),
    }, sort_keys=True) + "\n")

    manifest = build(
        subject_id="fixture-single",
        geometry_npz=geometry,
        source_scene_snapshot=snapshot,
        output_dir=tmp_path / "teacher",
        maximum_surface_samples=4,
    )
    surface_path = Path(manifest["surface_teacher"]["path"])
    with np.load(surface_path, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}
    arrays["surface_xyz_source_world"] = arrays["surface_xyz_source_world"].copy()
    arrays["surface_xyz_source_world"][0, 0] += 0.1
    np.savez_compressed(surface_path, **arrays)

    # Hash binding must fail before any scientific comparison is attempted.
    try:
        validate(manifest)
    except ValueError as exc:
        assert "SURFACE_DRIFT" in str(exc)
    else:
        raise AssertionError("mutated surface teacher was accepted")
