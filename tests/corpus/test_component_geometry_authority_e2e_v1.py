from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.build_component_geometry_authority_v1 import build
from tools.corpus.validate_component_geometry_authority_v1 import validate
from tools.corpus.build_component_geometry_authority_v1 import closest_valid_surface


@pytest.mark.parametrize('chunk_size', [1, 7, 64])
def test_bounded_distance_queries_preserve_exact_unbatched_results(tmp_path, chunk_size):
    import trimesh
    source, _, _ = _write_packet(tmp_path)
    with np.load(source, allow_pickle=False) as z:
        vertices, faces = z['rest_vertices_source_world'], z['faces']
    valid = np.array([True, False, True, True, True])
    query = np.random.default_rng(491).uniform(-1., 6., (179, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[valid], process=False)
    _, expected_distance, expected_face = trimesh.proximity.closest_point(mesh, query)
    distance, face = closest_valid_surface(query, vertices, faces, valid, chunk_size=chunk_size)
    np.testing.assert_array_equal(distance, expected_distance.astype(np.float32))
    np.testing.assert_array_equal(face, np.flatnonzero(valid)[expected_face].astype(np.int32))


def _write_packet(root: Path):
    # Closed oriented tetra + a separate open triangle. Both are legitimate source
    # surfaces, but only the tetra may carry sign/occupancy authority.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, -1.0, 0.0],
            [5.0, -1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )
    owner = np.zeros(len(faces), dtype=np.int32)
    geometry = root / "REST_SURFACE_GEOMETRY.npz"
    np.savez_compressed(
        geometry,
        rest_vertices_source_world=vertices,
        faces=faces,
        face_owner_privileged=owner,
    )
    snapshot = root / "SOURCE_SCENE_SNAPSHOT.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema": "Synthetic.SourceSceneSnapshot.v1",
                "source_world_to_canonical": np.eye(4, dtype=float).tolist(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    camera = root / "camera_bundle.json"
    camera.write_text(json.dumps({"schema": "Synthetic.CameraBundle.v1"}) + "\n")
    return geometry, snapshot, camera


def test_open_surface_does_not_erase_closed_component_sign_truth(tmp_path):
    geometry, snapshot, camera = _write_packet(tmp_path)
    output = tmp_path / "out"
    manifest = build(
        "synthetic-closed-plus-open",
        geometry,
        snapshot,
        camera,
        output,
        surface_samples=1024,
        query_samples=512,
    )

    assert manifest["counts"]["surface_component_count"] == 2
    assert manifest["counts"]["sign_valid_component_count"] == 1
    assert manifest["counts"]["unsigned_query_count"] == 512
    assert 0 < manifest["counts"]["signed_query_count"] < 512
    assert 0.0 < manifest["coverage"]["signed_query_fraction"] < 1.0

    with np.load(output / "COMPONENT_DISTANCE_SIGN_TARGETS_V1.npz", allow_pickle=False) as z:
        # Unknown sign is absence from the compact signed index. There is no dense
        # signed-distance placeholder array that a trainer could accidentally use.
        assert "signed_distance" not in z.files
        assert "occupancy" not in z.files
        signed_index = np.asarray(z["signed_query_index"], dtype=np.int64)
        signed_value = np.asarray(z["signed_distance_value"], dtype=np.float32)
        occupancy = np.asarray(z["occupancy_value"], dtype=np.bool_)
        assert len(signed_index) == len(signed_value) == len(occupancy)
        assert len(np.unique(signed_index)) == len(signed_index)

    court = validate(
        manifest,
        maximum_surface_samples=96,
        maximum_distance_samples=48,
        maximum_sign_samples=64,
    )
    assert court["status"] == "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
    assert court["family_candidate_outcomes"]["SURFACE_GEOMETRY"] == "QUALIFIED_CORE_CANDIDATE"
    assert court["family_candidate_outcomes"]["ORIENTED_NORMALS"] == "QUALIFIED_CORE_CANDIDATE"
    assert court["family_candidate_outcomes"]["SIGNED_DISTANCE_OCCUPANCY"] == "QUALIFIED_MASKED_CANDIDATE"
    assert court["authority"]["training_authorized"] is False
    assert court["authority"]["final_subject_disposition_minted"] is False
