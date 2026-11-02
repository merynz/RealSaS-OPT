from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.build_component_geometry_authority_v1 import build
from tools.corpus.validate_component_geometry_authority_v1 import validate as validate_geometry
from tools.training.validate_iris_component_geometry_consumer_v1 import validate_consumer


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "canonical" / "IRIS_COMPONENT_GEOMETRY_CONSUMER_CONTRACT_V1.json"


def packet(root: Path):
    vertices = np.asarray(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [3, -1, 0], [5, -1, 0], [3, 1, 0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6]],
        dtype=np.int32,
    )
    geometry = root / "REST_SURFACE_GEOMETRY.npz"
    np.savez_compressed(
        geometry,
        rest_vertices_source_world=vertices,
        faces=faces,
        face_owner_privileged=np.zeros(len(faces), dtype=np.int32),
    )
    snapshot = root / "SOURCE_SCENE_SNAPSHOT.json"
    snapshot.write_text(json.dumps({"source_world_to_canonical": np.eye(4).tolist()}) + "\n")
    camera = root / "camera_bundle.json"
    camera.write_text("{}\n")
    return geometry, snapshot, camera


def recipe(*, global_sign_policy="FAIL_CLOSED_EXACT_ONLY_WHERE_VALID", fstar_role="AUXILIARY_SOURCE_CONSISTENCY_ONLY__GLOBAL_SIGN_HEAD_SOFT_ANCHOR"):
    base = {
        "fstar_auxiliary": 0.02,
        "local_oriented_sdf": 0.5,
        "local_sdf_magnitude": 0.25,
        "local_sign_margin": 0.35,
        "normal_direction_slope": 0.1,
        "occupancy_when_valid": 0.25,
        "oriented_normal": 0.02,
        "signed_tsdf_when_valid": 0.5,
        "surface_bracket_regression": 0.25,
        "surface_bracket_sign": 0.25,
        "surface_distance": 1.0,
    }
    return {
        "schema": "Synthetic.IRISRecipe.v1",
        "geometry": {
            "primary_authority": "EXACT_UNSIGNED_DISTANCE__LOCAL_ORIENTED_SDF__VALIDITY_MASKED_GLOBAL_SIGN_OCCUPANCY__ORIENTED_NORMAL",
            "global_sign_policy": global_sign_policy,
            "fstar_role": fstar_role,
            "surface_bracket_epsilons": [0.005, 0.01, 0.02, 0.04],
            "loss_weights_by_role": {
                "geometry_base": base,
                "geometry_refine": {**base, "surface_distance": 0.75, "fstar_auxiliary": 0.01},
            },
        },
    }


def build_fixture(tmp_path: Path):
    geometry, snapshot, camera = packet(tmp_path)
    out = tmp_path / "geometry"
    manifest = build(
        "synthetic-consumer",
        geometry,
        snapshot,
        camera,
        out,
        surface_samples=512,
        query_samples=256,
    )
    manifest_path = out / "COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json"
    court = validate_geometry(manifest, maximum_surface_samples=64, maximum_distance_samples=32, maximum_sign_samples=48)
    court_path = out / "COMPONENT_GEOMETRY_AUTHORITY_COURT_V1.json"
    court_path.write_text(json.dumps(court, sort_keys=True, indent=2) + "\n")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe(), sort_keys=True, indent=2) + "\n")
    return manifest_path, court_path, recipe_path


def test_component_geometry_cannot_feed_a_global_sign_recipe(tmp_path):
    manifest_path, court_path, recipe_path = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="IRIS_COMPONENT_SIGN_IS_NOT_GLOBAL_SOLID_SIGN"):
        validate_consumer(CONTRACT, recipe_path, manifest_path, court_path)


def test_wrong_global_sign_policy_fails_closed(tmp_path):
    manifest_path, court_path, recipe_path = build_fixture(tmp_path)
    recipe_path.write_text(json.dumps(recipe(global_sign_policy="DENSE_ZERO_FILL")) + "\n")
    with pytest.raises(ValueError, match="IRIS_GEOMETRY_CONSUMER_GLOBAL_SIGN_POLICY"):
        validate_consumer(CONTRACT, recipe_path, manifest_path, court_path)


def test_fstar_cannot_be_promoted_to_primary(tmp_path):
    manifest_path, court_path, recipe_path = build_fixture(tmp_path)
    recipe_path.write_text(json.dumps(recipe(fstar_role="PRIMARY_GEOMETRY_AUTHORITY")) + "\n")
    with pytest.raises(ValueError, match="IRIS_GEOMETRY_CONSUMER_FSTAR_NOT_AUXILIARY"):
        validate_consumer(CONTRACT, recipe_path, manifest_path, court_path)


def test_missing_local_loss_fails_recipe_semantic_court(tmp_path):
    manifest_path, court_path, recipe_path = build_fixture(tmp_path)
    value = recipe()
    for weights in value["geometry"]["loss_weights_by_role"].values():
        weights.pop("surface_bracket_sign")
    recipe_path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="IRIS_GEOMETRY_CONSUMER_LOCAL_LOSS_MISSING"):
        validate_consumer(CONTRACT, recipe_path, manifest_path, court_path)
