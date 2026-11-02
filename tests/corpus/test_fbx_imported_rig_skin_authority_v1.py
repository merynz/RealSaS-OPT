from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.validate_fbx_imported_rig_skin_authority_v1 import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def write_case(root: Path, *, vertex_count: int = 2, weighted_vertices=(0, 1), replay_delta: float = 0.0):
    source = root / "subject.fbx"
    source.write_bytes(b"synthetic-fbx-bytes")
    matrix = np.eye(4).tolist()

    inspection_arrays = root / "inspection.npz"
    vertices = np.asarray(weighted_vertices, np.int32)
    np.savez_compressed(
        inspection_arrays,
        weight_vertex=vertices,
        weight_joint=np.zeros(len(vertices), np.int32),
        weight_value=np.ones(len(vertices), np.float32),
    )
    inspection = {
        "schema": "RealSaS.ImportedTeacherSourceInspection.v1",
        "subject_id": "fbx-subject",
        "source": ref(source),
        "runtime": {"blender": "4.0.2"},
        "import": {
            "operation": "FBX_IMPORT_WITHOUT_ANIMATION",
            "importer_options": {"automatic_bone_orientation": False, "use_anim": False},
        },
        "controls": [
            {
                "armature": "Rig",
                "name": "Root",
                "parent": None,
                "use_deform": True,
                "rest_matrix_world": matrix,
            }
        ],
        "meshes": [
            {
                "owner_id": 0,
                "name": "Mesh",
                "vertex_count": vertex_count,
                "triangle_count": 1,
                "armature_targets": ["Rig"],
            }
        ],
        "outputs": {"rest_geometry": ref(inspection_arrays)},
        "training_authorized": False,
    }
    inspection_path = root / "inspection.json"
    inspection_path.write_text(json.dumps(inspection, sort_keys=True) + "\n")

    replay_arrays = root / "replay.npz"
    replay_values = np.ones(len(vertices), np.float32)
    if len(replay_values):
        replay_values[0] += replay_delta
    np.savez_compressed(
        replay_arrays,
        weight_mesh_index=np.zeros(len(vertices), np.int32),
        weight_vertex_index=vertices,
        weight_control_index=np.zeros(len(vertices), np.int32),
        weight_value=replay_values,
    )
    replay = {
        "schema": "RealSaS.FbxRigSkinImporterReplay.v1",
        "status": "PASS_FBX_RIG_SKIN_IMPORTER_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "subject_id": "fbx-subject",
        "source": ref(source),
        "blender_version": "4.0.2",
        "import": {
            "operation": "FBX_IMPORT_WITHOUT_ANIMATION",
            "importer_options": {"automatic_bone_orientation": False, "use_anim": False},
            "pose_policy": "SOURCE_BIND_REST__NO_ANIMATION",
        },
        "controls": [
            {
                "armature": "Rig",
                "name": "Root",
                "parent": None,
                "use_deform": True,
                "rest_matrix_world": matrix,
            }
        ],
        "meshes": [
            {
                "mesh_index": 0,
                "name": "Mesh",
                "vertex_count": vertex_count,
                "armature_target": "Rig",
                "weighted_vertex_count": len(vertices),
            }
        ],
        "arrays": ref(replay_arrays),
        "authority": {"training_authorized": False},
    }
    replay_path = root / "replay.json"
    replay_path.write_text(json.dumps(replay, sort_keys=True) + "\n")
    return inspection_path, replay_path


def test_fbx_full_weight_coverage_is_core_candidate(tmp_path: Path):
    inspection, replay = write_case(tmp_path)
    result = validate(inspection, replay)
    assert result["family_candidate_outcomes"] == {
        "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
        "SKIN_WEIGHTS": "QUALIFIED_CORE_CANDIDATE",
    }
    assert result["skin_quality"]["simplex_valid_weight_coverage_fraction"] == 1.0
    assert result["authority"]["training_authorized"] is False
    assert result["equivalence"]["semantic_skin_transfer_used"] is False


def test_fbx_partial_weight_coverage_is_masked_not_zero_filled(tmp_path: Path):
    inspection, replay = write_case(tmp_path, vertex_count=2, weighted_vertices=(0,))
    result = validate(inspection, replay)
    assert result["family_candidate_outcomes"]["SKIN_WEIGHTS"] == "QUALIFIED_MASKED_CANDIDATE"
    assert result["skin_quality"]["simplex_valid_weight_coverage_fraction"] == 0.5
    assert result["skin_quality"]["invalid_or_missing_vertices_are_masked_not_zero_filled"] is True


def test_fbx_weight_drift_fails_closed(tmp_path: Path):
    inspection, replay = write_case(tmp_path, replay_delta=0.01)
    with pytest.raises(ValueError, match="FBX_RIG_SKIN_COURT_WEIGHT_VALUE_DRIFT"):
        validate(inspection, replay)
