from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mechanics.audit_stage47_source_mechanics_v1 import (
    SourceAuditError,
    audit_stage47_source_mechanics_v1,
)


def _make_stage47_pair(tmp_path: Path, *, mutate_pair=False):
    rest = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=np.float64)
    neg = rest + np.array([[0., 0., 0.], [0., -0.01, 0.], [0., -0.02, 0.]])
    pos = rest + np.array([[0., 0., 0.], [0., 0.01, 0.], [0., 0.02, 0.]])
    npz = tmp_path / "source_mechanics.npz"
    np.savez_compressed(
        npz, rest_vertices_source_world=rest, pose_identity_baseline=rest,
        sham_before=rest, sham_after=rest, isolated_position=np.stack((neg, pos)),
        root_control_position=rest + (0.13, -0.2, 0.01),
        source_weight_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        source_weight_values=np.ones(3, dtype=np.float64),
        source_weight_bone_indices_privileged=np.array([0, 1, 2], dtype=np.int64),
        source_weight_coverage=np.ones(3, dtype=bool),
    )
    npz_sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    action = {
        "kind": "ISOLATED_KINEMATIC_PARAMETER_PERTURBATION",
        "control_index_privileged": 0,
        "opaque_privileged_handle": "teacher-only-handle",
        "amplitude": 0.01,
        "amplitude_index": 0,
        "units": "RADIAN",
    }
    scene = {
        "schema": "RealSaS.TeacherSceneIR.v2", "subject_key": "RESEARCH_EXPOSED_FIXTURE",
        "source_sha256": "a"*64,
        "student_input_allowed": False,
        "qualification_minted": False,
        "geometry_payloads": [{"sha256": npz_sha}],
        "coordinate_contract": {"frame": "SOURCE_WORLD"},
        "constraint_capabilities": {"authored_physical_dof_or_joint_limit": "UNKNOWN"},
        "calibration": {
            "action_measure": "UNSEALED__NO_IMPLICIT_UNIFORM_LAW",
            "root_translation_source_units": [0.13, -0.2, 0.01],
            "actions": [{**action, "sign": -1},
                        {**action, "sign": (-1 if mutate_pair else 1)}],
        },
    }
    teacher = tmp_path / "teacher.json"
    teacher.write_text(json.dumps(scene, sort_keys=True), encoding="utf-8")
    return teacher, npz


def test_source_mechanical_intervention_audit_keeps_authority_separate(tmp_path):
    teacher, npz = _make_stage47_pair(tmp_path)
    report = audit_stage47_source_mechanics_v1(teacher, npz)
    assert report["status"] == "PASS_SOURCE_MEASUREMENT_INTEGRITY_ONLY"
    assert report["source_weight_covered_vertex_count"] == 3
    assert report["action_count"] == 2
    assert report["sign_pair_count"] == 1
    assert report["nonzero_source_consequence_actions"] == 2
    assert report["source_displacement_max_world"] == pytest.approx(0.02)
    assert report["source_pair_even_residual_median_world"] == pytest.approx(0.)
    assert report["student_training_authorized"] is False
    assert report["compiled_mechanics_correspondence_qualified"] is False


def test_source_audit_rejects_sha_pair_and_baseline_corruption(tmp_path):
    teacher, npz = _make_stage47_pair(tmp_path)
    with pytest.raises(SourceAuditError, match="SHA_DRIFT"):
        audit_stage47_source_mechanics_v1(
            teacher, npz, expected_geometry_sha256="0"*64,
        )
    report = json.loads(teacher.read_text())
    report["geometry_payloads"][0]["sha256"] = "0"*64
    teacher.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SourceAuditError, match="SOURCE_BYTES_NOT_BOUND"):
        audit_stage47_source_mechanics_v1(teacher, npz)
    teacher, npz = _make_stage47_pair(tmp_path, mutate_pair=True)
    with pytest.raises(SourceAuditError, match="ACTION_SIGN_PAIR_DRIFT"):
        audit_stage47_source_mechanics_v1(teacher, npz)
