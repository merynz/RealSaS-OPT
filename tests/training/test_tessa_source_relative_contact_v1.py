"""Source-relative contact can admit only exact sampled-frame identity.

These tests intentionally include intersecting authored source components.
They do not claim that a collision-free candidate has been constructed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.training.tessa_source_relative_contact_v1 import (
    build, sha256, PASS, MASKED,
)
from tools.training.validate_tessa_source_relative_contact_v1 import (
    validate, PASS as COURT_PASS, MASKED as COURT_MASKED,
)


def data(root):
    v = np.asarray([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [.2, .2, -.2], [.2, .2, .2], [.8, .2, 0.],
    ], dtype=np.float64)
    f = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    # The two independently indexed triangles intersect in the source rest.
    poses = np.stack([v.copy(), v + np.array([.05, .0, .0])], axis=0)
    source = root / "source.npz"
    candidate = root / "candidate.npz"
    np.savez_compressed(source, rest_vertices_source_world=v, faces=f,
                        isolated_position=poses)
    np.savez_compressed(candidate, vertices_source_world=v, faces=f,
                        isolated_position=poses)
    return source, candidate, v, f, poses


def measure(root, source, candidate):
    path = root / "CONTACT.json"
    result = build(source, candidate, path,
                   source_sha256=sha256(source),
                   candidate_sha256=sha256(candidate),
                   subject_id="test_subject")
    return result, path


def test_exact_indexed_source_fallback_with_existing_contact_has_a_scoped_witness(tmp_path):
    source, candidate, *_ = data(tmp_path)
    result, path = measure(tmp_path, source, candidate)
    assert result["status"] == PASS
    assert result["source_frame_count"] == 2
    assert result["authority"]["physical_collision_freedom_claimed"] is False
    assert result["authority"]["continuous_time_motion_qualified"] is False
    assert result["authority"]["training_authorized"] is False
    independent = validate(json.loads(path.read_text()))
    assert independent["status"] == COURT_PASS
    assert independent["authority"]["final_tessa_teacher_qualified"] is False


@pytest.mark.parametrize("mutation", ["rest", "pose", "face_order", "frame_count"])
def test_any_candidate_topology_or_authored_motion_drift_remains_masked(tmp_path, mutation):
    source, candidate, v, f, poses = data(tmp_path)
    cv, cf, cp = v.copy(), f.copy(), poses.copy()
    if mutation == "rest":
        cv[0, 2] += 1e-12
    elif mutation == "pose":
        cp[1, 3, 1] += 1e-12
    elif mutation == "face_order":
        cf = cf[::-1].copy()
    else:
        cp = cp[:1]
    np.savez_compressed(candidate, vertices_source_world=cv, faces=cf,
                        isolated_position=cp)
    result, path = measure(tmp_path, source, candidate)
    assert result["status"] == MASKED
    assert result["authority"]["source_relative_contact_equivalence_certified"] is False
    assert validate(json.loads(path.read_text()))["status"] == COURT_MASKED


def test_court_rejects_fabricated_resigned_pass(tmp_path):
    source, candidate, v, f, poses = data(tmp_path)
    poses = poses.copy()
    poses[1, 3, 2] += .2
    np.savez_compressed(candidate, vertices_source_world=v, faces=f,
                        isolated_position=poses)
    _, path = measure(tmp_path, source, candidate)
    forged = json.loads(path.read_text())
    forged["status"] = PASS
    forged["rest_and_sampled_frames_index_identical"] = True
    forged["authority"]["source_relative_contact_equivalence_certified"] = True
    forged["authority"]["new_or_increased_contact_excluded_on_identical_sampled_frames"] = True
    with pytest.raises(ValueError, match="RESIGNED_FALSE_PASS"):
        validate(forged)


def test_court_rejects_parent_data_mutation(tmp_path):
    source, candidate, v, f, poses = data(tmp_path)
    _, path = measure(tmp_path, source, candidate)
    np.savez_compressed(candidate, vertices_source_world=v + .5, faces=f,
                        isolated_position=poses)
    with pytest.raises(ValueError, match="PARENT_DRIFT"):
        validate(json.loads(path.read_text()))


def test_no_sampled_frames_cannot_mint_equivalence(tmp_path):
    source, candidate, v, f, poses = data(tmp_path)
    np.savez_compressed(candidate, vertices_source_world=v, faces=f,
                        isolated_position=np.empty((0, len(v), 3)))
    with pytest.raises(ValueError, match="SAMPLED_FRAME_LAYOUT"):
        measure(tmp_path, source, candidate)


def test_no_relabelling_to_continuous_motion_or_training(tmp_path):
    source, candidate, *_ = data(tmp_path)
    _, path = measure(tmp_path, source, candidate)
    forged = json.loads(path.read_text())
    forged["authority"]["continuous_time_motion_qualified"] = True
    with pytest.raises(ValueError, match="AUTHORITY_OVERCLAIM"):
        validate(forged)
