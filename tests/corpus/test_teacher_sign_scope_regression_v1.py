from __future__ import annotations

import numpy as np
import pytest
import trimesh

from models.iris.truth.component_geometry_authority_v1 import build_component_geometry_authority_v1
from models.iris.truth.component_geometry_training_targets_v1 import load_component_geometry_training_targets
from tools.corpus.build_component_geometry_authority_v1 import closest_valid_surface, compact_sign_targets
from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as teacher


def test_nested_solids_nearest_component_sign_is_not_global_sign(tmp_path):
    outer = trimesh.creation.box(extents=[6, 6, 6])
    inner = trimesh.creation.box(extents=[2, 2, 2])
    vertices = np.concatenate([outer.vertices, inner.vertices])
    faces = np.concatenate([outer.faces, inner.faces + len(outer.vertices)])
    owner = np.repeat([0, 1], [len(outer.faces), len(inner.faces)]).astype(np.int32)
    authority = build_component_geometry_authority_v1(vertices, faces, owner)
    query = np.asarray([[1.2, 0., 0.]])
    distance, closest = closest_valid_surface(query, vertices, faces, authority.face_valid)
    index, signed, occupancy, component, *_ = compact_sign_targets(
        query, distance, closest, authority, authority.topology_vertices
    )
    # The nearest inner box says outside (+.2), while the union solid says
    # inside (-1.8), measured to its outer boundary. Both sign and magnitude differ.
    assert np.allclose(signed, [.2])
    assert not occupancy[0]
    assert outer.contains(query)[0]
    assert np.allclose(trimesh.proximity.closest_point(outer, query)[1], [1.8])
    query_path, surface_path = tmp_path / "query.npz", tmp_path / "surface.npz"
    np.savez(query_path, query_xyz_canonical=query, unsigned_distance=distance,
             unsigned_distance_valid=np.ones(1, bool), signed_query_index=index,
             signed_distance_value=signed, occupancy_value=occupancy, signed_component_id=component)
    np.savez(surface_path, surface_xyz_canonical=np.asarray([[3., 0, 0]]),
             normal_canonical=np.asarray([[1., 0, 0]]), normal_valid=np.ones(1, bool))
    target = load_component_geometry_training_targets(query_path, surface_path)
    assert np.array_equal(target.component_sign_rows()["component_id"], component)
    with pytest.raises(ValueError, match="IRIS_COMPONENT_SIGN_IS_NOT_GLOBAL_SOLID_SIGN"):
        target.global_sign_rows()


@pytest.mark.parametrize("kind", ["SIMPLIFIED", "SOURCE_FALLBACK"])
def test_candidate_kind_cannot_override_a_measured_failure(monkeypatch, kind):
    v = np.asarray([[0., 0, 0], [1., 0, 0], [0., 1., 0], [0., 0, 1.]])
    f = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    _, fc = teacher.topology_components(len(v), f)
    top = teacher.topology_signature(len(v), f, fc)
    monkeypatch.setattr(teacher, "has_self_intersection", lambda *_: True)
    metrics, *_ = teacher.evaluate_candidate(
        v, f, v, f, fc, top, np.zeros(len(f), np.int32), set(),
        (v + [.01, 0, 0])[None],
        [{"control_index_privileged": 0, "amplitude": .01, "sign": 1}], [], 1., kind,
    )
    assert metrics["self_intersection_pass"] is False
    assert metrics["mechanical_pass"] is False
