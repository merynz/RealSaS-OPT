import hashlib
import json
import numpy as np

from tools.corpus.probe_closed_source_embedding_v1 import intersecting_faces
from tools.corpus.probe_source_coincidence_weights_v1 import probe


def test_intersection_probe_distinguishes_closed_tetrahedron_from_crossing_solids():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert not intersecting_faces(v, f).any()
    assert intersecting_faces(np.vstack([v, v + .1]), np.vstack([f, f + 4])).any()


def test_coincident_vertices_with_different_skin_are_not_a_mechanical_weld(tmp_path):
    geometry = tmp_path / "source.npz"
    np.savez(geometry, rest_vertices_source_world=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
             faces=np.array([[0, 1, 2], [3, 2, 1]]), face_owner_privileged=np.array([0, 0]),
             weight_vertex=np.array([0, 1, 2, 3]), weight_joint=np.array([0, 0, 0, 1]), weight_value=np.ones(4))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"subject_id": "fixture", "controls": [{}, {}], "outputs": {"rest_geometry": {
        "path": str(geometry), "sha256": hashlib.sha256(geometry.read_bytes()).hexdigest()}}}))
    result = probe(report)
    assert result["same_owner_exact_position_duplicate_group_count"] == 1
    assert result["weight_conflict_group_count"] == 1
    assert not result["full_deformation_equivalence_qualified"]
    assert not result["training_authorized"]
