import hashlib
import json
import numpy as np
import trimesh

from tools.corpus.build_source_union_solid_candidate_v1 import build, union_boundary
from tools.corpus.validate_source_union_solid_candidate_v1 import validate


def test_union_distance_excludes_internal_overlapping_source_faces():
    a = trimesh.creation.box(extents=[2, 2, 2])
    b = a.copy()
    b.apply_translation([1, 0, 0])
    v = np.vstack([a.vertices, b.vertices])
    f = np.vstack([a.faces, b.faces + len(a.vertices)])
    out_v, out_f, _, _ = union_boundary(v, f, np.repeat([0, 1], [len(a.faces), len(b.faces)]))
    target = trimesh.Trimesh(vertices=out_v, faces=out_f, process=False)
    q = np.array([[.5, 0, 0]])
    _, raw_distance, _ = trimesh.proximity.closest_point(trimesh.Trimesh(vertices=v, faces=f, process=False), q)
    _, boundary_distance, _ = trimesh.proximity.closest_point(target, q)
    np.testing.assert_allclose(raw_distance, [.5])
    np.testing.assert_allclose(boundary_distance, [1.])
    assert target.contains(q).all()


def test_nested_component_is_not_the_global_sign_or_distance_domain():
    outer, inner = trimesh.creation.box(extents=[4, 4, 4]), trimesh.creation.box(extents=[2, 2, 2])
    v = np.vstack([outer.vertices, inner.vertices])
    f = np.vstack([outer.faces, inner.faces + len(outer.vertices)])
    out_v, out_f, _, _ = union_boundary(v, f, np.repeat([0, 1], [len(outer.faces), len(inner.faces)]))
    target = trimesh.Trimesh(vertices=out_v, faces=out_f, process=False)
    q = np.array([[1.25, 0, 0]])
    assert not inner.contains(q).any()
    assert target.contains(q).all()
    np.testing.assert_allclose(trimesh.proximity.closest_point(target, q)[1], [.75])


def test_actual_candidate_passes_independent_source_union_court(tmp_path):
    a, b = trimesh.creation.box(extents=[2, 2, 2]), trimesh.creation.box(extents=[2, 2, 2])
    b.apply_translation([1, 0, 0])
    source = tmp_path / "source.gltf"
    source.write_text("immutable fixture source")
    def ref(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    geometry = tmp_path / "rest.npz"
    np.savez(geometry, rest_vertices_source_world=np.vstack([a.vertices, b.vertices]),
             faces=np.vstack([a.faces, b.faces + len(a.vertices)]), face_owner_privileged=np.repeat([0, 1], [len(a.faces), len(b.faces)]))
    imported = tmp_path / "import.json"
    imported.write_text(json.dumps({"source": ref(source), "outputs": {"rest_geometry": ref(geometry)}}))
    structure = tmp_path / "structure.json"
    structure.write_text(json.dumps({"import_report": ref(imported)}))
    row = {"subject_id": "fixture_union", "report": ref(structure)}
    probe = {"subject_id": "fixture_union", "components_with_self_intersection": 0,
             "components": [{"signed_volume_source_world": 8.}, {"signed_volume_source_world": 8.}]}
    result = build(row, probe, tmp_path / "out", query_count=128)
    court = validate(result["manifest"]["path"])
    assert court["status"].startswith("PASS_SOURCE_UNION_BOUNDARY_AND_GLOBAL_SDF_NUMERICS")
    assert court["validated"]["all_signed_query_rows"] == 128
    assert not court["training_authorized"]
    assert not court["deformation_qualified"]
