import numpy as np
import pytest

from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as teacher


def test_coincident_source_vertices_keep_distinct_authored_trajectories(monkeypatch):
    # Two coincident, oppositely wound source triangles. A nearest query cannot
    # distinguish their vertex/face addresses, but their interventions differ.
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]] * 2)
    f = np.array([[0,1,2],[3,5,4]])
    posed = v[None].copy(); posed[0,3:,2] += 0.2
    monkeypatch.setattr(teacher, 'nearest_mapping', lambda *_: pytest.fail('identity remapped by proximity'))
    vf, bar, dist, ff, mode = teacher.candidate_source_correspondence(v.copy(), f.copy(), v, f)
    assert mode == 'EXACT_INDEX_IDENTITY'
    np.testing.assert_array_equal(ff, [0,1])
    np.testing.assert_array_equal(dist, np.zeros(6))
    np.testing.assert_array_equal(teacher.transfer_positions(vf, bar, f, posed), posed)


@pytest.mark.parametrize('change', ['vertex', 'face_order'])
def test_identity_requires_exact_vertices_and_faces(monkeypatch, change):
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f = np.array([[0,1,2],[0,3,1]])
    cv, cf = v.copy(), f.copy()
    if change == 'vertex': cv[0,0] += 1e-12
    else: cf = cf[::-1].copy()
    calls = []
    def measured(points, *_):
        calls.append(points.copy())
        return np.zeros(len(points), dtype=int), np.tile([1.,0.,0.], (len(points),1)), np.zeros(len(points))
    monkeypatch.setattr(teacher, 'nearest_mapping', measured)
    *_, mode = teacher.candidate_source_correspondence(cv, cf, v, f)
    assert mode == 'MEASURED_NEAREST_SURFACE' and len(calls) == 2


@pytest.mark.parametrize('kind', ['SIMPLIFIED','SOURCE_FALLBACK'])
def test_exact_correspondence_keeps_measured_collision_rejection(monkeypatch, kind):
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    _, fc = teacher.topology_components(len(v), f)
    actions = [{'control_index_privileged':0, 'amplitude':.1, 'sign':1}]
    poses = (v + [.01,0.,0.])[None]
    monkeypatch.setattr(teacher, 'has_self_intersection', lambda *_: True)
    m, *_ = teacher.evaluate_candidate(v, f, v, f, fc, teacher.topology_signature(len(v),f,fc),
        np.zeros(len(f),int), set(), poses, actions, [], 1., kind)
    assert m['source_correspondence'] == 'EXACT_INDEX_IDENTITY'
    assert m['response_nrms'] == 0.0 and m['energy_nrms'] == 0.0
    assert m['self_intersection_pass'] is False and m['mechanical_pass'] is False
