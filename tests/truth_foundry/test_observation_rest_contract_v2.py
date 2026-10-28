from copy import deepcopy
import numpy as np
import pytest

from tools.truth_foundry.observation_rest_contract_v2 import compare_geometry, compare_snapshots, validate_rest_binding


def geometry():
    # The observation exporter uses scene order; teacher uses its explicit owner order.
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    b = a + [3., 0., 0.]
    imported = {'vertices': np.concatenate([b, a]).tolist(), 'faces': [[0, 1, 2], [3, 4, 5]],
                'objects': [{'name': 'B', 'vertex_count': 3}, {'name': 'A', 'vertex_count': 3}]}
    rest = {'vertices': np.concatenate([a, b]), 'faces': np.array([[0, 1, 2], [3, 4, 5]]),
            'owners': [{'source_import_name': 'A', 'vertex_count': 3}, {'source_import_name': 'B', 'vertex_count': 3}]}
    return rest, imported


def test_exact_geometry_preserves_declared_object_address_order():
    rest, imported = geometry()
    proof = compare_geometry(rest, imported)
    assert proof['vertices_exact_after_declared_object_order'] is True
    assert proof['triangles_exact_after_declared_object_order'] is True
    assert proof['nearest_neighbour_transfer_used'] is False


@pytest.mark.parametrize('change,code', [('position', 'VERTEX_DRIFT'), ('winding', 'TRIANGLE_DRIFT'),
                                      ('cross_object', 'CROSS_OBJECT_FACE'), ('missing', 'OBJECT_SET')])
def test_matching_counts_do_not_hide_geometry_or_topology_drift(change, code):
    rest, imported = geometry()
    if change == 'position': imported['vertices'][0][0] += .1
    if change == 'winding': imported['faces'][0] = [0, 2, 1]
    if change == 'cross_object': imported['faces'][0][2] = 3
    if change == 'missing': imported['objects'][0]['name'] = 'other'
    with pytest.raises(ValueError, match=code):
        compare_geometry(rest, imported)


def snapshot():
    identity = np.eye(4).tolist()
    node = {'node_id': 'source:node:0', 'source_import_name': 'root', 'type': 'EMPTY',
            'parent_node_id': None, 'parent_type': 'OBJECT', 'parent_bone': None,
            'node_to_source_world': identity, 'effective_node_to_parent': identity,
            'authored_import_matrix_basis': identity, 'matrix_parent_inverse': identity,
            'node_to_canonical': identity, 'active_action_count': 0, 'active_nla_track_count': 0, 'driver_count': 0}
    return {'schema': 'RealSaS.ImportedSourceSceneSnapshot.v2', 'source_sha256': 'a' * 64,
            'source_world_to_canonical': identity, 'canonical_to_source_world': identity,
            'nodes': [node], 'bones': []}


def test_sealed_scene_frames_and_parents_are_part_of_rest_binding():
    sealed = snapshot()
    assert compare_snapshots(sealed, deepcopy(sealed), 1e-6) == 0
    current = deepcopy(sealed)
    current['nodes'][0]['authored_import_matrix_basis'] = np.diag([2., 1., 1., 1.]).tolist()
    with pytest.raises(ValueError, match='FRAME_DRIFT'):
        compare_snapshots(sealed, current, 1e-6)
    current = deepcopy(sealed)
    current['nodes'][0]['parent_bone'] = 'other'
    with pytest.raises(ValueError, match='PARENT_DRIFT'):
        compare_snapshots(sealed, current, 1e-6)


def test_binding_requires_the_admission_raw_import_hash_and_retains_authority_limits():
    rest, imported = geometry()
    receipt = {'source_sha256': 'a' * 64, 'operation': 'fixture', 'importer_options': {},
               'pose_policy': 'SOURCE_BIND_REST__NO_ANIMATION', 'raw_source_mutated': False, 'excluded_import_controls': []}
    imported['bone_count'] = 0
    imported['import'] = receipt
    job = {'source': {'sha256': 'a' * 64}, 'subject_key': 'fixture', 'source_rest_candidate': {'sha256': 'b' * 64},
           'source_scene_snapshot': {'sha256': 'c' * 64}, 'observation_import_geometry': {'sha256': 'd' * 64},
           'observation_admission': {'sha256': 'e' * 64}, 'matrix_tolerance': 1e-6}
    admission = {'schema': 'RealSaS.IrisBindObservationAdmission.v1', 'status': 'PASS_IRIS_BIND_OBSERVATIONS',
                 'source_sha256': 'a' * 64, 'subject_id': 'fixture', 'candidate_sha256': 'b' * 64,
                 'imported_geometry_sha256': 'd' * 64, 'imported_vertex_count': 6, 'bone_count': 0}
    candidate = {'schema': 'RealSaS.Controlled8ViewSourceRestCandidate.v2', 'source_sha256': 'a' * 64,
                 'subject_id': 'fixture', 'source_scene_snapshot': {'sha256': 'c' * 64}, 'import': receipt}
    proof = validate_rest_binding(job, admission, candidate, snapshot(), snapshot(), imported, receipt, rest)
    assert proof['status'] == 'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY'
    assert proof['native_clip_rest_compatibility_claimed'] is False
    assert proof['student_supervision_authorized'] is False and proof['training_authorized'] is False
    job['observation_import_geometry']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='SEALED_IMPORT_BYTES_DRIFT'):
        validate_rest_binding(job, admission, candidate, snapshot(), snapshot(), imported, receipt, rest)
