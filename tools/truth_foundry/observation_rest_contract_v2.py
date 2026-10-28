"""Exact geometry replay and sealed scene-frame binding for teacher V2.

This proves a rest-coordinate bridge, not response-family qualification,
observability, native-clip compatibility or student/training eligibility.
"""
from __future__ import annotations

import numpy as np

from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot


def require(value, code):
    if not value:
        raise ValueError(code)


def compare_geometry(rest, imported):
    vertices = np.asarray(imported['vertices'], dtype=np.float64)
    faces = np.asarray(imported['faces'], dtype=np.int64)
    require(vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices)
            and np.isfinite(vertices).all(), 'TEACHER_REST_REPLAY_VERTEX_LAYOUT')
    require(faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces)
            and faces.min() >= 0 and faces.max() < len(vertices), 'TEACHER_REST_REPLAY_FACE_LAYOUT')
    rows = imported['objects']
    require(len({row['name'] for row in rows}) == len(rows), 'TEACHER_REST_REPLAY_DUPLICATE_OBJECT')
    owners = rest['owners']
    require({row['name'] for row in rows} == {row['source_import_name'] for row in owners}
            and len(rows) == len(owners), 'TEACHER_REST_REPLAY_OBJECT_SET')
    blocks = {}
    offset = covered = 0
    for row in rows:
        count = row['vertex_count']
        require(type(count) is int and count >= 0 and offset + count <= len(vertices), 'TEACHER_REST_REPLAY_OBJECT_LAYOUT')
        selected = (faces[:, 0] >= offset) & (faces[:, 0] < offset + count)
        local = faces[selected] - offset
        require(not len(local) or (local.min() >= 0 and local.max() < count), 'TEACHER_REST_REPLAY_CROSS_OBJECT_FACE')
        blocks[row['name']] = (vertices[offset:offset + count], local)
        covered += int(selected.sum())
        offset += count
    require(offset == len(vertices) and covered == len(faces), 'TEACHER_REST_REPLAY_OBJECT_COVERAGE')
    expected_v, expected_f = [], []
    offset = 0
    for owner in owners:
        v, f = blocks[owner['source_import_name']]
        require(owner['vertex_count'] == len(v), 'TEACHER_REST_REPLAY_OWNER_COUNT')
        expected_v.append(v)
        expected_f.append(f + offset)
        offset += len(v)
    require(np.array_equal(rest['vertices'], np.concatenate(expected_v)), 'TEACHER_REST_REPLAY_VERTEX_DRIFT')
    require(np.array_equal(rest['faces'], np.concatenate(expected_f)), 'TEACHER_REST_REPLAY_TRIANGLE_DRIFT')
    return {'vertex_count': len(vertices), 'triangle_count': len(faces), 'object_count': len(rows),
            'vertices_exact_after_declared_object_order': True, 'triangles_exact_after_declared_object_order': True,
            'nearest_neighbour_transfer_used': False}


def compare_snapshots(sealed, current, tolerance):
    validate_imported_scene_snapshot(sealed, tolerance=tolerance)
    validate_imported_scene_snapshot(current, tolerance=tolerance)
    require(sealed['source_sha256'] == current['source_sha256'], 'TEACHER_REST_SNAPSHOT_SOURCE_DRIFT')
    maximum = 0.

    def numeric(a, b):
        nonlocal maximum
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.allclose(a, b, atol=tolerance, rtol=0), 'TEACHER_REST_SNAPSHOT_FRAME_DRIFT')
        maximum = max(maximum, float(np.max(np.abs(a - b))))

    for key in ('source_world_to_canonical', 'canonical_to_source_world'):
        numeric(sealed[key], current[key])
    for section, identity, exact, arrays in [
        ('nodes', lambda row: row['node_id'],
         ('source_import_name', 'type', 'parent_node_id', 'parent_type', 'parent_bone'),
         ('node_to_source_world', 'effective_node_to_parent', 'authored_import_matrix_basis', 'matrix_parent_inverse', 'node_to_canonical')),
        ('bones', lambda row: (row['armature_node_id'], row['source_bone_name']),
         ('source_parent_bone_name', 'use_deform'),
         ('rest_joint_to_source_world', 'rest_joint_to_canonical', 'head_source_world', 'tail_source_world', 'head_canonical', 'tail_canonical')),
    ]:
        a = {identity(row): row for row in sealed[section]}
        b = {identity(row): row for row in current[section]}
        require(len(a) == len(sealed[section]) and len(b) == len(current[section]) and set(a) == set(b),
                'TEACHER_REST_SNAPSHOT_MEMBER_DRIFT')
        for key in a:
            require(all(a[key].get(field) == b[key].get(field) for field in exact), 'TEACHER_REST_SNAPSHOT_PARENT_DRIFT')
            for field in arrays:
                numeric(a[key][field], b[key][field])
    return maximum


def validate_rest_binding(job, admission, candidate, sealed_snapshot, current_snapshot, imported, import_receipt, rest):
    source_sha = job['source']['sha256']
    require(admission.get('schema') == 'RealSaS.IrisBindObservationAdmission.v1'
            and admission.get('status') == 'PASS_IRIS_BIND_OBSERVATIONS', 'TEACHER_REST_OBSERVATION_ADMISSION_REQUIRED')
    require(candidate.get('schema') == 'RealSaS.Controlled8ViewSourceRestCandidate.v2', 'TEACHER_REST_CANDIDATE_SCHEMA')
    for value in (admission, candidate):
        require(value['source_sha256'] == source_sha and value['subject_id'] == job['subject_key'], 'TEACHER_REST_SOURCE_BINDING_DRIFT')
    require(admission['candidate_sha256'] == job['source_rest_candidate']['sha256']
            and candidate['source_scene_snapshot']['sha256'] == job['source_scene_snapshot']['sha256'], 'TEACHER_REST_OBSERVATION_LINEAGE_DRIFT')
    replay_sha = job['observation_import_geometry']['sha256']
    require(replay_sha == admission['imported_geometry_sha256'], 'TEACHER_REST_SEALED_IMPORT_BYTES_DRIFT')
    sealed_import = candidate['import']
    for key in ('source_sha256', 'operation', 'importer_options', 'pose_policy', 'raw_source_mutated', 'excluded_import_controls'):
        require(import_receipt.get(key) == sealed_import.get(key) == imported['import'].get(key), 'TEACHER_REST_IMPORT_POLICY_DRIFT:' + key)
    require(import_receipt['pose_policy'] == 'SOURCE_BIND_REST__NO_ANIMATION'
            and import_receipt['raw_source_mutated'] is False, 'TEACHER_REST_IMPORT_SCOPE')
    geometry = compare_geometry(rest, imported)
    require(geometry['vertex_count'] == admission['imported_vertex_count']
            and len(current_snapshot['bones']) == imported['bone_count'] == admission['bone_count'], 'TEACHER_REST_ADMITTED_CARDINALITY_DRIFT')
    error = compare_snapshots(sealed_snapshot, current_snapshot, job['matrix_tolerance'])
    return {'schema': 'RealSaS.TeacherObservationRestBinding.v2', 'status': 'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY',
            'source_sha256': source_sha, 'observation_admission_sha256': job['observation_admission']['sha256'],
            'candidate_sha256': job['source_rest_candidate']['sha256'], 'snapshot_sha256': job['source_scene_snapshot']['sha256'],
            'imported_geometry_sha256': replay_sha, 'geometry': geometry, 'maximum_scene_frame_error': error,
            'matrix_tolerance_from_source_contract': job['matrix_tolerance'],
            'native_clip_rest_compatibility_claimed': False, 'student_supervision_authorized': False,
            'training_authorized': False, 'product_authority_changed': False}
