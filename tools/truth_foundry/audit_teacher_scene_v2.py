#!/usr/bin/env python3
"""Read-only numeric audit of Registry-exported observation-rest measurements.

This verifies bytes, runtime/rest bindings and reset arrays. It does not admit
student truth, infer native-clip compatibility or select a response action law.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from tools.truth_foundry.audit_teacher_scene_v1 import inspect_numeric, payload, require
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256
from tools.truth_foundry.observation_rest_contract_v2 import compare_geometry


def inspect(scene, available):
    require(scene['schema'] == 'RealSaS.TeacherSceneIR.v2'
            and scene['authority'] == 'UNQUALIFIED_DERIVED_MEASUREMENTS'
            and scene['qualification_minted'] is False and scene['student_input_allowed'] is False
            and scene['native_clip_refs'] == [] and scene['native_temporal_rest_compatible'] is False,
            'TEACHER_V2_AUDIT_AUTHORITY_DRIFT')
    rest = scene['observation_rest_binding']
    runtime = scene['observation_runtime_binding']
    require(rest['schema'] == 'RealSaS.TeacherObservationRestBinding.v2'
            and rest['status'] == 'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY'
            and rest['source_sha256'] == scene['source_sha256']
            and all(rest[key] is False for key in ('student_supervision_authorized', 'training_authorized',
                'native_clip_rest_compatibility_claimed', 'product_authority_changed')),
            'TEACHER_V2_AUDIT_REST_BINDING_DRIFT')
    require(runtime['status'] == 'PASS_EXACT_ADMITTED_IMPORT_BYTES'
            and runtime['imported_geometry_sha256'] == rest['imported_geometry_sha256']
            and runtime['evaluator_binary_sha256'] == scene['evaluator_binary_sha256']
            and runtime['binary_identity_equal'] is (runtime['observation_producer_binary_sha256'] == runtime['evaluator_binary_sha256']),
            'TEACHER_V2_AUDIT_RUNTIME_BINDING_DRIFT')
    replay = Path(available[rest['imported_geometry_sha256']])
    require(sha256(replay) == rest['imported_geometry_sha256'], 'TEACHER_V2_AUDIT_IMPORT_BYTES_DRIFT')
    imported = json.loads(replay.read_text())
    array_path = payload(scene['geometry_payloads'][0], available)
    # The producer extracts mesh blocks sorted by source import name. Rebuild
    # only that declared order; no nearest-neighbour or tolerant transfer.
    owners = [{'source_import_name': row['name'], 'vertex_count': row['vertex_count']}
              for row in sorted(imported['objects'], key=lambda row: row['name'])]
    with np.load(array_path, allow_pickle=False) as arrays:
        geometry = compare_geometry({'vertices': arrays['rest_vertices_source_world'], 'faces': arrays['faces'],
                                     'owners': owners}, imported)
        require(np.array_equal(arrays['rest_vertices_source_world'], arrays['pose_identity_baseline']),
                'TEACHER_V2_AUDIT_REST_IDENTITY_DRIFT')
    require(geometry == rest['geometry'], 'TEACHER_V2_AUDIT_REPORTED_GEOMETRY_DRIFT')
    result = inspect_numeric(scene, available)
    return {**result, 'observation_rest_binding': rest, 'observation_runtime_binding': runtime,
            'native_clip_rest_compatibility_claimed': False}


def inspect_export(observation, artifact_root=None):
    require(observation['schema'] == 'RealSaS.TeacherCalibrationObservation.v1'
            and observation['all_measurements_committed_and_exported'] is True
            and observation['measurement_target_goal_complete'] is True,
            'TEACHER_V2_AUDIT_COMPLETE_MEASUREMENT_REQUIRED')
    campaign = observation['campaign']
    require(campaign['execution_scope'] == 'SOURCE_ENTRY_TO_RESEARCH_COMPONENT'
            and campaign['full_closure_complete'] is False
            and all(member['final_state'] == 'COMPLETED' for member in campaign['members']),
            'TEACHER_V2_AUDIT_COMPONENT_SCOPE_DRIFT')
    expected = {member['attempt_id'] for member in campaign['members'] if member['role'] == 'COHORT'}
    actual = [member['attempt_id'] for member in observation['exports'] if member['role'] == 'COHORT']
    require(expected and set(actual) == expected and len(actual) == len(expected),
            'TEACHER_V2_AUDIT_COHORT_DRIFT')
    results = []
    for member in observation['exports']:
        if member['role'] != 'COHORT':
            continue
        available = {ref['sha256']: ref['path'] for ref in member['outputs']}
        if artifact_root is not None:
            root = Path(artifact_root).resolve() / member['attempt_id'] / 'outputs'
            available = {sha256(path): path for path in root.rglob('*') if path.is_file()}
        scenes = [ref for ref in member['outputs'] if ref['schema'] == 'RealSaS.TeacherSceneIR.v2']
        require(len(scenes) == 1, 'TEACHER_V2_AUDIT_EXACT_SCENE_REQUIRED')
        scene = json.loads(payload(scenes[0], available).read_text())
        results.append({'attempt_id': member['attempt_id'],
            'registry_stage_artifact_id': member['registry_stage_artifact_id'], **inspect(scene, available)})
    return {'schema': 'RealSaS.TeacherSceneNumericAudit.v2', 'campaign_id': observation['campaign_id'],
        'experiment_id': observation['experiment_id'], 'subjects': results, 'truth_qualification_minted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-root', help='Downloaded export root; resolve payloads by actual SHA, ignoring old host paths')
    args = parser.parse_args()
    result = inspect_export(json.loads(Path(args.observation).read_text()), args.artifact_root)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'subjects_inspected': len(result['subjects']), 'truth_qualification_minted': False}))
