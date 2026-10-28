"""Actual adapter/Blender witness with authored default != animation frame zero."""
import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import bpy
from mathutils import Matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_teacher_scene_smoke as legacy
from tools.observation import blender_controlled_8view_v2 as observation
from tools.observation.source_scene_contract_v2 import _json_and_chunks
from tools.iris_vnext import blender_authored_teacher_export_v1 as animated
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry
from tools.truth_foundry.blender_teacher_scene_v1 import sha256
from compiler.realsas_compiler_core.preproduct_authority_v1 import NormalizationDomainIR, normalization_domain_hash
from compiler.realsas_compiler_services.orchestrator.adapters.truth_foundry_v2 import measure_teacher_scene_stage


def ref(path):
    return {'path': str(path), 'sha256': sha256(path)}


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return ref(path)


def animated_default_source(base, target):
    doc, chunks = _json_and_chunks(base)
    body = bytearray(next(data for kind, data in chunks if kind == 0x004E4942)[:doc['buffers'][0]['byteLength']])
    def accessor(values, count, kind):
        body.extend(b'\0' * (-len(body) % 4))
        offset = len(body)
        body.extend(struct.pack('<' + 'f' * len(values), *values))
        view = len(doc.setdefault('bufferViews', []))
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(body) - offset})
        index = len(doc.setdefault('accessors', []))
        row = {'bufferView': view, 'componentType': 5126, 'count': count, 'type': kind}
        if kind == 'SCALAR': row.update(min=[min(values)], max=[max(values)])
        doc['accessors'].append(row)
        return index
    times = accessor([0., 1.], 2, 'SCALAR')
    positions = accessor([.4, 0., 0., .8, 0., 0.], 2, 'VEC3')
    scene = doc['scenes'][doc.get('scene', 0)]
    root = len(doc['nodes'])
    doc['nodes'].append({'name': 'AuthoredDefaultRoot', 'children': scene['nodes'], 'translation': [0., 0., 0.]})
    scene['nodes'] = [root]
    doc['animations'] = [{'name': 'DifferentFrameZero', 'samplers': [{'input': times, 'output': positions, 'interpolation': 'LINEAR'}],
                          'channels': [{'sampler': 0, 'target': {'node': root, 'path': 'translation'}}]}]
    doc['buffers'][0] = {'byteLength': len(body), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(body).decode()}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc))


def main(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    legacy.main(folder / 'legacy')
    source = folder / 'raw/animated.gltf'
    animated_default_source(folder / 'legacy/fixture.glb', source)
    imported = folder / 'parity/imported.json'
    subprocess.run([bpy.app.binary_path, '-b', '--factory-startup', '--python-exit-code', '2', '--python',
        str(ROOT / 'tools/observation/blender_source_parity_export_v1.py'), '--', str(source), str(imported)], check=True)
    receipt = observation.import_rest(source, folder / 'observation')
    expected_rest = geometry._extract_snapshot(animated._mesh_objects(), freeze_triangulation=True)
    snapshot = write(folder / 'snapshot.json', observation.scene_snapshot(sha256(source), Matrix.Identity(4)))
    candidate = write(folder / 'candidate.json', {'schema': 'RealSaS.Controlled8ViewSourceRestCandidate.v2',
        'subject_id': 'synthetic_default_frame', 'source_sha256': sha256(source), 'import': receipt,
        'source_scene_snapshot': {'sha256': snapshot['sha256']}})
    replay = json.loads(imported.read_text())
    admission = write(folder / 'admission.json', {'schema': 'RealSaS.IrisBindObservationAdmission.v1',
        'status': 'PASS_IRIS_BIND_OBSERVATIONS', 'subject_id': 'synthetic_default_frame',
        'source_sha256': sha256(source), 'candidate_sha256': candidate['sha256'],
        'imported_geometry_sha256': sha256(imported), 'imported_vertex_count': len(replay['vertices']),
        'bone_count': replay['bone_count'], 'scope': 'SYNTHETIC_FIXTURE_ONLY__NOT_REAL_CORPUS'})
    animated._import_animated(source)
    animated._neutralize_animation()
    old = geometry._extract_snapshot(animated._mesh_objects(), freeze_triangulation=True)
    drift = float(np.max(np.abs(old['vertices'] - expected_rest['vertices'])))
    assert drift > .1, f'synthetic fixture must expose animated/default rest drift: {drift}'
    norm = NormalizationDomainIR('1' * 64, '2' * 64, (0., 0., 0.), 1., (-1., -1., -1.), (1., 1., 1.), 'REALSAS_OBJECT_FRAME', '')
    norm = replace(norm, normalization_hash=normalization_domain_hash(norm))
    parents = [('01_SOURCE_BYTES_SEALED', 'RealSaS.SourceByteSeal.v1', {'schema': 'RealSaS.SourceByteSeal.v1', 'files': [ref(source)]}),
        ('03_SOURCE_MECHANICAL_AUDIT', 'RealSaS.SourceMechanicalAuditIR.v1', {'schema': 'RealSaS.SourceMechanicalAuditIR.v1'}),
        ('08_NORMALIZATION_DOMAIN_QUALIFIED', norm.schema_version, norm.to_dict())]
    ledger = []
    for sid, schema, value in parents:
        bound = write(folder / (sid + '.json'), value)
        ledger.append({'id': sid, 'status': 'PASS', 'outputs': [{**bound, 'schema': schema}]})
    policy = write(folder / 'policy.json', {'schema': 'RealSaS.TeacherKinematicCalibrationPolicy.v2',
        'rest_contract': 'EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY', 'angle_amplitudes_radians': [.01],
        'root_translation_source_units': [.1, -.2, .3]})
    binary = Path(bpy.app.binary_path).resolve()
    result = measure_teacher_scene_stage({'repo_root': ROOT, 'run_root': folder / 'run',
        'stage': {'id': '47_SOURCE_RESPONSE_CALIBRATION', 'policy': {'product_pass_authority': False}},
        'ledger': {'execution_class': 'WITNESS', 'stages': ledger}, 'run_manifest': {
            'subject_id': 'synthetic_default_frame', 'truth_foundry': {
                'mode': 'OBSERVATION_BIND_REST_KINEMATIC_CALIBRATION_ONLY', 'policy': policy,
                'evaluator_binary': ref(binary), 'evaluator_version': 'Blender ' + bpy.app.version_string,
                'maximum_seconds_per_subject': 90, 'producer_source_commit': '3' * 40},
            'source_files': [{'path': str(source), 'expected_sha256': sha256(source), 'source_kind': 'PRIMARY'}],
            'source_audit': {'profile': 'SOURCE_SCENE_BIND_V2', 'matrix_tolerance': 1e-6,
                'iris_observation_admission': admission, 'source_rest_candidate': candidate, 'source_scene_snapshot': snapshot}}})
    assert result['status'] == 'PASS'
    scene_path = Path(next(row['path'] for row in result['outputs'] if row['schema'] == 'RealSaS.TeacherSceneIR.v2'))
    scene = json.loads(scene_path.read_text())
    assert scene['native_clip_refs'] == [] and scene['native_temporal_rest_compatible'] is False
    assert scene['observation_bundle_sha256'] == norm.observation_set_binding_hash
    assert scene['normalization_sha256'] == norm.normalization_hash
    assert scene['observation_rest_binding']['status'] == 'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY'
    assert scene['calibration']['source_rest_to_pose_identity_max_abs'] == 0
    assert scene['calibration']['sham_repeat_max_abs'] == 0
    with np.load(scene_path.parent / 'geometry_and_response.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['rest_vertices_source_world'], expected_rest['vertices'])
        np.testing.assert_array_equal(arrays['faces'], expected_rest['faces'])
        assert arrays['isolated_position'].shape[0] == 6
        assert np.any(np.abs(arrays['isolated_position'] - arrays['pose_identity_baseline']) > 1e-4)
    write(folder / 'SMOKE_REPORT.json', {'schema': 'RealSaS.TeacherObservationRestSyntheticWitness.v2',
        'legacy_v1_kernel_witness_preserved': True, 'old_animated_to_observation_rest_max_abs': drift,
        'admitted_import_raw_hash_replayed': True, 'actual_compiler_wire_adapter_and_blender': True,
        'observation_rest_vertices_and_triangles_exact': True, 'real_corpus_qualified': False, 'training_executed': False})
    print('TEACHER_OBSERVATION_REST_SYNTHETIC_WITNESS_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args.output)
