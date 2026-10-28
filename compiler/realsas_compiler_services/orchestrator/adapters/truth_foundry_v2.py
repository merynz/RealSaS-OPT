"""Released observation-rest calibration; no student or product authority."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING

from compiler.realsas_compiler_core.hashing import content_sha256
from .adapter_io import load_file_ref, sha256_file, stage_output_payload
from .truth_foundry_v1 import _artifact, _require, measure_teacher_scene_stage as legacy_measure

if TYPE_CHECKING:
    from tools.observation import blender_source_parity_export_v1 as _import_program
    from tools.truth_foundry.blender_teacher_scene_v2 import build as _producer_program


def _run(executable, program, arguments, log_path, maximum_seconds, subject):
    with log_path.open('w') as log:
        process = subprocess.Popen([str(executable), '-b', '--factory-startup', '--python-exit-code', '2',
            '--python', str(program), '--', *map(str, arguments)], stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f'[TRUTH_REST_V2_HEARTBEAT] subject={subject} elapsed_s={elapsed:.1f}', flush=True)
                    _require(elapsed < maximum_seconds, 'TRUTH_REST_V2_TIME_BUDGET_EXCEEDED')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    _require(code == 0, 'TRUTH_REST_V2_BLENDER_FAILED__INSPECT_BOUND_LOG:' + str(log_path))


def measure_teacher_scene_stage(ctx):
    cfg = dict(ctx['run_manifest'].get('truth_foundry') or {})
    _require(ctx['stage']['policy'].get('product_pass_authority') is False, 'TRUTH_REST_V2_CANNOT_OWN_PRODUCT_PASS')
    if cfg.get('mode') == 'SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS':
        return legacy_measure(ctx)
    _require(cfg.get('mode') == 'OBSERVATION_BIND_REST_KINEMATIC_CALIBRATION_ONLY', 'TRUTH_REST_V2_MODE_REQUIRED')
    source = stage_output_payload(ctx, '01_SOURCE_BYTES_SEALED', 'RealSaS.SourceByteSeal.v1')
    stage_output_payload(ctx, '03_SOURCE_MECHANICAL_AUDIT', 'RealSaS.SourceMechanicalAuditIR.v1')
    normalization = stage_output_payload(ctx, '08_NORMALIZATION_DOMAIN_QUALIFIED', 'RealSaS.NormalizationDomainIR.v1')
    audit = ctx['run_manifest']['source_audit']
    _require(audit.get('profile') == 'SOURCE_SCENE_BIND_V2', 'TRUTH_REST_V2_SOURCE_SCENE_PROFILE_REQUIRED')
    rows = [row for row in ctx['run_manifest']['source_files'] if row.get('source_kind') == 'PRIMARY']
    _require(len(rows) == 1, 'TRUTH_REST_V2_PRIMARY_SOURCE_REQUIRED')
    source_ref = {'path': rows[0]['path'], 'sha256': rows[0]['expected_sha256']}
    _require(source_ref['sha256'] in {row['sha256'] for row in source['files']}, 'TRUTH_REST_V2_SOURCE_NOT_QUALIFIED_PARENT')
    for ref in (source_ref, cfg['evaluator_binary']):
        load_file_ref(ref, json_required=False)
    admission = load_file_ref(audit['iris_observation_admission'], expected_schema='RealSaS.IrisBindObservationAdmission.v1')
    candidate = load_file_ref(audit['source_rest_candidate'], expected_schema='RealSaS.Controlled8ViewSourceRestCandidate.v2')
    load_file_ref(audit['source_scene_snapshot'], expected_schema='RealSaS.ImportedSourceSceneSnapshot.v2')
    _require(admission['candidate_sha256'] == audit['source_rest_candidate']['sha256']
             and candidate['source_scene_snapshot']['sha256'] == audit['source_scene_snapshot']['sha256'],
             'TRUTH_REST_V2_OBSERVATION_LINEAGE_DRIFT')
    load_file_ref(cfg['policy'], expected_schema='RealSaS.TeacherKinematicCalibrationPolicy.v2')
    executable = Path(cfg['evaluator_binary']['path'])
    version = subprocess.check_output([str(executable), '--version'], text=True).splitlines()[0]
    _require(version == cfg['evaluator_version'], 'TRUTH_REST_V2_EVALUATOR_VERSION_DRIFT')
    root = ctx['run_root'] / 'artifacts' / ctx['stage']['id']
    root.mkdir(parents=True, exist_ok=True)
    replay = root / 'admitted_observation_import.json'
    maximum = float(cfg['maximum_seconds_per_subject'])
    if not replay.exists():
        _run(executable, ctx['repo_root'] / 'tools/observation/blender_source_parity_export_v1.py',
             [source_ref['path'], replay], root / 'observation.import.blender.log', maximum, ctx['run_manifest']['subject_id'])
    _require(replay.is_file() and sha256_file(replay) == admission['imported_geometry_sha256'],
             'TRUTH_REST_V2_ADMITTED_IMPORT_BYTES_DRIFT')
    job = {'schema': 'RealSaS.TeacherSceneCalibrationJob.v2', 'subject_key': ctx['run_manifest']['subject_id'],
        'source': source_ref, 'observation_admission': audit['iris_observation_admission'],
        'source_rest_candidate': audit['source_rest_candidate'], 'source_scene_snapshot': audit['source_scene_snapshot'],
        'observation_import_geometry': {'path': str(replay), 'sha256': sha256_file(replay)},
        'policy': cfg['policy'], 'matrix_tolerance': audit['matrix_tolerance'],
        'observation_bundle_sha256': normalization['observation_set_binding_hash'],
        'normalization_sha256': normalization['normalization_hash'],
        'evaluator_binary_sha256': cfg['evaluator_binary']['sha256'], 'producer_source_commit': cfg['producer_source_commit'],
        'training_allowed': False}
    job_hash = content_sha256(job)
    job_path = root / (job_hash + '.job.json')
    job_path.write_text(json.dumps(job, sort_keys=True, indent=2) + '\n')
    output = root / job_hash
    if not output.exists():
        _run(executable, ctx['repo_root'] / 'tools/truth_foundry/blender_teacher_scene_v2.py',
             ['--job', job_path, '--output', output], root / (job_hash + '.blender.log'), maximum, job['subject_key'])
    scene_path = output / 'TEACHER_SCENE.json'
    _require(scene_path.is_file(), 'TRUTH_REST_V2_UNSEALED_PARTIAL_OUTPUT__EXPLICIT_RECOVERY_REQUIRED')
    scene = json.loads(scene_path.read_text())
    _require(scene.get('schema') == 'RealSaS.TeacherSceneIR.v2' and scene.get('job_sha256') == job_hash
             and scene.get('authority') == 'UNQUALIFIED_DERIVED_MEASUREMENTS'
             and scene.get('qualification_minted') is False and scene.get('native_clip_refs') == [], 'TRUTH_REST_V2_SCENE_BINDING_DRIFT')
    binding = scene['observation_rest_binding']
    _require(binding['status'] == 'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY'
             and binding['imported_geometry_sha256'] == sha256_file(replay), 'TRUTH_REST_V2_REST_BINDING_MISSING')
    outputs = [_artifact(scene_path, 'RealSaS.TeacherSceneIR.v2'), _artifact(replay, 'RealSaS.ImportedObservationGeometryReplay.v1')]
    schemas = {'geometry_and_response.npz': 'RealSaS.TeacherCalibrationArrays.v2',
        'appearance.json': 'RealSaS.SourceAppearanceSamplerInputs.v1', 'image_pixels.npz': 'RealSaS.SourceImagePixels.v1',
        'source_scene.blend': 'RealSaS.PackedSourceEvaluatorScene.v2'}
    for ref in scene['geometry_payloads'] + scene['appearance_payloads']:
        path = load_file_ref(ref, json_required=False)
        _require(path.parent == output.resolve(), 'TRUTH_REST_V2_OUTPUT_PATH_ESCAPE')
        outputs.append(_artifact(path, schemas[path.name]))
    return {'status': 'PASS', 'outputs': outputs, 'diagnostics': {
        'teacher_scene_sha256': sha256_file(scene_path), 'action_count': len(scene['calibration']['actions']),
        'observation_rest_binding': binding, 'qualification_minted': False, 'training_executed': False,
        'native_clip_rest_compatibility_claimed': False}}
