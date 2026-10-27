"""Platform Engine producer for unqualified geometric teacher calibration.

Stage success means bounded measurement execution and immutable byte output.
It is not TruthQualificationReport admission or a learned/product PASS.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from .adapter_io import load_file_ref, sha256_file, stage_output_payload, write_json

if TYPE_CHECKING:
    # Pin the external Blender program in the static release source closure.
    from tools.truth_foundry.blender_teacher_scene_v1 import build as _blender_source_entry


def _require(condition,code):
    if not condition:raise QualificationError(code)


def _artifact(path,schema):
    return {'path':str(path),'sha256':sha256_file(path),'authority_class':'UNQUALIFIED_DERIVED_MEASUREMENTS','schema':schema}


def measure_teacher_scene_stage(ctx):
    cfg=dict(ctx['run_manifest'].get('truth_foundry') or {})
    _require(ctx['stage']['policy'].get('product_pass_authority') is False,'TRUTH_CALIBRATION_CANNOT_OWN_PRODUCT_PASS')
    root=ctx['run_root']/'artifacts'/ctx['stage']['id'];root.mkdir(parents=True,exist_ok=True)
    mode=cfg.get('mode')
    if mode=='SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS':
        _require(cfg.get('new_truth_claimed') is False,'TRUTH_REGRESSION_EXCLUSION_NOT_EXPLICIT')
        payload={'schema':'RealSaS.TruthCalibrationRegressionExclusion.v1','mode':mode,
                 'reason':'SEPARATE_FROZEN_REGRESSION_MEMBER__NOT_A_FIT_SUBJECT','new_truth_produced':False,
                 'qualification_minted':False}
        return {'status':'PASS','outputs':[write_json(root/'regression_exclusion.json',payload,
                authority_class='EXPLICIT_CORPUS_EXCLUSION_ONLY',schema=payload['schema'])],
                'diagnostics':{'new_truth_produced':False,'corpus_member':False}}
    _require(mode=='SOURCE_KINEMATIC_CALIBRATION_ONLY','TRUTH_CALIBRATION_MODE_REQUIRED')
    source=stage_output_payload(ctx,'01_SOURCE_BYTES_SEALED','RealSaS.SourceByteSeal.v1')
    stage_output_payload(ctx,'03_SOURCE_MECHANICAL_AUDIT','RealSaS.SourceMechanicalAuditIR.v1')
    normalization=stage_output_payload(ctx,'08_NORMALIZATION_DOMAIN_QUALIFIED','RealSaS.NormalizationDomainIR.v1')
    baseline_index=load_file_ref(cfg['native_baselines'],expected_schema='RealSaS.NativeTeacherBaselineBindings.v1')
    key=ctx['run_manifest']['subject_id'];rows=[r for r in baseline_index['subjects'] if r['subject_key']==key]
    _require(len(rows)==1,'TRUTH_CALIBRATION_EXACT_BASELINE_ROW_REQUIRED');baseline=rows[0]
    audit=ctx['run_manifest']['source_audit']
    source_rows=[r for r in ctx['run_manifest']['source_files'] if r.get('source_kind')=='PRIMARY']
    _require(len(source_rows)==1,'TRUTH_CALIBRATION_PRIMARY_SOURCE_REQUIRED')
    source_ref={'path':source_rows[0]['path'],'sha256':source_rows[0]['expected_sha256']}
    _require(source_ref['sha256'] in {r['sha256'] for r in source['files']},'TRUTH_CALIBRATION_SOURCE_NOT_QUALIFIED_PARENT')
    for ref in (source_ref,baseline['native_teacher'],audit['iris_observation_admission'],audit['source_scene_snapshot'],cfg['evaluator_binary']):
        load_file_ref(ref,json_required=False)
    load_file_ref(cfg['policy'],expected_schema='RealSaS.TeacherKinematicCalibrationPolicy.v1')
    report=load_file_ref(cfg['source_equivalence_report'],expected_schema='RealSaS.NativeTeacherReplayReport.v1')
    _require(report['status']=='EXACT_SEALED_REPLAY','TRUTH_CALIBRATION_NATIVE_REPLAY_UNPROVEN')
    executable=Path(cfg['evaluator_binary']['path'])
    version=subprocess.check_output([str(executable),'--version'],text=True).splitlines()[0]
    _require(version==cfg['evaluator_version'],'TRUTH_CALIBRATION_EVALUATOR_VERSION_DRIFT')
    job={'schema':'RealSaS.TeacherSceneCalibrationJob.v1','subject_key':key,'source':source_ref,
         'observation_admission':audit['iris_observation_admission'],'source_scene_snapshot':audit['source_scene_snapshot'],
         'native_teacher':baseline['native_teacher'],'source_equivalence_report':cfg['source_equivalence_report'],
         'observation_bundle_sha256':normalization['observation_set_binding_hash'],'policy':cfg['policy'],
         'normalization_sha256':normalization['normalization_hash'],
         'producer_source_commit':cfg['producer_source_commit'],'evaluator_binary_sha256':cfg['evaluator_binary']['sha256'],
         'training_allowed':False}
    job_hash=content_sha256(job);job_path=root/(job_hash+'.job.json');job_path.write_text(json.dumps(job,sort_keys=True,indent=2)+'\n')
    output=root/job_hash;scene_path=output/'TEACHER_SCENE.json'
    if not output.exists():
        program=ctx['repo_root']/'tools/truth_foundry/blender_teacher_scene_v1.py'
        log_path=root/(job_hash+'.blender.log')
        with log_path.open('w') as log:
            process=subprocess.Popen([str(executable),'-b','--factory-startup','--python',str(program),'--',
                    '--job',str(job_path),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic()
            try:
                while True:
                    try:code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        elapsed=time.monotonic()-started
                        print(f'[TRUTH_CALIBRATION_HEARTBEAT] subject={key} elapsed_s={elapsed:.1f}',flush=True)
                        _require(elapsed<float(cfg['maximum_seconds_per_subject']),'TRUTH_CALIBRATION_TIME_BUDGET_EXCEEDED')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        _require(code==0,'TRUTH_CALIBRATION_BLENDER_FAILED__INSPECT_BOUND_LOG:'+str(log_path))
    _require(scene_path.is_file(),'TRUTH_CALIBRATION_UNSEALED_PARTIAL_OUTPUT__EXPLICIT_RECOVERY_REQUIRED')
    scene=json.loads(scene_path.read_text())
    _require(scene.get('job_sha256')==job_hash and scene.get('authority')=='UNQUALIFIED_DERIVED_MEASUREMENTS'
             and scene.get('qualification_minted') is False,'TRUTH_CALIBRATION_SCENE_BINDING_DRIFT')
    outputs=[_artifact(scene_path,'RealSaS.TeacherSceneIR.v1')]
    schema={'geometry_and_response.npz':'RealSaS.TeacherCalibrationArrays.v1','appearance.json':'RealSaS.SourceAppearanceSamplerInputs.v1',
            'image_pixels.npz':'RealSaS.SourceImagePixels.v1','source_scene.blend':'RealSaS.PackedSourceEvaluatorScene.v1'}
    for ref in scene['geometry_payloads']+scene['appearance_payloads']:
        path=load_file_ref(ref,json_required=False)
        _require(path.parent==output.resolve(),'TRUTH_CALIBRATION_OUTPUT_PATH_ESCAPE')
        outputs.append(_artifact(path,schema[path.name]))
    return {'status':'PASS','outputs':outputs,'diagnostics':{'teacher_scene_sha256':sha256_file(scene_path),
            'action_count':len(scene['calibration']['actions']),'qualification_minted':False,'training_executed':False,
            'source_rest_to_pose_identity_max_abs':scene['calibration']['source_rest_to_pose_identity_max_abs'],
            'sham_repeat_max_abs':scene['calibration']['sham_repeat_max_abs'],
            'common_root_translation_max_abs':scene['calibration']['common_root_translation_max_abs']}}
