#!/usr/bin/env python3
"""Inspect recovered source calibration arrays without admitting student truth.

Payload resolution is by the Registry-exported content hash, never by the old
host path in a scene. This audit does not select an action law, derivative scale,
rank threshold, timebase, observation witness or physical interpretation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256
from models.iris.truth.response_operator_v1 import central_kinematic_derivative


def require(value,code):
    if not value:raise ValueError(code)


def payload(ref,available):
    path=Path(available[ref['sha256']])
    require(path.is_file() and sha256(path)==ref['sha256'] and path.stat().st_size==ref['size_bytes'],
            'TEACHER_AUDIT_PAYLOAD_BYTES_DRIFT')
    return path


def inspect(scene,available):
    require(scene['schema']=='RealSaS.TeacherSceneIR.v1' and scene['authority']=='UNQUALIFIED_DERIVED_MEASUREMENTS'
            and scene['qualification_minted'] is False and scene['student_input_allowed'] is False,
            'TEACHER_AUDIT_AUTHORITY_DRIFT')
    require(len(scene['geometry_payloads'])==1,'TEACHER_AUDIT_GEOMETRY_PAYLOAD_COUNT')
    ref=scene['geometry_payloads'][0];path=payload(ref,available)
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    require(set(arrays)==set(ref['arrays']),'TEACHER_AUDIT_ARRAY_SET_DRIFT')
    for name,array in arrays.items():
        desc=ref['arrays'][name]
        require(array.dtype.str==desc['dtype'] and list(array.shape)==desc['shape'],
                'TEACHER_AUDIT_ARRAY_LAYOUT_DRIFT:'+name)
        require(array.dtype.kind in 'biuf','TEACHER_AUDIT_NONNUMERIC_ARRAY:'+name)
    rest=arrays['rest_vertices_source_world'];faces=arrays['faces']
    require(rest.ndim==2 and rest.shape[1]==3 and len(rest)>0 and np.isfinite(rest).all(),'TEACHER_AUDIT_REST_LAYOUT')
    require(faces.ndim==2 and faces.shape[1]==3 and faces.dtype.kind in 'iu' and len(faces)>0
            and faces.min()>=0 and faces.max()<len(rest),'TEACHER_AUDIT_FACE_LAYOUT')
    n=len(rest);f=len(faces);controls=scene['privileged_control_refs'];actions=scene['calibration']['actions']
    require(len(controls)>0 and len({c['opaque_privileged_handle'] for c in controls})==len(controls),'TEACHER_AUDIT_CONTROL_IDENTITY')
    require(arrays['isolated_position'].shape==(len(actions),n,3),'TEACHER_AUDIT_ACTION_LAYOUT')
    for name in ('pose_identity_baseline','sham_before','sham_after','root_control_position'):
        require(arrays[name].shape==rest.shape and np.isfinite(arrays[name]).all(),'TEACHER_AUDIT_POSE_LAYOUT:'+name)
    require(np.isfinite(arrays['isolated_position']).all(),'TEACHER_AUDIT_NONFINITE_ACTION')
    require(arrays['triangle_uv'].shape==(f,3,2) and arrays['triangle_uv_valid'].shape==(f,)
            and arrays['triangle_uv_valid'].dtype==np.bool_,'TEACHER_AUDIT_UV_LAYOUT')
    require(np.isfinite(arrays['triangle_uv'][arrays['triangle_uv_valid']]).all(),'TEACHER_AUDIT_VALID_UV_NONFINITE')
    for name in ('face_owner_privileged','face_material_privileged'):
        require(arrays[name].shape==(f,) and arrays[name].dtype.kind in 'iu','TEACHER_AUDIT_FACE_METADATA_LAYOUT')
    offsets=arrays['source_weight_offsets'];indices=arrays['source_weight_bone_indices_privileged'];weights=arrays['source_weight_values']
    bone_count=len(scene['calibration']['native_control_frames_privileged'])
    require(offsets.shape==(n+1,) and offsets.dtype.kind in 'iu' and offsets[0]==0 and offsets[-1]==len(weights)
            and np.all(np.diff(offsets)>=0) and indices.shape==weights.shape and indices.dtype.kind in 'iu'
            and (not len(indices) or (indices.min()>=0 and indices.max()<bone_count))
            and np.isfinite(weights).all() and np.all(weights>0),'TEACHER_AUDIT_SOURCE_WEIGHT_CSR')
    require(arrays['source_weight_coverage'].shape==(n,) and arrays['source_weight_coverage'].dtype==np.bool_
            and np.array_equal(arrays['source_weight_coverage'],np.diff(offsets)>0),'TEACHER_AUDIT_SOURCE_WEIGHT_COVERAGE')
    action_index={};amplitudes={}
    for index,action in enumerate(actions):
        control=action['control_index_privileged'];amplitude=action['amplitude'];ai=action['amplitude_index'];sign=action['sign']
        require(type(control) is int and 0<=control<len(controls) and sign in (-1,1) and type(ai) is int and ai>=0
                and np.isfinite(amplitude) and amplitude>0 and action['units']=='RADIAN'
                and action['kind']=='ISOLATED_KINEMATIC_PARAMETER_PERTURBATION'
                and action['opaque_privileged_handle']==controls[control]['opaque_privileged_handle'],
                'TEACHER_AUDIT_ACTION_IDENTITY')
        key=(control,ai,sign);require(key not in action_index,'TEACHER_AUDIT_DUPLICATE_ACTION')
        require(ai not in amplitudes or amplitudes[ai]==amplitude,'TEACHER_AUDIT_AMPLITUDE_DRIFT')
        action_index[key]=index;amplitudes[ai]=amplitude
    require(set(action_index)=={(c,a,s) for c in range(len(controls)) for a in amplitudes for s in (-1,1)},
            'TEACHER_AUDIT_MISSING_CENTRAL_ACTION_PAIR')
    baseline=arrays['pose_identity_baseline'];derivatives={};per_amplitude=[]
    diagonal=float(np.linalg.norm(np.ptp(rest,axis=0)))
    require(diagonal>0,'TEACHER_AUDIT_DEGENERATE_SOURCE_EXTENT')
    for ai,h in sorted(amplitudes.items(),key=lambda row:row[1]):
        positive=np.stack([arrays['isolated_position'][action_index[c,ai,1]] for c in range(len(controls))],axis=1)
        negative=np.stack([arrays['isolated_position'][action_index[c,ai,-1]] for c in range(len(controls))],axis=1)
        j,_=central_kinematic_derivative(positive,negative,np.full(len(controls),h),np.ones((n,len(controls)),bool))
        derivatives[ai]=j
        response=np.linalg.norm(positive-negative,axis=-1)/(2*diagonal)
        per_amplitude.append({'amplitude_index':ai,'radians':h,
            'symmetric_response_over_source_diagonal_p50_p95_max':np.quantile(response,[.5,.95,1]).tolist(),
            'control_max_symmetric_response_over_source_diagonal':response.max(axis=0).tolist()})
    derivative_agreement=[]
    ordered=sorted(amplitudes,key=amplitudes.get)
    for first,second in zip(ordered,ordered[1:]):
        delta=np.linalg.norm(derivatives[first]-derivatives[second],axis=1)/diagonal
        derivative_agreement.append({'amplitude_indices':[first,second],
            'derivative_difference_over_source_diagonal_per_radian_p50_p95_max':np.quantile(delta,[.5,.95,1]).tolist()})
    translation=np.asarray(scene['calibration']['root_translation_source_units'],float)
    require(translation.shape==(3,) and np.isfinite(translation).all(),'TEACHER_AUDIT_ROOT_CONTROL_LAYOUT')
    measured={'source_rest_to_pose_identity_max_abs':float(np.max(np.abs(rest-baseline))),
              'sham_repeat_max_abs':float(np.max(np.abs(arrays['sham_after']-arrays['sham_before']))),
              'common_root_translation_max_abs':float(np.max(np.abs(arrays['root_control_position']-(baseline+translation))))}
    require(all(np.isclose(value,scene['calibration'][name],rtol=0,atol=1e-14) for name,value in measured.items()),
            'TEACHER_AUDIT_REPORTED_DIAGNOSTICS_DRIFT')
    return {'subject_key':scene['subject_key'],'source_sha256':scene['source_sha256'],'geometry_sha256':ref['sha256'],
        'vertices':n,'faces':f,'privileged_native_controls':len(controls),'isolated_actions':len(actions),
        'source_diagonal':diagonal,'reset_and_gauge_raw_source_units':measured,
        'valid_uv_faces':int(arrays['triangle_uv_valid'].sum()),
        'source_weight_covered_vertices':int(arrays['source_weight_coverage'].sum()),
        'calibration_ladder':per_amplitude,'adjacent_derivative_agreement':derivative_agreement,
        'student_observability':'UNQUALIFIED__PRIVILEGED_SOURCE_MEASUREMENTS_ONLY',
        'source_weight_lbs_equivalence':'NOT_QUALIFIED','action_law_selected':False,
        'finite_difference_amplitude_selected':False,'rank_threshold_selected':False,
        'dynamic_timebase_inferred':False,'physical_constraints_inferred':False,'truth_qualification_minted':False}


def inspect_export(observation):
    require(observation['schema']=='RealSaS.TeacherCalibrationObservation.v1','TEACHER_AUDIT_OBSERVATION_SCHEMA')
    results=[]
    for member in observation['exports']:
        if member['role']!='COHORT':continue
        outputs=member['outputs'];available={r['sha256']:r['path'] for r in outputs}
        scenes=[r for r in outputs if r['schema']=='RealSaS.TeacherSceneIR.v1']
        require(len(scenes)==1,'TEACHER_AUDIT_EXACT_SCENE_REQUIRED')
        scene_ref=scenes[0];scene_path=payload(scene_ref,available)
        result=inspect(json.loads(scene_path.read_text()),available)
        results.append({'attempt_id':member['attempt_id'],'registry_stage_artifact_id':member['registry_stage_artifact_id'],**result})
    return {'schema':'RealSaS.TeacherSceneNumericAudit.v1','campaign_id':observation['campaign_id'],
            'experiment_id':observation['experiment_id'],'subjects':results,'truth_qualification_minted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=inspect_export(json.loads(Path(a.observation).read_text()))
    Path(a.output).write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'subjects_inspected':len(result['subjects']),'truth_qualification_minted':False}),flush=True)
