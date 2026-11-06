#!/usr/bin/env python3
"""Independent original-source V13b support court (NO producer import).

Recompute FIT action supports with independent face-distance implementation;
heldout pose2 is evaluated only after the precommitted edit atoms are sealed.
This is NOT an artist owner, renderer, or training qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.corpus.court_corpus370_operational_edit_algebra_v1 import court as algebra_court

SCHEMA='RealSaS.Corpus370IndependentOperationalNativeSupportCourt.v1'
FLOOR=2e-4
POLICY='ALL_NATIVE_CONTROL_AXES__FIT_POSES_0_1_AGREE__MEAN_VERTEX_DISPLACEMENT_MAX_SIGN_2E-4'
NONPROMOTE=('graphics_owner_authorized','iris_training_authorized',
    'tessa_training_authorized','materialization_authorized','training_authorized')

def require(ok,why):
    if not ok: raise ValueError(why)

def sha_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def sha_json(obj):
    data=(json.dumps(obj,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def eval_pose(faces,baseline,controls,pose,extent):
    result=np.empty((controls.shape[1],3,len(faces)),dtype=bool)
    for j in range(controls.shape[1]):
        for axis in range(3):
            signed=[]
            for sign in range(2):
                delta=controls[pose,j,axis,sign]-baseline[pose]
                lengths=np.sqrt(np.sum(delta*delta,axis=-1))
                signed.append(np.sum(lengths[faces],axis=1)/3.)
            result[j,axis]=np.maximum(signed[0],signed[1])/extent>FLOOR
    return result

def baseline_mismatch(labels,tasks):
    groups={}
    for i,j in enumerate(labels):
        groups.setdefault(int(j),[]).append(i)
    conflicts=set()
    group_task_count=0
    minimum_face_mismatches=0
    for task in tasks:
        s,v=task['support'],task['validity']
        for group,faces in groups.items():
            yes=sum(1 for f in faces if v[f] and s[f])
            no=sum(1 for f in faces if v[f] and not s[f])
            if yes and no:
                conflicts.add(group)
                group_task_count+=1
                minimum_face_mismatches+=min(yes,no)
    return {'K':len(groups),'groups_with_known_task_conflict':len(conflicts),
        'group_task_conflict_count':group_task_count,
        'sum_minimum_observed_face_mismatches_per_task':minimum_face_mismatches}

def independent_real_court(a,m,mc,az,mz,paths,evidence,decision,report):
    require(a.get('schema')=='RealSaS.Corpus370NativeReactiveStainCandidate.v18a'
        and m.get('schema')=='RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b'
        and mc.get('schema')=='RealSaS.Corpus370AnonymousNativeMixedCarrierResponseCourt.v13b',
        'OD_REAL_SOURCE_SCHEMAS')
    require(a.get('private_candidate_npz_sha256')==sha_file(paths['v18a_npz'])
        and m.get('source_pose_articulation_array_sha256')==sha_file(paths['v13b_npz'])
        and mc.get('parents',{}).get('native_anonymous_intervention_vertex_array_sha256')==sha_file(paths['v13b_npz'])
        and mc.get('parents',{}).get('native_anonymous_intervention_manifest_sha256')==sha_file(paths['v13b_manifest']),
        'OD_REAL_SHA_CHAIN')
    require(a.get('parent_source_immutable_sha256',{}).get('v13b_source_intervention')==sha_file(paths['v13b_npz'])
        and a.get('parent_source_immutable_sha256',{}).get('source_rest_geometry')==m.get('rest_source_sha256')
        and a.get('source_sha256')==m.get('source_sha256')==mc.get('source_sha256'),
        'OD_REAL_SOURCE_IDENTITY')
    faces=np.asarray(az['faces'],dtype=np.int64)
    baseline=np.asarray(mz['baseline_vertices'],dtype=np.float64)
    ctrl=np.asarray(mz['native_control_perturbed_vertices'],dtype=np.float64)
    require(faces.ndim==2 and faces.shape[1]==3 and faces.min()>=0
        and len(faces)==a['source_face_count']==m['source_face_count']
        and baseline.ndim==3 and baseline.shape[0]==3 and baseline.shape[2]==3
        and ctrl.ndim==6 and ctrl.shape==(3,ctrl.shape[1],3,2,baseline.shape[1],3)
        and faces.max()<baseline.shape[1]
        and np.isfinite(baseline).all() and np.isfinite(ctrl).all(),
        'OD_REAL_NATIVE_ARRAY_DOMAIN')
    extent=float(np.sqrt(np.sum(np.ptp(baseline[0],axis=0)**2)))
    require(extent>1e-9 and np.isfinite(extent),'OD_REAL_SOURCE_EXTENT')
    arr0,arr1,arr2=(eval_pose(faces,baseline,ctrl,k,extent) for k in range(3))
    require(len(evidence['operations'])==ctrl.shape[1]*3
        and evidence['source_face_count']==len(faces)
        and evidence['source_original_sha256']==a['source_sha256']
        and evidence['source_rest_sha256']==m['rest_source_sha256']
        and evidence['workload_policy']==POLICY
        and report['evidence_sha256']==sha_json(evidence)
        and report['native_control_count']==ctrl.shape[1]
        and report['source_face_count']==len(faces)
        and report['task_count']==len(evidence['operations'])
        and np.isclose(report['source_extent_fit_pose_0'],extent,atol=1e-12,rtol=1e-13)
        and report['frozen_normalized_floor']==FLOOR
        and report['fit_pose_ordinals']==[0,1]
        and report['heldout_pose_ordinal']==2
        and report['heldout_used_in_fit'] is False
        and report['source_shader_measurement']=='NOT_PERFORMED'
        and report['palette_task_measurement']=='NOT_PERFORMED'
        and report['source_editor_edit_court']=='NOT_EVALUATED',
        'OD_REAL_SOURCE_RESEARCH_MANIFEST')
    require(all(report['parent_sha256'][key]==sha_file(paths[key])
        for key in ('v18a_manifest','v18a_npz','v13b_manifest','v13b_npz','v13b_court'))
        and all(report.get(k) is False for k in NONPROMOTE),
        'OD_REAL_SOURCE_RECEIPT_PARENTS')
    stability=[]
    for j in range(ctrl.shape[1]):
        for axis in range(3):
            t=evidence['operations'][j*3+axis]
            ident=f'ANON_NATIVE_{j:03d}_AXIS_{axis}'
            require(t['operation_id']==ident,'OD_REAL_NATIVE_ACTION_ADDRESS')
            s0,s1,s2=arr0[j,axis],arr1[j,axis],arr2[j,axis]
            valid=s0==s1
            support=s0&s1
            require(t['support']==support.tolist()
                and t['validity']==valid.tolist(),
                'OD_REAL_FIT_NATIVE_ACTUATOR_WRONG_SUPPORT')
            op_sha=sha_json({'policy':POLICY,'source_sha256':a['source_sha256'],
                'source_intervention_sha256':sha_file(paths['v13b_npz']),
                'operation_id':ident,'fit_pose0_support':s0.tolist(),
                'fit_pose1_support':s1.tolist()})
            require(t['evidence_sha256']==op_sha,
                'OD_REAL_OPERATION_SIGNED_EVIDENCE_DRIFT')
            known=int(np.count_nonzero(valid))
            same=int(np.count_nonzero(valid&(support==s2)))
            stability.append({'operation_id':ident,
                'fit_qualified_faces':known,
                'fit_disagreed_faces':len(faces)-known,
                'heldout_agreed_known_faces':same,
                'heldout_disagreed_known_faces':known-same,
                'fit_affected_face_count':int(np.count_nonzero(support&valid)),
                'heldout_affected_face_count':int(np.count_nonzero(s2))})
    algebra=algebra_court(evidence,decision)
    require(algebra['status']=='PASS_ALGEBRA_ONLY__NO_REAL_SOURCE_OWNER_AUTHORITY',
        'OD_REAL_ALGEBRA_DID_NOT_PASS')
    alternatives={'OPERATIONAL_FULL_WORKLOAD':
        baseline_mismatch(decision['face_atom_index'],evidence['operations'])}
    for threshold in ('0.045','0.1','0.18'):
        key='labels_'+threshold.replace('.','p')
        require(key in az.files,'OD_REAL_ORIGINAL_V18A_BASELINE_MISSING')
        labels=np.asarray(az[key],dtype=np.int64)
        require(labels.shape==(len(faces),) and labels.min()>=0,
            'OD_REAL_BASELINE_LABEL_SHAPE')
        alternatives['V18A_THRESHOLD_'+threshold]=baseline_mismatch(labels,evidence['operations'])
    return {'schema':SCHEMA,'original_source_sha256':a['source_sha256'],
        'subject_id_private':a['subject_id'],
        'parents_sha256':{k:sha_file(x) for k,x in paths.items()},
        'fit_evidence_sha256':sha_json(evidence),
        'edit_algebra_decision_sha256':sha_file(paths['decision']),
        'original_fit_support_report_sha256':sha_file(paths['source_report']),
        'source_face_count':len(faces),'native_control_count':int(ctrl.shape[1]),
        'task_count':len(evidence['operations']),
        'face_task_fit_validity_count':sum(x['fit_qualified_faces'] for x in stability),
        'face_task_fit_unknown_count':sum(x['fit_disagreed_faces'] for x in stability),
        'face_task_heldout_disagreement_count':sum(x['heldout_disagreed_known_faces'] for x in stability),
        'operation_stability':stability,
        'partition_comparisons':alternatives,'algebra_court':algebra,
        'native_source_renderer_verified':False,
        'source_palette_shaders_verified':False,
        'source_editor_edits_verified':False,
        'designer_drawable_ground_truth':'UNKNOWN',
        'synthetic_or_source_native_artist_owner_minted':False,
        'student_observation_equivalence_proven':False,
        'graphics_owner_authorized':False,
        'iris_training_authorized':False,
        'tessa_training_authorized':False,
        'materialization_authorized':False,
        'training_authorized':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v18a_manifest','v18a_npz','v13b_manifest','v13b_npz',
              'v13b_court','evidence','decision','source_report','output'):
        p.add_argument('--'+k.replace('_','-'),required=True,type=Path)
    x=p.parse_args()
    keys=('v18a_manifest','v18a_npz','v13b_manifest','v13b_npz',
          'v13b_court','decision','source_report')
    paths={k:getattr(x,k) for k in keys}
    a,m,mc,ev,decision,report=(json.loads(getattr(x,k).read_text(encoding='utf-8'))
        for k in ('v18a_manifest','v13b_manifest','v13b_court',
                  'evidence','decision','source_report'))
    with np.load(x.v18a_npz,allow_pickle=False) as az, \
         np.load(x.v13b_npz,allow_pickle=False) as mz:
        outcome=independent_real_court(a,m,mc,az,mz,paths,ev,decision,report)
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(outcome,sort_keys=True,indent=2,allow_nan=False)+'\n',
        encoding='utf-8')
    print(json.dumps({'status':outcome['schema'],
        'face_count':outcome['source_face_count'],
        'qualified_face_tasks':outcome['face_task_fit_validity_count'],
        'unknown_face_tasks':outcome['face_task_fit_unknown_count'],
        'heldout_changed_face_tasks':outcome['face_task_heldout_disagreement_count'],
        'candidate_K':outcome['partition_comparisons']['OPERATIONAL_FULL_WORKLOAD']['K'],
        'training_authorized':False},sort_keys=True))

if __name__=='__main__': main()
