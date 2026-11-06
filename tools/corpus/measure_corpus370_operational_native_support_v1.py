#!/usr/bin/env python3
"""Real V13b original native-actuator FIT support, not drawable owner truth.

Every original anonymous control-axis is retained, including empty/global actions.
FIT poses 0/1 generate selector evidence; heldout pose2 is not accessed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.corpus.compile_corpus370_operational_edit_algebra_v1 import INPUT_SCHEMA, canonical_bytes, digest

SCHEMA='RealSaS.Corpus370OperationalNativeSupportReport.v1'
POLICY='ALL_NATIVE_CONTROL_AXES__FIT_POSES_0_1_AGREE__MEAN_VERTEX_DISPLACEMENT_MAX_SIGN_2E-4'
FLOOR=2e-4

def require(v,why):
    if not v: raise ValueError(why)

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def face_sha(faces):
    return hashlib.sha256(np.ascontiguousarray(faces,dtype='<i8').tobytes()).hexdigest()

def verify_original(a,az,m,mz,mc,paths):
    require(a.get('schema')=='RealSaS.Corpus370NativeReactiveStainCandidate.v18a',
        'OD_SOURCE_V18A_SCHEMA')
    require(m.get('schema')=='RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b'
        and mc.get('schema')=='RealSaS.Corpus370AnonymousNativeMixedCarrierResponseCourt.v13b',
        'OD_SOURCE_V13B_SCHEMA')
    require(a.get('private_candidate_npz_sha256')==sha(paths['v18a_npz'])
        and m.get('source_pose_articulation_array_sha256')==sha(paths['v13b_npz'])
        and mc.get('parents',{}).get('native_anonymous_intervention_manifest_sha256')==sha(paths['v13b_manifest'])
        and mc.get('parents',{}).get('native_anonymous_intervention_vertex_array_sha256')==sha(paths['v13b_npz']),
        'OD_SOURCE_ORIGINAL_BYTES_NOT_BOUND')
    require(a.get('parent_source_immutable_sha256',{}).get('v13b_source_intervention')==sha(paths['v13b_npz'])
        and a.get('parent_source_immutable_sha256',{}).get('source_rest_geometry')==m.get('rest_source_sha256'),
        'OD_SOURCE_ORIGINAL_REST_JOIN_INVALID')
    require(a.get('source_sha256')==m.get('source_sha256')==mc.get('source_sha256')
        and a.get('subject_id')==m.get('subject_id')==mc.get('subject_id'),
        'OD_SOURCE_SUBJECT_MISMATCH')
    require(a.get('fit_source_authored_poses')==[0,1]
        and a.get('heldout_source_authored_pose')==2
        and m.get('original_source_teacher_pose_count')==3
        and mc.get('graphics_owner_authorized') is False
        and a.get('iris_training_authorized') is False,
        'OD_SOURCE_FIT_HELDOUT_OR_AUTHORITY_INVALID')
    faces=np.asarray(az['faces'],dtype=np.int64)
    base=np.asarray(mz['baseline_vertices'],dtype=np.float64)
    ctrl=np.asarray(mz['native_control_perturbed_vertices'],dtype=np.float64)
    require(faces.ndim==2 and faces.shape[1]==3 and len(faces)>2
        and faces.min()>=0 and base.ndim==3 and base.shape[0]==3
        and base.shape[2]==3 and faces.max()<base.shape[1]
        and ctrl.ndim==6 and ctrl.shape[0]==3
        and ctrl.shape[2:4]==(3,2)
        and ctrl.shape[4:]==(base.shape[1],3)
        and np.isfinite(base).all() and np.isfinite(ctrl).all(),
        'OD_SOURCE_NATIVE_ARRAY_SHAPE')
    require(len(faces)==a.get('source_face_count')==m.get('source_face_count'),
        'OD_SOURCE_FACE_CENSUS')
    return faces,base,ctrl

def supports_for_pose(faces,base,ctrl,pose,extent):
    delta=ctrl[pose]-base[pose][None,None,None,:,:]
    mags=np.linalg.norm(delta,axis=-1)
    per_face=mags[:,:,:,faces].mean(axis=-1)
    return np.max(per_face,axis=2)/extent

def compile_native_fit(faces,base,ctrl,a,m,paths):
    extent=float(np.linalg.norm(np.ptp(base[0],axis=0)))
    require(np.isfinite(extent) and extent>1e-9,'OD_SOURCE_EXTENT_INVALID')
    f0=supports_for_pose(faces,base,ctrl,0,extent)
    f1=supports_for_pose(faces,base,ctrl,1,extent)
    require(f0.shape==f1.shape==(ctrl.shape[1],3,len(faces)),
        'OD_SOURCE_FIT_MEASUREMENT_SHAPE')
    operations=[]
    for ci in range(ctrl.shape[1]):
        for ax in range(3):
            first,second=f0[ci,ax]>FLOOR,f1[ci,ax]>FLOOR
            valid=first==second
            ident=f'ANON_NATIVE_{ci:03d}_AXIS_{ax}'
            opsha=digest({'policy':POLICY,'source_sha256':a['source_sha256'],
                'source_intervention_sha256':sha(paths['v13b_npz']),
                'operation_id':ident,'fit_pose0_support':first.tolist(),
                'fit_pose1_support':second.tolist()})
            operations.append({'operation_id':ident,
                'support':(first & second).tolist(),
                'validity':valid.tolist(),'evidence_sha256':opsha})
    evidence={'schema':INPUT_SCHEMA,
        'source_original_sha256':a['source_sha256'],
        'source_rest_sha256':m['rest_source_sha256'],
        'source_face_order_sha256':face_sha(faces),
        'source_face_count':len(faces),'workload_policy':POLICY,
        'operations':operations}
    report={'schema':SCHEMA,'source_subject_id_private':a['subject_id'],
        'original_source_sha256':a['source_sha256'],
        'source_face_count':len(faces),'native_control_count':int(ctrl.shape[1]),
        'task_count':len(operations),'source_extent_fit_pose_0':extent,
        'frozen_normalized_floor':FLOOR,'fit_pose_ordinals':[0,1],
        'heldout_pose_ordinal':2,'heldout_used_in_fit':False,
        'parent_sha256':{k:sha(p) for k,p in paths.items()},
        'evidence_sha256':digest(evidence),
        'source_shader_measurement':'NOT_PERFORMED',
        'palette_task_measurement':'NOT_PERFORMED',
        'source_editor_edit_court':'NOT_EVALUATED',
        'graphics_owner_authorized':False,'iris_training_authorized':False,
        'tessa_training_authorized':False,'materialization_authorized':False,
        'training_authorized':False}
    return evidence,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v18a_manifest','v18a_npz','v13b_manifest','v13b_npz','v13b_court'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    x=p.parse_args()
    paths={k:getattr(x,k) for k in
           ('v18a_manifest','v18a_npz','v13b_manifest','v13b_npz','v13b_court')}
    a,m,mc=(read(paths[k]) for k in ('v18a_manifest','v13b_manifest','v13b_court'))
    with np.load(paths['v18a_npz'],allow_pickle=False) as az, \
         np.load(paths['v13b_npz'],allow_pickle=False) as mz:
        faces,base,ctrl=verify_original(a,az,m,mz,mc,paths)
        evidence,report=compile_native_fit(faces,base,ctrl,a,m,paths)
    for path,obj in ((x.evidence,evidence),(x.report,report)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(canonical_bytes(obj))
    print(json.dumps({'status':'ORIGINAL_FIT_SOURCE_NATIVE_WORKLOAD__OWNER_UNKNOWN',
        'source_face_count':len(faces),'task_count':len(evidence['operations']),
        'fit_unknown_masks_total':sum(len(t['validity'])-sum(t['validity'])
            for t in evidence['operations']),'training_authorized':False},
        sort_keys=True))

if __name__=='__main__': main()
