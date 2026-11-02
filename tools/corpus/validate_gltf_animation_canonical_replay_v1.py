#!/usr/bin/env python3
from __future__ import annotations

"""Independent replay court for GltfAnimationCanonicalReplay.v1.

The court re-decodes immutable source bytes through SourceGLTF and uses SciPy's
Rotation/Slerp for quaternion interpolation rather than the producer's quaternion
implementation. It reconstructs deterministic sample times, evaluates all joint
TRS channels through the source node hierarchy, and compares quantized replay
digests and imported-joint binding.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tools.corpus.gltf_bind_scene_v1 import SourceGLTF

PARENT_SCHEMA="RealSaS.GltfAnimationCanonicalReplay.v1"
PARENT_STATUS="PASS_GLTF_ANIMATION_CANONICAL_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.GltfAnimationCanonicalReplayCourt.v1"
STATUS="PASS_GLTF_ANIMATION_CANONICAL_REPLAY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8<<20),b''):h.update(c)
    return h.hexdigest()
def bound(row,code):
    row=dict(row or {});p=Path(str(row.get('path') or '')).resolve()
    if not p.is_file() or sha256(p)!=str(row.get('sha256') or ''):raise ValueError(code)
    return p

def qnorm(q):
    q=np.asarray(q,np.float64);n=float(np.linalg.norm(q))
    if not np.isfinite(n) or n<=1e-12:raise ValueError('ANIM_REPLAY_COURT_QUATERNION_ZERO')
    return q/n

def sample(times,values,interp,path,t):
    times=np.asarray(times,np.float64);values=np.asarray(values,np.float64)
    if t<=times[0]:i=0;u=0.
    elif t>=times[-1]:i=len(times)-1;u=0.
    else:i=int(np.searchsorted(times,t,side='right')-1);u=(float(t)-float(times[i]))/float(times[i+1]-times[i])
    if interp=='CUBICSPLINE':
        if i==len(times)-1:r=values[3*i+1]
        else:
            dt=float(times[i+1]-times[i]);p0=values[3*i+1];m0=values[3*i+2]*dt;p1=values[3*(i+1)+1];m1=values[3*(i+1)]*dt;u2=u*u;u3=u2*u;r=(2*u3-3*u2+1)*p0+(u3-2*u2+u)*m0+(-2*u3+3*u2)*p1+(u3-u2)*m1
        return qnorm(r) if path=='rotation' else r
    cur=values[i]
    if i==len(times)-1 or interp=='STEP':return qnorm(cur) if path=='rotation' else cur
    nxt=values[i+1]
    if path!='rotation':return (1-u)*cur+u*nxt
    qa,qb=qnorm(cur),qnorm(nxt)
    if np.dot(qa,qb)<0:qb=-qb
    s=Slerp([0.,1.],Rotation.from_quat(np.stack([qa,qb])))
    return qnorm(s([float(u)]).as_quat()[0])
def local(node,trs):
    if trs is None:return np.asarray(node['matrix'],np.float64).reshape(4,4).T
    m=np.eye(4);m[:3,:3]=Rotation.from_quat(qnorm(trs['rotation'])).as_matrix()@np.diag(np.asarray(trs['scale'],np.float64));m[:3,3]=trs['translation'];return m
def base(node):
    if 'matrix' in node:return None
    return {'translation':np.asarray(node.get('translation',[0,0,0]),np.float64),'rotation':qnorm(node.get('rotation',[0,0,0,1])),'scale':np.asarray(node.get('scale',[1,1,1]),np.float64)}
def choose_times(channels):
    x=[]
    for c in channels:
        t=c['time'];x.extend(t.tolist());
        if len(t)>1:x.extend(((t[:-1]+t[1:])*.5).tolist())
    u=np.asarray(sorted(set(map(float,x))),np.float64)
    if len(u)<=9:return u
    return u[np.unique(np.rint(np.linspace(0,len(u)-1,9)).astype(int))]
def validate(manifest_path:Path)->dict:
    manifest_path=manifest_path.resolve();m=json.loads(manifest_path.read_text())
    if m.get('schema')!=PARENT_SCHEMA or m.get('status')!=PARENT_STATUS:raise ValueError('ANIM_REPLAY_COURT_PARENT')
    inputs=dict(m.get('inputs') or {});source_path=bound(inputs.get('raw_source'),'ANIM_REPLAY_COURT_SOURCE_DRIFT');am_path=bound(inputs.get('animation_manifest'),'ANIM_REPLAY_COURT_ANIMATION_MANIFEST_DRIFT');ac_path=bound(inputs.get('animation_source_court'),'ANIM_REPLAY_COURT_ANIMATION_SOURCE_COURT_DRIFT');rm_path=bound(inputs.get('rig_manifest'),'ANIM_REPLAY_COURT_RIG_MANIFEST_DRIFT');ri_path=bound(inputs.get('rig_importer_court'),'ANIM_REPLAY_COURT_RIG_IMPORTER_DRIFT');teacher_path=bound(inputs.get('teacher_scene'),'ANIM_REPLAY_COURT_TEACHER_DRIFT')
    am=json.loads(am_path.read_text());ac=json.loads(ac_path.read_text());rm=json.loads(rm_path.read_text());ri=json.loads(ri_path.read_text());teacher=json.loads(teacher_path.read_text())
    if ac.get('status')!='PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_COURT__CANONICAL_REPLAY_REQUIRED' or not bool((ac.get('authority') or {}).get('source_animation_court_passed')):raise ValueError('ANIM_REPLAY_COURT_SOURCE_COURT_NOT_PASS')
    if ri.get('status')!='PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE' or not bool((ri.get('authority') or {}).get('importer_equivalence_passed')):raise ValueError('ANIM_REPLAY_COURT_RIG_IMPORT_NOT_PASS')
    if (am.get('source') or {}).get('sha256')!=sha256(source_path) or (rm.get('source') or {}).get('sha256')!=sha256(source_path):raise ValueError('ANIM_REPLAY_COURT_SOURCE_IDENTITY')
    source=SourceGLTF(source_path);doc=source.doc;nodes=list(doc.get('nodes') or []);joints=sorted({int(j) for skin in doc.get('skins') or [] for j in skin.get('joints') or []});joint_set=set(joints)
    frame_names={str(x.get('name') or '') for x in (teacher.get('calibration') or {}).get('native_control_frames_privileged') or []};joint_names={str(nodes[j].get('name') or f'node_{j}') for j in joints}
    if frame_names!=joint_names:raise ValueError('ANIM_REPLAY_COURT_IMPORTED_JOINT_SET')
    parent=source.parents;declared=list(m.get('animations') or []);animations=list(doc.get('animations') or [])
    if len(declared)!=len(animations):raise ValueError('ANIM_REPLAY_COURT_ANIMATION_COUNT')
    total_samples=0;non_joint=0
    for ai,animation in enumerate(animations):
        samplers=list(animation.get('samplers') or []);channels=[];seen=set()
        for ch in animation.get('channels') or []:
            s=samplers[int(ch['sampler'])];t=np.asarray(source.accessor(int(s['input'])),np.float64).reshape(-1);v=np.asarray(source.accessor(int(s['output'])),np.float64);target=dict(ch.get('target') or {});node=int(target['node']);path=str(target['path']);interp=str(s.get('interpolation','LINEAR'));is_joint=node in joint_set and path in {'translation','rotation','scale'}
            if is_joint:
                k=(node,path)
                if k in seen:raise ValueError('ANIM_REPLAY_COURT_DUPLICATE_TARGET_PATH')
                seen.add(k)
            else:non_joint+=1
            channels.append({'node':node,'path':path,'interp':interp,'time':t,'value':v,'is_joint':is_joint})
        times=choose_times(channels);h=hashlib.sha256();max_abs=0.;min_det=float('inf')
        bases=[base(n) for n in nodes]
        for c in channels:
            if c['is_joint'] and bases[c['node']] is None:raise ValueError('ANIM_REPLAY_COURT_ANIMATED_MATRIX_NODE')
        for t in times:
            trs=[None if x is None else {k:v.copy() for k,v in x.items()} for x in bases]
            for c in channels:
                if c['is_joint']:trs[c['node']][c['path']]=sample(c['time'],c['value'],c['interp'],c['path'],float(t))
            cache={}
            def world(i):
                if i in cache:return cache[i]
                mat=local(nodes[i],trs[i]);
                if i in parent:mat=world(parent[i])@mat
                cache[i]=mat;return mat
            h.update(np.asarray([float(t)],np.float64).tobytes())
            for j in joints:
                mat=world(j);det=float(np.linalg.det(mat[:3,:3]));max_abs=max(max_abs,float(np.max(np.abs(mat))));min_det=min(min_det,det)
                if not np.isfinite(mat).all() or abs(det)<=1e-12:raise ValueError('ANIM_REPLAY_COURT_NONFINITE_OR_SINGULAR')
                h.update(np.asarray([j],np.int64).tobytes());h.update(np.rint(mat*1e9).astype(np.int64).tobytes())
        row=declared[ai]
        if int(row.get('animation_index',-1))!=ai or list(map(float,row.get('sample_times') or []))!=list(map(float,times)) or int(row.get('sample_count',-1))!=len(times) or int(row.get('joint_count',-1))!=len(joints) or row.get('replay_digest_sha256')!=h.hexdigest():raise ValueError('ANIM_REPLAY_COURT_REPLAY_DRIFT')
        if abs(float(row.get('maximum_abs_world_matrix_element'))-max_abs)>1e-8 or abs(float(row.get('minimum_world_linear_determinant'))-min_det)>1e-8:raise ValueError('ANIM_REPLAY_COURT_SUMMARY_DRIFT')
        total_samples+=len(times)
    counts=dict(m.get('counts') or {})
    if counts!={'animation_count':len(animations),'joint_count':len(joints),'replay_sample_count':total_samples,'non_joint_or_non_trs_channel_count':non_joint}:raise ValueError('ANIM_REPLAY_COURT_COUNTS')
    outcome='QUALIFIED_CORE_CANDIDATE' if non_joint==0 else 'QUALIFIED_MASKED_CANDIDATE'
    if m.get('family_candidate_outcome')!=outcome:raise ValueError('ANIM_REPLAY_COURT_OUTCOME')
    return {'schema':SCHEMA,'status':STATUS,'subject_id':m.get('subject_id'),'counts':counts,'family_candidate_outcomes':{'ANIMATION_KINEMATICS':outcome},'independent_method':{'source_reader':'tools.corpus.gltf_bind_scene_v1.SourceGLTF','rotation_linear_interpolation':'scipy.spatial.transform.Slerp','producer_replay_helper_reused':False,'all_declared_replay_digests_rebuilt':True},'authority':{'source_animation_court_passed':True,'imported_rig_equivalence_passed':True,'canonical_replay_court_passed':True,'final_family_authority_minted':False,'training_authorized':False},'parent_manifest':{'path':str(manifest_path),'sha256':sha256(manifest_path)}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=validate(a.manifest);a.output.resolve().parent.mkdir(parents=True,exist_ok=True);a.output.resolve().write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':r['status'],'subject_id':r['subject_id'],'family_candidate_outcomes':r['family_candidate_outcomes']},sort_keys=True))
if __name__=='__main__':main()
