#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate source-authored glTF joint animation in a deterministic RealSaS replay.

Source animation bytes remain the authority. This adapter proves those channels can
be consumed as a coherent node-TRS hierarchy after source and imported rig/skin
courts have passed. It does not claim physical dynamics or motion quality.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np

ANIM_SCHEMA="RealSaS.GltfAnimationKinematicsAuthority.v1"
ANIM_COURT_SCHEMA="RealSaS.GltfAnimationKinematicsAuthorityCourt.v1"
RIG_SCHEMA="RealSaS.GltfRigSkinAuthority.v1"
RIG_IMPORT_SCHEMA="RealSaS.GltfImportedRigSkinEquivalenceCourt.v1"
SCHEMA="RealSaS.GltfAnimationCanonicalReplay.v1"
STATUS="PASS_GLTF_ANIMATION_CANONICAL_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8<<20),b''): h.update(c)
    return h.hexdigest()

def bound(row,code):
    row=dict(row or {}); p=Path(str(row.get('path') or '')).resolve()
    if not p.is_file() or sha256(p)!=str(row.get('sha256') or ''): raise ValueError(code)
    return p

def digest(v)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def qnorm(q):
    q=np.asarray(q,np.float64); n=float(np.linalg.norm(q))
    if not np.isfinite(n) or n<=1e-12: raise ValueError('ANIM_REPLAY_QUATERNION_ZERO')
    return q/n

def slerp(a,b,u):
    a=qnorm(a); b=qnorm(b); dot=float(np.dot(a,b))
    if dot<0: b=-b; dot=-dot
    dot=min(1.,max(-1.,dot))
    if dot>0.9995:return qnorm((1-u)*a+u*b)
    th=np.arccos(dot); s=np.sin(th)
    return qnorm(np.sin((1-u)*th)/s*a+np.sin(u*th)/s*b)

def sample(times,values,interp,path,t):
    times=np.asarray(times,np.float64); values=np.asarray(values,np.float64)
    if t<=times[0]: idx=0; u=0.
    elif t>=times[-1]: idx=len(times)-1; u=0.
    else:
        idx=int(np.searchsorted(times,t,side='right')-1); dt=float(times[idx+1]-times[idx]); u=(float(t)-float(times[idx]))/dt
    if interp=='CUBICSPLINE':
        if idx==len(times)-1: result=values[3*idx+1]
        else:
            dt=float(times[idx+1]-times[idx]); p0=values[3*idx+1]; m0=values[3*idx+2]*dt; p1=values[3*(idx+1)+1]; m1=values[3*(idx+1)]*dt
            u2=u*u; u3=u2*u; result=(2*u3-3*u2+1)*p0+(u3-2*u2+u)*m0+(-2*u3+3*u2)*p1+(u3-u2)*m1
        return qnorm(result) if path=='rotation' else result
    current=values[idx]
    if idx==len(times)-1 or interp=='STEP': return qnorm(current) if path=='rotation' else current
    nxt=values[idx+1]
    return slerp(current,nxt,u) if path=='rotation' else (1-u)*current+u*nxt

def quat_matrix(q):
    x,y,z,w=qnorm(q); return np.asarray([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),0],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),0],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y),0],[0,0,0,1]],np.float64)
def local(trs):
    T=np.eye(4); T[:3,3]=trs['translation']; S=np.diag([*trs['scale'],1.]); return T@quat_matrix(trs['rotation'])@S

def parents(nodes):
    out={}
    for p,n in enumerate(nodes):
        for c in n.get('children') or []:
            c=int(c)
            if c in out: raise ValueError('ANIM_REPLAY_MULTI_PARENT')
            out[c]=p
    return out

def base_trs(node):
    if 'matrix' in node: return None
    return {'translation':np.asarray(node.get('translation',[0,0,0]),np.float64),'rotation':qnorm(node.get('rotation',[0,0,0,1])),'scale':np.asarray(node.get('scale',[1,1,1]),np.float64)}
def sample_times(channel_rows,arrays):
    candidates=[]
    for row in channel_rows:
        t=np.asarray(arrays[row['time_key']],np.float64)
        candidates.extend(t.tolist())
        if len(t)>1:candidates.extend(((t[:-1]+t[1:])*.5).tolist())
    u=np.asarray(sorted(set(map(float,candidates))),np.float64)
    if len(u)<=9:return u
    ix=np.unique(np.rint(np.linspace(0,len(u)-1,9)).astype(int)); return u[ix]
def replay_digest(nodes,parent,joints,channels,arrays,times):
    h=hashlib.sha256(); max_abs=0.; min_det=float('inf')
    base=[base_trs(n) for n in nodes]
    for row in channels:
        if row['target_path'] in {'translation','rotation','scale'} and base[int(row['target_node'])] is None: raise ValueError('ANIM_REPLAY_ANIMATED_MATRIX_NODE')
    for t in times:
        trs=[None if x is None else {k:v.copy() for k,v in x.items()} for x in base]
        for row in channels:
            path=row['target_path']
            if path not in {'translation','rotation','scale'} or not row['target_is_skin_joint']: continue
            node=int(row['target_node']); trs[node][path]=sample(arrays[row['time_key']],arrays[row['value_key']],row['interpolation'],path,float(t))
        cache={}
        def world(i):
            if i in cache:return cache[i]
            m=local(trs[i]) if trs[i] is not None else np.asarray(nodes[i]['matrix'],np.float64).reshape(4,4).T
            if i in parent:m=world(parent[i])@m
            cache[i]=m; return m
        h.update(np.asarray([float(t)],np.float64).tobytes())
        for j in joints:
            m=world(j); max_abs=max(max_abs,float(np.max(np.abs(m)))); det=float(np.linalg.det(m[:3,:3])); min_det=min(min_det,det)
            if not np.isfinite(m).all() or abs(det)<=1e-12: raise ValueError('ANIM_REPLAY_NONFINITE_OR_SINGULAR')
            h.update(np.asarray([j],np.int64).tobytes()); h.update(np.rint(m*1e9).astype(np.int64).tobytes())
    return h.hexdigest(),max_abs,min_det

def build(animation_manifest_path:Path,animation_court_path:Path,rig_manifest_path:Path,rig_importer_court_path:Path,teacher_scene_path:Path,output:Path)->dict:
    am=json.loads(animation_manifest_path.resolve().read_text()); ac=json.loads(animation_court_path.resolve().read_text()); rm=json.loads(rig_manifest_path.resolve().read_text()); rc=json.loads(rig_importer_court_path.resolve().read_text()); teacher=json.loads(teacher_scene_path.resolve().read_text())
    if am.get('schema')!=ANIM_SCHEMA or ac.get('schema')!=ANIM_COURT_SCHEMA or not str(ac.get('status','')).startswith('PASS_'): raise ValueError('ANIM_REPLAY_SOURCE_COURT')
    if rm.get('schema')!=RIG_SCHEMA or rc.get('schema')!=RIG_IMPORT_SCHEMA or not str(rc.get('status','')).startswith('PASS_'): raise ValueError('ANIM_REPLAY_RIG_COURT')
    source=bound(am.get('source'),'ANIM_REPLAY_SOURCE_DRIFT'); rsource=bound(rm.get('source'),'ANIM_REPLAY_RIG_SOURCE_DRIFT')
    if source!=rsource or sha256(source)!=sha256(rsource): raise ValueError('ANIM_REPLAY_SOURCE_IDENTITY')
    if (rc.get('inputs') or {}).get('raw_source',{}).get('sha256')!=sha256(source): raise ValueError('ANIM_REPLAY_IMPORT_SOURCE_BINDING')
    if (rc.get('inputs') or {}).get('teacher_scene',{}).get('sha256')!=sha256(teacher_scene_path.resolve()): raise ValueError('ANIM_REPLAY_TEACHER_BINDING')
    gltf=json.loads(source.read_text()); nodes=list(gltf.get('nodes') or []); parent=parents(nodes)
    joints=sorted({int(j) for skin in gltf.get('skins') or [] for j in skin.get('joints') or []}); names=[str(nodes[j].get('name') or f'node_{j}') for j in joints]
    frames=list((teacher.get('calibration') or {}).get('native_control_frames_privileged') or []); frame_names=[str(x.get('name') or '') for x in frames]
    if set(names)!=set(frame_names) or len(names)!=len(frame_names): raise ValueError('ANIM_REPLAY_IMPORTED_JOINT_NAME_SET')
    arrays_path=bound((am.get('outputs') or {}).get('arrays'),'ANIM_REPLAY_ARRAY_DRIFT')
    with np.load(arrays_path,allow_pickle=False) as z: arrays={k:np.asarray(z[k]) for k in z.files}
    rows=[]; total_samples=0
    for animation in am.get('animations') or []:
        channels=list(animation.get('channels') or []); seen=set()
        for c in channels:
            if c.get('target_is_skin_joint') and c.get('target_path') in {'translation','rotation','scale'}:
                key=(int(c['target_node']),str(c['target_path']))
                if key in seen: raise ValueError('ANIM_REPLAY_DUPLICATE_TARGET_PATH')
                seen.add(key)
        times=sample_times(channels,arrays)
        d,max_abs,min_det=replay_digest(nodes,parent,joints,channels,arrays,times)
        rows.append({'animation_index':int(animation['animation_index']),'name':animation.get('name'),'sample_times':[float(x) for x in times],'sample_count':int(len(times)),'joint_count':len(joints),'replay_digest_sha256':d,'maximum_abs_world_matrix_element':max_abs,'minimum_world_linear_determinant':min_det})
        total_samples+=len(times)
    non_joint=sum(int(not c.get('target_is_skin_joint') or c.get('target_path') not in {'translation','rotation','scale'}) for a in am.get('animations') or [] for c in a.get('channels') or [])
    outcome='QUALIFIED_CORE_CANDIDATE' if non_joint==0 else 'QUALIFIED_MASKED_CANDIDATE'
    payload={'schema':SCHEMA,'status':STATUS,'subject_id':teacher.get('subject_key'),'inputs':{'animation_manifest':{'path':str(animation_manifest_path.resolve()),'sha256':sha256(animation_manifest_path.resolve())},'animation_source_court':{'path':str(animation_court_path.resolve()),'sha256':sha256(animation_court_path.resolve())},'rig_manifest':{'path':str(rig_manifest_path.resolve()),'sha256':sha256(rig_manifest_path.resolve())},'rig_importer_court':{'path':str(rig_importer_court_path.resolve()),'sha256':sha256(rig_importer_court_path.resolve())},'teacher_scene':{'path':str(teacher_scene_path.resolve()),'sha256':sha256(teacher_scene_path.resolve())},'raw_source':{'path':str(source),'sha256':sha256(source)}},'animations':rows,'counts':{'animation_count':len(rows),'joint_count':len(joints),'replay_sample_count':int(total_samples),'non_joint_or_non_trs_channel_count':non_joint},'policy':{'sample_time_selection':'UNIQUE_KEYS_AND_INTERVAL_MIDPOINTS__DETERMINISTIC_MAX_9_QUANTILE_SUBSET_PER_CLIP','source_channel_arrays_already_independently_courted':True,'imported_rig_name_parent_and_skin_equivalence_required':True,'authored_animation_is_kinematic_truth_not_physics':True,'motion_quality_claimed':False},'family_candidate_outcome':outcome,'authority':{'canonical_replay_candidate_only':True,'independent_replay_court_required':True,'final_family_authority_minted':False,'training_authorized':False}}
    payload['identity_sha256']=digest(payload); output=output.resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n'); return payload

def main():
    p=argparse.ArgumentParser(); p.add_argument('--animation-manifest',type=Path,required=True);p.add_argument('--animation-court',type=Path,required=True);p.add_argument('--rig-manifest',type=Path,required=True);p.add_argument('--rig-importer-court',type=Path,required=True);p.add_argument('--teacher-scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=build(a.animation_manifest,a.animation_court,a.rig_manifest,a.rig_importer_court,a.teacher_scene,a.output);print(json.dumps({'status':r['status'],'subject_id':r['subject_id'],'family_candidate_outcome':r['family_candidate_outcome'],'counts':r['counts']},sort_keys=True))
if __name__=='__main__':main()
