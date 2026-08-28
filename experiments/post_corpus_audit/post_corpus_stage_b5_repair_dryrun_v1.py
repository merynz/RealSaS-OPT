#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, subprocess, tarfile
from pathlib import Path
import numpy as np, requests

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
BLENDER_VERSION='5.2.0'; RELEASE_DIR='Blender5.2'
EXTRACTOR_LOCAL='/content/realsas_blender_evaluated_extract_v1.py'
MATERIAL=5e-4

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def ensure_blender():
    home=Path(f'/content/blender-{BLENDER_VERSION}-linux-x64'); bin=home/'blender'
    if bin.exists(): return bin
    work=Path('/content/realsas_blender_bootstrap'); work.mkdir(exist_ok=True)
    arc=work/f'blender-{BLENDER_VERSION}-linux-x64.tar.xz'; sha=work/f'blender-{BLENDER_VERSION}.sha256'
    bases=[f'https://mirror.blender.org/release/{RELEASE_DIR}',f'https://download.blender.org/release/{RELEASE_DIR}']
    def get(urls,dst,minbytes=1):
        errs=[]
        for u in urls:
            try:
                with requests.get(u,stream=True,timeout=(30,600),allow_redirects=True) as r:
                    r.raise_for_status(); tmp=dst.with_suffix(dst.suffix+'.part')
                    with open(tmp,'wb') as f:
                        for c in r.iter_content(8<<20):
                            if c: f.write(c)
                    if tmp.stat().st_size<minbytes: raise RuntimeError('too small')
                    os.replace(tmp,dst); return
            except Exception as e: errs.append(str(e))
        raise RuntimeError('download failed: '+' | '.join(errs[-4:]))
    get([f'{b}/{sha.name}' for b in bases],sha,64)
    expected=None
    for line in sha.read_text(errors='ignore').splitlines():
        ps=line.split()
        if len(ps)>=2 and Path(ps[-1].lstrip('*')).name==arc.name and re.fullmatch(r'[0-9a-fA-F]{64}',ps[0]): expected=ps[0].lower(); break
    if not expected: raise RuntimeError('checksum unresolved')
    if not arc.exists() or sha256_file(arc)!=expected:
        get([f'{b}/{arc.name}' for b in bases],arc,100_000_000)
        if sha256_file(arc)!=expected: raise RuntimeError('Blender SHA mismatch')
    with tarfile.open(arc,'r:xz') as tf: tf.extractall('/content')
    bin.chmod(bin.stat().st_mode|0o111); return bin

def affine_apply(M,P):
    P=np.asarray(P,np.float32); H=np.concatenate([P,np.ones((len(P),1),np.float32)],axis=1)
    return (H @ np.asarray(M,np.float32).T)[:,:3]

def canonicalize_arrays(vertices,joints=None,tails=None):
    V=np.asarray(vertices,np.float32); lo=V.min(0); hi=V.max(0); center=(lo+hi)*0.5; scale=float(np.max(hi-lo))
    if not np.isfinite(scale) or scale<=1e-8: raise ValueError('degenerate bbox')
    C=np.eye(4,dtype=np.float32); C[:3,:3]=np.eye(3,dtype=np.float32)/scale; C[:3,3]=-center/scale
    Ci=np.eye(4,dtype=np.float32); Ci[:3,:3]=np.eye(3,dtype=np.float32)*scale; Ci[:3,3]=center
    return affine_apply(C,V), None if joints is None else affine_apply(C,joints), None if tails is None else affine_apply(C,tails), C,Ci,center,scale

def compute_vertex_normals(V,F):
    V=np.asarray(V,np.float32); F=np.asarray(F,np.int32); N=np.zeros_like(V)
    tri=V[F]; fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for k in range(3): np.add.at(N,F[:,k],fn)
    n=np.linalg.norm(N,axis=1,keepdims=True); N=N/np.maximum(n,1e-12)
    return N.astype(np.float32)

def geometry_gate(V,F):
    V=np.asarray(V); F=np.asarray(F); reasons=[]; metrics={}
    if V.ndim!=2 or V.shape[1]!=3 or len(V)<3: reasons.append('bad_vertices')
    if F.ndim!=2 or F.shape[1]!=3 or len(F)<1: reasons.append('bad_faces')
    if not np.isfinite(V).all(): reasons.append('nonfinite_vertices')
    if len(F) and (F.min()<0 or F.max()>=len(V)): reasons.append('face_index_oob')
    if len(F) and not reasons:
        area=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)*0.5
        metrics['nondegenerate_fraction']=float(np.mean(area>1e-12))
        if metrics['nondegenerate_fraction']<0.95: reasons.append('too_many_degenerate_faces')
    return not reasons,reasons,metrics

def rig_gate(parents,heads=None):
    if parents is None: return False,['no_rig'],{}
    p=np.asarray(parents,dtype=np.int64); reasons=[]
    if p.ndim!=1 or len(p)<1: return False,['no_rig'],{}
    if np.any((p>=len(p))|(p<-1)): reasons.append('parent_oob')
    if not np.any(p==-1): reasons.append('no_root')
    for i in range(len(p)):
        seen=set(); j=i
        while j>=0:
            if j in seen: reasons.append('cycle'); break
            seen.add(j); j=int(p[j])
        if 'cycle' in reasons: break
    if heads is not None and not np.isfinite(np.asarray(heads)).all(): reasons.append('nonfinite_bones')
    return not reasons,reasons,{'bone_count':int(len(p))}

def skin_gate(W,vertex_count=None):
    if W is None: return False,['no_skin'],{}
    W=np.asarray(W,np.float32); reasons=[]; m={}
    if W.ndim!=2 or W.shape[0]<1 or W.shape[1]<1: return False,['no_skin'],m
    if vertex_count is not None and W.shape[0]!=vertex_count: reasons.append('skin_vertex_row_mismatch')
    if not np.isfinite(W).all(): reasons.append('nonfinite_skin')
    if np.nanmin(W)<-1e-6: reasons.append('negative_skin')
    rs=W.sum(1); m['zero_row_fraction']=float(np.mean(rs<=1e-8)); m['row_sum_mean']=float(np.mean(rs)); m['row_sum_p99_abs_err']=float(np.quantile(np.abs(rs-1),.99))
    if m['zero_row_fraction']>0.01: reasons.append('too_many_unweighted_vertices')
    return not reasons,reasons,m

def normalize_raw_npz(raw_npz,out_npz):
    with np.load(raw_npz,allow_pickle=False) as d:
        V=np.asarray(d['vertices_source'],np.float32); F=np.asarray(d['faces'],np.int32)
        heads=np.asarray(d['bone_heads_source']) if 'bone_heads_source' in d else None
        tails=np.asarray(d['bone_tails_source']) if 'bone_tails_source' in d else None
        carry={k:np.asarray(d[k]) for k in ['parents','deform_mask','skin','rest_local_source','rest_world_source'] if k in d}
    Vc,Hc,Tc,C,Ci,center,scale=canonicalize_arrays(V,heads,tails); N=compute_vertex_normals(Vc,F)
    arr={'vertices_source':V,'vertices':Vc,'faces':F,'vertex_normals':N,'canonical_transform':C,'inverse_canonical_transform':Ci,**carry}
    if Hc is not None: arr['bone_heads']=Hc
    if Tc is not None: arr['bone_tails']=Tc
    np.savez_compressed(out_npz,**arr)
    return {'center':center.tolist(),'scale':scale,'canonical_transform_max_abs_err':float(np.max(np.abs(affine_apply(C,V)-Vc))),'inverse_transform_max_abs_err':float(np.max(np.abs(affine_apply(Ci,Vc)-V)))}

def audit_normalized(npz_path):
    with np.load(npz_path,allow_pickle=False) as d:
        V=np.asarray(d['vertices']); F=np.asarray(d['faces']); Vsrc=np.asarray(d['vertices_source']); C=np.asarray(d['canonical_transform'])
        parents=np.asarray(d['parents']) if 'parents' in d else None; heads=np.asarray(d['bone_heads']) if 'bone_heads' in d else None; skin=np.asarray(d['skin']) if 'skin' in d else None
    gp,gr,gm=geometry_gate(V,F); rp,rr,rm=rig_gate(parents,heads); sp,sr,sm=skin_gate(skin,len(V))
    c_err=float(np.max(np.abs(affine_apply(C,Vsrc)-V))); cp=np.isfinite(c_err) and c_err<2e-6
    return {'geometry_pass':gp,'geometry_reasons':gr,'geometry_metrics':gm,'rig_pass':rp,'rig_reasons':rr,'rig_metrics':rm,'skin_pass':sp,'skin_reasons':sr,'skin_metrics':sm,'canonicalization_pass':cp,'canonicalization_max_abs_err':c_err,'iris_capable':gp and cp,'geppetto_capable':gp and cp and rp,'arachne_capable':gp and cp and rp and sp}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=ROOT_DEFAULT); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--timeout',type=int,default=300); args=ap.parse_args()
    root=Path(args.root); outdir=root/'reports'/'post_corpus_audit'; blender=ensure_blender(); extractor=Path(EXTRACTOR_LOCAL)
    if not extractor.exists(): raise RuntimeError('local evaluated extractor missing')
    cands=json.loads((outdir/'BLEND_GEOMETRY_REPAIR_CANDIDATES_V1.json').read_text())
    b4=json.loads((outdir/'POST_CORPUS_STAGE_B4_EVALUATED_SKIN_RESULT_V1.json').read_text())
    b4by={x['asset']:x for x in b4['results']}
    shape_quarantine=sorted([x for x in cands if x.get('active_shape_keys')],key=lambda x:x['asset'])
    repair=sorted([x for x in cands if not x.get('active_shape_keys') and (float(x.get('eval_surface_p95_rel') or 0)>MATERIAL or float(x.get('tri_surface_p95_rel') or 0)>MATERIAL)],key=lambda x:x['asset'])
    print(f'[stageB5] auto_repair={len(repair)} shape_quarantine={len(shape_quarantine)}',flush=True)
    work=Path('/content/realsas_stageb5_dryrun'); work.mkdir(exist_ok=True)
    def one(x):
        aid=x['asset']; src=root/x['raw_path']; od=work/aid; od.mkdir(exist_ok=True)
        raw=od/'raw.npz'; struct=od/'source_structure.json'; norm=od/'normalized.npz'
        try:
            cp=subprocess.run([str(blender),'-b','--factory-startup','--python',str(extractor),'--',str(src),str(raw),str(struct)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
            if cp.returncode!=0 or not raw.exists(): return {'asset':aid,'status':'FAIL','error':f'blender rc={cp.returncode}','stdout_tail':cp.stdout[-2500:]}
            meta=normalize_raw_npz(raw,norm); audit=audit_normalized(norm)
            old=(b4by.get(aid,{}).get('technical') or {})
            with np.load(norm,allow_pickle=False) as d:
                counts={'vertices':int(len(d['vertices'])),'faces':int(len(d['faces'])),'skin_rows':int(d['skin'].shape[0]) if 'skin' in d else None,'bones':int(d['skin'].shape[1]) if 'skin' in d else 0}
            return {'asset':aid,'status':'PASS','raw_path':x['raw_path'],'candidate':x,'old_capabilities':{'iris':bool(old.get('iris_capable')),'geppetto':bool(old.get('geppetto_capable')),'arachne':bool(old.get('arachne_capable'))},'new_audit':audit,'counts':counts,'normalization':meta,'source_structure':json.loads(struct.read_text()),'staged_npz_sha256':sha256_file(norm)}
        except Exception as e: return {'asset':aid,'status':'FAIL','error':f'{type(e).__name__}: {e}'}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        fs=[ex.submit(one,x) for x in repair]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            results.append(f.result())
            if i%10==0 or i==len(fs): print(f'[stageB5] {i}/{len(fs)}',flush=True)
    ok=[x for x in results if x['status']=='PASS']; fails=[x for x in results if x['status']!='PASS']
    iris_bad=[x['asset'] for x in ok if not x['new_audit']['iris_capable']]
    gep_bad=[x['asset'] for x in ok if not x['new_audit']['geppetto_capable']]
    ara_old=[x for x in ok if x['old_capabilities']['arachne']]
    ara_drops=[{'asset':x['asset'],'skin_metrics':x['new_audit']['skin_metrics'],'skin_reasons':x['new_audit']['skin_reasons']} for x in ara_old if not x['new_audit']['arachne_capable']]
    ara_gains=[x['asset'] for x in ok if (not x['old_capabilities']['arachne']) and x['new_audit']['arachne_capable']]
    summary={'schema':'RealSaS.PostCorpus.StageB5RepairDryRun.v1','auto_repair_requested':len(repair),'shape_key_quarantine_count':len(shape_quarantine),'shape_key_quarantine_assets':[{'asset':x['asset'],'keys':x.get('active_shape_keys')} for x in shape_quarantine],'ok':len(ok),'failures':len(fails),'failure_examples':fails[:20],'iris_capability_failures_after_repair':iris_bad,'geppetto_capability_failures_after_repair':gep_bad,'current_arachne_in_repair_set':len(ara_old),'arachne_drops_after_repair':ara_drops,'arachne_gains_after_repair':ara_gains,'results':results,'corpus_mutation':False}
    (outdir/'POST_CORPUS_STAGE_B5_REPAIR_DRYRUN_RESULT_V1.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Stage-B5 Repair Dry Run','',f'- auto repair candidates: **{len(repair)}**',f'- shape-key quarantine: **{len(shape_quarantine)}**',f'- extraction/audit ok: **{len(ok)}/{len(repair)}**',f'- IRIS failures after repair: **{len(iris_bad)}**',f'- Geppetto failures after repair: **{len(gep_bad)}**',f'- old Arachne-capable in repair set: **{len(ara_old)}**',f'- Arachne drops after repair: **{len(ara_drops)}**',f'- Arachne gains after repair: **{len(ara_gains)}**','','No corpus mutation was performed. Staged NPZs live only in /content/realsas_stageb5_dryrun.']
    (outdir/'POST_CORPUS_STAGE_B5_REPAIR_DRYRUN_REPORT_V1.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__': main()
