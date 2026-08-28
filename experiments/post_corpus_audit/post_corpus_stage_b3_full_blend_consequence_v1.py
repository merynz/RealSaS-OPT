#!/usr/bin/env python3
"""RealSaS Stage-B3 exhaustive selected .blend consequence audit.

Read-only with respect to corpus evidence. Reconstructs the complete selected .blend
set from canonical selection + admission.json via Drive API, then runs the frozen
Stage-B2 physical-surface consequence probe on every selected .blend source.

Outputs compact checkpoint/result/report/repair manifests under reports/post_corpus_audit.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, requests

from google.colab import auth
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
VARIANTS_PARENT_ID='1mZX14XTspqLSAc5j4AzpuJGTzGIgIQwH'
FOLDER_MIME='application/vnd.google-apps.folder'
PROBE_LOCAL='/content/realsas_blender_consequence_probe_v2.py'
BLENDER_LOCAL='/content/blender-5.2.0-linux-x64/blender'
BANDS=[1e-5,1e-4,5e-4,1e-3]


def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def auth_drive():
    auth.authenticate_user()
    creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    if not creds.valid: creds.refresh(GoogleAuthRequest())
    return build('drive','v3',credentials=creds,cache_discovery=False),creds

def list_all(svc,q,fields='id,name,mimeType,size,parents,modifiedTime',label=None):
    out=[]; token=None; pages=0
    while True:
        resp=svc.files().list(q=q,spaces='drive',fields=f'nextPageToken,files({fields})',pageSize=1000,pageToken=token,supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
        pages+=1; out.extend(resp.get('files',[])); token=resp.get('nextPageToken')
        if not token: break
    if label: print(f'[api] {label}: rows={len(out)} pages={pages}',flush=True)
    return out,pages

def media_get(file_id, token, timeout=90):
    u=f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true'
    r=requests.get(u,headers={'Authorization':f'Bearer {token}'},timeout=timeout); r.raise_for_status(); return r.content

def qstats(vals):
    a=np.asarray([float(x) for x in vals if x is not None and np.isfinite(float(x))],dtype=np.float64)
    if len(a)==0: return {'count':0}
    return {'count':int(len(a)),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(np.max(a)),**{f'gt_{t:g}':int(np.sum(a>t)) for t in BANDS}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=ROOT_DEFAULT)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('--json-workers',type=int,default=24)
    args=ap.parse_args()
    root=Path(args.root); outdir=root/'reports'/'post_corpus_audit'; outdir.mkdir(parents=True,exist_ok=True)
    probe=Path(PROBE_LOCAL); blender=Path(BLENDER_LOCAL)
    if not probe.exists(): raise RuntimeError(f'missing local probe: {probe}')
    if not blender.exists(): raise RuntimeError(f'missing Blender binary: {blender}; run Stage-B2 in the same runtime first')

    selraw=read_json(root/'metadata'/'CANONICAL_VARIANT_SELECTION.json')
    selected=selraw['selected'] if isinstance(selraw,dict) and isinstance(selraw.get('selected'),list) else (list(selraw.values()) if isinstance(selraw,dict) else selraw)
    selected_by_aid={x['canonical_asset_id']:x for x in selected}; selected_ids=set(selected_by_aid)
    print(f'[central] selected={len(selected_ids)}',flush=True)

    svc,creds=auth_drive();
    if not creds.valid: creds.refresh(GoogleAuthRequest())
    token=creds.token

    # Reconstruct selected variant-folder IDs.
    variant_rows,_=list_all(svc,f"'{VARIANTS_PARENT_ID}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",label='variant folders')
    by_name=collections.defaultdict(list)
    for r in variant_rows: by_name[r['name']].append(r)
    selected_variant_names={Path(str(x['variant_dir'])).name:x['canonical_asset_id'] for x in selected}
    folder_to_aid={}
    for name,aid in selected_variant_names.items():
        rs=by_name.get(name,[])
        if len(rs)==1: folder_to_aid[rs[0]['id']]=aid
    variant_ids=set(folder_to_aid)
    if len(variant_ids)!=len(selected_ids): raise RuntimeError(f'selected variant folder coverage {len(variant_ids)}/{len(selected_ids)}')

    # Find all admission.json and retain exactly selected-parent rows.
    adm_rows,_=list_all(svc,"name='admission.json' and trashed=false",label='admission.json')
    selected_adm=[]
    for r in adm_rows:
        pars=r.get('parents') or []
        if len(pars)==1 and pars[0] in variant_ids: selected_adm.append(r)
    if len(selected_adm)!=len(selected_ids): raise RuntimeError(f'admission metadata coverage {len(selected_adm)}/{len(selected_ids)}')

    def get_adm(r):
        try: return r,json.loads(media_get(r['id'],token).decode('utf-8')),None
        except Exception as e: return r,None,f'{type(e).__name__}: {e}'
    admissions={}; errs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.json_workers)) as ex:
        futs=[ex.submit(get_adm,r) for r in selected_adm]
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            r,obj,err=f.result(); parent=(r.get('parents') or [None])[0]; aid=folder_to_aid.get(parent)
            if err: errs.append({'asset':aid,'error':err})
            else: admissions[aid]=obj
            if i%500==0 or i==len(futs): print(f'[admission] {i}/{len(futs)}',flush=True)
    if errs or len(admissions)!=len(selected_ids): raise RuntimeError(f'admission parse failed: parsed={len(admissions)} errors={len(errs)}')

    blend=[]
    for aid,a in admissions.items():
        sid=a.get('source_identity') or {}; raw=str(sid.get('preserved_raw_source_path') or '')
        if Path(raw).suffix.lower()=='.blend':
            blend.append({'asset':aid,'raw_path':raw,'provider':a.get('source_registry_id') or selected_by_aid[aid].get('source_registry_id'),'source_sha256':sid.get('source_sha256')})
    blend=sorted(blend,key=lambda x:x['asset'])
    print(f'[stageB3] selected .blend={len(blend)}',flush=True)
    if len(blend)!=385: print(f'[stageB3] WARNING expected prior census 385, got {len(blend)}',flush=True)

    # Resume checkpoint if present and schema-compatible.
    ckpt=outdir/'POST_CORPUS_STAGE_B3_FULL_BLEND_CHECKPOINT_V1.json'
    done={}
    if ckpt.exists():
        try:
            z=read_json(ckpt)
            if z.get('schema')=='RealSaS.PostCorpus.StageB3Checkpoint.v1': done={x['asset']:x for x in z.get('results',[]) if x.get('asset')}
        except Exception: done={}
    todo=[x for x in blend if x['asset'] not in done]
    print(f'[stageB3] resume={len(done)} todo={len(todo)}',flush=True)
    local=Path('/content/realsas_stageb3_probe'); local.mkdir(exist_ok=True)

    def run_one(x):
        aid=x['asset']; src=root/x['raw_path']; out=local/f'{aid}.json'
        if not src.exists(): return {'asset':aid,'raw_path':x['raw_path'],'provider':x['provider'],'driver_error':'raw source missing'}
        cmd=[str(blender),'-b','--factory-startup','--python',str(probe),'--',str(src),str(out)]
        try:
            cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
            if cp.returncode!=0 or not out.exists(): return {'asset':aid,'raw_path':x['raw_path'],'provider':x['provider'],'driver_error':f'blender rc={cp.returncode}','stdout_tail':cp.stdout[-2500:]}
            z=json.loads(out.read_text()); z.update(asset=aid,raw_path=x['raw_path'],provider=x['provider'],source_sha256=x['source_sha256']); return z
        except Exception as e: return {'asset':aid,'raw_path':x['raw_path'],'provider':x['provider'],'driver_error':f'{type(e).__name__}: {e}'}

    processed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs=[ex.submit(run_one,x) for x in todo]
        for f in concurrent.futures.as_completed(futs):
            z=f.result(); done[z['asset']]=z; processed+=1
            if processed%10==0 or processed==len(futs): print(f'[stageB3] {processed}/{len(futs)} new; total={len(done)}/{len(blend)}',flush=True)
            if processed%25==0 or processed==len(futs):
                tmp=ckpt.with_suffix('.tmp'); tmp.write_text(json.dumps({'schema':'RealSaS.PostCorpus.StageB3Checkpoint.v1','results':[done[k] for k in sorted(done)]},indent=2),encoding='utf-8'); os.replace(tmp,ckpt)

    results=[done[k] for k in sorted(done)]
    ok=[x for x in results if not x.get('driver_error') and not x.get('errors')]
    fail=[x for x in results if x not in ok]
    if len(results)!=len(blend): raise RuntimeError(f'incomplete B3 results {len(results)}/{len(blend)}')

    def val(x,k): return float(x.get(k) or 0.0)
    eval_material=[x for x in ok if val(x,'asset_max_eval_surface_p95_rel')>5e-4]
    eval_clear=[x for x in ok if val(x,'asset_max_eval_surface_p95_rel')>1e-3]
    eval_potential=[x for x in ok if val(x,'asset_max_eval_surface_p95_rel')>1e-4]
    tri_material=[x for x in ok if val(x,'asset_max_triang_surface_p95_rel')>5e-4]
    tri_clear=[x for x in ok if val(x,'asset_max_triang_surface_p95_rel')>1e-3]
    tri_potential=[x for x in ok if val(x,'asset_max_triang_surface_p95_rel')>1e-4]
    nonarm=[x for x in ok if x.get('asset_non_armature_modifier_object_count',0)>0]
    active_shape=[x for x in ok if x.get('asset_active_shape_key_object_count',0)>0]
    risk_union={x['asset']:x for x in eval_material+tri_material+active_shape}

    # Causal/predictive coverage of simple source classifier.
    class_flag={x['asset'] for x in nonarm+active_shape}
    mat_flag={x['asset'] for x in eval_material}
    classifier={
      'material_eval_assets':len(mat_flag),
      'nonarm_or_active_shape_assets':len(class_flag),
      'material_eval_captured_by_nonarm_or_shape':len(mat_flag & class_flag),
      'material_eval_missed_by_nonarm_or_shape':len(mat_flag-class_flag),
      'flagged_but_not_material_eval':len(class_flag-mat_flag),
    }

    def compact(x):
        mods=sorted({m for o in x.get('mesh_objects',[]) for m in o.get('non_armature_modifier_types',[])})
        shapes=sorted({k.get('name') for o in x.get('mesh_objects',[]) for k in o.get('active_shape_keys',[]) if k.get('name')})
        return {'asset':x['asset'],'provider':x.get('provider'),'raw_path':x.get('raw_path'),'source_sha256':x.get('source_sha256'),
                'eval_surface_p95_rel':val(x,'asset_max_eval_surface_p95_rel'),'eval_surface_max_rel':val(x,'asset_max_eval_surface_max_rel'),
                'tri_surface_p95_rel':val(x,'asset_max_triang_surface_p95_rel'),'tri_surface_max_rel':val(x,'asset_max_triang_surface_max_rel'),
                'eval_topology_change_objects':int(x.get('asset_eval_topology_change_object_count',0)),
                'non_armature_modifiers':mods,'active_shape_keys':shapes}

    summary={
      'schema':'RealSaS.PostCorpus.StageB3FullBlendConsequence.v1','selected_blend_count':len(blend),'ok':len(ok),'failures':len(fail),'bands_rel_bbox_diag':BANDS,
      'evaluated_surface_p95_rel':qstats([x.get('asset_max_eval_surface_p95_rel') for x in ok]),
      'triangulation_surface_p95_rel':qstats([x.get('asset_max_triang_surface_p95_rel') for x in ok]),
      'eval_potential_gt_1e-4':len(eval_potential),'eval_material_gt_5e-4':len(eval_material),'eval_clear_gt_1e-3':len(eval_clear),
      'tri_potential_gt_1e-4':len(tri_potential),'tri_material_gt_5e-4':len(tri_material),'tri_clear_gt_1e-3':len(tri_clear),
      'topology_change_assets':sum(x.get('asset_eval_topology_change_object_count',0)>0 for x in ok),
      'non_armature_modifier_assets':len(nonarm),'active_nonbasis_shape_key_assets':len(active_shape),
      'classifier':classifier,
      'repair_candidate_count':len(risk_union),
      'repair_candidates':[compact(risk_union[k]) for k in sorted(risk_union)],
      'failures_detail':fail[:100],
      'corpus_mutation':False,
      'results':results,
    }
    out=outdir/'POST_CORPUS_STAGE_B3_FULL_BLEND_CONSEQUENCE_RESULT_V1.json'; out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    repair=outdir/'BLEND_GEOMETRY_REPAIR_CANDIDATES_V1.json'; repair.write_text(json.dumps(summary['repair_candidates'],indent=2),encoding='utf-8')
    md=outdir/'POST_CORPUS_STAGE_B3_FULL_BLEND_CONSEQUENCE_REPORT_V1.md'
    md.write_text('\n'.join([
      '# Stage-B3 Full Selected .blend Consequence Audit','',
      f"- selected .blend: **{len(blend)}**",f"- probe ok: **{len(ok)}/{len(blend)}**",f"- eval p95 >1e-4: **{len(eval_potential)}**",f"- eval p95 >5e-4: **{len(eval_material)}**",f"- eval p95 >1e-3: **{len(eval_clear)}**",
      f"- triangulation p95 >1e-4: **{len(tri_potential)}**",f"- triangulation p95 >5e-4: **{len(tri_material)}**",f"- triangulation p95 >1e-3: **{len(tri_clear)}**",
      f"- non-ARMATURE modifier assets: **{len(nonarm)}**",f"- active non-Basis shape-key assets: **{len(active_shape)}**",f"- repair candidate union: **{len(risk_union)}**",'',
      '## Classifier coverage','', '```json',json.dumps(classifier,indent=2),'```','',
      'No corpus mutation was performed. Repair/rerender authorization requires interpretation of this exhaustive result.'
    ])+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'results','repair_candidates','failures_detail'}},indent=2),flush=True)

if __name__=='__main__': main()
