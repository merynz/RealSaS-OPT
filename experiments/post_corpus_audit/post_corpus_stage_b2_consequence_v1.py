#!/usr/bin/env python3
"""RealSaS Stage-B2 consequence audit.
Read-only. Reopens the same Stage-B sample and measures geometric *surface consequence*,
not merely exact topology/index differences.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, subprocess, tarfile
from pathlib import Path
import requests, numpy as np

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
BLENDER_VERSION='5.2.0'; RELEASE_DIR='Blender5.2'
PROBE_LOCAL='/content/realsas_blender_consequence_probe_v2.py'

# Frozen interpretation bands, relative to object bounding-box diagonal.
# <=1e-5: numerical/negligible; >1e-4: potentially raster-relevant; >5e-4: material at ~1024 scale.
BANDS=[1e-5,1e-4,5e-4,1e-3]

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
        parts=line.split()
        if len(parts)>=2 and Path(parts[-1].lstrip('*')).name==arc.name and re.fullmatch(r'[0-9a-fA-F]{64}',parts[0]): expected=parts[0].lower(); break
    if not expected: raise RuntimeError('could not resolve Blender checksum')
    if not arc.exists() or sha256_file(arc)!=expected:
        get([f'{b}/{arc.name}' for b in bases],arc,100_000_000)
        if sha256_file(arc)!=expected: raise RuntimeError('Blender SHA mismatch')
    with tarfile.open(arc,'r:xz') as tf: tf.extractall('/content')
    if not bin.exists(): raise RuntimeError('Blender binary missing')
    bin.chmod(bin.stat().st_mode|0o111); return bin

def qstats(vals):
    a=np.asarray([float(x) for x in vals if x is not None and np.isfinite(float(x))],dtype=np.float64)
    if len(a)==0: return {'count':0}
    return {'count':int(len(a)),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(np.max(a)),
            **{f'gt_{t:g}':int(np.sum(a>t)) for t in BANDS}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=ROOT_DEFAULT); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--timeout',type=int,default=300); args=ap.parse_args()
    root=Path(args.root); outdir=root/'reports'/'post_corpus_audit'; local=Path('/content/realsas_stageb2_probe'); local.mkdir(exist_ok=True)
    blend_manifest=outdir/'BLEND_EVALUATED_MESH_AUDIT_SAMPLE_V1.json'; app_manifest=outdir/'APPEARANCE_RECOVERABILITY_SAMPLE_V1.json'
    if not blend_manifest.exists() or not app_manifest.exists(): raise RuntimeError('Stage-A manifests missing')
    probe=Path(PROBE_LOCAL)
    if not probe.exists(): raise RuntimeError(f'missing local consequence probe: {probe}')
    rows=json.loads(blend_manifest.read_text())+json.loads(app_manifest.read_text()); rows=list({x['asset']:x for x in rows if x.get('raw_path')}.values())
    print(f'[stageB2] unique sources={len(rows)}',flush=True)
    blender=ensure_blender()
    def run_one(x):
        aid=x['asset']; src=root/x['raw_path']; out=local/f'{aid}.json'
        if not src.exists(): return {'asset':aid,'raw_path':x['raw_path'],'driver_error':'raw source missing'}
        cmd=[str(blender),'-b','--factory-startup','--python',str(probe),'--',str(src),str(out)]
        try:
            cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
            if cp.returncode!=0 or not out.exists(): return {'asset':aid,'raw_path':x['raw_path'],'driver_error':f'blender rc={cp.returncode}','stdout_tail':cp.stdout[-2500:]}
            z=json.loads(out.read_text()); z['asset']=aid; z['raw_path']=x['raw_path']; return z
        except Exception as e: return {'asset':aid,'raw_path':x['raw_path'],'driver_error':f'{type(e).__name__}: {e}'}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs=[ex.submit(run_one,x) for x in rows]
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            results.append(f.result())
            if i%10==0 or i==len(futs): print(f'[stageB2] {i}/{len(futs)}',flush=True)
    ok=[x for x in results if not x.get('driver_error') and not x.get('errors')]
    blend=[x for x in ok if x.get('extension')=='.blend']
    summary={
      'schema':'RealSaS.PostCorpus.StageB2Consequence.v1',
      'sample_requested':len(rows),'sample_ok':len(ok),'failures':len(results)-len(ok),
      'blend_sample_ok':len(blend),
      'bands_rel_bbox_diag':BANDS,
      'evaluated_surface_p95_rel':qstats([x.get('asset_max_eval_surface_p95_rel') for x in blend]),
      'evaluated_surface_max_rel':qstats([x.get('asset_max_eval_surface_max_rel') for x in blend]),
      'triangulation_surface_p95_rel':qstats([x.get('asset_max_triang_surface_p95_rel') for x in blend]),
      'triangulation_surface_max_rel':qstats([x.get('asset_max_triang_surface_max_rel') for x in blend]),
      'blend_assets_with_eval_topology_change':sum(x.get('asset_eval_topology_change_object_count',0)>0 for x in blend),
      'blend_assets_with_non_armature_modifiers':sum(x.get('asset_non_armature_modifier_object_count',0)>0 for x in blend),
      'blend_assets_with_active_nonbasis_shape_keys':sum(x.get('asset_active_shape_key_object_count',0)>0 for x in blend),
      'results':results,'corpus_mutation':False,
    }
    j=outdir/'POST_CORPUS_STAGE_B2_CONSEQUENCE_RESULT_V1.json'; j.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=outdir/'POST_CORPUS_STAGE_B2_CONSEQUENCE_REPORT_V1.md'
    lines=['# Stage-B2 Consequence Audit','',f"- sample ok: **{len(ok)}/{len(rows)}**",f"- blend sample ok: **{len(blend)}**",f"- eval topology-change assets: **{summary['blend_assets_with_eval_topology_change']}**",f"- non-armature modifier assets: **{summary['blend_assets_with_non_armature_modifiers']}**",f"- active non-Basis shape-key assets: **{summary['blend_assets_with_active_nonbasis_shape_keys']}**",'', '## Relative surface-distance summaries','', '```json',json.dumps({k:v for k,v in summary.items() if k not in {'results','schema','corpus_mutation'}},indent=2),'```','']
    md.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
if __name__=='__main__': main()
