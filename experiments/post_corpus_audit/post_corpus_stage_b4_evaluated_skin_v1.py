#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, json, os, re, subprocess, tarfile, hashlib
from pathlib import Path
import requests, numpy as np

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
BLENDER_VERSION='5.2.0'; RELEASE_DIR='Blender5.2'
PROBE_LOCAL='/content/realsas_blender_evaluated_skin_probe_v1.py'

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

def read_tech(root, aid, selected):
    row=selected.get(aid) or {}; vd=row.get('variant_dir')
    p=root/vd/'technical_audit.json' if vd else None
    try: return json.loads(p.read_text()) if p and p.exists() else {}
    except Exception: return {}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=ROOT_DEFAULT); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--timeout',type=int,default=300); args=ap.parse_args()
    root=Path(args.root); outdir=root/'reports'/'post_corpus_audit'
    cands=json.loads((outdir/'BLEND_GEOMETRY_REPAIR_CANDIDATES_V1.json').read_text())
    selraw=json.loads((root/'metadata'/'CANONICAL_VARIANT_SELECTION.json').read_text())
    ss=selraw.get('selected',selraw) if isinstance(selraw,dict) else selraw
    selected={x['canonical_asset_id']:x for x in (ss if isinstance(ss,list) else ss.values())}
    blender=ensure_blender(); probe=Path(PROBE_LOCAL)
    if not probe.exists(): raise RuntimeError('local B4 probe missing')
    work=Path('/content/realsas_stageb4'); work.mkdir(exist_ok=True)
    def one(x):
        aid=x['asset']; src=root/x['raw_path']; o=work/f'{aid}.json'
        tech=read_tech(root,aid,selected)
        if not src.exists(): return {'asset':aid,'driver_error':'source missing','technical':tech}
        try:
            cp=subprocess.run([str(blender),'-b','--factory-startup','--python',str(probe),'--',str(src),str(o)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
            if cp.returncode!=0 or not o.exists(): return {'asset':aid,'driver_error':f'rc={cp.returncode}','stdout_tail':cp.stdout[-3000:],'technical':tech}
            z=json.loads(o.read_text()); z['asset']=aid; z['technical']=tech; z['repair_candidate']=x; return z
        except Exception as e: return {'asset':aid,'driver_error':f'{type(e).__name__}: {e}','technical':tech}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        fs=[ex.submit(one,x) for x in cands]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            results.append(f.result())
            if i%10==0 or i==len(fs): print(f'[stageB4] {i}/{len(fs)}',flush=True)
    ok=[x for x in results if not x.get('driver_error') and not x.get('errors')]
    ara=[x for x in ok if bool((x.get('technical') or {}).get('arachne_capable'))]
    gep=[x for x in ok if bool((x.get('technical') or {}).get('geppetto_capable'))]
    iris=[x for x in ok if bool((x.get('technical') or {}).get('iris_capable'))]
    def zf(x):
        v=(x.get('evaluated_global_skin') or {}).get('zero_row_fraction')
        return None if v is None else float(v)
    ara_bad=[x for x in ara if zf(x) is None or zf(x)>0.01 or not bool((x.get('evaluated_global_skin') or {}).get('finite',False)) or not bool((x.get('evaluated_global_skin') or {}).get('nonnegative',False))]
    topo=[x for x in ok if int(x.get('topology_changed_object_count',0))>0]
    topo_no_groups=[]
    for x in topo:
        for o in x.get('objects',[]):
            if o.get('topology_changed') and int((o.get('evaluated_group_meta') or {}).get('vertex_group_element_count',0))==0 and int(x.get('bone_count',0))>0:
                topo_no_groups.append({'asset':x['asset'],'object':o.get('name'),'modifiers':o.get('non_armature_modifiers')})
    summary={
      'schema':'RealSaS.PostCorpus.StageB4EvaluatedSkin.v1','repair_candidates':len(cands),'ok':len(ok),'failures':len(results)-len(ok),
      'current_capabilities':{'iris':len(iris),'geppetto':len(gep),'arachne':len(ara)},
      'topology_changed_candidates':len(topo),'topology_changed_objects_without_eval_group_elements':len(topo_no_groups),
      'arachne_candidates_with_eval_skin_gt_1pct_or_invalid':len(ara_bad),
      'arachne_bad_assets':[{'asset':x['asset'],'zero_row_fraction':zf(x),'bone_count':x.get('bone_count')} for x in ara_bad],
      'topology_no_group_examples':topo_no_groups[:100],
      'results':results,'corpus_mutation':False,
    }
    (outdir/'POST_CORPUS_STAGE_B4_EVALUATED_SKIN_RESULT_V1.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Stage-B4 Evaluated Skin Propagation','',f"- repair candidates: **{len(cands)}**",f"- ok: **{len(ok)}/{len(cands)}**",f"- current Arachne-capable: **{len(ara)}**",f"- topology-changed candidates: **{len(topo)}**",f"- topology-changed objects without evaluated group elements: **{len(topo_no_groups)}**",f"- Arachne evaluated-skin invalid/>1% zero rows: **{len(ara_bad)}**",'', 'No corpus mutation was performed.']
    (outdir/'POST_CORPUS_STAGE_B4_EVALUATED_SKIN_REPORT_V1.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
if __name__=='__main__': main()
