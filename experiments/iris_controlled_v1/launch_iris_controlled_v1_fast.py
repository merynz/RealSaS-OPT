from __future__ import annotations
import argparse,json,hashlib,subprocess,sys,shutil,time,os
from pathlib import Path

BASE_MANIFEST_SHA='e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098'
FAST_PREP_SHA='8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])
def verify_base_package(pkg):
    mp=pkg/'PACKAGE_MANIFEST_V1.json'
    got=sha(mp)
    if got!=BASE_MANIFEST_SHA: raise RuntimeError(f'base package manifest SHA mismatch: {got}')
    m=json.load(open(mp)); bad=[]
    for r in m['files']:
        p=pkg/r['path']; g=sha(p) if p.exists() else None
        if g!=r['sha256']: bad.append({'path':r['path'],'expected':r['sha256'],'got':g})
    if bad: raise RuntimeError('base package verification failed: '+json.dumps(bad[:5]))

def make_ceiling_seed(full_seed,out,n=64):
    s=json.load(open(full_seed)); rows=[r for r in s['records'] if r['split'] in ('FIT','TUNE')]
    rows=sorted(rows,key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest())[:n]
    x={'schema':'RealSaS.IRISControlledV1.CeilingSeed.FastV2.v1','source_seed_sha256':sha(full_seed),'records':rows,'record_count':len(rows)}
    atomic_json(out,x); return out

def copy_provenance(local_cache,run_dir):
    dst=run_dir/'prep_provenance'; dst.mkdir(parents=True,exist_ok=True)
    for name in ('IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json','IRIS_CONTROLLED_V1_CACHE_SEAL_FAST_V2.json','FAST_PREP_PROGRESS.json'):
        p=local_cache/name
        if p.exists(): shutil.copy2(p,dst/name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--package-dir',default=None)
    ap.add_argument('--local-cache',default='/content/IRIS_CONTROLLED_V1_FAST_CACHE')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--mode',choices=['ceiling','prepare-full','train','all'],default='all')
    ap.add_argument('--epochs',type=int,default=24); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); root=Path(a.root); pkg=Path(a.package_dir) if a.package_dir else root/'reports'/'iris_controlled_v1'; cache=Path(a.local_cache)
    verify_base_package(pkg)
    fast_prep=pkg/'prepare_iris_controlled_v1_fast.py'
    if not fast_prep.exists() or sha(fast_prep)!=FAST_PREP_SHA:
        raise RuntimeError(f'fast prep patch missing/SHA mismatch: expected {FAST_PREP_SHA}')
    seed=root/'cache'/'IRIS_CONTROLLED_V1'/'IRIS_CONTROLLED_V1_MANIFEST_SEED.json'
    if not seed.exists():
        run([sys.executable,pkg/'build_seed_manifest.py','--root',root,'--split-freeze',pkg/'IRIS_CONTROLLED_V1_SPLIT_FREEZE.json','--out',seed])
    run_dir=root/'runs'/'IRIS_CONTROLLED_V1_FAST_V2'; run_dir.mkdir(parents=True,exist_ok=True)
    ceiling_seed=Path('/content/IRIS_CONTROLLED_V1_FAST_CEILING_SEED.json')
    ceiling_manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'
    ceiling_result=run_dir/'IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_FAST_V2.json'

    if a.mode in ('ceiling','all'):
        make_ceiling_seed(seed,ceiling_seed,64)
        run([sys.executable,fast_prep,'--root',root,'--seed-manifest',ceiling_seed,'--out',cache,'--splits','FIT,TUNE','--workers',a.workers])
        run([sys.executable,pkg/'representation_ceiling_v1.py','--cache-manifest',ceiling_manifest,'--out',ceiling_result,'--max-assets','64'])
        gate=json.load(open(ceiling_result))
        if not gate.get('pass'): raise RuntimeError('representation ceiling failed; full prep/training forbidden')
        print('[fast-launch] CEILING PASS; full cache authorized',flush=True)
        if a.mode=='ceiling': return

    if a.mode in ('prepare-full','all'):
        run([sys.executable,fast_prep,'--root',root,'--seed-manifest',seed,'--out',cache,'--splits','FIT,TUNE','--workers',a.workers])
        full_manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'
        m=json.load(open(full_manifest))
        if m.get('record_count')!=3248: raise RuntimeError(f'full open cache expected 3248, got {m.get("record_count")}')
        copy_provenance(cache,run_dir)
        print('[fast-launch] FULL PREP PASS record_count=3248',flush=True)
        if a.mode=='prepare-full': return

    if a.mode in ('train','all'):
        if not ceiling_result.exists() or not json.load(open(ceiling_result)).get('pass'):
            raise RuntimeError('saved ceiling PASS required before optimizer')
        full_manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'
        if not full_manifest.exists() or json.load(open(full_manifest)).get('record_count')!=3248:
            raise RuntimeError('full local cache missing; run --mode prepare-full')
        cmd=[sys.executable,pkg/'train_iris_controlled_v1.py','--cache-manifest',full_manifest,'--out',run_dir,'--epochs',a.epochs,'--input-resolution','256','--workers','2']
        if a.resume: cmd.append('--resume')
        run(cmd)
    print(json.dumps({'status':'DONE','mode':a.mode,'local_cache':str(cache),'run_dir':str(run_dir),'sealed_splits_opened':False},indent=2))

if __name__=='__main__': main()
