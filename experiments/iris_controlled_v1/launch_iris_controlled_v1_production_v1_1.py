from __future__ import annotations
import argparse,json,hashlib,subprocess,sys,shutil
from pathlib import Path

BASE_MANIFEST_SHA='e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098'
FAST_PREP_SHA='8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7'
TRAINER_V1_1_SHA='961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.check_call([str(x) for x in cmd])

def verify_base(pkg):
    mp=pkg/'PACKAGE_MANIFEST_V1.json'
    if sha(mp)!=BASE_MANIFEST_SHA: raise RuntimeError('base package manifest SHA mismatch')
    m=json.load(open(mp)); bad=[]
    for r in m['files']:
        p=pkg/r['path']; got=sha(p) if p.exists() else None
        if got!=r['sha256']: bad.append((r['path'],r['sha256'],got))
    if bad: raise RuntimeError('base package verification failed: '+repr(bad[:3]))
    fp=pkg/'prepare_iris_controlled_v1_fast.py'
    if not fp.exists() or sha(fp)!=FAST_PREP_SHA: raise RuntimeError('fast prep SHA mismatch')
    tr=pkg/'train_iris_controlled_v1_v1_1.py'
    if not tr.exists() or sha(tr)!=TRAINER_V1_1_SHA: raise RuntimeError('trainer v1.1 SHA mismatch')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--package-dir',default=None)
    ap.add_argument('--local-cache',default='/content/IRIS_CONTROLLED_V1_FAST_CACHE')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=24)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--mode',choices=['prepare-full','train','all'],default='all')
    a=ap.parse_args(); root=Path(a.root); pkg=Path(a.package_dir) if a.package_dir else root/'reports'/'iris_controlled_v1'; cache=Path(a.local_cache)
    verify_base(pkg)
    ceiling_result=root/'runs'/'IRIS_CONTROLLED_V1_FAST_V2'/'IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_FAST_V2.json'
    if not ceiling_result.exists() or not json.load(open(ceiling_result)).get('pass'): raise RuntimeError('recorded 64-asset representation ceiling PASS required')
    seed=root/'cache'/'IRIS_CONTROLLED_V1'/'IRIS_CONTROLLED_V1_MANIFEST_SEED.json'
    if not seed.exists(): raise RuntimeError('seed manifest missing')
    fast_prep=pkg/'prepare_iris_controlled_v1_fast.py'; full_manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'
    run_dir=root/'runs'/'IRIS_CONTROLLED_V1_PRODUCTION_V1_1'; run_dir.mkdir(parents=True,exist_ok=True)
    if a.mode in ('prepare-full','all'):
        run([sys.executable,fast_prep,'--root',root,'--seed-manifest',seed,'--out',cache,'--splits','FIT,TUNE','--workers',str(a.workers)])
        m=json.load(open(full_manifest))
        if m.get('record_count')!=3248: raise RuntimeError(f'expected 3248 open records, got {m.get("record_count")}')
        seal=cache/'IRIS_CONTROLLED_V1_CACHE_SEAL_FAST_V2.json'
        if not seal.exists(): raise RuntimeError('full cache seal missing')
        prov=run_dir/'prep_provenance'; prov.mkdir(parents=True,exist_ok=True)
        for p in (full_manifest,seal,cache/'FAST_PREP_PROGRESS.json'):
            if p.exists(): shutil.copy2(p,prov/p.name)
        print('[production] FULL OPEN CACHE PASS 3248',flush=True)
        if a.mode=='prepare-full': return
    if a.mode in ('train','all'):
        if not full_manifest.exists(): raise RuntimeError('full cache manifest missing')
        m=json.load(open(full_manifest))
        if m.get('record_count')!=3248: raise RuntimeError('full cache incomplete')
        trainer=pkg/'train_iris_controlled_v1_v1_1.py'
        cmd=[sys.executable,trainer,'--cache-manifest',full_manifest,'--out',run_dir/'training','--epochs',str(a.epochs),'--input-resolution','256','--workers','2']
        if a.resume: cmd.append('--resume')
        run(cmd)
    print(json.dumps({'status':'DONE','mode':a.mode,'production_trainer':'v1_1','representation_ceiling_pass':True,'open_pilot_pass':True,'sealed_splits_opened':False},indent=2))
if __name__=='__main__': main()
