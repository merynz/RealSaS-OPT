from __future__ import annotations
import argparse,json,hashlib,subprocess,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])
def verify_package(pkg):
    mp=pkg/'PACKAGE_MANIFEST_V1.json'; m=json.load(open(mp))
    bad=[]
    for r in m['files']:
        p=pkg/r['path']; got=sha(p) if p.exists() else None
        if got!=r['sha256']: bad.append({'path':r['path'],'expected':r['sha256'],'got':got})
    if bad: raise RuntimeError('package SHA verification failed: '+json.dumps(bad[:5]))
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--package-dir',default=None); ap.add_argument('--mode',choices=['prepare','ceiling','train','all'],default='prepare')
    ap.add_argument('--epochs',type=int,default=24); ap.add_argument('--input-resolution',type=int,default=256); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    root=Path(a.root); pkg=Path(a.package_dir) if a.package_dir else root/'reports'/'iris_controlled_v1'; verify_package(pkg)
    sf=pkg/'IRIS_CONTROLLED_V1_SPLIT_FREEZE.json'; seed=root/'cache'/'IRIS_CONTROLLED_V1'/'IRIS_CONTROLLED_V1_MANIFEST_SEED.json'; cache=root/'cache'/'IRIS_CONTROLLED_V1'; runs=root/'runs'/'IRIS_CONTROLLED_V1'
    seed.parent.mkdir(parents=True,exist_ok=True); runs.mkdir(parents=True,exist_ok=True)
    if a.mode in ('prepare','all'):
        run([sys.executable,pkg/'build_seed_manifest.py','--root',root,'--split-freeze',sf,'--out',seed])
        run([sys.executable,pkg/'prepare_iris_controlled_v1.py','--root',root,'--seed-manifest',seed,'--out',cache,'--splits','FIT,TUNE'])
    manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST.json'
    if a.mode in ('ceiling','all'):
        if not manifest.exists(): raise RuntimeError('cache manifest absent; run --mode prepare first')
        gate=cache/'IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_V1.json'
        run([sys.executable,pkg/'representation_ceiling_v1.py','--cache-manifest',manifest,'--out',gate,'--max-assets','64'])
    if a.mode in ('train','all'):
        gate=cache/'IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_V1.json'
        if not gate.exists() or not json.load(open(gate)).get('pass'): raise RuntimeError('representation ceiling PASS required before optimizer')
        cmd=[sys.executable,pkg/'train_iris_controlled_v1.py','--cache-manifest',manifest,'--out',runs,'--epochs',a.epochs,'--input-resolution',a.input_resolution,'--workers',a.workers]
        if a.resume: cmd.append('--resume')
        run(cmd)
    print(json.dumps({'status':'DONE','mode':a.mode,'cache_manifest':str(manifest),'sealed_splits_opened':False},indent=2))
if __name__=='__main__': main()
