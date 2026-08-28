from __future__ import annotations
import argparse,json,hashlib,os,subprocess,sys,time
from pathlib import Path
import torch

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
PKG_REL='reports/iris_controlled_v1'
FAST_PREP_SHA='8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7'
TRAINER_SHA='961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3'
MATCHER_SHA='4c5b8c80ead2815e1bc9c512f2a5ec8669b5ee48ce22031ae46d083a577edc71'
MATCHER_EVAL_SHA='afb51ef36719512c46093201da94678ed0670b17032cf657482b299f9c788f91'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); os.replace(q,p)
def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])
def check(path,expected):
    g=sha(path)
    if g!=expected: raise RuntimeError(f'SHA mismatch {path}: {g} != {expected}')

def mini_seed(full_seed,out):
    s=json.load(open(full_seed)); open_rows=[r for r in s['records'] if r['split'] in ('FIT','TUNE')]
    ceiling=sorted(open_rows,key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest())[:64]
    chosen={r['asset_id']:r for r in ceiling}
    targets={'FIT':224,'TUNE':32}
    for split,target in targets.items():
        have=sum(r['split']==split for r in chosen.values())
        pool=sorted([r for r in open_rows if r['split']==split and r['asset_id'] not in chosen],key=lambda r:hashlib.sha256((r['asset_id']+'|mini256').encode()).hexdigest())
        for r in pool[:target-have]: chosen[r['asset_id']]=r
    rows=sorted(chosen.values(),key=lambda r:(r['split'],r['asset_id']))
    counts={k:sum(r['split']==k for r in rows) for k in targets}
    if len(rows)!=256 or counts!=targets: raise RuntimeError(f'mini seed wrong {len(rows)} {counts}')
    o={'schema':'RealSaS.IRISControlledV1.Mini256Seed.v1','date':'2026-08-23','source_seed_sha256':sha(full_seed),'record_count':256,'splits':counts,'contains_ceiling64':all(r['asset_id'] in chosen for r in ceiling),'records':rows}
    atomic_json(out,o); return o

def save_init(pkg,path):
    sys.path.insert(0,str(pkg)); from iris_model import IRISControlledV1
    torch.manual_seed(20260823); torch.cuda.manual_seed_all(20260823); m=IRISControlledV1()
    torch.save({'epoch':-1,'model':m.state_dict(),'best':float('inf'),'history':[],'contract':{'schema':'RealSaS.IRISControlledV1.Mini256Init.v1','seed':20260823}},path)

def classify(init_eval,best_eval,matcher):
    ip=init_eval['TUNE']['losses']['p_euclid']; bp=best_eval['TUNE']['losses']['p_euclid']; inn=init_eval['TUNE']['losses']['n_cos']; bnn=best_eval['TUNE']['losses']['n_cos']
    p_red=1-bp/max(ip,1e-12); n_red=1-bnn/max(inn,1e-12); m=matcher['combined']
    green=(p_red>=.50 and n_red>=.50 and m['final_top4']>=.90 and m['final_top8']>=.97 and m['adaptive_set_coverage']>=.98 and m['confident_singleton_precision']>=.97 and m['confident_singleton_coverage']>=.10)
    red=(p_red<.30 or n_red<.30 or m['final_top8']<.90 or m['adaptive_set_coverage']<.90)
    status='GREEN' if green else ('RED' if red else 'AMBER')
    return {'status':status,'p_error_reduction':p_red,'n_error_reduction':n_red,
      'criteria':{'GREEN':{'p_reduction_min':.50,'n_reduction_min':.50,'final_top4_min':.90,'final_top8_min':.97,'adaptive_set_coverage_min':.98,'singleton_precision_min':.97,'singleton_coverage_min':.10},
                  'RED_if':{'p_reduction_below':.30,'n_reduction_below':.30,'final_top8_below':.90,'adaptive_set_coverage_below':.90}},
      'matcher_combined':m}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=ROOT_DEFAULT); ap.add_argument('--workers',type=int,default=6); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--cache',default='/content/IRIS_CONTROLLED_V1_FAST_CACHE'); a=ap.parse_args()
    root=Path(a.root); pkg=root/PKG_REL; cache=Path(a.cache); run_dir=root/'runs'/'IRIS_CONTROLLED_V1_MINI256_V1'; run_dir.mkdir(parents=True,exist_ok=True)
    check(pkg/'prepare_iris_controlled_v1_fast.py',FAST_PREP_SHA); check(pkg/'train_iris_controlled_v1_v1_1.py',TRAINER_SHA); check(pkg/'iris_evidence_matcher_v1.py',MATCHER_SHA); check(pkg/'eval_iris_evidence_matcher_v1.py',MATCHER_EVAL_SHA)
    ceiling=root/'runs'/'IRIS_CONTROLLED_V1_FAST_V2'/'IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_FAST_V2.json'
    if not ceiling.exists() or not json.load(open(ceiling)).get('pass'): raise RuntimeError('64-asset representation ceiling PASS required')
    full_seed=root/'cache'/'IRIS_CONTROLLED_V1'/'IRIS_CONTROLLED_V1_MANIFEST_SEED.json'; seed=run_dir/'MINI256_SEED.json'; mini_seed(full_seed,seed)
    run([sys.executable,pkg/'prepare_iris_controlled_v1_fast.py','--root',root,'--seed-manifest',seed,'--out',cache,'--splits','FIT,TUNE','--workers',a.workers])
    manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'; man=json.load(open(manifest))
    if man.get('record_count')!=256: raise RuntimeError(f'expected mini256 manifest, got {man.get("record_count")}')
    init=run_dir/'init.pt'; save_init(pkg,init)
    init_eval=run_dir/'INIT_EVAL.json'; run([sys.executable,pkg/'iris_controlled_v1_pilot_eval.py','--cache-manifest',manifest,'--checkpoint',init,'--out',init_eval])
    train_dir=run_dir/'training'; run([sys.executable,pkg/'train_iris_controlled_v1_v1_1.py','--cache-manifest',manifest,'--out',train_dir,'--epochs',a.epochs,'--workers','0'])
    best=train_dir/'best.pt'; best_eval=run_dir/'BEST_EVAL.json'; run([sys.executable,pkg/'iris_controlled_v1_pilot_eval.py','--cache-manifest',manifest,'--checkpoint',best,'--out',best_eval])
    matcher_out=run_dir/'MATCHER_V1_EVAL.json'; run([sys.executable,pkg/'eval_iris_evidence_matcher_v1.py','--cache-manifest',manifest,'--checkpoint',best,'--out',matcher_out,'--cal-assets','8','--max-queries-per-asset','256'])
    ie=json.load(open(init_eval)); be=json.load(open(best_eval)); me=json.load(open(matcher_out)); dec=classify(ie,be,me)
    summary={'schema':'RealSaS.IRISControlledV1.Mini256Decision.v1','date':'2026-08-23','seed_sha256':sha(seed),'cache_manifest_sha256':sha(manifest),'best_checkpoint_sha256':sha(best),
      'records':256,'FIT':224,'TUNE':32,'epochs':a.epochs,'sealed_opened':False,'representation_ceiling_pass':True,
      'init_tune':ie['TUNE'],'best_tune':be['TUNE'],'matcher':me['combined'],'matcher_config':me['config'],'decision':dec,
      'next_policy':{'GREEN':'authorize 1024 intermediate before full 3248','AMBER':'do not scale; inspect matcher/component ablations','RED':'do not scale; localize learner/representation failure'}}
    atomic_json(run_dir/'MINI256_DECISION.json',summary); print('\n=== MINI256 DECISION ==='); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
