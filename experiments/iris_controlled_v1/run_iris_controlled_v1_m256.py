from __future__ import annotations
import argparse,json,hashlib,os,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from iris_model import IRISControlledV1

SEED=20260823
BASE_MANIFEST_SHA='e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098'
FAST_PREP_SHA='8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7'
TRAINER_V11_SHA='961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3'
BUILD_M256_SHA='0f6a923ea88b3cba49a30e08cd742b8577f34ec2c3173279db3374e2414aa610'
MATCHER_SHA='2ff2b6c37944fe99e23af16e5611d15871a299e193f85e6c54c7ceee42416c32'
EVAL_SHA='2ea571f70283d1b93b7ea1d18d74591a3a36df7e08ddb725ec3e7e27967c827b'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def run(cmd): print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])
def seed_all(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def verify(pkg):
    checks={
      'PACKAGE_MANIFEST_V1.json':BASE_MANIFEST_SHA,
      'prepare_iris_controlled_v1_fast.py':FAST_PREP_SHA,
      'train_iris_controlled_v1_v1_1.py':TRAINER_V11_SHA,
      'build_m256_seed.py':BUILD_M256_SHA,
      'iris_matcher_v1.py':MATCHER_SHA,
      'evaluate_iris_matcher_v1.py':EVAL_SHA,
    }
    bad=[]
    for name,expected in checks.items():
        p=pkg/name; got=sha(p) if p.exists() else None
        if got!=expected: bad.append({'file':name,'expected':expected,'got':got})
    if bad: raise RuntimeError('M256 authority verification failed: '+json.dumps(bad,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--package-dir',default=None); ap.add_argument('--cache-dir',default='/content/IRIS_CONTROLLED_V1_M256_CACHE')
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--epochs',type=int,default=12); a=ap.parse_args()
    root=Path(a.root); pkg=Path(a.package_dir) if a.package_dir else root/'reports'/'iris_controlled_v1'; cache=Path(a.cache_dir)
    verify(pkg)
    full_seed=root/'cache'/'IRIS_CONTROLLED_V1'/'IRIS_CONTROLLED_V1_MANIFEST_SEED.json'
    if not full_seed.exists():
        run([sys.executable,pkg/'build_seed_manifest.py','--root',root,'--split-freeze',pkg/'IRIS_CONTROLLED_V1_SPLIT_FREEZE.json','--out',full_seed])
    run_dir=root/'runs'/'IRIS_CONTROLLED_V1_M256_V1'; run_dir.mkdir(parents=True,exist_ok=True)
    mseed=run_dir/'IRIS_CONTROLLED_V1_M256_SEED.json'
    run([sys.executable,pkg/'build_m256_seed.py','--full-seed',full_seed,'--out',mseed])
    run([sys.executable,pkg/'prepare_iris_controlled_v1_fast.py','--root',root,'--seed-manifest',mseed,'--out',cache,'--splits','FIT,TUNE','--workers',a.workers])
    manifest=cache/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'
    mm=json.load(open(manifest)); counts={s:sum(r['split']==s for r in mm['records']) for s in ('FIT','TUNE')}
    if mm.get('record_count')!=256 or counts!={'FIT':208,'TUNE':48}: raise RuntimeError(f'M256 cache seal/count failure {mm.get("record_count")} {counts}')

    seed_all(); init=run_dir/'init.pt'; model=IRISControlledV1(); torch.save({'epoch':-1,'model':model.state_dict(),'contract':{'schema':'M256_INIT_WITNESS','seed':SEED}},init)
    init_eval=run_dir/'INIT_MATCHER_EVAL.json'
    run([sys.executable,pkg/'evaluate_iris_matcher_v1.py','--cache-manifest',manifest,'--checkpoint',init,'--out',init_eval,'--splits','TUNE'])

    train_dir=run_dir/'training'
    run([sys.executable,pkg/'train_iris_controlled_v1_v1_1.py','--cache-manifest',manifest,'--out',train_dir,'--epochs',a.epochs,'--workers','0','--seed',SEED])
    best=train_dir/'best.pt'
    if not best.exists(): raise RuntimeError('best checkpoint absent')
    best_eval=run_dir/'BEST_MATCHER_EVAL.json'
    run([sys.executable,pkg/'evaluate_iris_matcher_v1.py','--cache-manifest',manifest,'--checkpoint',best,'--out',best_eval,'--splits','TUNE'])

    I=json.load(open(init_eval))['splits']['TUNE']; B=json.load(open(best_eval))['splits']['TUNE']
    im=I['matcher']; bm=B['matcher']
    p0=I['losses']['p_euclid']; p1=B['losses']['p_euclid']; p_reduction=(p0-p1)/max(p0,1e-12)
    cg=bm['raw_coarse']['top8']-im['raw_coarse']['top8']; fg=bm['raw_fine']['top8']-im['raw_fine']['top8']
    learner={'p_error_reduction':p_reduction,'coarse_top8_gain':cg,'fine_top8_gain':fg,
             'pass':bool(p_reduction>=.50 and cg>=.40 and fg>=.40)}
    q=bm['qualification']; matcher={
        'truth_row_domain_recall':bm['truth_row_domain_recall'],'composite_top8':bm['composite']['top8'],
        'output_truth_coverage':q['output_truth_coverage'],'family_composite_top8_p10':bm['family_tail']['composite_top8_p10'],
        'singleton_coverage':q['singleton_coverage'],'singleton_precision':q['singleton_precision'],
    }
    matcher['pass']=bool(matcher['truth_row_domain_recall']>=.995 and matcher['composite_top8']>=.970 and matcher['output_truth_coverage']>=.970 and
                         matcher['family_composite_top8_p10']>=.900 and matcher['singleton_coverage']>=.100 and
                         matcher['singleton_precision'] is not None and matcher['singleton_precision']>=.950)
    if not learner['pass']: decision='LEARNER_FAIL'
    elif not matcher['pass']: decision='MATCHER_CONSUMER_FAIL_PARTIAL'
    else: decision='M256_PASS'
    summary={'schema':'RealSaS.IRISControlledV1.M256Decision.v1','date':'2026-08-23','records':256,'FIT':208,'TUNE':48,'pilot64_overlap':0,
             'epochs':a.epochs,'seed':SEED,'cache_manifest_sha256':sha(manifest),'best_checkpoint_sha256':sha(best),
             'init':{'p_euclid':p0,'raw_coarse':im['raw_coarse'],'raw_fine':im['raw_fine']},
             'best':{'p_euclid':p1,'n_error':B['losses']['n_cos'],'raw_coarse':bm['raw_coarse'],'raw_fine':bm['raw_fine'],'raw_P':bm['raw_P'],
                     'composite':bm['composite'],'reciprocal_top1_rate':bm['reciprocal_top1_rate'],'reciprocal_top4_rate':bm['reciprocal_top4_rate'],
                     'cycle_any_rate':bm['cycle_any_rate'],'qualification':q,'family_tail':bm['family_tail'],'hard_tail_count':bm['hard_tail_count']},
             'learner_gate':learner,'matcher_gate':matcher,'decision':decision,'sealed_opened':False,
             'interpretation':('Fresh M256 open gate. PASS supports scaling the current D3-free evidence architecture; it is not sealed/product proof.' if decision=='M256_PASS' else
                               'Failure is localized by learner_gate vs matcher_gate and is not an information-limit claim.')}
    atomic_json(run_dir/'M256_DECISION.json',summary); print(json.dumps(summary,indent=2),flush=True)
    if decision!='M256_PASS': raise SystemExit(2)
if __name__=='__main__': main()
