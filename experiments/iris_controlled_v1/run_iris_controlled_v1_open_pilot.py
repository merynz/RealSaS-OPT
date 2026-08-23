from __future__ import annotations
import argparse,json,subprocess,sys,torch
from pathlib import Path
from iris_model import IRISControlledV1

def run(cmd): print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',default='/content/IRIS_CONTROLLED_V1_FAST_CACHE/IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'); ap.add_argument('--out',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_CONTROLLED_V1_OPEN_PILOT_V1'); ap.add_argument('--epochs',type=int,default=8); a=ap.parse_args()
    m=json.load(open(a.cache_manifest)); splits=sorted(set(r['split'] for r in m['records']))
    if any(s not in ('FIT','TUNE') for s in splits): raise RuntimeError(f'sealed split present in pilot manifest: {splits}')
    if len(m['records'])>96: raise RuntimeError(f'pilot expects small ceiling cache, got {len(m["records"])} records')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); torch.manual_seed(20260823); model=IRISControlledV1(); torch.save({'model':model.state_dict(),'epoch':-1},out/'init.pt')
    pkg=Path('/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1')
    run([sys.executable,pkg/'iris_controlled_v1_pilot_eval.py','--cache-manifest',a.cache_manifest,'--checkpoint',out/'init.pt','--out',out/'INIT_EVAL.json'])
    run([sys.executable,pkg/'train_iris_controlled_v1_v1_1.py','--cache-manifest',a.cache_manifest,'--out',out/'training','--epochs',str(a.epochs),'--workers','0'])
    run([sys.executable,pkg/'iris_controlled_v1_pilot_eval.py','--cache-manifest',a.cache_manifest,'--checkpoint',out/'training'/'last.pt','--out',out/'LAST_EVAL.json'])
    init=json.load(open(out/'INIT_EVAL.json')); last=json.load(open(out/'LAST_EVAL.json'))
    summary={'schema':'RealSaS.IRISControlledV1.OpenPilotSummary.v1','records':len(m['records']),'splits':splits,'epochs':a.epochs,'sealed_opened':False,'tune_p_euclid_init':init['TUNE']['losses']['p_euclid'],'tune_p_euclid_last':last['TUNE']['losses']['p_euclid'],'tune_n_error_init':init['TUNE']['losses']['n_cos'],'tune_n_error_last':last['TUNE']['losses']['n_cos'],'tune_coarse_top8_init':init['TUNE']['coarse_retrieval']['top8'],'tune_coarse_top8_last':last['TUNE']['coarse_retrieval']['top8'],'tune_fine_top8_init':init['TUNE']['fine_retrieval']['top8'],'tune_fine_top8_last':last['TUNE']['fine_retrieval']['top8']}
    summary['diagnostic_pass']=bool(summary['tune_p_euclid_last'] < summary['tune_p_euclid_init']*0.80 and summary['tune_coarse_top8_last'] > summary['tune_coarse_top8_init']+0.05 and summary['tune_fine_top8_last'] > summary['tune_fine_top8_init']+0.05)
    (out/'PILOT_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
