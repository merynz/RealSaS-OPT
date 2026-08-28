from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from iris_model import IRISControlledV1
from iris_evidence_matcher_v1 import MatcherConfig,IRISEvidenceMatcherV1,build_view_evidence,calibrate_config,calibrate_singleton_margin


def seed_int(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)&0x7fffffff

def load_outputs(model,row,style,device):
    z=np.load(row['cache_path'],allow_pickle=False)
    images=torch.from_numpy(z['images'][style].astype(np.float32)/255.).permute(0,3,1,2)[None].to(device)
    yaw=torch.from_numpy(z['yaw_deg'].astype(np.float32))[None].to(device)
    with torch.no_grad(): out=model(images,yaw)
    cache={k:z[k] for k in ('track_visible','track_xy')}
    return out,cache

def build_asset_views(model,row,style,device):
    out,cache=load_outputs(model,row,style,device); return build_view_evidence(out,cache)

def sample_queries(views,asset_id,max_queries):
    qs=[]
    for sv,s in views.items():
        for si,tid in enumerate(s.track_ids):
            for tv,t in views.items():
                if tv==sv: continue
                truth=np.flatnonzero(t.track_ids==tid)
                if len(truth): qs.append((sv,si,tv,int(truth[0]),int(tid)))
    if len(qs)<=max_queries: return qs
    rng=np.random.default_rng(seed_int(asset_id+'|matcher_eval')); ids=np.sort(rng.choice(len(qs),size=max_queries,replace=False)); return [qs[i] for i in ids]

def eval_views(views,cfg,asset_id,max_queries=256):
    m=IRISEvidenceMatcherV1(views,cfg); rows=[]
    for sv,si,tv,truth,tid in sample_queries(views,asset_id,max_queries):
        base=m.base_rank(sv,si,tv,8); base_ids=[x['idx'] for x in base]
        rr=m.match(sv,si,tv); ids=[x['idx'] for x in rr['ranked']]
        def rank(lst): return (lst.index(truth)+1) if truth in lst else 999
        r0=rank(base_ids); r=rank(ids); ss=rr['set_size']; top=rr['ranked'][0] if rr['ranked'] else None
        rows.append({'base_rank':r0,'rank':r,'conf':rr['confident_singleton'],'set_hit':truth in ids[:ss], 'set_size':ss,
                     'top_recip':bool(top and top['reciprocal']),'top_cycle':int(top['cycle_support']) if top else 0})
    return rows

def aggregate(rows):
    n=max(len(rows),1); conf=[r for r in rows if r['conf']]
    return {
      'queries':len(rows),
      'base_top1':sum(r['base_rank']<=1 for r in rows)/n,'base_top4':sum(r['base_rank']<=4 for r in rows)/n,'base_top8':sum(r['base_rank']<=8 for r in rows)/n,
      'final_top1':sum(r['rank']<=1 for r in rows)/n,'final_top4':sum(r['rank']<=4 for r in rows)/n,'final_top8':sum(r['rank']<=8 for r in rows)/n,
      'mrr':sum((1/r['rank'] if r['rank']<999 else 0) for r in rows)/n,
      'confident_singleton_coverage':len(conf)/n,
      'confident_singleton_precision':(sum(r['rank']==1 for r in conf)/len(conf)) if conf else 0.0,
      'adaptive_set_coverage':sum(r['set_hit'] for r in rows)/n,
      'adaptive_set_mean_size':sum(r['set_size'] for r in rows)/n,
      'top1_reciprocal_fraction':sum(r['top_recip'] for r in rows)/n,
      'top1_cycle_support_mean':sum(r['top_cycle'] for r in rows)/n,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--cal-assets',type=int,default=8); ap.add_argument('--max-queries-per-asset',type=int,default=256); a=ap.parse_args()
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=device); model=IRISControlledV1().to(device); model.load_state_dict(ck['model']); model.eval()
    man=json.load(open(a.cache_manifest)); fit=sorted([r for r in man['records'] if r['split']=='FIT'],key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest()); tune=sorted([r for r in man['records'] if r['split']=='TUNE'],key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest())
    cal=[]
    for r in fit[:a.cal_assets]: cal.append(build_asset_views(model,r,0,device))
    cfg=calibrate_config(MatcherConfig(),cal,max_queries=3000); cfg=calibrate_singleton_margin(cfg,cal,target_precision=.99,max_queries=3000)
    style_results={}; all_rows=[]
    for style,name in ((0,'cel_clean'),(1,'ink_cel')):
        rr=[]; per_asset={}
        for i,r in enumerate(tune):
            views=build_asset_views(model,r,style,device); x=eval_views(views,cfg,r['asset_id'],a.max_queries_per_asset); rr.extend(x); per_asset[r['asset_id']]=aggregate(x)
            if (i+1)%8==0 or i+1==len(tune): print(f'[matcher-eval] {name} {i+1}/{len(tune)}',flush=True)
        style_results[name]={'aggregate':aggregate(rr),'per_asset':per_asset}; all_rows.extend(rr)
    out={'schema':'RealSaS.IRISEvidenceMatcherV1.MiniEval.v1','checkpoint':a.checkpoint,'cache_manifest':a.cache_manifest,
         'fit_calibration_assets':[r['asset_id'] for r in fit[:a.cal_assets]],'tune_assets':len(tune),'config':cfg.__dict__,
         'styles':style_results,'combined':aggregate(all_rows),'sealed_opened':False,
         'authority_note':'GT physical track ids are used only by evaluator scoring. Matcher decisions use predicted P/U/Z, observed coordinates, reciprocal and image-coordinate cycle consistency; no hidden owner/track identity authority is consumed.'}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
