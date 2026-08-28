from __future__ import annotations
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np


def rank_metrics(qp,qn,cp,cn,true_idx,p_noise=0.0,n_noise_deg=0.0,use_n=False,seed=0):
    rng=np.random.default_rng(seed); qp=np.asarray(qp,np.float32).copy(); qn=np.asarray(qn,np.float32).copy()
    if p_noise>0: qp += rng.normal(0,p_noise,size=qp.shape).astype(np.float32)
    if n_noise_deg>0:
        qn += rng.normal(0,math.radians(n_noise_deg),size=qn.shape).astype(np.float32)
        qn/=np.maximum(np.linalg.norm(qn,axis=-1,keepdims=True),1e-8)
    d=np.linalg.norm(cp-qp[None],axis=-1)
    if use_n:
        cos=np.clip((cn*qn[None]).sum(-1),-1,1); score=d+0.05*(1-cos)
    else: score=d
    order=np.argsort(score,kind='stable'); rank=int(np.flatnonzero(order==true_idx)[0])+1
    return rank,float(d[true_idx])


def asset_eval(path,p_noise,n_noise,use_n,seed):
    z=np.load(path,allow_pickle=False); vis=z['track_visible'].astype(bool); pv=z['track_p_view'].astype(np.float32); nv=z['track_n_view'].astype(np.float32)
    ranks=[]; errs=[]; ambiguity=[]
    for t in range(len(vis)):
        vv=np.flatnonzero(vis[t]);
        if len(vv)<2: continue
        s=int(vv[0]); qP=pv[t,s]; qN=nv[t,s]
        for tv in vv[1:]:
            cand=np.flatnonzero(vis[:,tv]); true_pos=np.flatnonzero(cand==t)
            if not len(true_pos): continue
            cp=pv[cand,tv]; cn=nv[cand,tv]; r,e=rank_metrics(qP,qN,cp,cn,int(true_pos[0]),p_noise,n_noise,use_n,seed+t*31+int(tv))
            ranks.append(r); errs.append(e); dist=np.linalg.norm(cp-qP[None],axis=-1); ambiguity.append(int((dist<=0.003).sum()))
    if not ranks: return None
    a=np.asarray(ranks); amb=np.asarray(ambiguity)
    return {'pairs':int(len(a)),'top1':float(np.mean(a<=1)),'top4':float(np.mean(a<=4)),'top8':float(np.mean(a<=8)),
            'mrr':float(np.mean(1/a)),'true_surface_error_median':float(np.median(errs)),
            'ambiguous_within_0p003_fraction':float(np.mean(amb>1))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-assets',type=int,default=64)
    a=ap.parse_args(); m=json.load(open(a.cache_manifest)); rows=[r for r in m['records'] if r['split'] in ('FIT','TUNE')]
    rows=sorted(rows,key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest())[:a.max_assets]
    arms=[('P_EXACT',0,0,False),('PN_EXACT',0,0,True),('P_NOISE_0p001',.001,0,False),('P_NOISE_0p0025',.0025,0,False),('PN_P0p0025_N10',.0025,10,True),('PN_P0p005_N20',.005,20,True)]
    res={}
    for name,pn,nn,use_n in arms:
        vals=[]
        for i,r in enumerate(rows):
            x=asset_eval(r['cache_path'],pn,nn,use_n,20260823+i)
            if x: vals.append(x)
        pairs=sum(x['pairs'] for x in vals); agg={k:sum(x[k]*x['pairs'] for x in vals)/max(pairs,1) for k in ['top1','top4','top8','mrr','ambiguous_within_0p003_fraction']}; agg['assets']=len(vals); agg['pairs']=pairs; agg['family_top1_p10']=float(np.percentile([x['top1'] for x in vals],10)) if vals else 0.; res[name]=agg
    passed=res['P_EXACT']['top8']>=0.995 and res['PN_EXACT']['top8']>=0.995 and res['P_EXACT']['family_top1_p10']>=0.95
    out={'schema':'RealSaS.IRISControlledV1.RepresentationCeiling.v1','assets_requested':a.max_assets,'arms':res,
         'primary_gate':{'P_EXACT_top8_min':0.995,'PN_EXACT_top8_min':0.995,'P_EXACT_family_top1_p10_min':0.95},'pass':bool(passed),
         'interpretation':'PASS means the legal observable P/(P,N) state can address the sampled persistent surface loci under controlled exact-camera observations. It does not prove a learner can recover that state from RGB.'}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
    if not passed: raise RuntimeError('representation ceiling gate failed; do not train')
if __name__=='__main__': main()
