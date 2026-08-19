from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import current_h_probe as m

ROOT=Path('/mnt/data/sufficiency/currentH')
FAMILIES=[9908,11032,12772,13203,14404,14702,14758,15290]
DIRNAME={9908:'09908',11032:'11032',12772:'12772',13203:'13203',14404:'14404',14702:'14702',14758:'14758',15290:'15290'}
T=0.05

def entropy_norm(scores):
    s=np.asarray(scores,dtype=np.float64)
    if len(s)<=1:return 0.0
    z=(s-s.max())/T; p=np.exp(z); p=p/p.sum()
    return float(-(p*np.log(np.maximum(p,1e-15))).sum()/np.log(len(s)))

def refined_top(desc_view,zA,mask,cache,coarse_k=8,max_final=16):
    coarse,zc=cache['coarse'],cache['zc']
    if len(coarse)==0:return np.empty((0,2),np.float32),np.empty(0,np.float32)
    score=zc@zA; seeds=coarse[np.argsort(-score,kind='stable')[:coarse_k]]; refined=[]
    for x,y in seeds:
        for dy in (-4,-2,0,2,4):
            for dx in (-4,-2,0,2,4):
                xx=int(round(float(x+dx))); yy=int(round(float(y+dy)))
                if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]: refined.append((xx,yy))
    if not refined:refined=[tuple(map(int,q)) for q in seeds]
    refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32)
    sr=m.normalize_rows(m.sample_desc_single_view(desc_view,refined))@zA
    ids=np.argsort(-sr,kind='stable')[:max_final]
    return refined[ids],sr[ids].astype(np.float32)

def make_H(cands):
    usable=[v for v in range(8) if len(cands[v])]; blocks=[]
    for ai in range(len(usable)):
      for bi in range(ai+1,len(usable)):
        v0,v1=usable[ai],usable[bi]; c0=np.asarray(cands[v0],np.float64); c1=np.asarray(cands[v1],np.float64)
        rr0=m.CAM_R[v0].cpu().numpy().astype(np.float64); uu0=m.CAM_U[v0].cpu().numpy().astype(np.float64)
        rr1=m.CAM_R[v1].cpu().numpy().astype(np.float64); uu1=m.CAM_U[v1].cpu().numpy().astype(np.float64)
        A=np.stack([rr0,uu0,rr1,uu1],0)
        if np.linalg.matrix_rank(A)<3: continue
        pinv=np.linalg.pinv(A)
        sx0=c0[:,0]/(m.IMAGE_NATIVE-1)-.5; sy0=.5-c0[:,1]/(m.IMAGE_NATIVE-1)
        sx1=c1[:,0]/(m.IMAGE_NATIVE-1)-.5; sy1=.5-c1[:,1]/(m.IMAGE_NATIVE-1)
        n0,n1=len(c0),len(c1); bb=np.empty((n0*n1,4),np.float64)
        bb[:,0]=np.repeat(sx0,n1); bb[:,1]=np.repeat(sy0,n1); bb[:,2]=np.tile(sx1,n0); bb[:,3]=np.tile(sy1,n0)
        P=bb@pinv.T; ok=np.isfinite(P).all(1)&(np.abs(P)<=.75).all(1)
        if ok.any():blocks.append(P[ok].astype(np.float32))
    return np.concatenate(blocks,0) if blocks else np.empty((0,3),np.float32)

def solver_diag(H,cands):
    usable=[v for v in range(8) if len(cands[v])]
    if len(H)==0 or len(usable)<2:return {'mean_reproj_px':float('inf'),'pair_hypotheses':int(len(H)),'usable_views':len(usable),'H_empty':True}
    means=np.zeros(len(H),np.float64)
    for v in usable:
        q=m.project_points(H,v); C=np.asarray(cands[v],np.float32)
        means += np.linalg.norm(q[:,None,:]-C[None,:,:],axis=2).min(1)
    means/=len(usable)
    return {'mean_reproj_px':float(means.min()),'pair_hypotheses':int(len(H)),'usable_views':len(usable),'H_empty':False}

def prep_family(fid):
    root=ROOT/DIRNAME[fid]; A=sorted((root/'A').glob('*.png')); B=sorted((root/'B').glob('*.png')); assert len(A)==8 and len(B)==8
    x=m.load_model_tensor(A,B)
    with torch.no_grad():out=m.MODEL(x)
    PA,VA,_,_,_,pa_diag=m.exact_problem_a_frontdoor(A,B,n=64,res=96)
    xyA=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    zA=m.descriptor_consensus(m.sample_field_np(out['descriptor'][0,0],xyA),VA)
    masks=[m.global_foreground_mask(p) for p in B]; cache=[m.prepare_b_search(out['descriptor'][0,1,v],masks[v]) for v in range(8)]
    views=[]; obs_metrics=[]
    for i in range(64):
        vv=[]
        for v in range(8):
            if not bool(VA[v,i]):vv.append(None);continue
            c8,s8=refined_top(out['descriptor'][0,1,v],zA[i],masks[v],cache[v],8,16)
            c32,s32=refined_top(out['descriptor'][0,1,v],zA[i],masks[v],cache[v],32,16)
            n=len(s8); margin4=float(s8[3]-s8[4]) if n>=5 else float('inf'); ent=entropy_norm(s8)
            vv.append({'c8':c8,'s8':s8,'c32':c32,'s32':s32,'margin4':margin4,'entropy16':ent}); obs_metrics.append((margin4,ent))
        views.append(vv)
    z=np.load(root/'B'/'observation_sidecar.npz'); center=np.asarray(z['camera_center'],np.float64); half=float(z['camera_half_extent'])
    TA=(np.asarray(z['surface_points_A'],np.float64)-center)/(2*half); TB=(np.asarray(z['surface_points_B'],np.float64)-center)/(2*half)
    sA=m.local_scale(TA); sB=m.local_scale(TB); D=np.linalg.norm(PA[:,None,:]-TA[None,:,:],axis=2); j=D.argmin(1); map_err=D[np.arange(len(PA)),j]; reliable=map_err<=2*sA[j]
    return {'fid':fid,'views':views,'obs_metrics':obs_metrics,'TB':TB,'sB':sB,'j':j,'reliable':reliable,'pa_diag':pa_diag}

def p_unc_margin(x,tr):
    a=np.asarray([v for v in tr if np.isfinite(v)],np.float64)
    return 0.0 if len(a)==0 or not np.isfinite(x) else float(np.mean(a>=x))
def p_unc_ent(x,tr):
    a=np.asarray([v for v in tr if np.isfinite(v)],np.float64)
    return 0.0 if len(a)==0 or not np.isfinite(x) else float(np.mean(a<=x))
def tier(u):return 4 if u<.50 else (8 if u<.80 else 16)
def up(k):return 8 if k<=4 else 16

def policy_ks(fam,i,tm,te):
    out=[]
    for v in range(8):
        r=fam['views'][i][v]
        if r is None:out.append(0);continue
        out.append(tier(max(p_unc_margin(r['margin4'],tm),p_unc_ent(r['entropy16'],te))))
    return out

def cands_for(fam,i,ks,coarse32=False):
    out=[]
    for v in range(8):
        r=fam['views'][i][v]
        if r is None:out.append(np.empty((0,2),np.float32));continue
        src=r['c32'] if coarse32 else r['c8']; out.append(src[:min(int(ks[v]),len(src))])
    return out

def eval_state(fam,i,ks,coarse32=False):
    c=cands_for(fam,i,ks,coarse32); H=make_H(c); dg=solver_diag(H,c); j=int(fam['j'][i]); target=fam['TB'][j]; scale=float(fam['sB'][j]); herr=float(np.linalg.norm(H-target[None],axis=1).min()) if len(H) else float('inf')
    return H,dg,herr,bool(herr<=scale),bool(herr<=2*scale)
def rec_from_state(i,ks,coarse32,st):
    H,dg,herr,c1,c2=st;return {'i':i,'ks':ks,'coarse32':coarse32,'contain_1x':c1,'contain_2x':c2,'herr':herr,'H_n':len(H),'diag':dg}

def fixed(fam,k):
    R=[]
    for i in range(64):
        ks=[k if fam['views'][i][v] is not None else 0 for v in range(8)];R.append(rec_from_state(i,ks,False,eval_state(fam,i,ks,False)))
    return R
def a1(fam,tm,te):
    R=[]
    for i in range(64):
        ks=policy_ks(fam,i,tm,te);R.append(rec_from_state(i,ks,False,eval_state(fam,i,ks,False)))
    return R
def geomcal(train,tm,te):
    vals=[]
    for fa in train:
        for r in a1(fa,tm,te):vals.append((r['diag']['mean_reproj_px'],r['diag']['pair_hypotheses']))
    repro=np.asarray([x for x,_ in vals if np.isfinite(x)],np.float64); pairs=np.asarray([y for _,y in vals],np.float64)
    return {'reproj_q75':float(np.quantile(repro,.75)),'reproj_q95':float(np.quantile(repro,.95)),'pair_q25':float(np.quantile(pairs,.25))}
def a2(fam,tm,te,g):
    R=[]
    for i in range(64):
        ks=policy_ks(fam,i,tm,te);coarse=False;st=eval_state(fam,i,ks,False)
        for _ in range(2):
            d=st[1]; trig=d['mean_reproj_px']>g['reproj_q75'] or d['usable_views']<3 or d['pair_hypotheses']<g['pair_q25'] or d['H_empty']
            if not trig:break
            nk=[up(k) if k else 0 for k in ks]
            if nk==ks:break
            ks=nk;st=eval_state(fam,i,ks,False)
        d=st[1]
        if all(k==0 or k>=16 for k in ks) and (d['H_empty'] or d['mean_reproj_px']>g['reproj_q95']):coarse=True;st=eval_state(fam,i,ks,True)
        R.append(rec_from_state(i,ks,coarse,st))
    return R

def summary(R,fam):
    rr=[r for r in R if fam['reliable'][r['i']]]; ks=[k for r in R for k in r['ks'] if k>0]
    return {'den':len(rr),'contain_1x':float(np.mean([r['contain_1x'] for r in rr])),'contain_2x':float(np.mean([r['contain_2x'] for r in rr])),'mean_K':float(np.mean(ks)),'median_K':float(np.median(ks)),'frac_K16':float(np.mean(np.asarray(ks)==16)),'coarse32_carrier_fraction':float(np.mean([r['coarse32'] for r in R])),'mean_H_n':float(np.mean([r['H_n'] for r in R])),'median_H_n':float(np.median([r['H_n'] for r in R])),'records':R}
def pooled(pf):
    den=sum(x['den'] for x in pf.values()); c2=sum(x['contain_2x']*x['den'] for x in pf.values())/den;c1=sum(x['contain_1x']*x['den'] for x in pf.values())/den;v=[x['contain_2x'] for x in pf.values()]
    return {'den':den,'contain_2x':float(c2),'contain_1x':float(c1),'worst_family':float(min(v)),'best_family':float(max(v)),'best_worst_gap_pp':float((max(v)-min(v))*100),'families_ge_0_90':int(sum(x>=.90 for x in v))}

def main():
    t=time.time();fams={}
    for f in FAMILIES:print('[PREP]',f,flush=True);fams[f]=prep_family(f)
    result={'schema':'RealSaS.N1D.AdaptiveHypothesisRetention.v1','date':'2026-08-19','families':FAMILIES,'arms':{}}
    for an,k in [('F4',4),('F8',8),('F16',16)]:
        pf={}
        for f in FAMILIES:print('[ARM]',an,f,flush=True);pf[str(f)]=summary(fixed(fams[f],k),fams[f])
        result['arms'][an]={'per_family':pf,'pooled':pooled(pf)}
    for an in ['A1','A2']:
        pf={}
        for held in FAMILIES:
            train=[fams[f] for f in FAMILIES if f!=held];tm=[x[0] for fa in train for x in fa['obs_metrics']];te=[x[1] for fa in train for x in fa['obs_metrics']]
            if an=='A1':print('[ARM]',an,held,flush=True);R=a1(fams[held],tm,te);extra={}
            else:g=geomcal(train,tm,te);print('[ARM]',an,held,g,flush=True);R=a2(fams[held],tm,te,g);extra={'geometry_calibration':g}
            s=summary(R,fams[held]);s.update(extra);pf[str(held)]=s
        result['arms'][an]={'per_family':pf,'pooled':pooled(pf)}
    for an,a in result['arms'].items():
        allk=[];co=[];hn=[]
        for f in FAMILIES:
            for r in a['per_family'][str(f)]['records']:
                allk.extend([k for k in r['ks'] if k>0]);co.append(r['coarse32']);hn.append(r['H_n'])
        a['efficiency']={'mean_K':float(np.mean(allk)),'median_K':float(np.median(allk)),'frac_K16':float(np.mean(np.asarray(allk)==16)),'coarse32_carrier_fraction':float(np.mean(co)),'mean_H_n':float(np.mean(hn)),'median_H_n':float(np.median(hn))}
    f16h=result['arms']['F16']['efficiency']['mean_H_n']
    for a in result['arms'].values():a['efficiency']['mean_H_ratio_to_F16']=float(a['efficiency']['mean_H_n']/f16h)
    gates={}
    for an,a in result['arms'].items():
        p=a['pooled'];e=a['efficiency'];cov=p['contain_2x']>=.975 and p['worst_family']>=.94 and p['families_ge_0_90']==8 and p['best_worst_gap_pp']<=7.0;eff=e['mean_K']<=10 and e['median_K']<=8 and e['frac_K16']<=.35 and e['coarse32_carrier_fraction']<=.10;gates[an]={'coverage_pass':bool(cov),'efficiency_pass':bool(eff),'both_pass':bool(cov and eff)}
    result['gates']=gates;decision='NO_COMPACT_ADAPTIVE_CONTRACT_PASSES__KEEP_F16_REFERENCE'
    if gates['F8']['both_pass']:
        decision='F8_FIXED_K8';f8=result['arms']['F8']
        for an in ['A1','A2']:
            aa=result['arms'][an]
            if gates[an]['both_pass'] and aa['efficiency']['mean_K']<=f8['efficiency']['mean_K']-1 and f8['pooled']['contain_2x']-aa['pooled']['contain_2x']<=.005 and f8['pooled']['worst_family']-aa['pooled']['worst_family']<=.01:decision=an+'_ADAPTIVE';break
    elif gates['A1']['both_pass']:decision='A1_ADAPTIVE'
    elif gates['A2']['both_pass']:decision='A2_ADAPTIVE'
    result['decision']=decision;result['seconds']=time.time()-t
    Path('/mnt/data/sufficiency/ADAPTIVE_HYPOTHESIS_RETENTION_V1_RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'decision':decision,'seconds':result['seconds'],'gates':gates,'arms':{k:{'pooled':v['pooled'],'efficiency':v['efficiency']} for k,v in result['arms'].items()}},indent=2),flush=True)
if __name__=='__main__':main()
