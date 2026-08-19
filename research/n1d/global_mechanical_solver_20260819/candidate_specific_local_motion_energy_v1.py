from __future__ import annotations
import sys,pickle,json,math,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/mnt/data/sufficiency')
import adaptive_h_v1 as a
import adaptive_fast_helpers as fh
m=a.m
F=a.FAMILIES; DIR=a.DIRNAME
FAMS=pickle.load(open('/mnt/data/sufficiency/adaptive_fams.pkl','rb'))
EPSPX=1e-3; IQREPS=1e-6; DIRTHR=1.0; PAIR_MARGIN=math.log(1.10); BINW=math.log(1.05)
OUTDIR=Path('/mnt/data/sufficiency/local_motion_energy_v1');OUTDIR.mkdir(exist_ok=True)

def bilinear_flow_samples(flow, xy):
    flow=np.asarray(flow,np.float64); xy=np.asarray(xy,np.float64);H,W=flow.shape[:2]
    x=np.clip(xy[:,0],0,W-1);y=np.clip(xy[:,1],0,H-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,W-1);y1=np.minimum(y0+1,H-1)
    wx=(x-x0)[:,None];wy=(y-y0)[:,None]
    return ((1-wx)*(1-wy)*flow[y0,x0]+wx*(1-wy)*flow[y0,x1]+(1-wx)*wy*flow[y1,x0]+wx*wy*flow[y1,x1]).astype(np.float64)

def candidate_H(fam,i):
    C=[]
    for v in range(8):
        r=fam['views'][i][v]
        C.append(np.empty((0,2),np.float32) if r is None else np.asarray(r['c8'])[:16])
    return fh.make_H_fast(C)

def amp_energy(H,PA,usable,obs):
    vals=[]
    for v in usable:
        q=fh.project_np(H,v)-fh.project_np(PA[None],v)[0]
        cm=np.linalg.norm(q,axis=1); om=float(np.linalg.norm(obs[v]))
        vals.append(np.abs(np.log(EPSPX+cm)-math.log(EPSPX+om)))
    return np.median(np.stack(vals,axis=1),axis=1) if len(vals)>=2 else np.full(len(H),np.inf)

def dir_energy(H,PA,usable,obs):
    vals=[]
    for v in usable:
        q=fh.project_np(H,v)-fh.project_np(PA[None],v)[0]
        qm=np.linalg.norm(q,axis=1); o=np.asarray(obs[v],float); om=float(np.linalg.norm(o))
        if om<DIRTHR: continue
        cos=(q@o)/(np.maximum(qm,EPSPX)*om)
        e=1.0-np.clip(cos,-1,1);e[qm<DIRTHR]=np.nan;vals.append(e)
    if not vals:return np.full(len(H),np.nan)
    A=np.stack(vals,axis=1);ok=np.isfinite(A).sum(1)>=2;out=np.full(len(H),np.nan)
    if ok.any(): out[ok]=np.nanmedian(A[ok],axis=1)
    return out

def robust_fusion(energies):
    zs=[]
    for E in energies:
        E=np.asarray(E,float); finite=np.isfinite(E)
        if finite.sum()<2: continue
        med=float(np.median(E[finite])); q25=float(np.quantile(E[finite],.25));q75=float(np.quantile(E[finite],.75));iqr=max(q75-q25,IQREPS)
        z=(E-med)/iqr; z[~finite]=np.nan;zs.append(z)
    if not zs:return np.full_like(energies[0],np.inf,dtype=float)
    A=np.stack(zs,axis=1);out=np.nanmedian(A,axis=1);out[~np.isfinite(out)]=np.inf;return out

def pair_order(E,amps,astar):
    n=len(amps)
    if n<2:return None
    idx=np.arange(n) if n<=91 else np.unique(np.rint(np.linspace(0,n-1,91)).astype(int))
    la=np.log(EPSPX+amps[idx]); oracle=abs(la-math.log(EPSPX+astar)); e=np.asarray(E)[idx];vals=[]
    for ii in range(len(idx)):
        for jj in range(ii+1,len(idx)):
            diff=oracle[ii]-oracle[jj]
            if abs(diff)<PAIR_MARGIN:continue
            de=e[ii]-e[jj]
            if not np.isfinite(de):continue
            vals.append(.5 if abs(de)<1e-12 else (1.0 if np.sign(de)==np.sign(diff) else 0.0))
    return float(np.mean(vals)) if vals else None

def bin_percentile(E,amps,astar):
    E=np.asarray(E,float);la=np.log(EPSPX+amps);lo=float(la.min());bins=np.floor((la-lo)/BINW).astype(int);ub=np.unique(bins)
    benergy={b:float(np.min(E[bins==b])) for b in ub if np.isfinite(E[bins==b]).any()}
    if not benergy:return None
    ob=int(math.floor((math.log(EPSPX+astar)-lo)/BINW))
    if ob not in benergy:ob=min(benergy,key=lambda b:abs(b-ob))
    oe=benergy[ob]; vals=np.array(list(benergy.values()),float)
    return float((np.sum(vals<oe)+0.5*np.sum(np.isclose(vals,oe,rtol=0,atol=1e-12)))/len(vals))

def summarize_records(records,arm):
    rr=[r for r in records if r.get('eval',False)]
    within25=[r[arm]['within25'] for r in rr];within50=[r[arm]['within50'] for r in rr];logerr=[r[arm]['abs_log_ratio'] for r in rr]
    pair=[r[arm]['pair_order'] for r in rr if r[arm]['pair_order'] is not None];pct=[r[arm]['oracle_bin_percentile'] for r in rr if r[arm]['oracle_bin_percentile'] is not None]
    return {'n':len(rr),'within25':float(np.mean(within25)) if rr else None,'within50':float(np.mean(within50)) if rr else None,'median_abs_log_ratio':float(np.median(logerr)) if rr else None,'mean_pairwise_order':float(np.mean(pair)) if pair else None,'median_oracle_bin_percentile':float(np.median(pct)) if pct else None}

def run_family(fid):
    fam=FAMS[fid];root=Path('/mnt/data/sufficiency/currentH')/DIR[fid];A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'))
    x=m.load_model_tensor(A,B)
    with torch.no_grad(): out=m.MODEL(x)
    PA,VA,_,_,_,_=m.exact_problem_a_frontdoor(A,B,n=64,res=96);xyA=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    flows=m.forward_dis_flows(A,B);dis=np.stack([bilinear_flow_samples(flows[v],xyA[v]) for v in range(8)],0)
    trans=m.sample_field_np(out['transport_offset_srcA'][0],xyA).astype(np.float64)*(255.0/31.0)
    d3=m.sample_field_np(out['delta_point_map_srcA'][0],xyA).astype(np.float64);delta2=np.empty((8,64,2),float)
    for v in range(8): delta2[v]=fh.project_np(PA+d3[v],v)-fh.project_np(PA,v)
    z=np.load(root/'B'/'observation_sidecar.npz');center=np.asarray(z['camera_center'],float);half=float(z['camera_half_extent']);TA=(np.asarray(z['surface_points_A'],float)-center)/(2*half);TB=(np.asarray(z['surface_points_B'],float)-center)/(2*half);tamp=np.linalg.norm(TB[fam['j']]-TA[fam['j']],axis=1);active=tamp>.005
    records=[];arms=['A_DIS','A_TRANSPORT','A_DELTA3D','A_ROBUST_FUSION']
    for i in range(64):
        H=candidate_H(fam,i);target=fam['TB'][fam['j'][i]];scale=float(fam['sB'][fam['j'][i]]);herr=np.linalg.norm(H-target[None],axis=1) if len(H) else np.array([]);contained=bool(len(H) and herr.min()<=2*scale);ev=bool(fam['reliable'][i] and active[i] and contained)
        rec={'i':i,'reliable':bool(fam['reliable'][i]),'active':bool(active[i]),'contained':contained,'eval':ev,'H_n':int(len(H))}
        if not ev:records.append(rec);continue
        hstar=H[int(herr.argmin())];astar=float(np.linalg.norm(hstar-PA[i]));amps=np.linalg.norm(H-PA[i][None],axis=1);usable=[v for v in range(8) if bool(VA[v,i])]
        Edis=amp_energy(H,PA[i],usable,dis[:,i]); Etr=amp_energy(H,PA[i],usable,trans[:,i]); Ed3=amp_energy(H,PA[i],usable,delta2[:,i]); Ef=robust_fusion([Edis,Etr,Ed3])
        Ddis=dir_energy(H,PA[i],usable,dis[:,i]);Dtr=dir_energy(H,PA[i],usable,trans[:,i]);Dd3=dir_energy(H,PA[i],usable,delta2[:,i])
        emap={'A_DIS':Edis,'A_TRANSPORT':Etr,'A_DELTA3D':Ed3,'A_ROBUST_FUSION':Ef};rec['astar']=astar
        for name,E in emap.items():
            k=int(np.nanargmin(E));asel=float(amps[k]);ratio=max(asel,astar)/max(min(asel,astar),EPSPX);rec[name]={'selected_amp':asel,'within25':bool(ratio<=1.25),'within50':bool(ratio<=1.5),'abs_log_ratio':float(abs(math.log((asel+EPSPX)/(astar+EPSPX)))),'pair_order':pair_order(E,amps,astar),'oracle_bin_percentile':bin_percentile(E,amps,astar)}
        hidx=int(herr.argmin());rec['direction_diag']={}
        for name,D in [('DIS',Ddis),('TRANSPORT',Dtr),('DELTA3D',Dd3)]:
            finite=np.isfinite(D)
            if finite.any():
                kk=int(np.nanargmin(D));rec['direction_diag'][name]={'oracle_dir_energy':float(D[hidx]) if np.isfinite(D[hidx]) else None,'selected_norm_endpoint_error':float(np.linalg.norm(H[kk]-target)/max(scale,1e-12))}
            else:rec['direction_diag'][name]={'oracle_dir_energy':None,'selected_norm_endpoint_error':None}
        records.append(rec)
    return {'family':fid,'n_eval':int(sum(r['eval'] for r in records)),'arms':{arm:summarize_records(records,arm) for arm in arms},'direction_diag':{name:{'median_oracle_dir_energy':float(np.median([r['direction_diag'][name]['oracle_dir_energy'] for r in records if r.get('eval') and r['direction_diag'][name]['oracle_dir_energy'] is not None])) if any(r.get('eval') and r['direction_diag'][name]['oracle_dir_energy'] is not None for r in records) else None,'median_selected_norm_endpoint_error':float(np.median([r['direction_diag'][name]['selected_norm_endpoint_error'] for r in records if r.get('eval') and r['direction_diag'][name]['selected_norm_endpoint_error'] is not None])) if any(r.get('eval') and r['direction_diag'][name]['selected_norm_endpoint_error'] is not None for r in records) else None} for name in ['DIS','TRANSPORT','DELTA3D']},'records':records}

def aggregate():
    famres={str(f):json.load(open(OUTDIR/f'{f}.json')) for f in F};arms=['A_DIS','A_TRANSPORT','A_DELTA3D','A_ROBUST_FUSION'];res={'schema':'RealSaS.N1D.CandidateSpecificLocalMotionEnergy.v1','date':'2026-08-19','prereg_commit':'6af5012a37673b5e1826b3d84304463927ff6e23','implementation_addendum_commit':'7a4d8ce8f4147f90cb054d079bd2c07be45df5b3','families':famres,'arms':{}}
    for arm in arms:
        per={f:famres[str(f)]['arms'][arm] for f in F};n=sum(q['n'] for q in per.values());w25=sum(q['within25']*q['n'] for q in per.values())/n;w50=sum(q['within50']*q['n'] for q in per.values())/n;pair=[q['mean_pairwise_order'] for q in per.values() if q['mean_pairwise_order'] is not None];mlog=np.median([r[arm]['abs_log_ratio'] for fr in famres.values() for r in fr['records'] if r.get('eval')])
        pooled={'n':n,'within25':float(w25),'within50':float(w50),'worst_family_within25':float(min(q['within25'] for q in per.values())),'worst_family_within50':float(min(q['within50'] for q in per.values())),'mean_family_pairwise_order':float(np.mean(pair)),'worst_family_pairwise_order':float(min(pair)),'median_abs_log_ratio':float(mlog),'hardtail_13203_within25':float(per[13203]['within25']),'hardtail_15290_within25':float(per[15290]['within25'])}
        gate=bool(pooled['within25']>=.70 and pooled['worst_family_within25']>=.55 and pooled['within50']>=.85 and pooled['worst_family_within50']>=.75 and pooled['mean_family_pairwise_order']>=.70 and pooled['worst_family_pairwise_order']>=.60 and pooled['median_abs_log_ratio']<=math.log(1.25) and pooled['hardtail_13203_within25']>=.55 and pooled['hardtail_15290_within25']>=.55)
        res['arms'][arm]={'per_family':per,'pooled':pooled,'gate':gate}
    passes=[x for x in arms if res['arms'][x]['gate']]
    if not passes:dec='NO_AMPLITUDE_ARM_PASSES__LOCAL_MOTION_FRONTDOOR_INSUFFICIENT_UNDER_V1'
    elif 'A_ROBUST_FUSION' in passes:
        f=res['arms']['A_ROBUST_FUSION']['pooled'];simple=[x for x in ['A_DIS','A_TRANSPORT','A_DELTA3D'] if x in passes]
        if simple:
            best=max(simple,key=lambda x:res['arms'][x]['pooled']['worst_family_within25']);b=res['arms'][best]['pooled'];dec='A_ROBUST_FUSION' if f['worst_family_within25']>=b['worst_family_within25']+.05 and f['worst_family_pairwise_order']>=b['worst_family_pairwise_order'] else best
        else:dec='A_ROBUST_FUSION'
    else:dec=passes[0]
    res['decision']=dec;p=OUTDIR/'AGGREGATE.json';p.write_text(json.dumps(res,indent=2,allow_nan=False));print(json.dumps({'decision':dec,'arms':{x:{'pooled':res['arms'][x]['pooled'],'gate':res['arms'][x]['gate'],'per_family_within25':{str(f):res['arms'][x]['per_family'][f]['within25'] for f in F}} for x in arms}},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',type=int);ap.add_argument('--aggregate',action='store_true');args=ap.parse_args()
    if args.aggregate:aggregate()
    else:
        r=run_family(args.family);p=OUTDIR/f'{args.family}.json';p.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({'family':args.family,'n_eval':r['n_eval'],'arms':r['arms'],'direction_diag':r['direction_diag']},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())
