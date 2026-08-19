from __future__ import annotations
import argparse, importlib.util, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

SIGNED_T = 0.30
ABS_T = 0.20
LP = 0.5
LS = 0.75
K = 6
SMOOTH = 0.4  # retained from research lineage; V11 final uses w_seed, not smoothed ws
HERE = Path(__file__).resolve().parent

def loadmod(name, path):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v8=loadmod('v8_base_frozen',HERE/'v8_base_frozen.py')
v5=loadmod('v5_seed_geometry_frozen',HERE/'v5_seed_geometry_frozen.py')

def sha(p): return v8.sha256_file(Path(p))

def model_current(m,A,B,P,V,XY):
    cfg=m.SEESConfig(image_size=128,base_dim=64,token_dim=192,descriptor_dim=64,transformer_depth=4,transformer_heads=6,camera_residual_enabled=False,input_channels=4,max_delta=.35,correspondence_radius_f4=5,correspondence_temperature=.03,correspondence_position_lambda=.02,correspondence_gain_threshold=.10,correspondence_gain_width=.10)
    model=m.IRISSEESN1(cfg);ck=torch.load(m.CHECKPOINT,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'] if isinstance(ck,dict) and 'model' in ck else ck,strict=True);model.eval()
    with torch.no_grad(): out=model(m.load_model_tensor(A,B))
    cur=m.consensus_np(m.sample_field_np(out['delta_point_map_srcA'][0],XY),V)
    return cur,np.linalg.norm(cur,axis=1).astype(np.float64),out

def obs(m,H):
    rr=m.rank_asc(np.array([h[1] for h in H]));dr=m.rank_desc(np.array([h[2] for h in H]));return m.OBS_ALPHA*rr+(1-m.OBS_ALPHA)*dr

def linfit(m,P,D,wp):
    AB=np.stack([np.hstack([-m.crossmat(x),np.eye(3)]) for x in P]);ids=np.argsort(-wp)[:max(8,len(P)//2)];M=np.vstack([max(wp[i],.05)*AB[i] for i in ids]);y=np.hstack([D[i] for i in ids]);return np.linalg.lstsq(M,y,rcond=None)[0]

def solve_seed(m,P,pools,camp,diag,theta0,seedB):
    sc=float(np.quantile(camp,.95))+1e-9;prior=np.sqrt(np.clip(camp/sc,0,1));N=len(P)
    Dist=cdist(P,P);np.fill_diagonal(Dist,np.inf);nb=np.argsort(Dist,axis=1)[:,:K];loc=np.median(Dist[np.isfinite(Dist)]);gw=np.exp(-Dist[np.arange(N)[:,None],nb]/max(loc,1e-6));gw/=gw.sum(1,keepdims=True)
    ob=[obs(m,H) for H in pools];par=np.r_[theta0[:3],theta0[3:]].astype(float);labels=np.zeros(N,int);w=np.array(prior,float)
    def q_of(z):R=Rotation.from_rotvec(z[:3]).as_matrix();return P@(R-np.eye(3)).T+z[3:]
    for _ in range(9):
        q=q_of(par);new=[];nw=[]
        for i,H in enumerate(pools):
            D=np.stack([h[0] for h in H])-P[i];den=max(float(q[i]@q[i]),1e-12);wc=np.clip((D@q[i])/den,0,1);perp=np.linalg.norm(D-wc[:,None]*q[i],axis=1)/diag/m.MECH_SCALE;pc=LP*(wc-prior[i])**2;sm=LS*np.sum(gw[i][None,:]*(wc[:,None]-w[nb[i]][None,:])**2,axis=1);cost=perp+m.OBS_BETA*ob[i]+pc+sm;j=int(np.argmin(cost));new.append(j);nw.append(wc[j])
        labels=np.array(new);w=np.array(nw);Dsel=np.stack([pools[i][labels[i]][0]-P[i] for i in range(N)]);ids=np.argsort(-(0.5*prior+0.5*w))[:max(12,int(.65*N))];fitw=np.clip(0.25+0.75*np.maximum(prior[ids],w[ids]),.25,1)
        def fun(z):return ((Dsel[ids]-w[ids,None]*q_of(z)[ids])*np.sqrt(fitw[:,None])).ravel()
        par=least_squares(fun,par,max_nfev=120,xtol=1e-7,ftol=1e-7,gtol=1e-7).x
    q=q_of(par);den=np.sum(q*q,axis=1)+1e-12;wseed=np.clip(np.sum((seedB-P)*q,axis=1)/den,0,1);PB=(P+wseed[:,None]*q).astype(np.float32)
    Dsel=np.stack([pools[i][labels[i]][0]-P[i] for i in range(N)]);fit=np.linalg.norm(Dsel-w[:,None]*q,axis=1)/diag/m.MECH_SCALE;cand=float(np.average(fit+np.array([ob[i][labels[i]] for i in range(N)])+LP*(w-prior)**2,weights=np.maximum(prior,.02)));pair=float(np.mean([np.sum(gw[i]*(w[i]-w[nb[i]])**2) for i in range(N)]));q95=float(np.quantile(np.linalg.norm(q,axis=1),.95));s95=float(np.quantile(np.linalg.norm(seedB-P,axis=1),.95))+1e-9;ratio=q95/s95;scale=abs(np.log(max(ratio,1e-9)));score=cand+LS*pair+scale
    return PB,par,wseed,{'total':score,'candidate':cand,'pair':pair,'ratio':ratio,'scale':scale}

def seed_predict(A,B):
    m=v5;PA,VA,Y,oidx,hb,prb,diag=m.exact_problem_a(A,B)
    cfg=m.SEESConfig(image_size=128,base_dim=64,token_dim=192,descriptor_dim=64,transformer_depth=4,transformer_heads=6,camera_residual_enabled=False,input_channels=4,max_delta=.35,correspondence_radius_f4=5,correspondence_temperature=.03,correspondence_position_lambda=.02,correspondence_gain_threshold=.10,correspondence_gain_width=.10)
    model=m.IRISSEESN1(cfg);ck=torch.load(m.CHECKPOINT,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'] if isinstance(ck,dict) and 'model' in ck else ck,strict=True);model.eval()
    with torch.no_grad():out=model(m.load_model_tensor(A,B))
    xy=np.stack([m.project_points(PA,v) for v in range(8)]).astype(np.float32);z=m.descriptor_consensus(m.sample_field_np(out['descriptor'][0,0],xy),VA);cur=m.consensus_np(m.sample_field_np(out['delta_point_map_srcA'][0],xy),VA);camp=np.linalg.norm(cur,axis=1).astype(float);Bm=[m.foreground_mask(p) for p in B];bc=[m.prepare_b_search(out['descriptor'][0,1,v],Bm[v]) for v in range(8)];pools=[]
    for i in range(len(PA)):
        c=[];s=[]
        for vv in range(8):
            if VA[vv,i]:q,ss=m.top4_with_scores(out['descriptor'][0,1,vv],z[i],Bm[vv],bc[vv]);c.append(q);s.append(ss)
            else:c.append(np.empty((0,2),np.float32));s.append(np.empty(0,np.float32))
        pools.append(m.candidate_pool(c,s,PA[i],Y[i]))
    P=PA[oidx];V=VA[:,oidx];seed=Y[oidx];xyo=xy[:,oidx];co=camp[oidx];pools=[pools[int(i)] for i in oidx]
    wp=np.sqrt(np.clip(co/(np.quantile(co,.95)+1e-9),0,1));theta=linfit(m,P,seed-P,wp);PB,par,wseed,u=solve_seed(m,P,pools,co,diag,theta,seed)
    VB=m.exact_b_visibility(hb,prb,PB,diag);xyB=np.stack([m.project_points(PB,v) for v in range(8)]).astype(np.float32);NA=m.consensus_normals(m.sample_field_np(out['normal_A'][0,0],xyo),V);NB=m.consensus_normals(m.sample_field_np(out['normal_B'][0,1],xyB),VB)
    pred={'P_A':P.astype(np.float32),'P_B':PB,'N_A':NA,'N_B':NB,'V_A':V,'V_B':VB,'XY_A':xyo.astype(np.float32),'XY_B':xyB,'U_obs_seed_weight':wseed.astype(np.float32)}
    return pred,par,u,wseed

def run(root,f,e,out):
    root=Path(root);A=sorted((root/str(f)/'A').glob('*.png'));B=sorted((root/str(f)/e/'B').glob('*.png'))
    if len(A)!=8 or len(B)!=8:raise RuntimeError((f,e,len(A),len(B)))
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);base=td/f'{f}_{e}.npz'
        subprocess.run([sys.executable,str(HERE/'v8_base_frozen.py'),'--family',str(f),'--episode',e,'--root',str(root),'--out',str(td)],check=True,stdout=subprocess.DEVNULL)
        z=np.load(base);P=np.array(z['P_A'],np.float32);V=np.array(z['V_A'],np.uint8);XY=np.array(z['XY_A'],np.float32);PB0=np.array(z['P_B'],np.float32)
        cur,camp,_=model_current(v8,A,B,P,V,XY);D=PB0-P;a=np.linalg.norm(D,axis=1);b=np.linalg.norm(cur,axis=1);valid=(a>1e-6)&(b>1e-6);co=np.sum(D*cur,axis=1)/np.maximum(a*b,1e-9);w=np.maximum(camp,1e-4);signed=float(np.average(co[valid],weights=w[valid])) if valid.any() else 0.;absolute=float(np.average(np.abs(co[valid]),weights=w[valid])) if valid.any() else 0.;switch=bool(signed<SIGNED_T and absolute>=ABS_T)
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        if not switch:
            shutil.copy2(base,out);route='V8_BASE';extra={'source_v8_prediction_sha256':sha(base)}
        else:
            pred,par,u,wseed=seed_predict(A,B);np.savez_compressed(out,**pred);route='SEED_BASIN_WSEED';extra={'seed_solver_uobs':u,'par':par.tolist(),'w_nontrivial':int((wseed>.05).sum())}
    meta={'schema':'RealSaS.N1D.HybridV11.FrozenPrediction.v1','family':f,'episode':e,'route':route,'gate':{'signed':signed,'abs':absolute,'signed_threshold':SIGNED_T,'abs_threshold':ABS_T},'checkpoint_sha256':v8.CHECKPOINT_EXPECTED,'prediction_sha256':sha(out),'raster_sha256':{str(p):sha(p) for p in A+B},'truth_access':'NONE',**extra};out.with_suffix('.json').write_text(json.dumps(meta,indent=2,sort_keys=True,default=float));return meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.root,a.family,a.episode,a.out),sort_keys=True,default=float))
if __name__=='__main__':main()
