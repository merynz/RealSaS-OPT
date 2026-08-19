from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0,'/mnt/data/sufficiency')
import current_h_probe as m
import current_h_breadth_fast as bf
sys.path.insert(0,str(m.SRC))
from realsas_iris_sees.targets import build_observation_target_from_sidecar
from realsas_gfdr_v2 import compute_gfdr_v2


def unit(x,eps=1e-8):
    x=np.asarray(x,np.float64); return (x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)).astype(np.float32)

def consensus_np(view_values,V):
    w=np.asarray(V,np.float64)[...,None]
    return ((np.asarray(view_values,np.float64)*w).sum(0)/np.maximum(w.sum(0),1.)).astype(np.float32)

def consensus_normals(n_view,V): return unit(consensus_np(n_view,V))

def exact_visibility(h,proj,P,diag):
    V=np.zeros((8,len(P)),np.uint8)
    for i,p in enumerate(np.asarray(P,np.float32)):
        for v,_ in m.visible_views_problem_a(h,proj,torch.from_numpy(p),diag): V[v,i]=1
    return V

def spearman(a,b):
    a=np.asarray(a);b=np.asarray(b);q=np.isfinite(a)&np.isfinite(b)
    if q.sum()<3:return float('nan')
    r=spearmanr(a[q],b[q]).statistic
    return float(r) if np.isfinite(r) else float('nan')

def gfdr_compare(pred,true,target,pred_world):
    mask=target['persistent_obs'].astype(bool); amp=np.linalg.norm(target['scene_flow'],axis=1); active=mask&(amp>.005)
    diag=float(np.linalg.norm(target['P_A'].max(0)-target['P_A'].min(0)))+1e-8; idx=true['R_neighbor_idx']
    f_act=spearman(pred['F_activity'][active],true['F_activity'][active]) if active.sum()>=3 else float('nan')
    ids=np.where(active)[0]
    if len(ids)>=3:
        tri=np.triu_indices(len(ids),1);ii,jj=ids[tri[0]],ids[tri[1]];f_kernel=spearman(pred['F_coresponse_kernel'][ii,jj],true['F_coresponse_kernel'][ii,jj])
    else:f_kernel=float('nan')
    true_pa=target['P_A'].astype(np.float64);true_pb=target['P_B'].astype(np.float64);pred_pa=np.asarray(pred_world['P_A'],np.float64);pred_pb=np.asarray(pred_world['P_B'],np.float64);dp=pred_pb-pred_pa;dt=true_pb-true_pa;derr=[];rtol=float(np.finfo(np.float32).eps*max(idx.shape[1],3))
    for i in np.where(mask)[0]:
        ids2=idx[i];ids2=ids2[mask[ids2]]
        if len(ids2)<2:continue
        A=true_pa[ids2]-true_pa[i];U,ss,Vt=np.linalg.svd(A,full_matrices=False)
        if not len(ss) or ss[0]<=0:continue
        rr=int((ss[:2]>=rtol*ss[0]).sum())
        if rr<1:continue
        Cpinv=(U[:,:rr]/ss[:rr]).T;Gp=Cpinv@(dp[ids2]-dp[i]);Gt=Cpinv@(dt[ids2]-dt[i]);derr.append(float(np.linalg.norm(Gp-Gt)/(1+np.linalg.norm(Gt))))
    d_surface=float(np.median(derr)) if derr else float('nan')
    pd=pred['F_delta_normalized'];tt=true['F_delta_normalized'];rp=pd[idx]-pd[:,None,:];rt=tt[idx]-tt[:,None,:];edge=mask[:,None]&mask[idx];re=np.linalg.norm(rp-rt,axis=-1);rerr=float(np.mean(re[edge])) if edge.any() else float('nan')
    gids=np.where(mask&(true['G_support']>=.5))[0];gcount=len(gids)
    if gcount:
        pd=pred['G_axis_direction'][gids];tdv=true['G_axis_direction'][gids];pn=np.linalg.norm(pd,axis=1,keepdims=True);tn=np.linalg.norm(tdv,axis=1,keepdims=True);pdu=np.divide(pd,pn,out=np.zeros_like(pd),where=pn>1e-8);tdu=np.divide(tdv,tn,out=np.zeros_like(tdv),where=tn>1e-8);gcos=float(np.median(np.abs(np.sum(pdu*tdu,axis=1))))
        dpoint=pred['G_axis_point'][gids]-true['G_axis_point'][gids];line=np.linalg.norm(np.cross(dpoint,tdu),axis=1)/diag;gline=float(np.median(line[np.isfinite(line)])) if np.isfinite(line).any() else float('nan')
    else:gcos=gline=float('nan')
    vals=amp[mask];rms=float(np.sqrt(np.mean(vals*vals))) if len(vals) else 0.;moved=float(np.mean(vals>.005)) if len(vals) else 0.
    stratum='near_zero' if rms<=.005 or moved<=.05 else ('strong' if rms>=.03 or moved>=.25 else 'finite')
    return {'F_activity_spearman':f_act,'F_kernel_spearman':f_kernel,'D_tangent_action_error_median':d_surface,'R_differential_body_error':rerr,'G_axis_dir_abs_cos_median':gcos,'G_axis_line_error_diag_median':gline,'G_supported_carriers':gcount,'true_flow_rms':rms,'moved_fraction_005':moved,'stratum':stratum}

def map_truth_ids(PA,t):
    pers=np.asarray(t.get('persistent_obs',np.ones(len(t['P_A']),np.uint8))).astype(bool); eligible=np.where(pers)[0]
    if len(eligible)<len(PA):eligible=np.arange(len(t['P_A']))
    cost=cdist(PA,t['P_A'][eligible]);rr,cc=linear_sum_assignment(cost);order=np.argsort(rr);return eligible[cc[order]]

def run_family(root:Path,coarse_k=8,final_k=16):
    A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'));assert len(A)==8 and len(B)==8
    x=m.load_model_tensor(A,B)
    with torch.no_grad(): out=m.MODEL(x)
    PA,VA,B_hull,B_proj,diag,pa_diag=m.exact_problem_a_frontdoor(A,B,n=64,res=96)
    xyA=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    zA=m.descriptor_consensus(m.sample_field_np(out['descriptor'][0,0],xyA),VA)
    masks=[m.global_foreground_mask(p) for p in B];cache=[m.prepare_b_search(out['descriptor'][0,1,v],masks[v]) for v in range(8)]
    t=build_observation_target_from_sidecar(root/'B'/'observation_sidecar.npz'); ids=map_truth_ids(PA,t); targetPB=np.asarray(t['P_B'])[ids]
    PB=[];hcount=[];oracle_err=[]
    for i in range(len(PA)):
        cands=[]
        for v in range(8):
            if not bool(VA[v,i]): cands.append(np.empty((0,2),np.float32));continue
            cands.append(bf.per_view_topk(out['descriptor'][0,1,v],zA[i],masks[v],cache[v],coarse_k,final_k))
        H=bf.make_H(cands);hcount.append(len(H))
        if len(H):
            d=np.linalg.norm(H-targetPB[i][None],axis=1);j=int(np.argmin(d));PB.append(H[j]);oracle_err.append(float(d[j]))
        else:
            current_view=m.sample_field_np(out['delta_point_map_srcA'][0],xyA[:,i:i+1,:])[:,0,:];vv=VA[:,i].astype(np.float64)[:,None];delta=(current_view*vv).sum(0)/max(vv.sum(),1.);p=PA[i]+delta;PB.append(p);oracle_err.append(float(np.linalg.norm(p-targetPB[i])))
    PB=np.asarray(PB,np.float32)
    VB=exact_visibility(B_hull,B_proj,PB,diag);xyB=np.stack([m.project_points(PB,v) for v in range(8)],0).astype(np.float32)
    NA=consensus_normals(m.sample_field_np(out['normal_A'][0,0],xyA),VA);NB=consensus_normals(m.sample_field_np(out['normal_B'][0,1],xyB),VB)
    pred_world={'P_A':PA.astype(np.float32),'P_B':PB,'N_A':NA,'N_B':NB,'V_A':VA.astype(np.uint8),'V_B':VB.astype(np.uint8)}
    truth_ids=map_truth_ids(PA,t); target={k:np.asarray(t[k])[truth_ids] for k in ('P_A','P_B','N_A','N_B','scene_flow','persistent_obs')};target['V_A']=np.asarray(t['V_A'])[:,truth_ids];target['V_B']=np.asarray(t['V_B'])[:,truth_ids]
    true=compute_gfdr_v2(target['P_A'],target['P_B'],target['N_A'],target['N_B'],target['V_A'],target['V_B'],Z=None,U=None)
    pg=compute_gfdr_v2(PA,PB,NA,NB,VA,VB,Z=None,U=None)
    met=gfdr_compare(pg,true,target,pred_world)
    ddiag=float(np.linalg.norm(target['P_A'].max(0)-target['P_A'].min(0)))+1e-8
    met['P_A_error_diag_median']=float(np.median(np.linalg.norm(PA-target['P_A'],axis=1))/ddiag)
    met['P_B_error_diag_median']=float(np.median(np.linalg.norm(PB-target['P_B'],axis=1))/ddiag)
    met['oracle_H_error_median']=float(np.median(oracle_err));met['median_H_n']=float(np.median(hcount));met['empty_H_count']=int(sum(n==0 for n in hcount));met['family']=int(root.name)
    return met

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('roots',nargs='+',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    rows=[run_family(r) for r in a.roots]
    keys=['F_activity_spearman','F_kernel_spearman','D_tangent_action_error_median','R_differential_body_error','G_axis_dir_abs_cos_median','G_axis_line_error_diag_median']
    agg={k:float(np.nanmedian([r[k] for r in rows])) for k in keys}
    gates={'F_activity':agg['F_activity_spearman']>=.5,'F_kernel':agg['F_kernel_spearman']>=.3,'D':agg['D_tangent_action_error_median']<=.75,'R':agg['R_differential_body_error']<=1.,'G_direction':agg['G_axis_dir_abs_cos_median']>=.7,'G_line':agg['G_axis_line_error_diag_median']<=.1}
    result={'schema':'RealSaS.N1D.PostStageB.RepresentationSufficiency.TestA.RevisedHOracle.v1','arm':{'coarse_k':8,'final_k':16,'oracle':'choose nearest endpoint only inside generated H; truth never injects free XYZ'},'rows':rows,'aggregate_median':agg,'gates':gates,'gate_pass_count':int(sum(gates.values())),'all_six_pass':bool(all(gates.values()))}
    txt=json.dumps(result,indent=2,allow_nan=False);print(txt)
    if a.out:a.out.write_text(txt)
