from __future__ import annotations
import sys,json,itertools,time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,'/mnt/data/sufficiency')
import current_h_probe as m


def per_view_topk(desc_view,zA,mask,cache,coarse_k,final_k):
    coarse,zc=cache['coarse'],cache['zc']
    if len(coarse)==0:return np.empty((0,2),np.float32)
    score=zc@zA
    seeds=coarse[np.argsort(-score,kind='stable')[:coarse_k]]
    refined=[]
    for x,y in seeds:
        for dy in (-4,-2,0,2,4):
            for dx in (-4,-2,0,2,4):
                xx=int(round(float(x+dx))); yy=int(round(float(y+dy)))
                if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]: refined.append((xx,yy))
    if not refined: refined=[tuple(map(int,q)) for q in seeds]
    refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32)
    sr=m.normalize_rows(m.sample_desc_single_view(desc_view,refined))@zA
    ids=np.argsort(-sr,kind='stable')[:final_k]
    return refined[ids]

def make_H(cands):
    usable=[v for v in range(8) if len(cands[v])]
    blocks=[]
    for v0,v1 in itertools.combinations(usable,2):
        c0=np.asarray(cands[v0],np.float64); c1=np.asarray(cands[v1],np.float64)
        rr0=m.CAM_R[v0].cpu().numpy().astype(np.float64); uu0=m.CAM_U[v0].cpu().numpy().astype(np.float64)
        rr1=m.CAM_R[v1].cpu().numpy().astype(np.float64); uu1=m.CAM_U[v1].cpu().numpy().astype(np.float64)
        A=np.stack([rr0,uu0,rr1,uu1],axis=0)
        if np.linalg.matrix_rank(A)<3: continue
        pinv=np.linalg.pinv(A)
        sx0=c0[:,0]/(m.IMAGE_NATIVE-1)-0.5; sy0=0.5-c0[:,1]/(m.IMAGE_NATIVE-1)
        sx1=c1[:,0]/(m.IMAGE_NATIVE-1)-0.5; sy1=0.5-c1[:,1]/(m.IMAGE_NATIVE-1)
        n0,n1=len(c0),len(c1)
        bb=np.empty((n0*n1,4),np.float64)
        bb[:,0]=np.repeat(sx0,n1); bb[:,1]=np.repeat(sy0,n1); bb[:,2]=np.tile(sx1,n0); bb[:,3]=np.tile(sy1,n0)
        P=bb@pinv.T
        ok=np.isfinite(P).all(1)&(np.abs(P)<=.75).all(1)
        if ok.any(): blocks.append(P[ok].astype(np.float32))
    return np.concatenate(blocks,axis=0) if blocks else np.empty((0,3),np.float32)

def run(root:Path):
    A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'))
    x=m.load_model_tensor(A,B)
    with torch.no_grad():out=m.MODEL(x)
    PA,VA,_,_,_,pa_diag=m.exact_problem_a_frontdoor(A,B,n=64,res=96)
    xyA=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
    zA=m.descriptor_consensus(m.sample_field_np(out['descriptor'][0,0],xyA),VA)
    masks=[m.global_foreground_mask(p) for p in B]
    cache=[m.prepare_b_search(out['descriptor'][0,1,v],masks[v]) for v in range(8)]
    z=np.load(root/'B'/'observation_sidecar.npz'); center=np.asarray(z['camera_center'],np.float64);half=float(z['camera_half_extent'])
    TA=(np.asarray(z['surface_points_A'],np.float64)-center)/(2*half); TB=(np.asarray(z['surface_points_B'],np.float64)-center)/(2*half)
    sA=m.local_scale(TA);sB=m.local_scale(TB);D=np.linalg.norm(PA[:,None,:]-TA[None,:,:],axis=2);j=D.argmin(1);map_err=D[np.arange(len(PA)),j];reliable=map_err<=2*sA[j]
    arms={'R_retention_only':(8,16),'S_broader_search':(32,16)}
    result={}
    for name,(coarse_k,final_k) in arms.items():
        rec=[]
        for i in range(len(PA)):
            cands=[]
            for v in range(8):
                if not bool(VA[v,i]):cands.append(np.empty((0,2),np.float32));continue
                cands.append(per_view_topk(out['descriptor'][0,1,v],zA[i],masks[v],cache[v],coarse_k,final_k))
            H=make_H(cands); target=TB[j[i]]; scale=float(sB[j[i]])
            herr=float(np.min(np.linalg.norm(H-target[None],axis=1))) if len(H) else float('inf')
            tdist=[]
            for v in range(8):
                if not bool(VA[v,i]) or not len(cands[v]):continue
                tq=m.project_points(target[None],v)[0];tdist.append(float(np.min(np.linalg.norm(cands[v]-tq[None],axis=1))))
            rec.append({'i':i,'mapping_reliable':bool(reliable[i]),'H_n':int(len(H)),'H_norm_err':herr/max(scale,1e-12),'contain_1x':bool(herr<=scale),'contain_2x':bool(herr<=2*scale),'target_candidate_views_le4px':int(sum(d<=4 for d in tdist)),'target_candidate_median_min_px':float(np.median(tdist)) if tdist else None})
        rr=[r for r in rec if r['mapping_reliable']]
        result[name]={'coarse_k':coarse_k,'final_k':final_k,'den':len(rr),'contain_1x':float(np.mean([r['contain_1x'] for r in rr])),'contain_2x':float(np.mean([r['contain_2x'] for r in rr])),'median_H_norm_err':float(np.median([r['H_norm_err'] for r in rr])),'median_H_n':float(np.median([r['H_n'] for r in rr])),'median_target_candidate_views_le4px':float(np.median([r['target_candidate_views_le4px'] for r in rr])),'median_target_candidate_min_px':float(np.median([r['target_candidate_median_min_px'] for r in rr])),'records':rec}
    return {'family':int(root.name),'mapping_reliable_fraction':float(np.mean(reliable)),'pa_diag':pa_diag,'arms':result}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args();t=time.time();r=run(a.root);r['seconds']=time.time()-t;txt=json.dumps(r,indent=2,allow_nan=False);a.out.write_text(txt) if a.out else print(txt);print(json.dumps({k:v for k,v in r.items() if k!='arms'}|{'arms':{n:{k:v for k,v in q.items() if k!='records'} for n,q in r['arms'].items()}},indent=2))
