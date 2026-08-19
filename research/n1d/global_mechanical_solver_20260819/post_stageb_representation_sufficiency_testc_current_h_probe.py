from __future__ import annotations
import sys, math, itertools, json, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

SRC=Path('/mnt/data/sufficiency/src/unpacked/realsas_iris_sees_n1d_canonical')
sys.path.insert(0,str(SRC))
from realsas_iris_sees import IRISSEESN1, SEESConfig
from realsas_iris_sees.sampling import sample_dense

# Canonical frozen model
CFG=SEESConfig(image_size=128,base_dim=64,token_dim=192,descriptor_dim=64,transformer_depth=4,transformer_heads=6,camera_residual_enabled=False,input_channels=4,max_delta=.35,correspondence_radius_f4=5,correspondence_temperature=.03,correspondence_position_lambda=.02,correspondence_gain_threshold=.10,correspondence_gain_width=.10)
DEVICE=torch.device('cpu')
MODEL=IRISSEESN1(CFG).to(DEVICE)
ck=torch.load('/mnt/data/sufficiency/BEST.pt',map_location='cpu',weights_only=False)
state=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck
load=MODEL.load_state_dict(state,strict=True)
assert not load.missing_keys and not load.unexpected_keys
MODEL.eval()

YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)
IMAGE_NATIVE=256
ACTIVITY_THRESHOLD=.005

def camera_basis_torch(dtype=torch.float32):
    rows=[]
    for deg in YAW_DEG:
        t=math.radians(deg)
        radial=torch.tensor([math.sin(t),-math.cos(t),0.],dtype=dtype)
        fw=-radial; up=torch.tensor([0.,0.,1.],dtype=dtype)
        right=torch.cross(fw,up,dim=0); right=right/right.norm()
        rows.append((right,up,fw))
    return torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),torch.stack([x[2] for x in rows])
CAM_R,CAM_U,CAM_F=camera_basis_torch()

def reproject_normalized(P,v):
    vi=torch.as_tensor(v,dtype=torch.long); rr=CAM_R[vi]; uu=CAM_U[vi]
    return torch.stack([2*(P*rr).sum(-1),-2*(P*uu).sum(-1)],-1)

def project_points(P,view,image_size=256):
    p=torch.as_tensor(np.asarray(P,np.float32)); q=reproject_normalized(p,torch.full((len(p),),int(view),dtype=torch.long)).cpu().numpy()
    return np.stack([(q[:,0]+1)*.5*(image_size-1),(q[:,1]+1)*.5*(image_size-1)],1)

def masks_problem_a(paths,size=128):
    out=[]
    for p in paths:
        a=np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.NEAREST),dtype=np.float32)/255.
        out.append(~((a[...,1]>.72)&(a[...,0]<.30)&(a[...,2]<.30)))
    return np.stack(out)

def hull_from_masks_problem_a(ms,res=96,bound=.62):
    vals=torch.linspace(-bound,bound,res); zz,yy,xx=torch.meshgrid(vals,vals,vals,indexing='ij'); pts=torch.stack([xx,yy,zz],-1).reshape(-1,3)
    keep=torch.ones(len(pts),dtype=torch.bool); S=ms.shape[-1]
    for v in range(8):
        q=reproject_normalized(pts,torch.full((len(pts),),v,dtype=torch.long)); gx=((q[:,0]+1)*.5*(S-1)).round().long(); gy=((q[:,1]+1)*.5*(S-1)).round().long(); inside=(gx>=0)&(gx<S)&(gy>=0)&(gy<S); ok=torch.zeros(len(pts),dtype=torch.bool); ii=torch.where(inside)[0]; ok[ii]=torch.from_numpy(ms[v,gy[ii].numpy(),gx[ii].numpy()]); keep &= ok
    occ=keep.reshape(res,res,res).numpy(); interior=occ.copy()
    for ax in range(3): interior &= np.roll(occ,1,ax)&np.roll(occ,-1,ax)
    interior[[0,-1],:,:]=False; interior[:,[0,-1],:]=False; interior[:,:,[0,-1]]=False
    surf=pts.reshape(res,res,res,3).numpy()[occ & ~interior]
    return torch.from_numpy(surf).float()

def prepare_problem_a_hull(paths,res=96):
    h=hull_from_masks_problem_a(masks_problem_a(paths),res); proj=torch.stack([reproject_normalized(h,torch.full((len(h),),v,dtype=torch.long)) for v in range(8)]); return h,proj

def raycast_problem_a(h,proj,view,xy,k=24):
    q=torch.as_tensor(xy,dtype=h.dtype); d=((proj[view]-q)**2).sum(-1); kk=min(k,len(h)); vals,idx=torch.topk(d,kk,largest=False); thr=vals[0]+(2.5/128.)**2; idx=idx[vals<=thr]
    if idx.numel()==0: idx=torch.topk(d,1,largest=False).indices
    dep=h[idx]@CAM_F[view]; return h[idx[dep.argmin()]]

def fps_problem_a(P,n=96):
    start=int(torch.argmin(P[:,0]*1e3+P[:,1]*10+P[:,2])); sel=[start]; d=((P-P[start])**2).sum(-1)
    while len(sel)<min(n,len(P)):
        j=int(d.argmax()); sel.append(j); d=torch.minimum(d,((P-P[j])**2).sum(-1))
    return sel

def visible_views_problem_a(h,proj,p,diag,thr=.045):
    out=[]
    for v in range(8):
        xy=reproject_normalized(p[None],torch.tensor([v]))[0]
        if (xy.abs()>1).any(): continue
        front=raycast_problem_a(h,proj,v,xy)
        if float((front-p).norm()/diag)<=thr: out.append((v,xy))
    return out

def _u8_rgb(path): return np.asarray(Image.open(path).convert('RGB'),np.uint8)
def forward_dis_flows(Apaths,Bpaths):
    out=[]
    for a,b in zip(Apaths,Bpaths):
        ga=cv2.cvtColor(_u8_rgb(a),cv2.COLOR_RGB2GRAY); gb=cv2.cvtColor(_u8_rgb(b),cv2.COLOR_RGB2GRAY); dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setUseSpatialPropagation(True); out.append(dis.calc(ga,gb,None))
    return out

def sample_dis_flow(fl,xy):
    H,W=fl.shape[:2]; x=(float(xy[0])+1)*.5*(W-1); y=(float(xy[1])+1)*.5*(H-1); x0=int(np.floor(x)); y0=int(np.floor(y)); x1=min(W-1,x0+1); y1=min(H-1,y0+1); x0=max(0,x0); y0=max(0,y0); wx=x-x0; wy=y-y0
    f=((1-wx)*(1-wy)*fl[y0,x0]+wx*(1-wy)*fl[y0,x1]+(1-wx)*wy*fl[y1,x0]+wx*wy*fl[y1,x1]); return torch.tensor([f[0]*2/(W-1),f[1]*2/(H-1)],dtype=torch.float32)

def cluster_medoid_problem_a(P,thr):
    if len(P)<2:return None,0
    T=torch.stack(P); D=torch.cdist(T,T); cnt=(D<thr).sum(1); best=int(cnt.argmax()); ids=torch.where(D[best]<thr)[0]
    if ids.numel()<2:return None,int(ids.numel())
    A_=T[ids]; j=int(torch.cdist(A_,A_).sum(1).argmin()); return A_[j].cpu().numpy(),int(ids.numel())

def exact_problem_a_frontdoor(Apaths,Bpaths,n=64,res=96):
    ha,pra=prepare_problem_a_hull(Apaths,res); hb,prb=prepare_problem_a_hull(Bpaths,res); hh=torch.cat([ha,hb],0); diag=float((hh.max(0).values-hh.min(0).values).norm())+1e-9; flows=forward_dis_flows(Apaths,Bpaths)
    rows=[]; considered=visfail=consfail=0
    for ix in fps_problem_a(ha,min(len(ha),n*8)):
        considered+=1; pA=ha[ix]; vvs=visible_views_problem_a(ha,pra,pA,diag)
        if len(vvs)<2: visfail+=1; continue
        props=[]
        for v,xyA in vvs:
            dxy=sample_dis_flow(flows[v],xyA); props.append(raycast_problem_a(hb,prb,v,(xyA+dxy).clamp(-1,1)))
        pB,nc=cluster_medoid_problem_a(props,.08*diag)
        if pB is None or nc<2: consfail+=1; continue
        rows.append((pA.numpy(),vvs,pB,nc))
        if len(rows)>=n:break
    if len(rows)!=n: raise RuntimeError(('Problem-A failed',len(rows),considered,visfail,consfail))
    X=np.stack([r[0] for r in rows]).astype(np.float32); V=np.stack([[any(vv==v for vv,_ in r[1]) for v in range(8)] for r in rows]).T.astype(np.uint8)
    return X,V,hb,prb,diag,{'carriers':len(rows),'considered':considered,'visibility_rejects':visfail,'consensus_rejects':consfail,'mean_usable_views':float(V.sum(0).mean())}

def read_rgb_native(path): return np.asarray(Image.open(path).convert('RGB'),np.uint8)
def global_foreground_mask(path):
    a=read_rgb_native(path); return ~((a[...,0]==0)&(a[...,1]==255)&(a[...,2]==0))
def load_model_tensor(Apaths,Bpaths):
    rows=[]
    for paths in (Apaths,Bpaths):
        vv=[]
        for p in paths:
            im=Image.open(p).convert('RGB').resize((128,128),Image.Resampling.BILINEAR); a=np.asarray(im,np.float32)/255.; fg=~((a[...,0]<1e-6)&(a[...,1]>.999)&(a[...,2]<1e-6)); rgb=a.copy(); rgb[~fg]=0.; rgba=np.concatenate([rgb,fg[...,None].astype(np.float32)],-1); vv.append(torch.from_numpy(rgba).permute(2,0,1))
        rows.append(torch.stack(vv))
    return torch.stack(rows).unsqueeze(0).to(DEVICE)
def sample_field_np(field,xy):
    ft=field.unsqueeze(0) if field.ndim==4 else field; xt=torch.from_numpy(np.asarray(xy,np.float32)).unsqueeze(0).to(ft.device); return sample_dense(ft,xt,image_size=256)[0].float().detach().cpu().numpy()
def normalize_rows(x,eps=1e-8):
    x=np.asarray(x,np.float64); return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)
def consensus_np(view_values,V):
    w=np.asarray(V,np.float64)[...,None]; return ((np.asarray(view_values,np.float64)*w).sum(0)/np.maximum(w.sum(0),1.)).astype(np.float32)
def descriptor_consensus(descA,VA): return normalize_rows(consensus_np(descA,VA))
def foreground_candidates(mask,stride=4):
    yy,xx=np.nonzero(mask); keep=(xx%stride==0)&(yy%stride==0); q=np.stack([xx[keep],yy[keep]],1).astype(np.float32)
    if len(q)<8:q=np.stack([xx,yy],1).astype(np.float32)
    return q
def sample_desc_single_view(desc_view,xy):
    ft=desc_view[None,None]; xt=torch.from_numpy(xy.astype(np.float32))[None,None].to(ft.device); return sample_dense(ft,xt,image_size=256)[0,0].float().detach().cpu().numpy()
def prepare_b_search(desc_view,mask):
    coarse=foreground_candidates(mask,4); zc=normalize_rows(sample_desc_single_view(desc_view,coarse)) if len(coarse) else np.empty((0,desc_view.shape[0]),np.float64); return {'coarse':coarse,'zc':zc}
def per_view_top4(desc_view,zA,mask,cache):
    coarse,zc=cache['coarse'],cache['zc'];
    if len(coarse)==0:return np.empty((0,2),np.float32),np.empty(0,np.float32)
    score=zc@zA; seeds=coarse[np.argsort(-score,kind='stable')[:8]]; refined=[]
    for x,y in seeds:
        for dy in (-4,-2,0,2,4):
            for dx in (-4,-2,0,2,4):
                xx=int(round(float(x+dx))); yy=int(round(float(y+dy)))
                if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]:refined.append((xx,yy))
    if not refined:refined=[tuple(map(int,q)) for q in seeds]
    refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32); sr=normalize_rows(sample_desc_single_view(desc_view,refined))@zA; ids=np.argsort(-sr,kind='stable')[:4]; return refined[ids],sr[ids].astype(np.float32)
def linear_triangulate(obs):
    A=[];b=[]
    for v,x,y in obs:
        rr=CAM_R[int(v)].cpu().numpy().astype(np.float64); uu=CAM_U[int(v)].cpu().numpy().astype(np.float64); sx=float(x)/(IMAGE_NATIVE-1)-.5; sy=.5-float(y)/(IMAGE_NATIVE-1); A.append(rr);b.append(sx);A.append(uu);b.append(sy)
    A=np.asarray(A);b=np.asarray(b)
    if np.linalg.matrix_rank(A)<3:return None
    p,res,rank,sv=np.linalg.lstsq(A,b,rcond=None); return p if rank>=3 else None

def solve_top4_multiview_export(cands,scores):
    usable=[v for v in range(8) if len(cands[v])]
    if len(usable)<2:return None,{'reason':'lt2_visible_views','usable_views':len(usable)},np.empty((0,3),np.float32)
    H=[]
    for v0,v1 in itertools.combinations(usable,2):
      for j0 in range(len(cands[v0])):
       for j1 in range(len(cands[v1])):
        p=linear_triangulate([(v0,*cands[v0][j0]),(v1,*cands[v1][j1])])
        if p is not None and np.isfinite(p).all() and np.all(np.abs(p)<=.75):H.append(p)
    if not H:return None,{'reason':'no_rank3_pair_hypothesis','usable_views':len(usable)},np.empty((0,3),np.float32)
    H=np.asarray(H,np.float64); best_p=None;best_mean=float('inf');best_nearest=None
    for p in H:
        errs=[];nearest=[]
        for v in usable:
            q=project_points(p[None],v)[0]; d=np.linalg.norm(cands[v]-q[None],axis=1); j=int(np.argmin(d));errs.append(float(d[j]));nearest.append((v,j))
        mean_err=float(np.mean(errs))
        if mean_err<best_mean:best_mean,best_p,best_nearest=mean_err,p,nearest
    p_refit=linear_triangulate([(v,*cands[v][j]) for v,j in best_nearest])
    if p_refit is None or not np.isfinite(p_refit).all() or not np.all(np.abs(p_refit)<=.75):p_refit=best_p; refit='seed_retained'
    else:refit='all_visible_nearest_ls'
    final_errs=[]
    for v in usable:
        q=project_points(p_refit[None],v)[0];d=np.linalg.norm(cands[v]-q[None],axis=1);final_errs.append(float(np.min(d)))
    return p_refit.astype(np.float32),{'reason':'ok','usable_views':len(usable),'mean_reproj_px':float(np.mean(final_errs)),'pair_hypotheses':len(H),'refit':refit},H.astype(np.float32)

def local_scale(P,k=4):
    P=np.asarray(P,np.float64); D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2); np.fill_diagonal(D,np.inf); nn=np.partition(D,k-1,axis=1)[:,:k]; return np.median(nn,axis=1)

def run_family(root:Path):
    A=sorted((root/'A').glob('*.png')); B=sorted((root/'B').glob('*.png')); assert len(A)==8 and len(B)==8
    x=load_model_tensor(A,B)
    with torch.no_grad():out=MODEL(x)
    PA,VA,B_hull,B_proj,diag,pa_diag=exact_problem_a_frontdoor(A,B,n=64,res=96)
    xyA=np.stack([project_points(PA,v) for v in range(8)],0).astype(np.float32)
    descA=sample_field_np(out['descriptor'][0,0],xyA); zA=descriptor_consensus(descA,VA)
    masksB=[global_foreground_mask(p) for p in B]; bcache=[prepare_b_search(out['descriptor'][0,1,v],masksB[v]) for v in range(8)]
    PB=[]; Hs=[]; diags=[]
    for i in range(len(PA)):
        cands=[];scores=[]
        for v in range(8):
            if not bool(VA[v,i]):cands.append(np.empty((0,2),np.float32));scores.append(np.empty(0,np.float32));continue
            q,s=per_view_top4(out['descriptor'][0,1,v],zA[i],masksB[v],bcache[v]);cands.append(q);scores.append(s)
        p,d,H=solve_top4_multiview_export(cands,scores)
        if p is None:
            current_view=sample_field_np(out['delta_point_map_srcA'][0],xyA[:,i:i+1,:])[:,0,:]
            vv=VA[:,i].astype(np.float64)[:,None]; delta=(current_view*vv).sum(0)/max(vv.sum(),1.)
            p=PA[i]+delta.astype(np.float32);d['mode']='current_fallback'
        else:d['mode']='global_foreground'
        PB.append(p);Hs.append(H);diags.append(d)
    PB=np.asarray(PB,np.float32)
    z=np.load(root/'B'/'observation_sidecar.npz')
    center=np.asarray(z['camera_center'],np.float64); half=float(z['camera_half_extent'])
    TA=(np.asarray(z['surface_points_A'],np.float64)-center)/(2*half); TB=(np.asarray(z['surface_points_B'],np.float64)-center)/(2*half)
    sA=local_scale(TA);sB=local_scale(TB)
    D=np.linalg.norm(PA[:,None,:]-TA[None,:,:],axis=2); j=D.argmin(1); map_err=D[np.arange(len(PA)),j]; reliable=map_err<=2*sA[j]
    records=[]
    for i in range(len(PA)):
        H=Hs[i]; target=TB[j[i]]
        h_err=float(np.min(np.linalg.norm(H-target[None,:],axis=1))) if len(H) else float('inf')
        commit_err=float(np.linalg.norm(PB[i]-target)); scale=float(sB[j[i]])
        records.append({'i':i,'truth_idx':int(j[i]),'map_err':float(map_err[i]),'map_scale_A':float(sA[j[i]]),'mapping_reliable':bool(reliable[i]),'H_n':int(len(H)),'H_min_err':h_err,'B_scale':scale,'H_norm_err':h_err/max(scale,1e-12),'contain_1x':bool(h_err<=scale),'contain_2x':bool(h_err<=2*scale),'commit_err':commit_err,'commit_norm_err':commit_err/max(scale,1e-12),'mode':diags[i].get('mode'),'mean_reproj_px':diags[i].get('mean_reproj_px')})
    rr=[r for r in records if r['mapping_reliable']]
    return {'family':int(root.name),'pa_diag':pa_diag,'mapping_reliable_fraction':float(np.mean(reliable)),'den':len(rr),'contain_1x':float(np.mean([r['contain_1x'] for r in rr])) if rr else None,'contain_2x':float(np.mean([r['contain_2x'] for r in rr])) if rr else None,'median_H_norm_err':float(np.median([r['H_norm_err'] for r in rr])) if rr else None,'median_commit_norm_err':float(np.median([r['commit_norm_err'] for r in rr])) if rr else None,'global_solution_fraction':float(np.mean([r['mode']=='global_foreground' for r in records])),'records':records}

if __name__=='__main__':
 import argparse,time
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path);args=ap.parse_args();t=time.time();r=run_family(args.root);r['seconds']=time.time()-t;text=json.dumps(r,indent=2,allow_nan=False);print(text if not args.out else json.dumps({k:v for k,v in r.items() if k!='records'},indent=2));
 if args.out: args.out.write_text(text)
