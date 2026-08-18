from __future__ import annotations
import argparse, hashlib, json, itertools, math, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

SOURCE_ROOT=Path('/mnt/data/activity_source/realsas_iris_sees_n1d_canonical')
if not SOURCE_ROOT.exists():
    SOURCE_ROOT=Path('/mnt/data/stageb_prep/n1dcanon/realsas_iris_sees_n1d_canonical')
sys.path.insert(0,str(SOURCE_ROOT))
from realsas_iris_sees import IRISSEESN1, SEESConfig
from realsas_iris_sees.sampling import sample_dense

YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)
IMAGE_NATIVE=256
CHECKPOINT=Path('/mnt/data/BEST_copy.pt')
CHECKPOINT_EXPECTED='0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18'
# Frozen after design witness 10763/e01 only.
OBS_ALPHA=.75
OBS_BETA=1.0
MECH_SCALE=.025
FIT_FRACTION=.50
ITERATIONS=6


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def camera_basis_torch(dtype=torch.float32):
    rows=[]
    for deg in YAW_DEG:
        t=math.radians(deg); radial=torch.tensor([math.sin(t),-math.cos(t),0.],dtype=dtype)
        fw=-radial;up=torch.tensor([0.,0.,1.],dtype=dtype);right=torch.cross(fw,up,dim=0);right=right/right.norm();rows.append((right,up,fw))
    return torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),torch.stack([x[2] for x in rows])
CAM_R,CAM_U,CAM_F=camera_basis_torch()

def reproject(P,v):
    vi=torch.as_tensor(v,dtype=torch.long);rr=CAM_R[vi];uu=CAM_U[vi]
    return torch.stack([2*(P*rr).sum(-1),-2*(P*uu).sum(-1)],-1)

def project_points(P,view,image_size=256):
    p=torch.as_tensor(np.asarray(P,np.float32));q=reproject(p,torch.full((len(p),),view,dtype=torch.long)).numpy()
    return np.stack([(q[:,0]+1)*.5*(image_size-1),(q[:,1]+1)*.5*(image_size-1)],1)

def masks_paths(paths,size=128):
    out=[]
    for p in paths:
        a=np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.NEAREST),dtype=np.float32)/255.
        out.append(~((a[...,1]>.72)&(a[...,0]<.30)&(a[...,2]<.30)))
    return np.stack(out)

def hull_from_masks(ms,res=96,bound=.62):
    vals=torch.linspace(-bound,bound,res);zz,yy,xx=torch.meshgrid(vals,vals,vals,indexing='ij');pts=torch.stack([xx,yy,zz],-1).reshape(-1,3);keep=torch.ones(len(pts),dtype=torch.bool);S=ms.shape[-1]
    for v in range(8):
        q=reproject(pts,torch.full((len(pts),),v,dtype=torch.long));gx=((q[:,0]+1)*.5*(S-1)).round().long();gy=((q[:,1]+1)*.5*(S-1)).round().long();inside=(gx>=0)&(gx<S)&(gy>=0)&(gy<S);ok=torch.zeros(len(pts),dtype=torch.bool);ii=torch.where(inside)[0];ok[ii]=torch.from_numpy(ms[v,gy[ii].numpy(),gx[ii].numpy()]);keep&=ok
    occ=keep.reshape(res,res,res).numpy();interior=occ.copy()
    for ax in range(3):interior &= np.roll(occ,1,ax)&np.roll(occ,-1,ax)
    interior[[0,-1],:,:]=False;interior[:,[0,-1],:]=False;interior[:,:,[0,-1]]=False
    return torch.from_numpy(pts.reshape(res,res,res,3).numpy()[occ & ~interior]).float()

def prepare(paths,res=96):
    h=hull_from_masks(masks_paths(paths),res);return h,torch.stack([reproject(h,torch.full((len(h),),v,dtype=torch.long)) for v in range(8)])

def raycast(h,proj,view,xy,k=24):
    q=torch.as_tensor(xy,dtype=h.dtype);d=((proj[view]-q)**2).sum(-1);kk=min(k,len(h));vals,idx=torch.topk(d,kk,largest=False);thr=vals[0]+(2.5/128.)**2;idx=idx[vals<=thr]
    if idx.numel()==0:idx=torch.topk(d,1,largest=False).indices
    dep=h[idx]@CAM_F[view];return h[idx[dep.argmin()]]

def fps(P,n=96):
    start=int(torch.argmin(P[:,0]*1e3+P[:,1]*10+P[:,2]));sel=[start];d=((P-P[start])**2).sum(-1)
    while len(sel)<min(n,len(P)):
        j=int(d.argmax());sel.append(j);d=torch.minimum(d,((P-P[j])**2).sum(-1))
    return sel

def visible_views(h,proj,p,diag,thr=.045):
    out=[]
    for v in range(8):
        xy=reproject(p[None],torch.tensor([v]))[0]
        if (xy.abs()>1).any():continue
        front=raycast(h,proj,v,xy)
        if float((front-p).norm()/diag)<=thr:out.append((v,xy))
    return out

def u8path(p):return np.asarray(Image.open(p).convert('RGB'),np.uint8)
def forward_flows(A,B):
    out=[]
    for a,b in zip(A,B):
        ga=cv2.cvtColor(u8path(a),cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(u8path(b),cv2.COLOR_RGB2GRAY);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);dis.setUseSpatialPropagation(True);out.append(dis.calc(ga,gb,None))
    return out

def sample_flow(fl,xy):
    H,W=fl.shape[:2];x=(float(xy[0])+1)*.5*(W-1);y=(float(xy[1])+1)*.5*(H-1);x0=int(np.floor(x));y0=int(np.floor(y));x1=min(W-1,x0+1);y1=min(H-1,y0+1);x0=max(0,x0);y0=max(0,y0);wx=x-x0;wy=y-y0;f=(1-wx)*(1-wy)*fl[y0,x0]+wx*(1-wy)*fl[y0,x1]+(1-wx)*wy*fl[y1,x0]+wx*wy*fl[y1,x1];return torch.tensor([f[0]*2/(W-1),f[1]*2/(H-1)],dtype=torch.float32)

def cluster_medoid(P,thr):
    if len(P)<2:return None,0
    T=torch.stack(P);D=torch.cdist(T,T);cnt=(D<thr).sum(1);best=int(cnt.argmax());ids=torch.where(D[best]<thr)[0]
    if ids.numel()<2:return None,int(ids.numel())
    A_=T[ids];j=int(torch.cdist(A_,A_).sum(1).argmin());return A_[j].cpu().numpy(),int(ids.numel())

def exact_problem_a(A,B):
    ha,pra=prepare(A,96);hb,prb=prepare(B,96);hh=torch.cat([ha,hb],0);diag=float((hh.max(0).values-hh.min(0).values).norm())+1e-9;flows=forward_flows(A,B);rows=[]
    for ix in fps(ha,min(len(ha),64*8)):
        pA=ha[ix];vvs=visible_views(ha,pra,pA,diag)
        if len(vvs)<2:continue
        props=[]
        for v,xyA in vvs:props.append(raycast(hb,prb,v,(xyA+sample_flow(flows[v],xyA)).clamp(-1,1)))
        pB,nc=cluster_medoid(props,.08*diag)
        if pB is None or nc<2:continue
        rows.append((pA.numpy(),vvs,pB,nc))
        if len(rows)>=64:break
    if len(rows)!=64:raise RuntimeError(f'Problem-A carriers {len(rows)} !=64')
    X=np.stack([r[0] for r in rows]).astype(np.float32);V=np.stack([[any(vv==v for vv,_ in r[1]) for v in range(8)] for r in rows]).T.astype(np.uint8);Y=np.stack([r[2] for r in rows]).astype(np.float32)
    return X,V,Y,hb,prb,diag

def foreground_mask(path):
    a=np.asarray(Image.open(path).convert('RGB'),np.uint8);return ~((a[...,0]==0)&(a[...,1]==255)&(a[...,2]==0))
def load_model_tensor(Apaths,Bpaths):
    rows=[]
    for paths in (Apaths,Bpaths):
        vv=[]
        for p in paths:
            im=Image.open(p).convert('RGB').resize((128,128),Image.Resampling.BILINEAR);a=np.asarray(im,np.float32)/255.;fg=~((a[...,0]<1e-6)&(a[...,1]>0.999)&(a[...,2]<1e-6));rgb=a.copy();rgb[~fg]=0.;rgba=np.concatenate([rgb,fg[...,None].astype(np.float32)],-1);vv.append(torch.from_numpy(rgba).permute(2,0,1))
        rows.append(torch.stack(vv))
    return torch.stack(rows).unsqueeze(0)
def sample_field_np(field,xy):
    ft=field.unsqueeze(0) if field.ndim==4 else field;xt=torch.from_numpy(np.asarray(xy,np.float32)).unsqueeze(0);return sample_dense(ft,xt,image_size=256)[0].float().detach().cpu().numpy()
def normrows(x,eps=1e-8):x=np.asarray(x,np.float64);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)
def consensus_np(x,V):
    w=np.asarray(V,np.float64)[...,None];return ((np.asarray(x,np.float64)*w).sum(0)/np.maximum(w.sum(0),1.)).astype(np.float32)
def descriptor_consensus(desc,V):return normrows(consensus_np(desc,V))
def consensus_normals(n_view,V):return normrows(consensus_np(n_view,V)).astype(np.float32)
def foreground_candidates(mask,stride=4):
    yy,xx=np.nonzero(mask);k=(xx%stride==0)&(yy%stride==0);q=np.stack([xx[k],yy[k]],1).astype(np.float32);return q if len(q)>=8 else np.stack([xx,yy],1).astype(np.float32)
def sample_desc_single_view(desc_view,xy):return sample_dense(desc_view[None,None],torch.from_numpy(xy.astype(np.float32))[None,None],image_size=256)[0,0].float().detach().cpu().numpy()
def prepare_b_search(desc_view,mask):
    q=foreground_candidates(mask,4);return {'coarse':q,'zc':normrows(sample_desc_single_view(desc_view,q)) if len(q) else np.empty((0,desc_view.shape[0]))}
def top4_with_scores(desc_view,zA,mask,cache):
    coarse,zc=cache['coarse'],cache['zc'];score=zc@zA;seeds=coarse[np.argsort(-score,kind='stable')[:8]];refined=[]
    for x,y in seeds:
        for dy in (-4,-2,0,2,4):
            for dx in (-4,-2,0,2,4):
                xx=int(round(float(x+dx)));yy=int(round(float(y+dy)))
                if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]:refined.append((xx,yy))
    if not refined:refined=[tuple(map(int,q)) for q in seeds]
    refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32);sr=normrows(sample_desc_single_view(desc_view,refined))@zA;ids=np.argsort(-sr,kind='stable')[:4];return refined[ids],sr[ids]

def linear_triangulate(obs):
    AA=[];bb=[]
    for v,x,y in obs:
        rr=CAM_R[v].numpy();uu=CAM_U[v].numpy();sx=float(x)/(IMAGE_NATIVE-1)-.5;sy=float(y)/(IMAGE_NATIVE-1)-.5;AA += [rr,-uu];bb += [sx,sy]
    AA=np.asarray(AA);bb=np.asarray(bb)
    if np.linalg.matrix_rank(AA)<3:return None
    p,res,rank,sv=np.linalg.lstsq(AA,bb,rcond=None);return p if rank>=3 else None

def candidate_pool(cands,cscores,PAi,seed_i):
    usable=[v for v in range(8) if len(cands[v])];H=[]
    for v0,v1 in itertools.combinations(usable,2):
        for q0 in cands[v0]:
            for q1 in cands[v1]:
                p=linear_triangulate([(v0,*q0),(v1,*q1)])
                if p is None or not np.isfinite(p).all() or not np.all(np.abs(p)<=.75):continue
                nearest=[];errs=[];dsc=[]
                for v in usable:
                    qq=project_points(p[None],v)[0];dd=np.linalg.norm(cands[v]-qq[None],axis=1);j=int(np.argmin(dd));nearest.append((v,j));errs.append(float(dd[j]));dsc.append(float(cscores[v][j]))
                H.append((p.astype(np.float32),float(np.mean(errs)),float(np.mean(dsc))))
                pref=linear_triangulate([(v,*cands[v][j]) for v,j in nearest])
                if pref is not None and np.isfinite(pref).all() and np.all(np.abs(pref)<=.75):
                    e2=[];d2=[]
                    for v in usable:
                        qq=project_points(pref[None],v)[0];dd=np.linalg.norm(cands[v]-qq[None],axis=1);j=int(np.argmin(dd));e2.append(float(dd[j]));d2.append(float(cscores[v][j]))
                    H.append((pref.astype(np.float32),float(np.mean(e2)),float(np.mean(d2))))
    def score_point(p):
        er=[];dd=[]
        for v in usable:
            qq=project_points(np.asarray(p)[None],v)[0];ds=np.linalg.norm(cands[v]-qq[None],axis=1);j=int(np.argmin(ds));er.append(float(ds[j]));dd.append(float(cscores[v][j]))
        return float(np.mean(er)),float(np.mean(dd))
    for p in (seed_i,PAi):
        e,d=score_point(p);H.append((p.astype(np.float32),e,d))
    uniq={}
    for h in H:
        key=tuple(np.round(h[0]/.0015).astype(int));rank=(h[1],-h[2])
        if key not in uniq or rank<(uniq[key][1],-uniq[key][2]):uniq[key]=h
    return sorted(uniq.values(),key=lambda h:(h[1],-h[2],*h[0]))

def rank_asc(x):
    o=np.argsort(x,kind='stable');r=np.empty(len(x),np.float64);r[o]=np.arange(len(x));return r/max(len(x)-1,1)
def rank_desc(x):
    o=np.argsort(-x,kind='stable');r=np.empty(len(x),np.float64);r[o]=np.arange(len(x));return r/max(len(x)-1,1)
def crossmat(x):return np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])

def solve_rank1_global(PA,pools,diag):
    AB=np.stack([np.hstack([-crossmat(x),np.eye(3)]) for x in PA]);us=[];ev=[]
    for i,H in enumerate(pools):
        rr=rank_asc(np.array([h[1] for h in H]));dr=rank_desc(np.array([h[2] for h in H]));u=OBS_ALPHA*rr+(1-OBS_ALPHA)*dr;us.append(u);DD=np.stack([h[0] for h in H])-PA[i];null=int(np.argmin(np.linalg.norm(DD,axis=1)));non=np.where(np.linalg.norm(DD,axis=1)>.005*diag)[0];ev.append(float(u[null]-(np.min(u[non]) if len(non) else 0.)))
    ev=np.asarray(ev);er=rank_desc(ev);cw=np.clip(1-er,.05,1.)**2;ids=np.argsort(-ev)[:max(8,int(round(len(PA)*FIT_FRACTION)))];sels=[int(np.argmin(np.array([h[1] for h in H]))) for H in pools]
    def chosen_D():return np.stack([pools[i][sels[i]][0]-PA[i] for i in range(len(PA))])
    def initial(D):
        M=np.vstack([AB[i] for i in ids]);y=np.hstack([D[i] for i in ids]);sw=np.sqrt(np.repeat(cw[ids],3));return np.linalg.lstsq(M*sw[:,None],y*sw,rcond=None)[0]
    def refit(D,th,nalt=6):
        for _ in range(nalt):
            q=np.einsum('nij,j->ni',AB,th);w=np.clip(np.sum(D*q,axis=1)/(np.sum(q*q,axis=1)+1e-12),0,1);rows=[];yy=[];ww=[]
            for i in ids:
                if w[i]<.03:continue
                rows.append(w[i]*AB[i]);yy.append(D[i]);ww.extend([cw[i]]*3)
            if len(rows)<4:break
            M=np.vstack(rows);y=np.hstack(yy);sw=np.sqrt(np.asarray(ww));th=np.linalg.lstsq(M*sw[:,None],y*sw,rcond=None)[0]
        return th
    D=chosen_D();th=refit(D,initial(D))
    for _ in range(ITERATIONS):
        q=np.einsum('nij,j->ni',AB,th);new=[]
        for i,H in enumerate(pools):
            PP=np.stack([h[0] for h in H]);DD=PP-PA[i];den=float(q[i]@q[i])+1e-12;wij=np.clip(DD@q[i]/den,0,1);resid=np.linalg.norm(DD-wij[:,None]*q[i],axis=1)/diag;cost=resid/MECH_SCALE+OBS_BETA*us[i];new.append(int(np.argmin(cost)))
        sels=new;D=chosen_D();th=refit(D,th)
    PB=np.stack([pools[i][sels[i]][0] for i in range(len(PA))]).astype(np.float32);q=np.einsum('nij,j->ni',AB,th);D=PB-PA;w=np.clip(np.sum(D*q,axis=1)/(np.sum(q*q,axis=1)+1e-12),0,1)
    return PB,th.astype(np.float32),w.astype(np.float32),ev.astype(np.float32),np.asarray(sels,np.int32)

def exact_b_visibility(hb,prb,PB,diag):
    V=np.zeros((8,len(PB)),np.uint8)
    for i,p in enumerate(PB):
        for v,xy in visible_views(hb,prb,torch.from_numpy(np.asarray(p,np.float32)),diag):V[v,i]=1
    return V

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',required=True);ap.add_argument('--root',type=Path,default=Path('/mnt/data/stagea_new'));ap.add_argument('--out',type=Path,default=Path('/mnt/data/stagea_new/predictions_frozen_v1'));args=ap.parse_args()
    if sha256_file(CHECKPOINT)!=CHECKPOINT_EXPECTED:raise RuntimeError('checkpoint SHA mismatch')
    A=sorted((args.root/str(args.family)/'A').glob('*.png'));B=sorted((args.root/str(args.family)/args.episode/'B').glob('*.png'))
    if len(A)!=8 or len(B)!=8:raise RuntimeError((args.family,args.episode,len(A),len(B)))
    PA,VA,PAseed,hb,prb,diag=exact_problem_a(A,B)
    cfg=SEESConfig(image_size=128,base_dim=64,token_dim=192,descriptor_dim=64,transformer_depth=4,transformer_heads=6,camera_residual_enabled=False,input_channels=4,max_delta=.35,correspondence_radius_f4=5,correspondence_temperature=.03,correspondence_position_lambda=.02,correspondence_gain_threshold=.10,correspondence_gain_width=.10)
    model=IRISSEESN1(cfg);ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);state=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck;model.load_state_dict(state,strict=True);model.eval()
    with torch.no_grad():out=model(load_model_tensor(A,B))
    xyA=np.stack([project_points(PA,v) for v in range(8)]).astype(np.float32);zA=descriptor_consensus(sample_field_np(out['descriptor'][0,0],xyA),VA);Bm=[foreground_mask(p) for p in B];bc=[prepare_b_search(out['descriptor'][0,1,v],Bm[v]) for v in range(8)];pools=[]
    for i in range(64):
        c=[];s=[]
        for v in range(8):
            if VA[v,i]:q,ss=top4_with_scores(out['descriptor'][0,1,v],zA[i],Bm[v],bc[v]);c.append(q);s.append(ss)
            else:c.append(np.empty((0,2),np.float32));s.append(np.empty(0,np.float32))
        pools.append(candidate_pool(c,s,PA[i],PAseed[i]))
    PB,theta,w,ev,sels=solve_rank1_global(PA,pools,diag);VB=exact_b_visibility(hb,prb,PB,diag);xyB=np.stack([project_points(PB,v) for v in range(8)]).astype(np.float32);NA=consensus_normals(sample_field_np(out['normal_A'][0,0],xyA),VA);NB=consensus_normals(sample_field_np(out['normal_B'][0,1],xyB),VB)
    args.out.mkdir(parents=True,exist_ok=True);npz=args.out/f'{args.family}_{args.episode}.npz';np.savez_compressed(npz,P_A=PA.astype(np.float32),P_B=PB,N_A=NA,N_B=NB,V_A=VA,V_B=VB,XY_A=xyA,XY_B=xyB,U_obs_motion_evidence=ev,U_obs_rank1_weight=w)
    meta={'schema':'RealSaS.IRIS.SEES.N1D.GlobalMechanicalSetSolverPrediction.v1','family':args.family,'episode':args.episode,'checkpoint_sha256':CHECKPOINT_EXPECTED,'solver':{'obs_alpha':OBS_ALPHA,'obs_beta':OBS_BETA,'mechanical_scale':MECH_SCALE,'fit_fraction':FIT_FRACTION,'iterations':ITERATIONS,'init':'minimum_mean_reprojection','model':'linearized rank-1 LBS d_i=w_i(omega x X_i+t)'},'problem_A':{'carriers':64,'mean_usable_views':float(VA.sum(0).mean())},'candidate_pool':{'min':int(min(map(len,pools))),'median':float(np.median(list(map(len,pools)))),'max':int(max(map(len,pools)))},'theta':theta.tolist(),'rank1_w_nontrivial':int((w>.05).sum()),'prediction_sha256':sha256_file(npz),'raster_sha256':{str(p):sha256_file(p) for p in A+B},'truth_access':'NONE'}
    js=npz.with_suffix('.json');js.write_text(json.dumps(meta,indent=2,sort_keys=True));print(json.dumps(meta,sort_keys=True))
if __name__=='__main__':main()
