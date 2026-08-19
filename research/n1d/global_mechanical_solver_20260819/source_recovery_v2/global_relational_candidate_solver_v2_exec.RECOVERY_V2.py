from __future__ import annotations
import math, itertools, json, pickle, hashlib, time, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

ROOT=Path('/mnt/data/sufficiency')
SRC=ROOT/'cansrc/realsas_iris_sees_n1d_canonical'
sys.path.insert(0,str(SRC))
from realsas_iris_sees import IRISSEESN1, SEESConfig
from realsas_iris_sees.sampling import sample_dense

FAMILIES=[9908,11032,12772,13203,14404,14702,14758,15290]
DIR={9908:'09908',11032:'11032',12772:'12772',13203:'13203',14404:'14404',14702:'14702',14758:'14758',15290:'15290'}
EXPECTED_F16={9908:1.0,11032:0.9491525424,12772:1.0,13203:0.9508196721,14404:0.9682539683,14702:1.0,14758:0.9838709677,15290:1.0}
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)
IMAGE_NATIVE=256
ACTIVITY_THRESHOLD=.005
OUT=ROOT/'global_rel_solver_v2'; OUT.mkdir(exist_ok=True)
CACHE=OUT/'cache'; CACHE.mkdir(exist_ok=True)

CFG=SEESConfig(image_size=128,base_dim=64,token_dim=192,descriptor_dim=64,transformer_depth=4,transformer_heads=6,camera_residual_enabled=False,input_channels=4,max_delta=.35,correspondence_radius_f4=5,correspondence_temperature=.03,correspondence_position_lambda=.02,correspondence_gain_threshold=.10,correspondence_gain_width=.10)
MODEL=IRISSEESN1(CFG).cpu()
ck=torch.load(str(ROOT/'srcpack/BEST.pt'),map_location='cpu',weights_only=False)
state=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck
load=MODEL.load_state_dict(state,strict=True); assert not load.missing_keys and not load.unexpected_keys
MODEL.eval()

def sha256_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
assert sha256_file(ROOT/'srcpack/BEST.pt')=='0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18'

def camera_basis(dtype=torch.float32):
 rows=[]
 for deg in YAW_DEG:
  t=math.radians(deg); radial=torch.tensor([math.sin(t),-math.cos(t),0.],dtype=dtype); fw=-radial; up=torch.tensor([0.,0.,1.],dtype=dtype); right=torch.cross(fw,up,dim=0); right=right/right.norm(); rows.append((right,up,fw))
 return torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),torch.stack([x[2] for x in rows])
CAM_R,CAM_U,CAM_F=camera_basis()
RNP=CAM_R.numpy().astype(np.float64); UNP=CAM_U.numpy().astype(np.float64)
PAIR_PINV={}
for v0,v1 in itertools.combinations(range(8),2):
 A=np.stack([RNP[v0],UNP[v0],RNP[v1],UNP[v1]],0)
 if np.linalg.matrix_rank(A)>=3: PAIR_PINV[(v0,v1)]=np.linalg.pinv(A)

def reproject_norm(P,v):
 vi=torch.as_tensor(v,dtype=torch.long); rr=CAM_R[vi]; uu=CAM_U[vi]
 return torch.stack([2*(P*rr).sum(-1),-2*(P*uu).sum(-1)],-1)
def project_np(P,v):
 P=np.asarray(P,np.float64); sx=P@RNP[v]; sy=P@UNP[v]
 return np.stack([(sx+.5)*(IMAGE_NATIVE-1),(.5-sy)*(IMAGE_NATIVE-1)],1)
def masks_problem_a(paths,size=128):
 out=[]
 for p in paths:
  a=np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.NEAREST),dtype=np.float32)/255.
  out.append(~((a[...,1]>.72)&(a[...,0]<.30)&(a[...,2]<.30)))
 return np.stack(out)
def hull_from_masks(ms,res=96,bound=.62):
 vals=torch.linspace(-bound,bound,res); zz,yy,xx=torch.meshgrid(vals,vals,vals,indexing='ij'); pts=torch.stack([xx,yy,zz],-1).reshape(-1,3); keep=torch.ones(len(pts),dtype=torch.bool); S=ms.shape[-1]
 for v in range(8):
  q=reproject_norm(pts,torch.full((len(pts),),v,dtype=torch.long)); gx=((q[:,0]+1)*.5*(S-1)).round().long(); gy=((q[:,1]+1)*.5*(S-1)).round().long(); inside=(gx>=0)&(gx<S)&(gy>=0)&(gy<S); ok=torch.zeros(len(pts),dtype=torch.bool); ii=torch.where(inside)[0]; ok[ii]=torch.from_numpy(ms[v,gy[ii].numpy(),gx[ii].numpy()]); keep &= ok
 occ=keep.reshape(res,res,res).numpy(); interior=occ.copy()
 for ax in range(3): interior &= np.roll(occ,1,ax)&np.roll(occ,-1,ax)
 interior[[0,-1],:,:]=False; interior[:,[0,-1],:]=False; interior[:,:,[0,-1]]=False
 return torch.from_numpy(pts.reshape(res,res,res,3).numpy()[occ & ~interior]).float()
def prepare_hull(paths,res=96):
 h=hull_from_masks(masks_problem_a(paths),res); proj=torch.stack([reproject_norm(h,torch.full((len(h),),v,dtype=torch.long)) for v in range(8)]); return h,proj
def raycast(h,proj,view,xy,k=24):
 q=torch.as_tensor(xy,dtype=h.dtype); d=((proj[view]-q)**2).sum(-1); kk=min(k,len(h)); vals,idx=torch.topk(d,kk,largest=False); thr=vals[0]+(2.5/128.)**2; idx=idx[vals<=thr]
 if idx.numel()==0: idx=torch.topk(d,1,largest=False).indices
 dep=h[idx]@CAM_F[view]; return h[idx[dep.argmin()]]
def fps(P,n=96):
 start=int(torch.argmin(P[:,0]*1e3+P[:,1]*10+P[:,2])); sel=[start]; d=((P-P[start])**2).sum(-1)
 while len(sel)<min(n,len(P)):
  j=int(d.argmax()); sel.append(j); d=torch.minimum(d,((P-P[j])**2).sum(-1))
 return sel
def visible_views(h,proj,p,diag,thr=.045):
 out=[]
 for v in range(8):
  xy=reproject_norm(p[None],torch.tensor([v]))[0]
  if (xy.abs()>1).any(): continue
  front=raycast(h,proj,v,xy)
  if float((front-p).norm()/diag)<=thr: out.append((v,xy))
 return out
def u8rgb(p): return np.asarray(Image.open(p).convert('RGB'),np.uint8)
def forward_dis_flows(A,B):
 out=[]
 for a,b in zip(A,B):
  ga=cv2.cvtColor(u8rgb(a),cv2.COLOR_RGB2GRAY); gb=cv2.cvtColor(u8rgb(b),cv2.COLOR_RGB2GRAY)
  dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setUseSpatialPropagation(True); out.append(dis.calc(ga,gb,None))
 return out

def sample_dis_flow(fl,xy):
 H,W=fl.shape[:2]; x=(float(xy[0])+1)*.5*(W-1); y=(float(xy[1])+1)*.5*(H-1)
 x0=int(np.floor(x)); y0=int(np.floor(y)); x1=min(W-1,x0+1); y1=min(H-1,y0+1); x0=max(0,x0); y0=max(0,y0); wx=x-x0; wy=y-y0
 f=((1-wx)*(1-wy)*fl[y0,x0]+wx*(1-wy)*fl[y0,x1]+(1-wx)*wy*fl[y1,x0]+wx*wy*fl[y1,x1])
 return torch.tensor([f[0]*2/(W-1),f[1]*2/(H-1)],dtype=torch.float32)

def cluster_medoid(P,thr):
 if len(P)<2:return None,0
 T=torch.stack(P);D=torch.cdist(T,T);cnt=(D<thr).sum(1);best=int(cnt.argmax());ids=torch.where(D[best]<thr)[0]
 if ids.numel()<2:return None,int(ids.numel())
 A_=T[ids];j=int(torch.cdist(A_,A_).sum(1).argmin());return A_[j].cpu().numpy(),int(ids.numel())

def exact_problem_a_frontdoor(A,B,n=64,res=96):
 ha,pra=prepare_hull(A,res);hb,prb=prepare_hull(B,res);hh=torch.cat([ha,hb],0);diag=float((hh.max(0).values-hh.min(0).values).norm())+1e-9;flows=forward_dis_flows(A,B)
 rows=[];considered=visfail=consfail=0
 for ix in fps(ha,min(len(ha),n*8)):
  considered+=1;pA=ha[ix];vvs=visible_views(ha,pra,pA,diag)
  if len(vvs)<2:visfail+=1;continue
  props=[]
  for v,xyA in vvs:
   dxy=sample_dis_flow(flows[v],xyA);props.append(raycast(hb,prb,v,(xyA+dxy).clamp(-1,1)))
  pB,nc=cluster_medoid(props,.08*diag)
  if pB is None or nc<2:consfail+=1;continue
  rows.append((pA.numpy(),vvs,pB,nc))
  if len(rows)>=n:break
 if len(rows)!=n:raise RuntimeError(('Problem-A failed to produce 64 carriers',len(rows),considered,visfail,consfail))
 X=np.stack([r[0] for r in rows]).astype(np.float32)
 V=np.stack([[any(vv==v for vv,_ in r[1]) for v in range(8)] for r in rows]).T.astype(np.uint8)
 diag_record={'carriers':len(rows),'considered':considered,'visibility_rejects':visfail,'consensus_rejects':consfail,'mean_usable_views':float(V.sum(0).mean()),'median_usable_views':float(np.median(V.sum(0))),'mean_problem_a_consensus_views':float(np.mean([r[3] for r in rows]))}
 return X,V,diag_record

def foreground_mask(path):
 a=np.asarray(Image.open(path).convert('RGB'),np.uint8);return ~((a[...,0]==0)&(a[...,1]==255)&(a[...,2]==0))

def load_model_tensor(A,B):
 rows=[]
 for paths in (A,B):
  vv=[]
  for p in paths:
   im=Image.open(p).convert('RGB').resize((128,128),Image.Resampling.BILINEAR);a=np.asarray(im,np.float32)/255.;fg=~((a[...,0]<1e-6)&(a[...,1]>.999)&(a[...,2]<1e-6));rgb=a.copy();rgb[~fg]=0.;rgba=np.concatenate([rgb,fg[...,None].astype(np.float32)],axis=-1);vv.append(torch.from_numpy(rgba).permute(2,0,1))
  rows.append(torch.stack(vv))
 return torch.stack(rows).unsqueeze(0)

def sample_field_np(field,xy):
 ft=field.unsqueeze(0) if field.ndim==4 else field;xt=torch.from_numpy(np.asarray(xy,np.float32)).unsqueeze(0).to(ft.device)
 return sample_dense(ft,xt,image_size=256)[0].float().detach().cpu().numpy()

def normalize_rows(x,eps=1e-8):
 x=np.asarray(x,np.float64);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)

def consensus_np(view_values,V):
 w=np.asarray(V,np.float64)[...,None];return ((np.asarray(view_values,np.float64)*w).sum(0)/np.maximum(w.sum(0),1.)).astype(np.float32)

def foreground_candidates(mask,stride=4):
 yy,xx=np.nonzero(mask);keep=(xx%stride==0)&(yy%stride==0);q=np.stack([xx[keep],yy[keep]],axis=1).astype(np.float32)
 if len(q)<8:q=np.stack([xx,yy],axis=1).astype(np.float32)
 return q

def sample_desc_single_view(desc_view,xy):
 ft=desc_view[None,None];xt=torch.from_numpy(np.asarray(xy,np.float32))[None,None].to(ft.device)
 return sample_dense(ft,xt,image_size=256)[0,0].float().detach().cpu().numpy()

def prepare_b_search(desc_view,mask):
 coarse=foreground_candidates(mask,4);zc=normalize_rows(sample_desc_single_view(desc_view,coarse)) if len(coarse) else np.empty((0,desc_view.shape[0]),np.float64)
 return {'coarse':coarse,'zc':zc}

def per_view_top16(desc_view,zA,mask):
 cache=prepare_b_search(desc_view,mask);coarse,zc=cache['coarse'],cache['zc']
 if len(coarse)==0:return np.empty((0,2),np.float32)
 score=zc@zA;seeds=coarse[np.argsort(-score,kind='stable')[:8]];refined=[]
 for x,y in seeds:
  for dy in (-4,-2,0,2,4):
   for dx in (-4,-2,0,2,4):
    xx=int(round(float(x+dx)));yy=int(round(float(y+dy)))
    if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]:refined.append((xx,yy))
 if not refined:refined=[tuple(map(int,q)) for q in seeds]
 refined=np.array(sorted(set(refined),key=lambda q:(q[1],q[0])),np.float32);sr=normalize_rows(sample_desc_single_view(desc_view,refined))@zA;ids=np.argsort(-sr,kind='stable')[:16]
 return refined[ids]

def make_H_fast(cands):
 usable=[v for v in range(8) if len(cands[v])];blocks=[]
 for ai in range(len(usable)):
  for bi in range(ai+1,len(usable)):
   v0,v1=usable[ai],usable[bi];c0=np.asarray(cands[v0],np.float64);c1=np.asarray(cands[v1],np.float64);pinv=PAIR_PINV.get((v0,v1))
   if pinv is None:continue
   sx0=c0[:,0]/(IMAGE_NATIVE-1)-.5;sy0=.5-c0[:,1]/(IMAGE_NATIVE-1);sx1=c1[:,0]/(IMAGE_NATIVE-1)-.5;sy1=.5-c1[:,1]/(IMAGE_NATIVE-1);n0,n1=len(c0),len(c1);bb=np.empty((n0*n1,4),np.float64);bb[:,0]=np.repeat(sx0,n1);bb[:,1]=np.repeat(sy0,n1);bb[:,2]=np.tile(sx1,n0);bb[:,3]=np.tile(sy1,n1);P=bb@pinv.T;ok=np.isfinite(P).all(1)&(np.abs(P)<=.75).all(1)
   if ok.any():blocks.append(P[ok].astype(np.float32))
 return np.concatenate(blocks,0) if blocks else np.empty((0,3),np.float32)

def bilinear_flow_samples(fl,xy):
 fl=np.asarray(fl,np.float32);xy=np.asarray(xy,np.float64);H,W=fl.shape[:2];x=np.clip(xy[:,0],0,W-1);y=np.clip(xy[:,1],0,H-1);x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);x1=np.minimum(W-1,x0+1);y1=np.minimum(H-1,y0+1);wx=(x-x0)[:,None];wy=(y-y0)[:,None]
 return ((1-wx)*(1-wy)*fl[y0,x0]+wx*(1-wy)*fl[y0,x1]+(1-wx)*wy*fl[y1,x0]+wx*wy*fl[y1,x1]).astype(np.float32)

def local_scale(P,k=12):
 P=np.asarray(P,np.float64);n=len(P);kk=min(max(1,int(k)),n-1);d2=((P[:,None,:]-P[None,:,:])**2).sum(-1);idx=np.argsort(d2,axis=1)[:,1:kk+1];dist=np.take_along_axis(np.sqrt(np.maximum(d2,0.0)),idx,axis=1);s=np.median(dist,axis=1);g=np.median(s[s>1e-12]) if np.any(s>1e-12) else 1.0;return np.where(s>1e-12,s,g)
def u_raw(H,PAi,usable,cands):
 vals=[]
 for v in usable:
  q=project_np(H,v); C=np.asarray(cands[v],float); d=np.sqrt(((q[:,None,:]-C[None,:,:])**2).sum(2));vals.append(d.min(1))
 return np.mean(np.stack(vals,1),1)
def dedup_1e5(H):
 if not len(H):return H
 keys=np.round(H,5);_,idx=np.unique(keys,axis=0,return_index=True);return H[np.sort(idx)]
def m256_from_H(H,U):
 n=len(H)
 order=np.lexsort((H[:,2],H[:,1],H[:,0],U)); initial=list(order[:min(32,n)])
 if n<=256:return np.array(initial+list(order[len(initial):]),int)[:n]
 selected=initial[:]; mask=np.ones(n,bool);mask[selected]=False; rem=np.where(mask)[0]; mind=np.full(n,np.inf)
 S=H[np.array(selected)]
 if len(S):
  for s in S:mind=np.minimum(mind,np.linalg.norm(H-s[None],axis=1))
 while len(selected)<256:
  rr=np.where(mask)[0]; mx=np.max(mind[rr]); cand=rr[np.isclose(mind[rr],mx,rtol=0,atol=1e-12)]
  if len(cand)>1:
   cc=sorted(cand.tolist(),key=lambda j:(U[j],float(H[j,0]),float(H[j,1]),float(H[j,2]),j));j=cc[0]
  else:j=int(cand[0])
  selected.append(j);mask[j]=False;mind=np.minimum(mind,np.linalg.norm(H-H[j][None],axis=1))
 return np.array(selected,int)
def robust_z(x):
 x=np.asarray(x,float);med=np.median(x);iq=max(float(np.quantile(x,.75)-np.quantile(x,.25)),1e-6);return (x-med)/iq

def build_family(fid):
 p=CACHE/f'{fid}.pkl'
 if p.exists():return pickle.load(open(p,'rb'))
 root=ROOT/'currentH'/DIR[fid];A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'));assert len(A)==len(B)==8
 x=load_model_tensor(A,B)
 with torch.no_grad():out=MODEL(x)
 PA,VA,pdiag=exact_problem_a_frontdoor(A,B,64,96);xyA=np.stack([project_np(PA,v) for v in range(8)],0).astype(np.float32)
 descA=sample_field_np(out['descriptor'][0,0],xyA); zA=normalize_rows(consensus_np(descA,VA));masks=[foreground_mask(q) for q in B]
 flows=forward_dis_flows(A,B);dis=np.stack([bilinear_flow_samples(flows[v],xyA[v]) for v in range(8)],0)
 z=np.load(root/'B'/'observation_sidecar.npz');center=np.asarray(z['camera_center'],float);half=float(z['camera_half_extent']);TA=(np.asarray(z['surface_points_A'],float)-center)/(2*half);TB=(np.asarray(z['surface_points_B'],float)-center)/(2*half);sA=local_scale(TA);sB=local_scale(TB);D=np.linalg.norm(PA[:,None,:]-TA[None,:,:],axis=2);j=D.argmin(1);maperr=D[np.arange(64),j];reliable=maperr<=2*sA[j];target=TB[j];truthamp=np.linalg.norm(target-TA[j],axis=1);active=truthamp>ACTIVITY_THRESHOLD
 nodes=[]
 for i in range(64):
  cands=[]
  for v in range(8):cands.append(per_view_top16(out['descriptor'][0,1,v],zA[i],masks[v]) if VA[v,i] else np.empty((0,2),np.float32))
  H=make_H_fast(cands);herr=np.linalg.norm(H-target[i][None],axis=1) if len(H) else np.array([]);full_cont=bool(len(H) and herr.min()<=2*sB[j[i]]);Hs=dedup_1e5(H);usable=[v for v in range(8) if VA[v,i] and len(cands[v])];U=u_raw(Hs,PA[i],usable,cands);mi=m256_from_H(Hs,U);M=Hs[mi];Um=U[mi];merr=np.linalg.norm(M-target[i][None],axis=1) if len(M) else np.array([]);mcont=bool(len(M) and merr.min()<=2*sB[j[i]])
  nodes.append({'cands':cands,'H_n':len(H),'full_cont2':full_cont,'M':M,'Uraw':Um,'M_cont2':mcont,'M_cont1':bool(len(M) and merr.min()<=sB[j[i]]),'scale':float(sB[j[i]]),'target':target[i],'reliable':bool(reliable[i]),'active':bool(active[i]),'j':int(j[i])})
 obj={'fid':fid,'PA':PA,'VA':VA,'dis':dis,'nodes':nodes,'pdiag':pdiag}
 pickle.dump(obj,open(p,'wb'),protocol=4);return obj

def preflight(fams):
 per={};fullper={};s1={}
 for fid,f in fams.items():
  rr=[n for n in f['nodes'] if n['reliable']];per[fid]=float(np.mean([n['M_cont2'] for n in rr]));fullper[fid]=float(np.mean([n['full_cont2'] for n in rr]));s1[fid]=float(np.mean([n['M_cont1'] for n in rr]))
 den=sum(n['reliable'] for f in fams.values() for n in f['nodes']);num=sum(n['reliable'] and n['M_cont2'] for f in fams.values() for n in f['nodes']);pooled=num/den;worst=min(per.values());best=max(per.values());gate=pooled>=.970 and worst>=.900 and sum(x>=.90 for x in per.values())==8 and best-worst<=.10
 parity={fid:abs(fullper[fid]-EXPECTED_F16[fid])<1e-8 for fid in FAMILIES}
 return {'pooled_2x':pooled,'den':den,'per_family_2x':per,'per_family_1x':s1,'full_F16_per_family_2x':fullper,'full_F16_parity':parity,'full_F16_parity_all':all(parity.values()),'worst_family_2x':worst,'best_worst_gap':best-worst,'families_ge_090':sum(x>=.90 for x in per.values()),'gate':bool(gate)}

def build_graph(f):
 PA=f['PA'];VA=f['VA'];D=np.linalg.norm(PA[:,None,:]-PA[None,:,:],axis=2);np.fill_diagonal(D,np.inf);E=set()
 for i in range(64):
  for j in np.argsort(D[i],kind='stable')[:4]:E.add(tuple(sorted((i,int(j)))))
 edges=[]
 for i,j in sorted(E):
  common=[v for v in range(8) if VA[v,i] and VA[v,j] and np.isfinite(f['dis'][v,i]).all() and np.isfinite(f['dis'][v,j]).all()]
  if len(common)>=2:edges.append((i,j,common))
 return edges

def pair_R(f,i,j,views):
 ni,nj=f['nodes'][i],f['nodes'][j];Mi,Mj=ni['M'],nj['M'];PA=f['PA'];vals=[]
 for v in views:
  di=project_np(Mi,v)-project_np(PA[i][None],v)[0];dj=project_np(Mj,v)-project_np(PA[j][None],v)[0];obs=f['dis'][v,i]-f['dis'][v,j];A=di[:,None,:]-dj[None,:,:]-obs[None,None,:];vals.append(np.linalg.norm(A,axis=2))
 raw=np.median(np.stack(vals,2),2);med=np.median(raw);iq=max(float(np.quantile(raw,.75)-np.quantile(raw,.25)),1e-6);return (raw-med)/iq

def solve_family(f):
 U=[robust_z(n['Uraw']) for n in f['nodes']];x=np.array([int(np.argmin(u)) for u in U],int);x0=x.copy();edges=build_graph(f);deg=np.zeros(64,int)
 for i,j,_ in edges:deg[i]+=1;deg[j]+=1
 adj=[[] for _ in range(64)];mats={}
 for i,j,vs in edges:
  R=pair_R(f,i,j,vs);w=1/max(deg[i],1)+1/max(deg[j],1);mats[(i,j)]=(R,w);adj[i].append((j,R,w,True));adj[j].append((i,R,w,False))
 changes=[]
 for sweep in range(20):
  ch=0;order=range(64) if sweep%2==0 else range(63,-1,-1)
  for i in order:
   cost=U[i].copy()
   for j,R,w,forward in adj[i]:cost += w*(R[:,x[j]] if forward else R[x[j],:])
   q=int(np.argmin(cost))
   if q!=x[i]:x[i]=q;ch+=1
  changes.append(ch)
  if ch==0:break
 def objective(xx):
  e=sum(U[i][xx[i]] for i in range(64))
  for i,j,_ in edges:R,w=mats[(i,j)];e+=w*R[xx[i],xx[j]]
  return float(e)
 return x0,x,{'edges':len(edges),'degrees':deg.tolist(),'sweeps':len(changes),'changed_nodes':changes,'objective_init':objective(x0),'objective_final':objective(x)}

def eval_arm(f,x,primary=True):
 rec=[]
 for i,n in enumerate(f['nodes']):
  eligible=n['reliable'] and n['M_cont2'] and ((n['active'] and n['full_cont2']) if primary else True)
  if not eligible:continue
  p=n['M'][x[i]];err=float(np.linalg.norm(p-n['target']));sc=n['scale'];rec.append((err/sc,err<=sc,err<=2*sc))
 if not rec:return {'n':0,'contain1':None,'contain2':None,'median_norm_err':None,'p90_norm_err':None}
 a=np.array([r[0] for r in rec]);return {'n':len(rec),'contain1':float(np.mean([r[1] for r in rec])),'contain2':float(np.mean([r[2] for r in rec])),'median_norm_err':float(np.median(a)),'p90_norm_err':float(np.quantile(a,.9))}
def aggregate_evals(per,arm,key='primary'):
 qs={fid:per[fid][key][arm] for fid in FAMILIES};n=sum(q['n'] for q in qs.values());c1=sum(q['contain1']*q['n'] for q in qs.values())/n;c2=sum(q['contain2']*q['n'] for q in qs.values())/n;errs=[]
 for fid in FAMILIES:
  pass
 vals=[q['contain2'] for q in qs.values()];return {'n':n,'contain1':c1,'contain2':c2,'worst_family_contain1':min(q['contain1'] for q in qs.values()),'worst_family_contain2':min(vals),'best_family_contain2':max(vals),'best_worst_gap':max(vals)-min(vals),'per_family':qs}
def all_errors(f,x,primary):
 e=[]
 for i,n in enumerate(f['nodes']):
  eligible=n['reliable'] and n['M_cont2'] and ((n['active'] and n['full_cont2']) if primary else True)
  if eligible:e.append(float(np.linalg.norm(n['M'][x[i]]-n['target'])/n['scale']))
 return e

def main():
 t=time.time();fams={}
 for fid in FAMILIES:
  st=time.time();fams[fid]=build_family(fid);print('BUILT',fid,'sec',round(time.time()-st,2),flush=True)
 pf=preflight(fams);(OUT/'PREFLIGHT.json').write_text(json.dumps(pf,indent=2));print('PREFLIGHT',json.dumps(pf,indent=2),flush=True)
 if not pf['full_F16_parity_all']:
  raise SystemExit('SOURCE_PARITY_FAIL')
 if not pf['gate']:
  result={'schema':'RealSaS.N1D.GlobalRelationalCandidateSolver.v2','verdict':'INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL_V2','preflight':pf,'seconds':time.time()-t};(OUT/'RESULT.json').write_text(json.dumps(result,indent=2));return
 per={};xs={}
 for fid,f in fams.items():
  st=time.time();u,g,diag=solve_family(f);xs[fid]=(u,g);per[fid]={'diag':diag,'primary':{'U_ONLY':eval_arm(f,u,True),'G_REL_DIS':eval_arm(f,g,True)},'secondary':{'U_ONLY':eval_arm(f,u,False),'G_REL_DIS':eval_arm(f,g,False)}};print('SOLVED',fid,'sec',round(time.time()-st,2),per[fid]['primary'],flush=True)
 agg={}
 for key in ['primary','secondary']:
  agg[key]={}
  for arm in ['U_ONLY','G_REL_DIS']:
   q=aggregate_evals(per,arm,key);errs=[]
   for fid in FAMILIES:errs+=all_errors(fams[fid],xs[fid][0 if arm=='U_ONLY' else 1],key=='primary')
   q['median_norm_err']=float(np.median(errs));q['p90_norm_err']=float(np.quantile(errs,.9));agg[key][arm]=q
 G=agg['primary']['G_REL_DIS'];U=agg['primary']['U_ONLY'];hard=[11032,13203,15290];gains={fid:per[fid]['primary']['G_REL_DIS']['contain2']-per[fid]['primary']['U_ONLY']['contain2'] for fid in hard};abs_gate=G['contain2']>=.75 and G['worst_family_contain2']>=.60 and all(per[fid]['primary']['G_REL_DIS']['contain2']>=.60 for fid in hard) and G['contain1']>=.50 and G['worst_family_contain1']>=.35 and G['median_norm_err']<=1.0 and G['best_worst_gap']<=.30; causal=(G['contain2']-U['contain2']>=.08) and ((G['worst_family_contain2']-U['worst_family_contain2']>=.08) or (sum(g>=.10 for g in gains.values())>=2 and all(g>=-.05 for g in gains.values())));safety=agg['secondary']['G_REL_DIS']['contain2']>=agg['secondary']['U_ONLY']['contain2']-.03;verdict='GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_PASS' if abs_gate and causal and safety else 'RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2'
 result={'schema':'RealSaS.N1D.GlobalRelationalCandidateSolver.v2','date':'2026-08-19','prereg':'GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_PREREG.md','checkpoint_sha256':sha256_file(ROOT/'srcpack/BEST.pt'),'preflight':pf,'per_family':per,'aggregate':agg,'hardtail_contain2_gains':gains,'gates':{'absolute':bool(abs_gate),'causal':bool(causal),'safety':bool(safety)},'verdict':verdict,'seconds':time.time()-t};p=OUT/'RESULT.json';p.write_text(json.dumps(result,indent=2,allow_nan=False));print('VERDICT',verdict,flush=True);print(json.dumps({'preflight':pf,'primary':agg['primary'],'secondary':agg['secondary'],'hardtail_gains':gains,'gates':result['gates']},indent=2),flush=True);print('RESULT_SHA256',sha256_file(p),flush=True)
if __name__=='__main__':main()
