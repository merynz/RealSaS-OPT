#!/usr/bin/env python3
"""
RealSaS exploratory multi-asset P hard-tail discriminability comparison.

Inference-side evidence:
  * eight cel_clean 512 RGBA renders
  * known orthographic yaw cameras, half_extent=0.54
  * reference raster XY (query location)

Forbidden from inference:
  * teacher triangle IDs / barycentrics
  * teacher geometry / visibility
  * model predictions

Teacher raster+geometry are used only after scoring to evaluate the true forward depth.
No optimizer/training steps.
"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image

H=W=512
HALF_EXTENT=0.54
REF_VIEWS=[0,2,4,6]
ALL_VIEWS=list(range(8))
DEPTHS=np.linspace(-HALF_EXTENT,HALF_EXTENT,321,dtype=np.float64)
ROOT=Path('/mnt/data/multi_oracle')
ASSETS={
 'f089': {'asset_id':'asset_f089abadcd071194617d640b','model_p95_512':0.44874305129051195},
 'ea593':{'asset_id':'asset_ea593d044e14f20abe6d2818','model_p95_512':0.34310040324926333},
 '662ed':{'asset_id':'asset_662ed7f1e328bd85959157cf','model_p95_512':0.24911818504333483},
}

def basis(v):
    th=np.deg2rad(v*45.0)
    return (np.array([np.cos(th),-np.sin(th),0.0]),
            np.array([np.sin(th), np.cos(th),0.0]),
            np.array([0.0,0.0,1.0]))
BASE={v:basis(v) for v in ALL_VIEWS}

def load_asset(k):
    d=ROOT/k
    imgs={v:np.asarray(Image.open(d/f'V{v}.png').convert('RGBA'),dtype=np.float32)/255. for v in ALL_VIEWS}
    g=np.load(d/'primary_geometry.npz')
    return imgs,g

def raster_truth(d,view,V,F):
    r=np.load(d/f'V{view}_raster.npz')
    inds=r['pixel_linear_index'].astype(np.int64)
    tri=r['triangle_id'].astype(np.int64)
    uv=r['barycentric_uv'].astype(np.float64)
    ff=F[tri]
    w=np.stack([uv[:,0],uv[:,1],1.-uv[:,0]-uv[:,1]],1)
    P=(w[:,:,None]*V[ff]).sum(1)
    row,col=inds//1024,inds%1024
    x=(col//2).astype(np.int64); y=(row//2).astype(np.int64)
    pix=y*512+x
    _,ix=np.unique(pix,return_index=True)
    return x[ix],y[ix],P[ix]

def bilinear(im,xs,ys):
    h,w,c=im.shape
    x0=np.floor(xs).astype(np.int64); y0=np.floor(ys).astype(np.int64)
    x1=x0+1; y1=y0+1
    ok=(x0>=0)&(y0>=0)&(x1<w)&(y1<h)
    out=np.zeros((len(xs),c),np.float64)
    if ok.any():
        xa,xb,ya,yb=x0[ok],x1[ok],y0[ok],y1[ok]
        xv,yv=xs[ok],ys[ok]
        wx=xv-xa; wy=yv-ya
        out[ok]=(im[ya,xa]*(1-wx)[:,None]*(1-wy)[:,None]+
                 im[ya,xb]*wx[:,None]*(1-wy)[:,None]+
                 im[yb,xa]*(1-wx)[:,None]*wy[:,None]+
                 im[yb,xb]*wx[:,None]*wy[:,None])
    return out,ok

def per_target_costs(imgs,rv,x,y):
    n=len(x); D=len(DEPTHS)
    right,forward,up=BASE[rv]
    sr=HALF_EXTENT*((x+.5)/W*2.-1.)
    su=HALF_EXTENT*(1.-(y+.5)/H*2.)
    P=sr[:,None,None]*right+su[:,None,None]*up+DEPTHS[None,:,None]*forward
    ref=imgs[rv][y,x]
    targets=[]; costs=[]
    for tv in ALL_VIEWS:
        if tv==rv: continue
        tr=P@BASE[tv][0]; tu=P[:,:,2]
        xs=(tr/HALF_EXTENT+1.)*.5*W-.5
        ys=(1.-tu/HALF_EXTENT)*.5*H-.5
        s,ok=bilinear(imgs[tv],xs.ravel(),ys.ravel())
        s=s.reshape(n,D,4); ok=ok.reshape(n,D)
        rgb=np.mean(np.abs(s[:,:,:3]-ref[:,None,:3]),axis=2)
        alpha=s[:,:,3]
        c=rgb+0.75*np.maximum(0.,0.5-alpha)*2.
        c[~ok]=1.5
        targets.append(tv); costs.append(c)
    return np.stack(costs,2),targets

def aggregate(per,targets,rv,mode):
    if mode=='mean7': return per.mean(2)
    if mode=='median7': return np.median(per,axis=2)
    if mode=='best6': return np.sort(per,axis=2)[:,:,:6].mean(2)
    if mode=='best4': return np.sort(per,axis=2)[:,:,:4].mean(2)
    if mode=='perp2_mean':
        idx=[i for i,t in enumerate(targets) if ((t-rv)%8) in (2,6)]
        assert len(idx)==2
        return per[:,:,idx].mean(2)
    if mode=='parallax_weighted':
        w=np.asarray([abs(math.sin(math.radians((t-rv)*45.0))) for t in targets],np.float64)
        assert w.sum()>0
        return (per*w[None,None,:]).sum(2)/w.sum()
    raise KeyError(mode)

def local_texture(img,x,y,r=2):
    rgb=img[:,:,:3]; out=[]
    for xx,yy in zip(x,y):
        p=rgb[max(0,yy-r):min(H,yy+r+1),max(0,xx-r):min(W,xx+r+1)]
        out.append(float(np.mean(np.std(p.reshape(-1,3),axis=0))))
    return np.asarray(out)

def qq(a,p): return float(np.quantile(a,p)) if len(a) else None

def eval_cost(C,td):
    bi=C.argmin(1); pd=DEPTHS[bi]; e=np.abs(pd-td)
    ti=np.abs(DEPTHS[None,:]-td[:,None]).argmin(1)
    truth_cost=C[np.arange(len(td)),ti]
    rank=1+np.sum(C < truth_cost[:,None]-1e-12,axis=1)
    far=np.abs(DEPTHS[None,:]-td[:,None])>=.012
    alt=np.min(np.where(far,C,np.inf),axis=1)
    margin=alt-truth_cost
    return e,rank,margin

MODES=['mean7','best6','best4','median7','parallax_weighted','perp2_mean']
out={
 'schema':'RealSaS.IRIS.PHardTail.MultiAssetAll8CPUOracle.v1',
 'status':'EXPLORATORY_COMPARATIVE__NOT_FORMAL_GATE',
 'reference_views':REF_VIEWS,'target_view_domain':ALL_VIEWS,
 'candidate_depth_count':len(DEPTHS),'half_extent':HALF_EXTENT,
 'optimizer_steps':0,
 'teacher_geometry_used_for_inference':False,
 'teacher_visibility_used_for_inference':False,
 'teacher_raster_geometry_used_for_evaluation_only':True,
 'assets':{}
}
for ai,(k,meta) in enumerate(ASSETS.items()):
    imgs,g=load_asset(k); V=g['vertices'].astype(np.float64); F=g['faces'].astype(np.int64)
    accum={m:{'e':[],'r':[],'margin':[],'low':[],'hi':[]} for m in MODES}
    views={}
    for rv in REF_VIEWS:
        x,y,P=raster_truth(ROOT/k,rv,V,F)
        rng=np.random.default_rng(20260826+ai*100+rv)
        n=min(1400,len(x)); ix=rng.choice(len(x),n,replace=False)
        x=x[ix];y=y[ix];P=P[ix]
        td=P@BASE[rv][1]
        tex=local_texture(imgs[rv],x,y); threshold=np.quantile(tex,.35); low=tex<=threshold
        per,targets=per_target_costs(imgs,rv,x,y)
        vo={'n':int(n),'texture_p35':float(threshold)}
        for m in MODES:
            C=aggregate(per,targets,rv,m); e,r,margin=eval_cost(C,td)
            accum[m]['e'].append(e);accum[m]['r'].append(r);accum[m]['margin'].append(margin)
            accum[m]['low'].append(e[low]);accum[m]['hi'].append(e[~low])
            vo[m]={
                'p50':qq(e,.5),'p90':qq(e,.9),'p95':qq(e,.95),
                'top1':float(np.mean(r<=1)),'top4':float(np.mean(r<=4)),'top8':float(np.mean(r<=8)),
                'truth_positive_margin_frac':float(np.mean(margin>0)),
                'truth_margin_median':qq(margin,.5),
                'low_texture_p90':qq(e[low],.9),'higher_texture_p90':qq(e[~low],.9),
            }
        views[f'V{rv}']=vo
    ag={}
    for m in MODES:
        e=np.concatenate(accum[m]['e']); r=np.concatenate(accum[m]['r']); mar=np.concatenate(accum[m]['margin'])
        low=np.concatenate(accum[m]['low']); hi=np.concatenate(accum[m]['hi'])
        ag[m]={
            'n':int(len(e)),'p50':qq(e,.5),'p90':qq(e,.9),'p95':qq(e,.95),
            'top1':float(np.mean(r<=1)),'top4':float(np.mean(r<=4)),'top8':float(np.mean(r<=8)),
            'truth_positive_margin_frac':float(np.mean(mar>0)),'truth_margin_median':qq(mar,.5),
            'low_texture_p90':qq(low,.9),'higher_texture_p90':qq(hi,.9),
        }
    out['assets'][k]={'asset_id':meta['asset_id'],'model_p95_512':meta['model_p95_512'],'views':views,'aggregate':ag}
    print('\n',k,meta['asset_id'])
    for m in MODES:
        z=ag[m]; print(f" {m:18s} p50={z['p50']:.5f} p90={z['p90']:.5f} p95={z['p95']:.5f} top4={z['top4']:.3f} top8={z['top8']:.3f}")

Path('/mnt/data/MULTI_ASSET_ALL8_CPU_ORACLE_V1_METRICS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
