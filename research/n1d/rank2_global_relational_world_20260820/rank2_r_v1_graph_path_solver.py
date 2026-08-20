from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SCHEMA='RealSaS.N1D.Rank2GlobalRelationalWorld.V1GraphPath'
TOP_FRAC=.10; K_MIN=8; K_MAX=32; N_ANCHORS=20; K_GRAPH=8
N_RESTARTS=12; MAX_SWEEPS=30; HUBER_DELTA=.25; EPS=1e-10


def sha256_file(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rank01(x,descending=False):
    x=np.asarray(x,float); order=np.argsort(-x if descending else x,kind='stable'); r=np.empty(len(x)); r[order]=np.arange(len(x)); return r/max(len(x)-1,1)

def huber_abs(x,delta=HUBER_DELTA):
    x=np.abs(np.asarray(x,float)); return np.where(x<=delta,.5*x*x,delta*(x-.5*delta))

def load_state(path):
    z=np.load(path,allow_pickle=False); req={'P_A','P_B','H_xyz','H_reproj_px','H_desc_score','H_offsets'}
    miss=req-set(z.files)
    if miss: raise RuntimeError(sorted(miss))
    return tuple(np.asarray(z[k]) for k in ('P_A','P_B','H_xyz','H_reproj_px','H_desc_score','H_offsets'))

def robust_scale(P):
    P=np.asarray(P,float); D=np.linalg.norm(P[:,None]-P[None,:],axis=2); v=D[np.triu_indices(len(P),1)]; v=v[v>1e-9]; return float(np.median(v)) if len(v) else 1.

def build_domains(PA,PB,H,rp,ds,off):
    s=robust_scale(PA); out=[]
    for i in range(64):
        a,b=map(int,(off[i],off[i+1])); pts=np.asarray(H[a:b],float); n=len(pts)
        if n==0: raise RuntimeError(('empty',i))
        obs=.75*rank01(rp[a:b])+.25*rank01(ds[a:b],True)
        k=min(n,min(K_MAX,max(K_MIN,int(np.ceil(TOP_FRAC*n)))))
        keep=np.argsort(obs,kind='stable')[:k]; Q=pts[keep]; g=obs[keep]
        c=np.median(Q,axis=0); cov=np.cov((Q-c).T) if len(Q)>1 else np.zeros((3,3)); eig=np.maximum(np.linalg.eigvalsh(cov),0)
        spread=float(np.sqrt(eig[-1])/max(s,1e-9))
        ac=int(np.argmin(np.linalg.norm(Q-np.asarray(PB[i],float)[None,:],axis=1)))
        out.append({'points':Q,'g':g,'orig_global':(a+keep).astype(int),'spread':spread,'anchor_choice':ac,'raw_n':n})
    return out

def build_graph(PA):
    P=np.asarray(PA,float); D=np.linalg.norm(P[:,None]-P[None,:],axis=2); np.fill_diagonal(D,np.inf)
    edges=set()
    for i in range(64):
        for j in np.argsort(D[i],kind='stable')[:K_GRAPH]:
            a,b=sorted((i,int(j))); edges.add((a,b))
    return np.asarray(sorted(edges),dtype=np.int64)

def anchors_from_domains(dom):
    return np.asarray([x[2] for x in sorted((dom[i]['spread'],float(np.min(dom[i]['g'])),i) for i in range(64))[:N_ANCHORS]],int)

def graph_address(points,edges,anchors):
    P=np.asarray(points,float); row=[]; col=[]; data=[]
    for i,j in edges:
        w=float(np.linalg.norm(P[i]-P[j])); w=max(w,1e-12)
        row += [int(i),int(j)]; col += [int(j),int(i)]; data += [w,w]
    G=csr_matrix((data,(row,col)),shape=(64,64))
    A=dijkstra(G,directed=False,indices=np.asarray(anchors,int))
    return np.asarray(A,float).T

def R_energy(addrA,addrB,path_scale):
    mask=np.isfinite(addrA)&np.isfinite(addrB)
    if not np.any(mask): return float('inf')
    x=(addrB[mask]-addrA[mask])/max(path_scale,1e-12)
    return float(np.mean(huber_abs(x)))

def choice_points(dom,c): return np.stack([dom[i]['points'][int(c[i])] for i in range(64)])
def G_energy(dom,c): return float(np.mean([dom[i]['g'][int(c[i])] for i in range(64)]))
def raw_energies(dom,c,edges,anchors,addrA,path_scale):
    eg=G_energy(dom,c); addrB=graph_address(choice_points(dom,c),edges,anchors); er=R_energy(addrA,addrB,path_scale); return eg,er

def graph_matrix(points,edges,remove=None):
    P=np.asarray(points,float); row=[]; col=[]; data=[]
    for a,b in edges:
        a=int(a); b=int(b)
        if remove is not None and (a==remove or b==remove): continue
        w=max(float(np.linalg.norm(P[a]-P[b])),1e-12)
        row += [a,b]; col += [b,a]; data += [w,w]
    return csr_matrix((data,(row,col)),shape=(64,64))

def exact_candidate_addresses(points,edges,anchors,i,candidates):
    """Exact shortest-path address after changing only point i.
    Uses graph-with-i-removed distances. Positive edge weights imply a shortest path
    either avoids i or enters/leaves i once. Returns K x 64 x A address tensors.
    """
    P=np.asarray(points,float); i=int(i); cand=np.asarray(candidates,float)
    G0=graph_matrix(P,edges,remove=i)
    D0=np.asarray(dijkstra(G0,directed=False),float)
    nbs=[]
    for a,b in edges:
        a=int(a); b=int(b)
        if a==i: nbs.append(b)
        elif b==i: nbs.append(a)
    nbs=np.asarray(sorted(set(nbs)),int)
    if not len(nbs): raise RuntimeError(('isolated coordinate',i))
    baseA=D0[np.asarray(anchors,int)]
    outs=[]
    for q in cand:
        w=np.linalg.norm(P[nbs]-q[None,:],axis=1)
        enter=np.min(D0[np.asarray(anchors,int)[:,None],nbs[None,:]]+w[None,:],axis=1)
        exitv=np.min(w[:,None]+D0[nbs,:],axis=0)
        A=np.minimum(baseA,enter[:,None]+exitv[None,:])
        A[:,i]=enter
        outs.append(A.T)
    return np.stack(outs)

def calibration(dom,c0,edges,anchors,addrA,path_scale):
    pts=choice_points(dom,c0); eg0=G_energy(dom,c0); er0=R_energy(addrA,graph_address(pts,edges,anchors),path_scale)
    dg=[]; dr=[]; aset=set(map(int,anchors))
    for i in range(64):
        if i in aset: continue
        old=int(c0[i]); addrs=exact_candidate_addresses(pts,edges,anchors,i,dom[i]['points'])
        for q,Aq in enumerate(addrs):
            if q==old: continue
            eg=eg0+(float(dom[i]['g'][q])-float(dom[i]['g'][old]))/64.0; er=R_energy(addrA,Aq,path_scale)
            a=abs(eg-eg0); b=abs(er-er0)
            if a>EPS and np.isfinite(a): dg.append(a)
            if b>EPS and np.isfinite(b): dr.append(b)
    gs=float(np.median(dg)) if dg else 1.; rs=float(np.median(dr)) if dr else 1.
    return max(gs,1e-8),max(rs,1e-8),eg0,er0,len(dg),len(dr)
def objective(dom,c,edges,anchors,addrA,path_scale,gs,rs):
    eg,er=raw_energies(dom,c,edges,anchors,addrA,path_scale); return eg/gs+er/rs,eg,er

def coordinate_exact_costs(dom,c,edges,anchors,addrA,path_scale,gs,rs,i):
    pts=choice_points(dom,c); old=int(c[i]); eg0=G_energy(dom,c)
    addrs=exact_candidate_addresses(pts,edges,anchors,i,dom[i]['points'])
    vals=[]; ers=[]
    for q,Aq in enumerate(addrs):
        eg=eg0+(float(dom[i]['g'][q])-float(dom[i]['g'][old]))/64.0
        er=R_energy(addrA,Aq,path_scale); vals.append(eg/gs+er/rs); ers.append(er)
    return np.asarray(vals),np.asarray(ers)
def init_choice(dom,anchors,restart,family):
    c=np.array([int(np.argmin(d['g'])) for d in dom],int)
    if restart:
        rng=np.random.default_rng(np.uint64((family*1009+restart*9176)&0xffffffff))
        for i,d in enumerate(dom):
            p=np.exp(-4*(d['g']-np.min(d['g']))); p/=p.sum(); c[i]=int(rng.choice(len(p),p=p))
    for a in anchors: c[int(a)]=int(dom[int(a)]['anchor_choice'])
    return c
def optimize(dom,edges,anchors,addrA,path_scale,gs,rs,family,restart):
    c=init_choice(dom,anchors,restart,family); aset=set(map(int,anchors)); sweeps=0
    for sw in range(MAX_SWEEPS):
        changed=0
        for i in range(64):
            if i in aset: continue
            old=int(c[i]); costs,_=coordinate_exact_costs(dom,c,edges,anchors,addrA,path_scale,gs,rs,i)
            best=int(np.argmin(costs))
            if best!=old: c[i]=best; changed+=1
        sweeps=sw+1
        if not changed: break
    total,eg,er=objective(dom,c,edges,anchors,addrA,path_scale,gs,rs); return c,{'restart':restart,'sweeps':sweeps,'total':total,'E_G':eg,'E_R_graphpath':er}
def local_margin(dom,c,edges,anchors,addrA,path_scale,gs,rs):
    base=objective(dom,c,edges,anchors,addrA,path_scale,gs,rs)[0]; aset=set(map(int,anchors)); m=float('inf'); arg=None
    for i in range(64):
        if i in aset: continue
        old=int(c[i]); costs,_=coordinate_exact_costs(dom,c,edges,anchors,addrA,path_scale,gs,rs,i)
        for q,e in enumerate(costs):
            if q==old: continue
            d=float(e-base)
            if d<m: m=d; arg=(i,old,q)
    return m,arg
def solve(state_path,family):
    PA,PB,H,rp,ds,off=load_state(state_path); dom=build_domains(PA,PB,H,rp,ds,off); edges=build_graph(PA); anchors=anchors_from_domains(dom)
    addrA=graph_address(PA,edges,anchors); finite=addrA[np.isfinite(addrA)&(addrA>1e-12)]; path_scale=float(np.median(finite)) if len(finite) else 1.
    c0=init_choice(dom,anchors,0,family); gs,rs,eg0,er0,ng,nr=calibration(dom,c0,edges,anchors,addrA,path_scale)
    runs=[]; choices=[]
    for r in range(N_RESTARTS):
        c,info=optimize(dom,edges,anchors,addrA,path_scale,gs,rs,family,r); choices.append(c.copy()); runs.append(info)
    order=sorted(range(len(runs)),key=lambda r:(runs[r]['total'],tuple(choices[r].tolist()))); bi=order[0]; best=choices[bi]; info=runs[bi]
    agree=sum(np.array_equal(c,best) for c in choices); margin,arg=local_margin(dom,best,edges,anchors,addrA,path_scale,gs,rs)
    base_addr=graph_address(PB,edges,anchors); baseR=R_energy(addrA,base_addr,path_scale)
    return {'schema':SCHEMA,'family':int(family),'state_sha256':sha256_file(state_path),'truth_access':'NONE','parameters':{'top_frac':TOP_FRAC,'k_min':K_MIN,'k_max':K_MAX,'graph_k':K_GRAPH,'anchors':N_ANCHORS,'restarts':N_RESTARTS,'max_sweeps':MAX_SWEEPS,'huber_delta':HUBER_DELTA},'graph_edges':edges.tolist(),'anchors':anchors.tolist(),'calibration':{'G_scale':gs,'R_scale':rs,'minG_E_G':eg0,'minG_E_R':er0,'G_perturb_n':ng,'R_perturb_n':nr,'path_scale':path_scale},'selected_original_global_index':[int(dom[i]['orig_global'][int(best[i])]) for i in range(64)],'objective':{**info,'baseline_R_descriptive':baseR},'stability':{'exact_restart_agreement':int(agree),'restart_count':N_RESTARTS,'all_restarts_exact':bool(agree==N_RESTARTS),'local_counterfactual_min_margin':float(margin),'local_counterfactual_arg':None if arg is None else list(map(int,arg)),'locally_strict':bool(margin>1e-12)},'restarts':runs}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError('refusing overwrite')
    r=solve(a.state,a.family); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'family':r['family'],'calibration':r['calibration'],'objective':r['objective'],'stability':r['stability']},indent=2))
if __name__=='__main__': main()
