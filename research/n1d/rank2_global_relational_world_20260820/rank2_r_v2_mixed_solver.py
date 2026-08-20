from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SCHEMA='RealSaS.N1D.Rank2GlobalRelationalWorld.V2MixedSingletonSet'
TOP_FRAC=.10; K_MIN=8; K_MAX=32; K_GRAPH=8
N_RESTARTS=12; MAX_SWEEPS=30; HUBER_DELTA=.25; EPS=1e-10
RANK_TOL=1e-8
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)


def camera_basis():
    rows=[]
    for deg in YAW_DEG:
        t=math.radians(deg)
        radial=np.array([math.sin(t),-math.cos(t),0.],dtype=np.float64)
        fw=-radial; up=np.array([0.,0.,1.],dtype=np.float64)
        right=np.cross(fw,up); right/=np.linalg.norm(right)
        rows.append((right,up))
    return np.stack([x[0] for x in rows]), np.stack([x[1] for x in rows])
CAM_R,CAM_U=camera_basis()


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rank01(x,descending=False):
    x=np.asarray(x,float); o=np.argsort(-x if descending else x,kind='stable'); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r/max(len(x)-1,1)
def huber_abs(x,delta=HUBER_DELTA):
    x=np.abs(np.asarray(x,float)); return np.where(x<=delta,.5*x*x,delta*(x-.5*delta))
def load_state(path):
    z=np.load(path,allow_pickle=False)
    req={'P_A','P_B','V_A','V_B','H_xyz','H_reproj_px','H_desc_score','H_offsets'}
    miss=req-set(z.files)
    if miss: raise RuntimeError(sorted(miss))
    return {k:np.asarray(z[k]) for k in req}
def robust_scale(P):
    P=np.asarray(P,float); D=np.linalg.norm(P[:,None]-P[None,:],axis=2); v=D[np.triu_indices(len(P),1)]; v=v[v>1e-9]; return float(np.median(v)) if len(v) else 1.

def projective_rank(Vcol):
    views=np.where(np.asarray(Vcol)>0)[0]
    if not len(views): return 0
    A=[]
    for v in views:
        A += [CAM_R[int(v)], -CAM_U[int(v)]]
    return int(np.linalg.matrix_rank(np.asarray(A),tol=RANK_TOL))
def fixed_variable_masks(VA,VB):
    ra=np.array([projective_rank(VA[:,i]) for i in range(64)],dtype=np.int64)
    rb=np.array([projective_rank(VB[:,i]) for i in range(64)],dtype=np.int64)
    fixed=(ra>=3)&(rb>=3); variable=~fixed
    return fixed,variable,ra,rb

def build_domains(H,rp,ds,off,variable):
    out=[None]*64
    for i in np.where(variable)[0]:
        a,b=map(int,(off[i],off[i+1])); pts=np.asarray(H[a:b],float); n=len(pts)
        if n==0: raise RuntimeError(('empty',int(i)))
        obs=.75*rank01(rp[a:b])+.25*rank01(ds[a:b],True)
        k=min(n,min(K_MAX,max(K_MIN,int(np.ceil(TOP_FRAC*n)))))
        keep=np.argsort(obs,kind='stable')[:k]
        out[int(i)]={'points':pts[keep],'g':obs[keep],'orig_global':(a+keep).astype(int),'raw_n':n}
    return out
def build_graph(PA):
    P=np.asarray(PA,float); D=np.linalg.norm(P[:,None]-P[None,:],axis=2); np.fill_diagonal(D,np.inf)
    edges=set()
    for i in range(64):
        for j in np.argsort(D[i],kind='stable')[:K_GRAPH]:
            a,b=sorted((i,int(j))); edges.add((a,b))
    return np.asarray(sorted(edges),dtype=np.int64)
def graph_matrix(points,edges,remove=None):
    P=np.asarray(points,float); row=[]; col=[]; data=[]
    for a,b in edges:
        a=int(a); b=int(b)
        if remove is not None and (a==remove or b==remove): continue
        w=max(float(np.linalg.norm(P[a]-P[b])),1e-12)
        row += [a,b]; col += [b,a]; data += [w,w]
    return csr_matrix((data,(row,col)),shape=(64,64))
def graph_address(points,edges,anchors):
    G=graph_matrix(points,edges)
    A=dijkstra(G,directed=False,indices=np.asarray(anchors,int))
    return np.asarray(A,float).T

def mixed_points(PB,domains,variable,choice):
    P=np.asarray(PB,float).copy()
    for i in np.where(variable)[0]: P[i]=domains[int(i)]['points'][int(choice[int(i)])]
    return P
def G_energy(domains,variable,choice):
    ids=np.where(variable)[0]
    if not len(ids): return 0.0
    return float(np.mean([domains[int(i)]['g'][int(choice[int(i)])] for i in ids]))
def R_energy(addrA,addrB,variable,path_scale):
    rows=np.where(variable)[0]
    if not len(rows): return 0.0
    a=addrA[rows]; b=addrB[rows]
    mask=np.isfinite(a)&np.isfinite(b)
    if not np.any(mask): return float('inf')
    x=(b[mask]-a[mask])/max(path_scale,1e-12)
    return float(np.mean(huber_abs(x)))

def exact_candidate_addresses(points,edges,anchors,i,candidates):
    P=np.asarray(points,float); i=int(i); cand=np.asarray(candidates,float)
    G0=graph_matrix(P,edges,remove=i); D0=np.asarray(dijkstra(G0,directed=False),float)
    nbs=[]
    for a,b in edges:
        a=int(a); b=int(b)
        if a==i: nbs.append(b)
        elif b==i: nbs.append(a)
    nbs=np.asarray(sorted(set(nbs)),int)
    if not len(nbs): raise RuntimeError(('isolated coordinate',i))
    anchors=np.asarray(anchors,int); baseA=D0[anchors]
    outs=[]
    for q in cand:
        w=np.linalg.norm(P[nbs]-q[None,:],axis=1)
        enter=np.min(D0[anchors[:,None],nbs[None,:]]+w[None,:],axis=1)
        exitv=np.min(w[:,None]+D0[nbs,:],axis=0)
        A=np.minimum(baseA,enter[:,None]+exitv[None,:]); A[:,i]=enter
        outs.append(A.T)
    return np.stack(outs)
def initial_choice(domains,variable,restart,family):
    c=np.full(64,-1,dtype=np.int64)
    rng=None if restart==0 else np.random.default_rng(np.uint64((family*1009+restart*9176)&0xffffffff))
    for i in np.where(variable)[0]:
        d=domains[int(i)]; g=d['g']
        if restart==0: c[i]=int(np.argmin(g))
        else:
            p=np.exp(-4*(g-np.min(g))); p/=p.sum(); c[i]=int(rng.choice(len(p),p=p))
    return c
def raw_energies(PB,domains,variable,choice,edges,anchors,addrA,path_scale):
    P=mixed_points(PB,domains,variable,choice)
    return G_energy(domains,variable,choice), R_energy(addrA,graph_address(P,edges,anchors),variable,path_scale)
def calibration(PB,domains,variable,c0,edges,anchors,addrA,path_scale):
    ids=np.where(variable)[0]
    if not len(ids): return 1.,1.,0.,0.,0,0
    pts=mixed_points(PB,domains,variable,c0); eg0=G_energy(domains,variable,c0); er0=R_energy(addrA,graph_address(pts,edges,anchors),variable,path_scale)
    dg=[]; dr=[]
    for i in ids:
        old=int(c0[i]); addrs=exact_candidate_addresses(pts,edges,anchors,int(i),domains[int(i)]['points'])
        for q,Aq in enumerate(addrs):
            if q==old: continue
            eg=eg0+(float(domains[int(i)]['g'][q])-float(domains[int(i)]['g'][old]))/len(ids)
            er=R_energy(addrA,Aq,variable,path_scale)
            a=abs(eg-eg0); b=abs(er-er0)
            if a>EPS and np.isfinite(a): dg.append(a)
            if b>EPS and np.isfinite(b): dr.append(b)
    gs=float(np.median(dg)) if dg else 1.; rs=float(np.median(dr)) if dr else 1.
    return max(gs,1e-8),max(rs,1e-8),eg0,er0,len(dg),len(dr)
def objective(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs):
    eg,er=raw_energies(PB,domains,variable,c,edges,anchors,addrA,path_scale); return eg/gs+er/rs,eg,er
def coordinate_costs(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs,i):
    ids=np.where(variable)[0]; pts=mixed_points(PB,domains,variable,c); old=int(c[i]); eg0=G_energy(domains,variable,c)
    addrs=exact_candidate_addresses(pts,edges,anchors,int(i),domains[int(i)]['points'])
    vals=[]
    for q,Aq in enumerate(addrs):
        eg=eg0+(float(domains[int(i)]['g'][q])-float(domains[int(i)]['g'][old]))/len(ids)
        er=R_energy(addrA,Aq,variable,path_scale); vals.append(eg/gs+er/rs)
    return np.asarray(vals)
def optimize(PB,domains,variable,edges,anchors,addrA,path_scale,gs,rs,family,restart):
    c=initial_choice(domains,variable,restart,family); sweeps=0
    for sw in range(MAX_SWEEPS):
        changed=0
        for i in np.where(variable)[0]:
            old=int(c[i]); costs=coordinate_costs(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs,int(i)); best=int(np.argmin(costs))
            if best!=old: c[i]=best; changed+=1
        sweeps=sw+1
        if not changed: break
    total,eg,er=objective(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs)
    return c,{'restart':restart,'sweeps':sweeps,'total':total,'E_G':eg,'E_R_mixed_graphpath':er}
def local_margin(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs):
    if not np.any(variable): return float('inf'),None
    base=objective(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs)[0]; m=float('inf'); arg=None
    for i in np.where(variable)[0]:
        old=int(c[i]); costs=coordinate_costs(PB,domains,variable,c,edges,anchors,addrA,path_scale,gs,rs,int(i))
        for q,e in enumerate(costs):
            if q==old: continue
            d=float(e-base)
            if d<m: m=d; arg=(int(i),old,q)
    return m,arg

def solve(state_path,family):
    s=load_state(state_path); PA=np.asarray(s['P_A'],float); PB=np.asarray(s['P_B'],float)
    fixed,variable,rankA,rankB=fixed_variable_masks(s['V_A'],s['V_B'])
    domains=build_domains(s['H_xyz'],s['H_reproj_px'],s['H_desc_score'],s['H_offsets'],variable)
    edges=build_graph(PA); anchors=np.where(fixed)[0].astype(int)
    if not len(anchors): raise RuntimeError('no fixed rank-3 anchors')
    addrA=graph_address(PA,edges,anchors); vals=addrA[np.isfinite(addrA)&(addrA>1e-12)]; path_scale=float(np.median(vals)) if len(vals) else 1.
    if not np.any(variable):
        selected=np.full(64,-1,dtype=int)
        return {'schema':SCHEMA,'family':int(family),'state_sha256':sha256_file(state_path),'truth_access':'NONE','rank':{'A':rankA.tolist(),'B':rankB.tolist(),'fixed_mask':fixed.tolist(),'variable_mask':variable.tolist(),'fixed_n':int(fixed.sum()),'variable_n':0},'graph_edges':edges.tolist(),'anchors':anchors.tolist(),'selected_original_global_index':selected.tolist(),'calibration':{'G_scale':1.,'R_scale':1.,'minG_E_G':0.,'minG_E_R':0.,'G_perturb_n':0,'R_perturb_n':0,'path_scale':path_scale},'objective':{'restart':0,'sweeps':0,'total':0.,'E_G':0.,'E_R_mixed_graphpath':0.},'stability':{'exact_restart_agreement':N_RESTARTS,'restart_count':N_RESTARTS,'all_restarts_exact':True,'local_counterfactual_min_margin':None,'local_counterfactual_arg':None,'locally_strict':True},'restarts':[]}
    c0=initial_choice(domains,variable,0,family); gs,rs,eg0,er0,ng,nr=calibration(PB,domains,variable,c0,edges,anchors,addrA,path_scale)
    runs=[]; choices=[]
    for r in range(N_RESTARTS):
        c,info=optimize(PB,domains,variable,edges,anchors,addrA,path_scale,gs,rs,family,r); choices.append(c.copy()); runs.append(info)
    order=sorted(range(len(runs)),key=lambda r:(runs[r]['total'],tuple(choices[r].tolist()))); bi=order[0]; best=choices[bi]; info=runs[bi]
    agree=sum(np.array_equal(c,best) for c in choices); margin,arg=local_margin(PB,domains,variable,best,edges,anchors,addrA,path_scale,gs,rs)
    selected=np.full(64,-1,dtype=int)
    for i in np.where(variable)[0]: selected[i]=int(domains[int(i)]['orig_global'][int(best[i])])
    return {'schema':SCHEMA,'family':int(family),'state_sha256':sha256_file(state_path),'truth_access':'NONE','parameters':{'top_frac':TOP_FRAC,'k_min':K_MIN,'k_max':K_MAX,'graph_k':K_GRAPH,'restarts':N_RESTARTS,'max_sweeps':MAX_SWEEPS,'huber_delta':HUBER_DELTA,'rank_tol':RANK_TOL},'rank':{'A':rankA.tolist(),'B':rankB.tolist(),'fixed_mask':fixed.tolist(),'variable_mask':variable.tolist(),'fixed_n':int(fixed.sum()),'variable_n':int(variable.sum())},'graph_edges':edges.tolist(),'anchors':anchors.tolist(),'candidate_counts_variable':{str(int(i)):int(domains[int(i)]['raw_n']) for i in np.where(variable)[0]},'selected_original_global_index':selected.tolist(),'calibration':{'G_scale':gs,'R_scale':rs,'minG_E_G':eg0,'minG_E_R':er0,'G_perturb_n':ng,'R_perturb_n':nr,'path_scale':path_scale},'objective':info,'stability':{'exact_restart_agreement':int(agree),'restart_count':N_RESTARTS,'all_restarts_exact':bool(agree==N_RESTARTS),'local_counterfactual_min_margin':float(margin),'local_counterfactual_arg':None if arg is None else list(map(int,arg)),'locally_strict':bool(margin>1e-12)},'restarts':runs}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError('refusing overwrite')
    r=solve(a.state,a.family); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'family':r['family'],'rank':{'fixed_n':r['rank']['fixed_n'],'variable_n':r['rank']['variable_n']},'calibration':r['calibration'],'objective':r['objective'],'stability':r['stability']},indent=2))
if __name__=='__main__': main()
