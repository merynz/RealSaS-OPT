from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA='RealSaS.N1D.Rank2R.V3A.ProposalIdentity.v1'
CONTRACT_COMMIT='fa5b18530cfa72a67b2ecb314337b64f2c2b69c4'
N_ANCHORS=6
HUBER_DELTA=.25
ABSTAIN_COST=2.5
FORBIDDEN_COST=1e6


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def array_digest(arrays)->str:
    h=hashlib.sha256()
    for name,a in arrays:
        a=np.ascontiguousarray(a)
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()
def rank01(vals,descending=False):
    x=np.asarray(vals,float); o=np.argsort(-x if descending else x,kind='stable'); r=np.empty(len(x),float); r[o]=np.arange(len(x),dtype=float); return r/max(len(x)-1,1)
def huber_abs(x,delta=HUBER_DELTA):
    x=np.abs(np.asarray(x,float)); return np.where(x<=delta,.5*x*x,delta*(x-.5*delta))
def scene_scale(P):
    P=np.asarray(P,float)
    if len(P)<2:return 1.
    D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2); v=D[np.triu_indices(len(P),1)]; v=v[v>1e-9]
    return float(np.median(v)) if len(v) else 1.
def load_obs_module(science_dir:Path):
    sys.path.insert(0,str(science_dir))
    spec=importlib.util.spec_from_file_location('v3a_obs_exact',science_dir/'observable_phase.py'); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod

def extract(root:Path,science_dir:Path,state_path:Path,family:int,episode:str):
    obs=load_obs_module(science_dir)
    meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE': raise RuntimeError('observable state truth contract invalid')
    z=np.load(state_path,allow_pickle=False)
    m,A,B,PA,VA,seedB,output_idx,hb,prb,diag,field,xyA=obs._model_context(root,family,episode,meta['route'])
    P=np.asarray(PA[output_idx],np.float32); V=np.asarray(VA[:,output_idx],np.uint8)
    if not np.array_equal(P,np.asarray(z['P_A'],np.float32)): raise RuntimeError('P_A replay mismatch')
    if not np.array_equal(V,np.asarray(z['V_A'],np.uint8)): raise RuntimeError('V_A replay mismatch')
    XYA=np.asarray(z['XY_A'],np.float32)
    if not np.array_equal(np.stack([m.project_points(PA[output_idx],v) for v in range(8)]).astype(np.float32),XYA): raise RuntimeError('XY_A replay mismatch')
    zA=m.descriptor_consensus(m.sample_field_np(field['descriptor'][0,0],xyA),VA)
    masks=[m.foreground_mask(p) for p in B]
    cache=[m.prepare_b_search(field['descriptor'][0,1,v],masks[v]) for v in range(8)]
    coords=np.full((64,8,4,2),np.nan,np.float32); scores=np.full((64,8,4),np.nan,np.float32); valid=np.zeros((64,8,4),np.uint8)
    for oi,gi0 in enumerate(output_idx):
        gi=int(gi0)
        for v in range(8):
            if bool(VA[v,gi]):
                q,s=m.top4_with_scores(field['descriptor'][0,1,v],zA[gi],masks[v],cache[v]); n=min(4,len(q))
                coords[oi,v,:n]=q[:n]; scores[oi,v,:n]=s[:n]; valid[oi,v,:n]=1
    digest=array_digest([('coords',coords),('scores',scores),('valid',valid),('V_A',V),('XY_A',XYA),('output_idx',np.asarray(output_idx,np.int32))])
    return coords,scores,valid,V,XYA,np.asarray(output_idx,np.int32),digest

def row_sites(coords,scores,valid,i,v):
    best={}
    for k in range(4):
        if not valid[i,v,k]:continue
        key=tuple(map(int,np.rint(coords[i,v,k]).astype(int)))
        sc=float(scores[i,v,k])
        if key not in best or sc>best[key]:best[key]=sc
    items=sorted(best.items(),key=lambda kv:(-kv[1],kv[0][1],kv[0][0]))
    return items

def solve_view(coords,scores,valid,VA,XYA,v):
    carriers=[int(i) for i in np.where(VA[v]>0)[0]]
    rows={i:row_sites(coords,scores,valid,i,v) for i in carriers}
    if any(len(rows[i])<2 for i in carriers): raise RuntimeError(('fewer than 2 proposals',v,[i for i in carriers if len(rows[i])<2]))
    # high-confidence anchors: top1-top2 descriptor margin, greedy unique top1 site
    ranked=sorted(carriers,key=lambda i:(-(rows[i][0][1]-rows[i][1][1]),i))
    anchors=[]; reserved=set()
    for i in ranked:
        site=rows[i][0][0]
        if site in reserved:continue
        anchors.append(i); reserved.add(site)
        if len(anchors)==N_ANCHORS:break
    if len(anchors)!=N_ANCHORS: raise RuntimeError(('insufficient unique anchors',v,len(anchors)))
    anchor_site={i:rows[i][0][0] for i in anchors}
    all_sites=sorted({site for i in carriers for site,_ in rows[i]},key=lambda p:(p[1],p[0]))
    scaleA=scene_scale(XYA[v,carriers]); scaleB=scene_scale(np.asarray(all_sites,float))
    assigned={i:anchor_site[i] for i in anchors}; abstained=[]
    non=[i for i in carriers if i not in set(anchors)]
    avail=[s for s in all_sites if s not in reserved]
    site_col={s:j for j,s in enumerate(avail)}
    C=np.full((len(non),len(avail)+len(non)),FORBIDDEN_COST,float)
    row_debug={}
    for rr,i in enumerate(non):
        items=rows[i]; sites=[x[0] for x in items]; sc=np.asarray([x[1] for x in items],float)
        gr=rank01(sc,descending=True)
        rraw=[]
        for site in sites:
            vals=[]
            bxy=np.asarray(site,float)
            for a in anchors:
                da=(np.asarray(XYA[v,i],float)-np.asarray(XYA[v,a],float))/max(scaleA,1e-9)
                db=(bxy-np.asarray(anchor_site[a],float))/max(scaleB,1e-9)
                ra=np.linalg.norm(da); rb=np.linalg.norm(db); ua=da/max(ra,1e-9); ub=db/max(rb,1e-9)
                feat=np.r_[db-da,rb-ra,ub-ua]
                vals.extend(huber_abs(feat).tolist())
            rraw.append(float(np.mean(vals)))
        rrk=rank01(np.asarray(rraw),descending=False)
        costs=gr+rrk
        for site,cost in zip(sites,costs):
            if site in reserved:continue
            if site in site_col:C[rr,site_col[site]]=float(cost)
        C[rr,len(avail)+rr]=ABSTAIN_COST
        row_debug[i]={'g_top1':list(items[0][0]),'g_margin':float(items[0][1]-items[1][1]),'sites':[list(s) for s in sites],'descriptor_scores':sc.tolist(),'G_rank':gr.tolist(),'R_raw':rraw,'R_rank':rrk.tolist(),'combined':costs.tolist()}
    if non:
        ri,ci=linear_sum_assignment(C)
        if len(ri)!=len(non):raise RuntimeError(('assignment row coverage',v,len(ri),len(non)))
        obj=0.
        for r,c in zip(ri,ci):
            i=non[int(r)]; val=float(C[r,c]); obj+=val
            if c<len(avail) and val<FORBIDDEN_COST/2:assigned[i]=avail[int(c)]
            elif c==len(avail)+int(r):abstained.append(i)
            else:raise RuntimeError(('forbidden assignment',v,i,c,val))
    else:obj=0.
    if len(set(assigned.values()))!=len(assigned):raise RuntimeError(('injectivity failure',v))
    # audit max real matching cardinality (anchors not fixed) using deterministic augmenting paths
    # simple DFS Kuhn over candidate site keys
    owner={}
    def aug(i,seen):
        for site,_ in rows[i]:
            if site in seen:continue
            seen.add(site)
            if site not in owner or aug(owner[site],seen):owner[site]=i;return True
        return False
    matched=sum(bool(aug(i,set())) for i in carriers)
    return {'view':int(v),'visible_n':len(carriers),'unique_site_n':len(all_sites),'max_real_matching_n':int(matched),'anchors':anchors,'anchor_sites':{str(i):list(anchor_site[i]) for i in anchors},'g_top1':{str(i):list(rows[i][0][0]) for i in carriers},'assigned':{str(i):list(assigned[i]) for i in sorted(assigned)},'abstained':sorted(abstained),'abstain_n':len(abstained),'objective_nonanchor':float(obj),'scale_A':scaleA,'scale_B':scaleB,'row_debug':{str(k):v for k,v in row_debug.items()}}

def solve_family(root:Path,science_dir:Path,state_path:Path,family:int,episode:str,out_evidence:Path):
    coords,scores,valid,VA,XYA,output_idx,digest=extract(root,science_dir,state_path,family,episode)
    out_evidence.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_evidence,coords=coords,scores=scores,valid=valid,V_A=VA,XY_A=XYA,output_idx=output_idx)
    views=[solve_view(coords,scores,valid,VA,XYA,v) for v in range(8)]
    return {'schema':SCHEMA,'contract_commit':CONTRACT_COMMIT,'family':int(family),'episode':episode,'truth_access':'NONE','state_sha256':sha256_file(state_path),'proposal_evidence_content_sha256':digest,'proposal_evidence_file_sha256':sha256_file(out_evidence),'views':views,'audit':{'full_real_matching_views':int(sum(x['max_real_matching_n']==x['visible_n'] for x in views)),'visible_total':int(sum(x['visible_n'] for x in views)),'assigned_real_total':int(sum(len(x['assigned']) for x in views)),'abstain_total':int(sum(x['abstain_n'] for x in views))}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--science-dir',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',default='e01');ap.add_argument('--out-evidence',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists() or a.out_evidence.exists():raise RuntimeError('refusing overwrite')
    r=solve_family(a.root,a.science_dir,a.state,a.family,a.episode,a.out_evidence);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True));print(json.dumps({'family':r['family'],'audit':r['audit'],'views':[{'v':x['view'],'visible':x['visible_n'],'match':x['max_real_matching_n'],'abstain':x['abstain_n']} for x in r['views']]},indent=2))
if __name__=='__main__':main()
