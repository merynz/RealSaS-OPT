from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA='RealSaS.N1D.Rank2R.V3C.MinMarginalGR.v1'
CONTRACT_COMMIT='a5f9f3b8d9d201b0c2175c14c2e24ce0139faeab'
V3A_SELECTION_SHA256='82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
N_ANCHORS=6
HUBER_DELTA=.25
ABSTAIN_COST=2.5
FORBIDDEN_COST=1e6
DELTA_TOL=1e-10


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def array_digest(arrays)->str:
    h=hashlib.sha256()
    for name,a in arrays:
        a=np.ascontiguousarray(a); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.tobytes())
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
    sys.path.insert(0,str(science_dir)); spec=importlib.util.spec_from_file_location('v3c_obs_exact',science_dir/'observable_phase.py'); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; sys.modules['v3c_obs_exact']=mod; spec.loader.exec_module(mod); return mod

def row_sites(coords,scores,valid,i,v):
    best={}
    for k in range(4):
        if not valid[i,v,k]: continue
        key=tuple(map(int,np.rint(coords[i,v,k]).astype(int))); sc=float(scores[i,v,k])
        if key not in best or sc>best[key]: best[key]=sc
    return sorted(best.items(),key=lambda kv:(-kv[1],kv[0][1],kv[0][0]))

def build_view_problem(coords,scores,valid,VA,XYA,v):
    carriers=[int(i) for i in np.where(VA[v]>0)[0]]; rows={i:row_sites(coords,scores,valid,i,v) for i in carriers}
    if any(len(rows[i])<2 for i in carriers): raise RuntimeError(('fewer than 2 proposals',v,[i for i in carriers if len(rows[i])<2]))
    ranked=sorted(carriers,key=lambda i:(-(rows[i][0][1]-rows[i][1][1]),i)); anchors=[]; reserved=set()
    for i in ranked:
        site=rows[i][0][0]
        if site in reserved: continue
        anchors.append(i); reserved.add(site)
        if len(anchors)==N_ANCHORS: break
    if len(anchors)!=N_ANCHORS: raise RuntimeError(('insufficient unique anchors',v,len(anchors)))
    anchor_site={i:rows[i][0][0] for i in anchors}; all_sites=sorted({site for i in carriers for site,_ in rows[i]},key=lambda p:(p[1],p[0]))
    scaleA=scene_scale(XYA[v,carriers]); scaleB=scene_scale(np.asarray(all_sites,float)); non=[i for i in carriers if i not in set(anchors)]; avail=[s for s in all_sites if s not in reserved]; site_col={s:j for j,s in enumerate(avail)}
    C=np.full((len(non),len(avail)+len(non)),FORBIDDEN_COST,float); debug={}; row_index={i:r for r,i in enumerate(non)}
    for rr,i in enumerate(non):
        items=rows[i]; sites=[x[0] for x in items]; sc=np.asarray([x[1] for x in items],float); gr=rank01(sc,descending=True); rraw=[]
        for site in sites:
            vals=[]; bxy=np.asarray(site,float)
            for a in anchors:
                da=(np.asarray(XYA[v,i],float)-np.asarray(XYA[v,a],float))/max(scaleA,1e-9); db=(bxy-np.asarray(anchor_site[a],float))/max(scaleB,1e-9)
                ra=np.linalg.norm(da); rb=np.linalg.norm(db); ua=da/max(ra,1e-9); ub=db/max(rb,1e-9); vals.extend(huber_abs(np.r_[db-da,rb-ra,ub-ua]).tolist())
            rraw.append(float(np.mean(vals)))
        rrk=rank01(np.asarray(rraw),descending=False); costs=gr+rrk
        for site,cost in zip(sites,costs):
            if site in reserved: continue
            if site in site_col: C[rr,site_col[site]]=float(cost)
        C[rr,len(avail)+rr]=ABSTAIN_COST
        debug[i]={'sites':sites,'scores':sc,'G_rank':gr,'R_raw':np.asarray(rraw),'R_rank':rrk,'combined':costs}
    ri,ci=linear_sum_assignment(C) if len(non) else (np.empty(0,int),np.empty(0,int)); base=float(C[ri,ci].sum()) if len(non) else 0.
    assigned={i:anchor_site[i] for i in anchors}; abstained=[]
    for r,c in zip(ri,ci):
        i=non[int(r)]; val=float(C[r,c])
        if c<len(avail) and val<FORBIDDEN_COST/2: assigned[i]=avail[int(c)]
        elif c==len(avail)+int(r): abstained.append(i)
        else: raise RuntimeError(('forbidden base assignment',v,i,int(c),val))
    return {'carriers':carriers,'rows':rows,'anchors':anchors,'reserved':reserved,'anchor_site':anchor_site,'non':non,'avail':avail,'site_col':site_col,'C':C,'row_index':row_index,'base_objective':base,'assigned':assigned,'abstained':sorted(abstained),'debug':debug}

def forced_delta(problem,i,site):
    rr=problem['row_index'][i]
    if site not in problem['site_col']: return float('inf')
    cc=problem['site_col'][site]; C=problem['C']; val=float(C[rr,cc])
    if val>=FORBIDDEN_COST/2: return float('inf')
    rem_rows=[r for r in range(C.shape[0]) if r!=rr]; rem_cols=[c for c in range(C.shape[1]) if c!=cc]
    if rem_rows:
        sub=C[np.ix_(rem_rows,rem_cols)]; ri,ci=linear_sum_assignment(sub); forced=val+float(sub[ri,ci].sum())
    else: forced=val
    delta=forced-problem['base_objective']
    if delta < -DELTA_TOL: raise RuntimeError(('negative minmarginal',i,site,delta))
    return max(0.,float(delta))

def proposal_supports(coords,scores,valid,VA,XYA,selection_fam):
    per_view=[]; min_delta=float('inf'); max_replay_obj=0.
    frozen={int(x['view']):x for x in selection_fam['views']}
    for v in range(8):
        pr=build_view_problem(coords,scores,valid,VA,XYA,v); fr=frozen[v]
        if pr['anchors']!=[int(x) for x in fr['anchors']]: raise RuntimeError(('anchor replay',v))
        if {str(i):list(pr['anchor_site'][i]) for i in pr['anchors']}!=fr['anchor_sites']: raise RuntimeError(('anchor site replay',v))
        if {str(i):list(pr['assigned'][i]) for i in sorted(pr['assigned'])}!=fr['assigned'] or pr['abstained']!=[int(x) for x in fr['abstained']]: raise RuntimeError(('assignment replay',v))
        max_replay_obj=max(max_replay_obj,abs(pr['base_objective']-float(fr['objective_nonanchor'])))
        sup={}
        for i in pr['carriers']:
            items=pr['rows'][i]
            if i in set(pr['anchors']):
                sc=np.asarray([x[1] for x in items],float); rk=rank01(sc,descending=True); sup[i]=[(items[k][0],float(rk[k]),0.0) for k in range(len(items))]
            else:
                vals=[]
                for site,_ in items:
                    d=forced_delta(pr,i,site)
                    if np.isfinite(d): vals.append((site,d))
                if not vals: sup[i]=[]; continue
                ds=np.asarray([x[1] for x in vals],float); rk=rank01(ds,descending=False)
                sup[i]=[(vals[k][0],float(rk[k]),float(vals[k][1])) for k in range(len(vals))]
                min_delta=min(min_delta,float(np.min(ds)))
        per_view.append({'view':v,'supports':sup,'base_objective':pr['base_objective']})
    return per_view,{'assignment_replay':'PASS','max_objective_abs_replay':max_replay_obj,'min_finite_delta':None if not np.isfinite(min_delta) else min_delta}

def baseline_observable_score(m,PB,coords,scores,valid,VA,i):
    er=[]; ds=[]
    for v in range(8):
        if not bool(VA[v,i]): continue
        mask=np.asarray(valid[i,v],bool); Q=np.asarray(coords[i,v][mask],float); S=np.asarray(scores[i,v][mask],float)
        if not len(Q): continue
        q=np.asarray(m.project_points(np.asarray(PB[i],np.float32)[None,:],v)[0],float); dist=np.linalg.norm(Q-q[None,:],axis=1); j=int(np.argmin(dist)); er.append(float(dist[j])); ds.append(float(S[j]))
    if not er: raise RuntimeError(('no baseline proposal support',i))
    return float(np.mean(er)),float(np.mean(ds))

def candidate_R_raw(m,p,i,VA,per_view):
    vals=[]
    for v in range(8):
        if not bool(VA[v,i]): continue
        candidates=per_view[v]['supports'].get(i,[])
        if not candidates: continue
        sites=np.asarray([x[0] for x in candidates],float); ranks=np.asarray([x[1] for x in candidates],float); q=np.asarray(m.project_points(np.asarray(p,np.float32)[None,:],v)[0],float); j=int(np.argmin(np.linalg.norm(sites-q[None,:],axis=1))); vals.append(float(ranks[j]))
    if not vals: raise RuntimeError(('no R support views',i))
    return float(np.mean(vals))

def solve_family(science_dir,state_path,evidence_path,selection_path,family,out_world):
    if sha256_file(selection_path)!=V3A_SELECTION_SHA256: raise RuntimeError('V3A selection SHA mismatch')
    sel=json.loads(selection_path.read_text()); famrec=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
    if famrec is None: raise RuntimeError('selection family missing')
    if sha256_file(state_path)!=famrec['state_sha256']: raise RuntimeError('state SHA mismatch')
    z=np.load(state_path,allow_pickle=False); meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE': raise RuntimeError('state truth contract invalid')
    ez=np.load(evidence_path,allow_pickle=False); coords=np.asarray(ez['coords']); scores=np.asarray(ez['scores']); valid=np.asarray(ez['valid']); VA=np.asarray(ez['V_A']); XYA=np.asarray(ez['XY_A'])
    dig=array_digest([('coords',coords),('scores',scores),('valid',valid),('V_A',VA),('XY_A',XYA),('output_idx',np.asarray(ez['output_idx'],np.int32))])
    if dig!=famrec['proposal_evidence_content_sha256']: raise RuntimeError('proposal evidence digest mismatch')
    if not np.array_equal(VA,np.asarray(z['V_A'])) or not np.array_equal(XYA,np.asarray(z['XY_A'])): raise RuntimeError('proposal/state observation mismatch')
    per_view,replay=proposal_supports(coords,scores,valid,VA,XYA,famrec)
    obs=load_obs_module(science_dir); m=obs._route_module(meta['route'])
    PA=np.asarray(z['P_A'],np.float32); PB=np.asarray(z['P_B'],np.float32); H=np.asarray(z['H_xyz'],np.float32); HR=np.asarray(z['H_reproj_px'],np.float64); HD=np.asarray(z['H_desc_score'],np.float64); off=np.asarray(z['H_offsets'],np.int64)
    PBnew=PB.copy(); rows=[]; appended_baseline=0
    for i in range(64):
        a,b=int(off[i]),int(off[i+1]); Q=[np.asarray(x,np.float32) for x in H[a:b]]; rp=[float(x) for x in HR[a:b]]; ds=[float(x) for x in HD[a:b]]; source_idx=[a+j for j in range(b-a)]
        exact=[j for j,p in enumerate(Q) if np.array_equal(p,PB[i])]
        baseline_appended=False
        if not exact:
            e,d=baseline_observable_score(m,PB,coords,scores,valid,VA,i); Q.append(PB[i].copy()); rp.append(e); ds.append(d); source_idx.append(-1); baseline_appended=True; appended_baseline+=1
        rp=np.asarray(rp,float); ds=np.asarray(ds,float); graw=.75*rank01(rp,False)+.25*rank01(ds,True); grank=rank01(graw,False)
        rraw=np.asarray([candidate_R_raw(m,p,i,VA,per_view) for p in Q],float); rrank=rank01(rraw,False); total=grank+rrank; best=int(np.argmin(total)); PBnew[i]=Q[best]
        rows.append({'carrier':i,'domain_n':len(Q),'baseline_appended':baseline_appended,'selected_domain_index':best,'selected_original_global_index':int(source_idx[best]),'selected_is_baseline_exact':bool(np.array_equal(Q[best],PB[i])),'selected_xyz':np.asarray(Q[best],float).tolist(),'selected_G_raw':float(graw[best]),'selected_G_rank':float(grank[best]),'selected_R_raw':float(rraw[best]),'selected_R_rank':float(rrank[best]),'selected_total':float(total[best]),'baseline_domain_index':int(next(j for j,p in enumerate(Q) if np.array_equal(p,PB[i]))),'domain_min_total_margin':float(np.partition(total,1)[1]-total[best]) if len(total)>1 else None})
    out_world.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out_world,P_A=PA,P_B_baseline=PB,P_B_V3C=PBnew)
    changed=np.linalg.norm(PBnew-PB,axis=1)
    return {'schema':SCHEMA,'contract_commit':CONTRACT_COMMIT,'family':int(family),'truth_access':'NONE','authorities':{'state_sha256':sha256_file(state_path),'proposal_evidence_content_sha256':dig,'selection_sha256':V3A_SELECTION_SHA256,'observable_panel_manifest_sha256':OBS_PANEL_SHA256},'minmarginal_replay':replay,'world':{'world_file_sha256':sha256_file(out_world),'baseline_appended_n':appended_baseline,'baseline_exact_n':int(np.sum(changed==0)),'changed_n_gt_1e_4':int(np.sum(changed>1e-4)),'mean_distance_from_baseline':float(np.mean(changed)),'max_distance_from_baseline':float(np.max(changed))},'rows':rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--science-dir',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--out-world',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists() or a.out_world.exists(): raise RuntimeError('refusing overwrite')
    r=solve_family(a.science_dir,a.state,a.evidence,a.selection,a.family,a.out_world); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'family':r['family'],'replay':r['minmarginal_replay'],'world':r['world']},indent=2,sort_keys=True))
if __name__=='__main__':main()
