from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCHEMA='RealSaS.N1D.Rank2G.G2.ReciprocalCycleTruthEvaluation.v1'
FAMILIES=(10287,15521,16638,15284,13015,14703,16414,16528,16036,15226,14315,12949,14405,11447,13843,14885)
EPS=1e-12


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));mod=importlib.util.module_from_spec(spec)
    if spec.loader is None:raise RuntimeError(('module loader missing',path))
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def unique_sites(ez,i,v):
    coords=np.asarray(ez['coords']);scores=np.asarray(ez['scores']);valid=np.asarray(ez['valid'])
    best={}
    for k in range(4):
        if not valid[i,v,k]:continue
        site=tuple(map(int,np.rint(coords[i,v,k]).astype(int)));sc=float(scores[i,v,k])
        if site not in best or sc>best[site]:best[site]=sc
    return [x[0] for x in sorted(best.items(),key=lambda kv:(-kv[1],kv[0][1],kv[0][0]))]


def view_entry(sel,v):
    x=next((x for x in sel['views'] if int(x['view'])==int(v)),None)
    if x is None:raise RuntimeError(('view missing',sel.get('family'),v))
    return x


def assigned_site(view,carrier):
    a={int(k):tuple(map(int,val)) for k,val in view['assigned'].items()}
    abst=set(map(int,view.get('abstained',[])))
    if carrier in abst:return None
    return a.get(carrier)


def unary_top(view,carrier,key='G_rank'):
    row=view['rows'][str(carrier)];vals=np.asarray(row[key],float);sites=[tuple(map(int,x)) for x in row['sites']]
    return sites[int(np.argmin(vals))]


def evaluate_family(root,science_dir,state_path,evidence_path,g1_path,g2_path,sidecar,family,episode):
    meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE':raise RuntimeError(('observable state truth contract invalid',family))
    obs=load_module(f'g2_obs_{family}',science_dir/'observable_phase.py')
    ev=load_module(f'g2_eval_{family}',science_dir/'evaluation_phase.py')

    # Critical evaluator authority: reconstruct Problem-A from frozen rasters/route.
    # Do NOT trust a serialized Rank2 P_A reconstruction for truth mapping.
    m,A,B,PA_support,VA_support,seedB,output_idx,hb,prb,diag,field,xyA=obs._model_context(root,family,episode,meta['route'])
    PA=np.asarray(PA_support[output_idx],np.float32)
    VA=np.asarray(VA_support[:,output_idx],np.uint8)
    XYA=np.stack([m.project_points(PA,v) for v in range(8)]).astype(np.float32)

    ez=np.load(evidence_path,allow_pickle=False)
    if not np.array_equal(np.asarray(ez['V_A'],np.uint8),VA):raise RuntimeError(('V_A evidence/raster replay mismatch',family))
    if not np.array_equal(np.asarray(ez['XY_A'],np.float32),XYA):raise RuntimeError(('XY_A evidence/raster replay mismatch',family))

    g1=json.loads(g1_path.read_text());g2=json.loads(g2_path.read_text())
    for name,sel in [('G1',g1),('G2',g2)]:
        if int(sel['family'])!=int(family):raise RuntimeError((name,'family mismatch',family))
        if sel.get('truth_access')!='NONE':raise RuntimeError((name,'selection truth contract invalid',family))

    # Evaluator-only truth begins here.
    t512=ev._truth(sidecar)
    ids,map_err=ev._map64(PA,t512)
    t=ev._subset_truth(t512,ids)
    sA=ev.local_scale(t512['P_A'],4)[ids]
    reliable=np.asarray(map_err)<=2*np.asarray(sA)
    truthB=np.asarray(t['P_B'],np.float32);truthVB=np.asarray(t['V_B'],np.uint8)
    truthXY=np.stack([m.project_points(truthB,v) for v in range(8)]).astype(np.float32)

    rows=[]
    for v in range(8):
        a=view_entry(g1,v);b=view_entry(g2,v)
        if int(a['visible_n'])!=int(b['visible_n']) or int(a['K'])!=int(b['K']):
            raise RuntimeError(('G1/G2 cardinality mismatch',family,v))
        for i0 in np.where(VA[v]>0)[0]:
            i=int(i0)
            if not reliable[i] or not bool(truthVB[v,i]):continue
            sites=unique_sites(ez,i,v)
            if not sites:continue
            target=np.asarray(truthXY[v,i],float)
            dist=np.asarray([np.linalg.norm(np.asarray(s,float)-target) for s in sites],float)
            best=float(np.min(dist));oracle={sites[k] for k in np.where(np.isclose(dist,best,rtol=0,atol=1e-9))[0]}
            s1=assigned_site(a,i);s2=assigned_site(b,i)
            d1=None if s1 is None else float(np.linalg.norm(np.asarray(s1,float)-target))
            d2=None if s2 is None else float(np.linalg.norm(np.asarray(s2,float)-target))
            g1u=unary_top(a,i,'G_rank')
            cyc=unary_top(b,i,'G_reciprocal_cycle_rank')
            g2u=unary_top(b,i,'G_rank')
            rows.append({
                'family':int(family),'view':int(v),'carrier':i,
                'oracle_best_sites':[list(x) for x in sorted(oracle)],'oracle_best_distance_px':best,
                'g1_site':None if s1 is None else list(s1),'g1_abstain':s1 is None,'g1_hit':False if s1 is None else s1 in oracle,'g1_regret_px':None if d1 is None else float(d1-best),
                'g2_site':None if s2 is None else list(s2),'g2_abstain':s2 is None,'g2_hit':False if s2 is None else s2 in oracle,'g2_regret_px':None if d2 is None else float(d2-best),
                'g1_unary_top':list(g1u),'cycle_top':list(cyc),'g2_unary_top':list(g2u),
                'g1_cycle_disagreement':bool(g1u!=cyc),
            })
    return {'family':int(family),'mapping_reliable_n':int(reliable.sum()),'rows':rows,'sidecar_sha256':sha256_file(sidecar)}


def metrics(rows,prefix):
    n=len(rows);hit=sum(bool(r[f'{prefix}_hit']) for r in rows);reg=[r[f'{prefix}_regret_px'] for r in rows if r[f'{prefix}_regret_px'] is not None]
    return {'n':n,'hit_n':hit,'hit_rate':hit/n if n else None,'assigned_n':len(reg),'abstain_n':n-len(reg),'mean_regret_px_assigned':float(np.mean(reg)) if reg else None,'median_regret_px_assigned':float(np.median(reg)) if reg else None}


def family_guard(rows):
    by=defaultdict(list)
    for r in rows:by[r['family']].append(r)
    out=[];nonworse=0
    for fam in FAMILIES:
        rs=by[int(fam)];a=[r['g1_regret_px'] for r in rs if r['g1_regret_px'] is not None];b=[r['g2_regret_px'] for r in rs if r['g2_regret_px'] is not None]
        if len(a)!=len(b):raise RuntimeError(('family assigned coverage mismatch',fam,len(a),len(b)))
        m1=float(np.mean(a)) if a else None;m2=float(np.mean(b)) if b else None
        nw=bool(m1 is not None and m2<=m1+EPS);nonworse+=int(nw)
        out.append({'family':int(fam),'n':len(rs),'g1_mean_regret_px':m1,'g2_mean_regret_px':m2,'nonworse':nw})
    return {'nonworse_count':nonworse,'families':out}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--science-dir',type=Path,required=True);ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--evidence-dir',type=Path,required=True);ap.add_argument('--g1-dir',type=Path,required=True);ap.add_argument('--g2-dir',type=Path,required=True);ap.add_argument('--sidecar-dir',type=Path,required=True)
    ap.add_argument('--episode',default='e01');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise RuntimeError('refusing overwrite')
    fams=[];rows=[]
    for fam in FAMILIES:
        pf='09908' if fam==9908 else str(fam)
        state=a.panel/pf/f'{fam}_{a.episode}_OBSERVABLE_STATE.npz'
        evidence=a.evidence_dir/f'{fam}_G2_RECIPROCAL_CYCLE_EVIDENCE.npz'
        g1=a.g1_dir/f'{fam}_G1_SELECTION.json';g2=a.g2_dir/f'{fam}_G2_SELECTION.json';side=a.sidecar_dir/str(fam)/'observation_sidecar.npz'
        fr=evaluate_family(a.root,a.science_dir,state,evidence,g1,g2,side,fam,a.episode);fams.append({k:v for k,v in fr.items() if k!='rows'});rows.extend(fr['rows'])
    dis=[r for r in rows if r['g1_cycle_disagreement']]
    result={'schema':SCHEMA,'families':fams,'overall':{'g1':metrics(rows,'g1'),'g2':metrics(rows,'g2')},'g1_cycle_disagreement':{'n':len(dis),'g1':metrics(dis,'g1'),'g2':metrics(dis,'g2')},'family_mean_regret_guard':family_guard(rows),'rows':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,sort_keys=True))


if __name__=='__main__':main()
