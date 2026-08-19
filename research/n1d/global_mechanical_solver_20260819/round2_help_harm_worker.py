from __future__ import annotations
import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import one_pass_min_sum_message_solver_p0 as p0
import two_round_cavity_min_sum_p1 as p1
import round2_help_harm_localization_v1 as d

OUT=Path('/mnt/data/sufficiency/round2_help_harm_localization_v1');OUT.mkdir(exist_ok=True)

def process(fid:int):
    f=m.build_family(fid)
    U,edges,neigh,vm,m1,m2,x1,x2=d.build_messages(f)
    # Exact parent comparison. p0 round1 and P1 round1/2 are independently replayed.
    xu,_,xp0=p0.solve(f)
    px1,px2=p1.solve_two_round(f)
    diag=d.node_diagnostics(f,edges,neigh,vm,m1)
    rows=[];r1=[];r2=[]
    for i,n in enumerate(f['nodes']):
        if not (n['reliable'] and n['M_cont2'] and n['active'] and n['full_cont2']):continue
        e1=float(np.linalg.norm(n['M'][x1[i]]-n['target'])/n['scale']);e2=float(np.linalg.norm(n['M'][x2[i]]-n['target'])/n['scale']);delta=e2-e1
        lab='HELP' if delta<=-.25 else ('HARM' if delta>=.25 else 'NEUTRAL')
        rows.append({'fid':int(fid),'i':int(i),'err1':e1,'err2':e2,'delta_err':delta,'c2_1':bool(e1<=2),'c2_2':bool(e2<=2),'label':lab,**diag[i]});r1.append(e1);r2.append(e2)
    out={'fid':fid,'index_parity':{'x1_vs_p0':int(np.sum(x1==xp0)),'x1_vs_p1':int(np.sum(x1==px1)),'x2_vs_p1':int(np.sum(x2==px2)),'n':64},'family_metrics':{'n':len(r1),'c2_1':float(np.mean(np.asarray(r1)<=2)),'c2_2':float(np.mean(np.asarray(r2)<=2)),'median_delta_err':float(np.median(np.asarray(r2)-np.asarray(r1)))},'rows':rows}
    (OUT/f'partial_{fid}.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('fid',type=int);args=ap.parse_args();process(args.fid)
