from __future__ import annotations
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import round2_help_harm_localization_v1 as d
import v3_rank_runner as m
OUT=Path('/mnt/data/sufficiency/round2_help_harm_localization_v1');FAMS=m.FAMILIES

def main():
    t=time.time();parts={fid:json.load(open(OUT/f'partial_{fid}.json')) for fid in FAMS};rows=[];fam_metrics={};idx1=idx1p1=idx2=0
    for fid in FAMS:
        z=parts[fid];rows+=z['rows'];fam_metrics[str(fid)]=z['family_metrics'];idx1+=z['index_parity']['x1_vs_p0'];idx1p1+=z['index_parity']['x1_vs_p1'];idx2+=z['index_parity']['x2_vs_p1']
    c21=float(np.mean([r['c2_1'] for r in rows]));c22=float(np.mean([r['c2_2'] for r in rows]));worst1=min(v['c2_1'] for v in fam_metrics.values());worst2=min(v['c2_2'] for v in fam_metrics.values())
    valid=bool(len(rows)==261 and idx1==512 and idx1p1==512 and idx2==512 and abs(c21-.7777777777777778)<1e-12 and abs(c22-.8160919540229885)<1e-12 and abs(worst1-.4)<1e-12 and abs(worst2-.52)<1e-12 and abs(fam_metrics['12772']['c2_2']-.7272727272727273)<1e-12 and abs(fam_metrics['11032']['c2_2']-.52)<1e-12)
    scores={k:d.score_feature(rows,k) for k in ['D1','D2','D3','D4']};passing=[k for k,v in scores.items() if v['supported']]
    names={'D1':'PAIR_MESSAGE_AMBIGUITY_SUPPORTED','D2':'TRIANGLE_LOOP_OVERLAP_SUPPORTED','D3':'INCIDENT_MESSAGE_CONFLICT_SUPPORTED','D4':'REL_FLOW_CYCLE_INCONSISTENCY_SUPPORTED'}
    if not valid:verdict='INVALID_P1_PARITY_FAIL'
    elif len(passing)>=2:verdict='MIXED_PROPAGATION_HETEROGENEITY_SUPPORTED'
    elif len(passing)==1:verdict=names[passing[0]]
    else:verdict='NO_SIMPLE_ROUND2_HELP_HARM_LOCALIZER_V1'
    label_counts={lab:int(sum(r['label']==lab for r in rows)) for lab in ['HELP','NEUTRAL','HARM']};flips={'bad_to_good':int(sum((not r['c2_1']) and r['c2_2'] for r in rows)),'good_to_bad':int(sum(r['c2_1'] and (not r['c2_2']) for r in rows))}
    result={'schema':'RealSaS.N1D.Round2HelpHarmLocalization.v1','date':'2026-08-19','prereg_commit':'0bad17f775f8e8e67cce1b1cefaebe1a7a54e2b2','validity':{'all':valid,'primary_n':len(rows),'round1_vs_p0_index_parity':idx1,'round1_vs_p1_index_parity':idx1p1,'round2_vs_p1_index_parity':idx2,'round1_c2':c21,'round2_c2':c22,'round1_worst_c2':worst1,'round2_worst_c2':worst2,'12772_round2_c2':fam_metrics['12772']['c2_2'],'11032_round2_c2':fam_metrics['11032']['c2_2']},'labels':label_counts,'contain2_flips':flips,'scores':scores,'passing':passing,'family_metrics':fam_metrics,'verdict':verdict,'rows':rows,'seconds_combine':time.time()-t}
    path=OUT/'RESULT.json';path.write_text(json.dumps(result,indent=2,allow_nan=False));print('VALID',json.dumps(result['validity'],indent=2));print('LABELS',label_counts,'FLIPS',flips);print('SCORES',json.dumps(scores,indent=2));print('VERDICT',verdict);print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
