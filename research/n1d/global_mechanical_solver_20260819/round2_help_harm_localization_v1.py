from __future__ import annotations
import sys,json,hashlib,time,itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import multi_edge_incident_audit_v1 as a
import one_pass_min_sum_message_solver_p0 as p0
import two_round_cavity_min_sum_p1 as p1

ROOT=Path('/mnt/data/sufficiency');OUT=ROOT/'round2_help_harm_localization_v1';OUT.mkdir(exist_ok=True);FAMS=m.FAMILIES

def normmsg(v):
    v=np.asarray(v,np.float64); return v-float(np.min(v))

def build_messages(f):
    nodes=f['nodes'];U=[a.robust_z(n['Uraw']) for n in nodes];edges=m.graph(f);deg=np.zeros(64,int)
    for i,j,_ in edges:deg[i]+=1;deg[j]+=1
    factors={};neigh=[[] for _ in range(64)];views_map={}
    for i,j,vs in edges:
        R=a.robust_z(m.pair_raw(f,i,j,vs));w=1/max(int(deg[i]),1)+1/max(int(deg[j]),1)
        factors[(i,j)]=(R,float(w));views_map[(i,j)]=list(vs);neigh[i].append(j);neigh[j].append(i)
    msg1={}
    for (i,j),(R,w) in factors.items():
        msg1[(i,j)]=normmsg(np.min(U[i][:,None]+w*R,axis=0))
        msg1[(j,i)]=normmsg(np.min(U[j][None,:]+w*R,axis=1))
    B1=[]
    for i in range(64):
        b=U[i].copy()
        for j in neigh[i]:b+=msg1[(j,i)]
        B1.append(b)
    x1=np.array([int(np.argmin(b)) for b in B1],int)
    msg2={}
    for (i,j),(R,w) in factors.items():
        ci=U[i].copy();cj=U[j].copy()
        for q in neigh[i]:
            if q!=j:ci+=msg1[(q,i)]
        for q in neigh[j]:
            if q!=i:cj+=msg1[(q,j)]
        msg2[(i,j)]=normmsg(np.min(ci[:,None]+w*R,axis=0))
        msg2[(j,i)]=normmsg(np.min(cj[None,:]+w*R,axis=1))
    B2=[]
    for i in range(64):
        b=U[i].copy()
        for j in neigh[i]:b+=msg2[(j,i)]
        B2.append(b)
    x2=np.array([int(np.argmin(b)) for b in B2],int)
    return U,edges,neigh,views_map,msg1,msg2,x1,x2

def edge_g(f,i,j,views):
    A=[];b=[]
    for v in views:
        A.append((m.IMAGE_NATIVE-1)*m.RNP[v]);A.append(-(m.IMAGE_NATIVE-1)*m.UNP[v])
        r=np.asarray(f['dis'][v,i]-f['dis'][v,j],np.float64);b.extend([float(r[0]),float(r[1])])
    A=np.asarray(A,np.float64);b=np.asarray(b,np.float64)
    return np.linalg.lstsq(A,b,rcond=None)[0]

def midrank_percentile_scalar(x,arr):
    arr=np.asarray(arr,float);return float((np.sum(arr<x)+.5*np.sum(arr==x))/len(arr))

def node_diagnostics(f,edges,neigh,views_map,msg1):
    gs={}
    for i,j,vs in edges:gs[(i,j)]=edge_g(f,i,j,vs)
    edge_set=set((i,j) for i,j,_ in edges)
    def gorient(x,y):
        if x<y:return gs[(x,y)]
        return -gs[(y,x)]
    scores=[]
    for i,n in enumerate(f['nodes']):
        incoming=[msg1[(j,i)] for j in neigh[i]]
        amb=[]
        for mm in incoming:
            iq=float(np.quantile(mm,.75)-np.quantile(mm,.25));thr=float(np.min(mm))+.25*max(iq,1e-6);amb.append(float(np.mean(mm<=thr)))
        D1=float(np.median(amb)) if amb else 0.0
        den=0;num=0;ni=set(neigh[i])
        for j in neigh[i]:
            for q in neigh[j]:
                if q==i:continue
                den+=1
                if q in ni:num+=1
        D2=float(num/den) if den else 0.0
        if len(incoming)>=2:
            vote_idx=[int(np.argmin(mm)) for mm in incoming];votes=np.asarray(n['M'])[vote_idx]
            numd=float(np.median(pdist(votes))) if len(votes)>=2 else 0.0
            alld=pdist(np.asarray(n['M'],float));dend=float(np.median(alld)) if len(alld) else 0.0
            D3=float(numd/(dend+1e-9))
        else:D3=0.0
        closures=[]
        ns=sorted(neigh[i])
        for j,k in itertools.combinations(ns,2):
            ek=(min(j,k),max(j,k))
            if ek not in edge_set:continue
            gij=orient=gorient(i,j);gjk=gorient(j,k);gki=gorient(k,i)
            closures.append(float(np.linalg.norm(gij+gjk+gki)/(np.linalg.norm(gij)+np.linalg.norm(gjk)+np.linalg.norm(gki)+1e-9)))
        D4=float(np.median(closures)) if closures else 0.0
        scores.append({'D1':D1,'D2':D2,'D3':D3,'D4':D4,'degree':len(neigh[i]),'triangle_n':len(closures)})
    return scores

def score_feature(rows,key):
    s=np.asarray([r[key] for r in rows],float);d=np.asarray([r['delta_err'] for r in rows],float)
    rho=float(spearmanr(s,d).statistic) if len(np.unique(s))>1 else 0.0
    sel=[r for r in rows if r['label']!='NEUTRAL']; nlab=len(sel);auc=None
    if nlab>=20 and len(set(r['label'] for r in sel))==2:
        auc=float(roc_auc_score([1 if r['label']=='HARM' else 0 for r in sel],[r[key] for r in sel]))
    meds={lab:(float(np.median([r[key] for r in rows if r['label']==lab])) if any(r['label']==lab for r in rows) else None) for lab in ['HELP','NEUTRAL','HARM']}
    fam={}
    for fid in FAMS:
        rr=[r for r in rows if r['fid']==fid];fam[str(fid)]={'median_score':float(np.median([r[key] for r in rr])),'median_delta_err':float(np.median([r['delta_err'] for r in rr])),'n':len(rr)}
    fm=np.asarray([fam[str(fid)]['median_score'] for fid in FAMS],float);p12772=midrank_percentile_scalar(fam['12772']['median_score'],fm)
    supported=bool(nlab>=20 and rho>=.20-1e-12 and auc is not None and auc>=.65-1e-12 and meds['HARM'] is not None and meds['HELP'] is not None and meds['HARM']>meds['HELP'])
    return {'spearman_rho':rho,'help_harm_n':nlab,'auroc_harm_vs_help':auc,'medians':meds,'per_family':fam,'family_12772_score_percentile':p12772,'supported':supported}

def main():
    t=time.time();fams={fid:m.build_family(fid) for fid in FAMS};pf=m.preflight(fams);valid=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-.975609756097561)<1e-12 and abs(pf['worst']-.9322033898305084)<1e-12)
    rows=[];idx1eq=idx2eq=0;fam_metrics={}
    for fid in FAMS:
        f=fams[fid];U,edges,neigh,vm,m1,m2,x1,x2=build_messages(f);px1,px2=p1.solve_two_round(f);idx1eq+=int(np.sum(x1==px1));idx2eq+=int(np.sum(x2==px2));diag=node_diagnostics(f,edges,neigh,vm,m1)
        r1=[];r2=[]
        for i,n in enumerate(f['nodes']):
            if not (n['reliable'] and n['M_cont2'] and n['active'] and n['full_cont2']):continue
            e1=float(np.linalg.norm(n['M'][x1[i]]-n['target'])/n['scale']);e2=float(np.linalg.norm(n['M'][x2[i]]-n['target'])/n['scale']);delta=e2-e1
            lab='HELP' if delta<=-.25 else ('HARM' if delta>=.25 else 'NEUTRAL')
            row={'fid':int(fid),'i':int(i),'err1':e1,'err2':e2,'delta_err':delta,'c2_1':bool(e1<=2),'c2_2':bool(e2<=2),'label':lab,**diag[i]};rows.append(row);r1.append(e1);r2.append(e2)
        fam_metrics[str(fid)]={'n':len(r1),'c2_1':float(np.mean(np.asarray(r1)<=2)),'c2_2':float(np.mean(np.asarray(r2)<=2)),'median_delta_err':float(np.median(np.asarray(r2)-np.asarray(r1)))}
        print('FAMILY',fid,'n',len(r1),'c2',fam_metrics[str(fid)]['c2_1'],'->',fam_metrics[str(fid)]['c2_2'],'med_delta',fam_metrics[str(fid)]['median_delta_err'],flush=True)
    c21=float(np.mean([r['c2_1'] for r in rows]));c22=float(np.mean([r['c2_2'] for r in rows]));worst1=min(v['c2_1'] for v in fam_metrics.values());worst2=min(v['c2_2'] for v in fam_metrics.values())
    valid=bool(valid and len(rows)==261 and idx1eq==512 and idx2eq==512 and abs(c21-.7777777777777778)<1e-12 and abs(c22-.8160919540229885)<1e-12 and abs(worst1-.4)<1e-12 and abs(worst2-.52)<1e-12 and abs(fam_metrics['12772']['c2_2']-.7272727272727273)<1e-12 and abs(fam_metrics['11032']['c2_2']-.52)<1e-12)
    scores={k:score_feature(rows,k) for k in ['D1','D2','D3','D4']};passing=[k for k,v in scores.items() if v['supported']]
    names={'D1':'PAIR_MESSAGE_AMBIGUITY_SUPPORTED','D2':'TRIANGLE_LOOP_OVERLAP_SUPPORTED','D3':'INCIDENT_MESSAGE_CONFLICT_SUPPORTED','D4':'REL_FLOW_CYCLE_INCONSISTENCY_SUPPORTED'}
    if not valid:verdict='INVALID_P1_PARITY_FAIL'
    elif len(passing)>=2:verdict='MIXED_PROPAGATION_HETEROGENEITY_SUPPORTED'
    elif len(passing)==1:verdict=names[passing[0]]
    else:verdict='NO_SIMPLE_ROUND2_HELP_HARM_LOCALIZER_V1'
    label_counts={lab:int(sum(r['label']==lab for r in rows)) for lab in ['HELP','NEUTRAL','HARM']}
    flips={'bad_to_good':int(sum((not r['c2_1']) and r['c2_2'] for r in rows)),'good_to_bad':int(sum(r['c2_1'] and (not r['c2_2']) for r in rows))}
    result={'schema':'RealSaS.N1D.Round2HelpHarmLocalization.v1','date':'2026-08-19','prereg_commit':'0bad17f775f8e8e67cce1b1cefaebe1a7a54e2b2','validity':{'all':valid,'primary_n':len(rows),'round1_index_parity':idx1eq,'round2_index_parity':idx2eq,'round1_c2':c21,'round2_c2':c22,'round1_worst_c2':worst1,'round2_worst_c2':worst2,'12772_round2_c2':fam_metrics['12772']['c2_2'],'11032_round2_c2':fam_metrics['11032']['c2_2']},'labels':label_counts,'contain2_flips':flips,'scores':scores,'passing':passing,'family_metrics':fam_metrics,'verdict':verdict,'rows':rows,'seconds':time.time()-t}
    path=OUT/'RESULT.json';path.write_text(json.dumps(result,indent=2,allow_nan=False));print('VALID',json.dumps(result['validity'],indent=2));print('LABELS',label_counts,'FLIPS',flips);print('SCORES',json.dumps(scores,indent=2));print('VERDICT',verdict);print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
