#!/usr/bin/env python3
"""Join existing 57,089 family evidence cells + 25,438 historical head routes.

This exports *historical reported* and importer observations with original
per-family source SHA receipts. It never upgrades a historical machine state to
current independent 8K/head supervision admission.
"""
import argparse,csv,hashlib,json,gzip,io
from collections import Counter,defaultdict
from pathlib import Path

FAMILY_FIELDS=['source_key','corpus','family','historical_evidence_state','historical_authority','machine_evidence_state','machine_report_run_id','machine_report_sha256','machine_court_sha256s','machine_court_reference_count','machine_evidence_sha256s','historical_report_fraction','reason_codes','current_independent_court_pass','training_authorized']
ROUTE_FIELDS=['source_key','corpus','head_or_historical_model','historical_source','candidate_required_families','historical_blocking_families','historical_eligibility','current_head_target_court','training_authorized']

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def load(path):
 with Path(path).open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write_gz(path,fields,rows):
 with Path(path).open('wb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as gz:
   with io.TextIOWrapper(gz,encoding='utf-8',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)

def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 p_master=root/'MASTER3993_51909_TRUTH_FAMILY_CELLS_V2.csv'
 p_master_head=root/'MASTER3993_23958_SIX_HEAD_ROUTES_V2.csv'
 p_core=root/'truth_foundry_inputs/CORE370_5180_MACHINE_TRUTH_FAMILY_CELLS_V1.csv'
 p_hist=root/'truth_foundry_inputs/CORE370_5180_HISTORICAL_TRUTH_FAMILY_CELLS_V1.csv'
 p_core_head=root/'truth_foundry_inputs/CORE370_1480_MACHINE_MODEL_ROUTE_V1.csv'
 p_core_source=root/'truth_foundry_inputs/CORE370_370_MACHINE_SUBJECTS_V1.csv'
 pins=[p_master,p_master_head,p_core,p_hist,p_core_head,p_core_source]
 a,b,c,h,m,core_src=(load(p) for p in pins)
 if [len(a),len(b),len(c),len(h),len(m),len(core_src)]!=[51909,23958,5180,5180,1480,370]:
  raise ValueError('FROZEN_FAMILY_SOURCE_COUNT_MISMATCH')
 # Explicit per-subject court history is not 2026 independent pass.
 historical={(x['subject_id'],x['family']):x for x in h}
 machine={(x['subject_id'],x['family']):x for x in c}
 if len(historical)!=5180 or len(machine)!=5180:raise ValueError('CORE_DUPLICATE_FAMILY')
 current={r['subject_id']:r for r in core_src}
 if len(current)!=370:raise ValueError('CORE_DUPLICATE_SOURCE')
 family=[]
 for r in a:
  if r['training_authorized']!='False':raise ValueError('MASTER_FALSE_TRAIN_AUTHORITY')
  family.append({'source_key':'MASTER3993:'+r['asset_id'],'corpus':'MASTER3993',
   'family':r['family'],'historical_evidence_state':r['evidence_state'],
   'historical_authority':r['authority'],'machine_evidence_state':'NOT_MAPPED_PER_ROW',
   'machine_report_run_id':'','machine_report_sha256':'','machine_court_sha256s':'',
   'machine_court_reference_count':'','machine_evidence_sha256s':'',
   'historical_report_fraction':'','reason_codes':'',
   'current_independent_court_pass':False,'training_authorized':False})
 for (uid,f),r in machine.items():
  old=historical[(uid,f)]
  if r['core370_training_authorized']!='False' or old['training_authorized']!='False':
   raise ValueError('CORE_FALSE_TRAIN_AUTHORITY')
  family.append({'source_key':'CORE370:'+uid,'corpus':'CORE370','family':f,
   'historical_evidence_state':old['historical_human_report_state'],
   'historical_authority':old['current_family_authority'],
   'machine_evidence_state':r['historical_machine_state'],
   'machine_report_run_id':r['machine_run_id'],
   'machine_report_sha256':r['machine_report_sha256'],
   'machine_court_sha256s':r['court_sha256s'],
   'machine_court_reference_count':r['court_reference_count'],
   'machine_evidence_sha256s':r['evidence_sha256s'],
   'historical_report_fraction':old['historical_validity_fraction'],
   'reason_codes':r['reason_codes'],
   'current_independent_court_pass':False,'training_authorized':False})
 family.sort(key=lambda r:(r['source_key'],r['family']))
 keys={(r['source_key'],r['family']) for r in family}
 if len(keys)!=57089:raise ValueError('FAMILY_UNIQUE_KEY_COUNT')
 routes=[]
 for r in b:
  if r['head_training_admitted']!='False':raise ValueError('MASTER_HEAD_FALSE_AUTHORITY')
  routes.append({'source_key':'MASTER3993:'+r['asset_id'],'corpus':'MASTER3993',
   'head_or_historical_model':r['head'],'historical_source':'MASTER_TRUTH_FAMILY_ROUTE_V2',
   'candidate_required_families':r['required_family_courts'],
   'historical_blocking_families':r['head_truth_state'],
   'historical_eligibility':'UNKNOWN','current_head_target_court':'UNKNOWN',
   'training_authorized':False})
 for r in m:
  if r['training_authorized']!='False':raise ValueError('CORE_MODEL_FALSE_AUTHORITY')
  routes.append({'source_key':'CORE370:'+r['subject_id'],'corpus':'CORE370',
   'head_or_historical_model':r['model'],'historical_source':'CORE370_MACHINE_MODEL_ROUTE_V1',
   'candidate_required_families':r['historical_required_families'],
   'historical_blocking_families':r['historical_blocking_families'],
   'historical_eligibility':r['historical_machine_corpus_eligible'],
   'current_head_target_court':'UNKNOWN','training_authorized':False})
 if len(routes)!=25438:raise ValueError('HEAD_ROUTE_COUNT')
 r_f=out/'SOURCE_FAMILY_EVIDENCE_57089_V3.csv.gz'
 r_h=out/'SOURCE_HEAD_HISTORICAL_ROUTES_25438_V3.csv.gz'
 write_gz(r_f,FAMILY_FIELDS,family)
 write_gz(r_h,ROUTE_FIELDS,routes)
 observed_master=Counter((r['family'],r['historical_evidence_state']) for r in family if r['corpus']=='MASTER3993')
 observed_core=Counter((r['family'],r['machine_evidence_state']) for r in family if r['corpus']=='CORE370')
 summary={
  'schema':'RealSaS.TruthFoundryFamilyHistoryOverlay.v3',
  'source_records_with_family_ledger':4363,'family_evidence_rows':len(family),
  'master_family_evidence_rows':51909,'core_family_evidence_rows':5180,
  'original_historical_head_routes':len(routes),
  'current_independent_new_courts_minted':0,'per_head_training_admission_minted':0,
  'source_identity_sha_and_rights_reverified_in_this_overlay':0,
  'machine_current_revalidation_satisfied_in_this_overlay':False,
  'training_authorized':False,
  'master_family_stage_distribution':{f+'|'+s:n for (f,s),n in observed_master.items()},
  'core_family_stage_distribution':{f+'|'+s:n for (f,s),n in observed_core.items()},
  'source_input_sha256':{p.name:digest(p) for p in pins},
  'outputs_sha256':{r_f.name:digest(r_f),r_h.name:digest(r_h)},
 }
 (out/'SUMMARY_FAMILY_V3.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();s=run(a.root,a.out)
 print(json.dumps({k:s[k] for k in ('source_records_with_family_ledger','family_evidence_rows','original_historical_head_routes','current_independent_new_courts_minted')},indent=2))