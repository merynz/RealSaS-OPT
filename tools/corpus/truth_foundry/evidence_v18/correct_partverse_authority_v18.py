#!/usr/bin/env python3
"""Correct 21,084 V17 PartVerse evidence *state labels* without fabricating truth.

The original raw facet values, per-source SHA receipts, V17 files, and all
other states remain unchanged. A state is not stronger than its underlying
measured field merely because the provider has a numeric part court.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,pathlib,collections
V17_SHA={
 'states':'57249761c5b93207bdbee68adc7c48c5a1360245317e13b2b1be1870433e3fbd',
 'values':'4d1f0637ba6503a5e7ea0f8fc8ac0ba471fee6ee0705875e8f1df7d819d13088',
 'delta':'349a8cb73cfc586d26fc5412a29fb90eddcd424f7516c65edb0769cc32850975',
}
RECLASS={
 'attribution_obligations':'PROVIDER_LICENSE_CLASS_CLAIM__ATTRIBUTION_NOT_COURTED',
 'original_asset_provider_locator':'RECORDED_PROVIDER_LOCATOR_CLAIM__NOT_SOURCE_VERIFIED',
 'original_asset_recovery_status':'HISTORICAL_PROCESSED_PART_NUMERIC_RECEIPT__ORIGINAL_UNHASHED',
}
INPUT_FILES={
 'states':'FACET_STATES_EFFECTIVE_V17_16397x264.csv.gz',
 'values':'FACET_VALUES_EFFECTIVE_V17_16397x264.csv.gz',
 'delta':'FACET_DELTA_INVENTORY_V17.csv.gz',
}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def run(v17,out):
 out.mkdir(parents=True,exist_ok=True)
 for key,filename in INPUT_FILES.items():
  if sha(v17/filename)!=V17_SHA[key]:raise ValueError('V17_SOURCE_SHA_DRIFT:'+key)
 targets={}
 with gzip.open(v17/INPUT_FILES['delta'],'rt',newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   if row['domain']=='PARTVERSE' and row['facet'] in RECLASS:
    key=(row['source_key'],row['facet'])
    if key in targets:raise ValueError('DUPLICATE_TARGET')
    value=json.loads(row['value_json'])
    if value['source_evidence_domain']!='PARTVERSE':raise ValueError('MISBOUND_DOMAIN')
    if value['model_training_admission'] is not False:raise ValueError('BAD_AUTHORITY')
    targets[key]={'state':RECLASS[row['facet']],'source_evidence_sha256':row['source_evidence_sha256']}
 expected=7028*len(RECLASS)
 if len(targets)!=expected:raise ValueError('EXPECTED_21084_CORRECTIONS')
 s_out=out/'FACET_STATES_EFFECTIVE_V18_16397x264.csv.gz'
 v_out=out/'FACET_VALUES_EFFECTIVE_V18_16397x264.csv.gz'
 change_out=out/'AUTHORITY_CORRECTIONS_V18.csv.gz'
 changes=0;rows=0;seen=set();counts=collections.Counter()
 with gzip.open(v17/INPUT_FILES['states'],'rt',newline='',encoding='utf-8') as fi,\
      gzip.open(v17/INPUT_FILES['values'],'rt',newline='',encoding='utf-8') as fv,\
      gzip.open(s_out,'wt',newline='',encoding='utf-8') as fs,\
      gzip.open(v_out,'wt',newline='',encoding='utf-8') as fo,\
      gzip.open(change_out,'wt',newline='',encoding='utf-8') as fc:
  sr,vr=csv.DictReader(fi),csv.DictReader(fv)
  if sr.fieldnames!=vr.fieldnames or len(sr.fieldnames)!=265:raise ValueError('V17_SCHEMA_DRIFT')
  sw,vw=csv.DictWriter(fs,fieldnames=sr.fieldnames),csv.DictWriter(fo,fieldnames=vr.fieldnames)
  cw=csv.DictWriter(fc,fieldnames=['source_key','facet','old_state','new_state','source_evidence_sha256','reason'])
  for w in (sw,vw,cw):w.writeheader()
  for ss,vv in zip(sr,vr):
   key=ss['source_key'];rows+=1
   if key!=vv['source_key'] or key in seen:raise ValueError('SOURCE_IDENTITY_DRIFT')
   seen.add(key)
   for facet in RECLASS:
    target=targets.get((key,facet))
    if not target:continue
    if ss[facet]!='OBSERVED_PROCESSED_PART_GEOMETRY_ONLY':raise ValueError('OLD_STATE_DRIFT')
    payload=json.loads(vv[facet])
    if payload['source_evidence_sha256']!=target['source_evidence_sha256']:raise ValueError('SOURCE_SHA_MISMATCH')
    cw.writerow(dict(source_key=key,facet=facet,old_state=ss[facet],new_state=target['state'],source_evidence_sha256=payload['source_evidence_sha256'],reason='RECLASSIFY_PARTVERSE_NON_GEOMETRY_PROVENANCE'))
    ss[facet]=target['state'];changes+=1;counts[facet]+=1
   sw.writerow(ss);vw.writerow(vv)
  if next(sr,None) is not None or next(vr,None) is not None:raise ValueError('ROW_OVERFLOW')
 if rows!=16397 or changes!=expected:raise ValueError('COUNT_DRIFT')
 if set(counts)!=(set(RECLASS)) or any(v!=7028 for v in counts.values()):raise ValueError('FACET_COUNT_DRIFT')
 report={
  'schema':'RealSaS.TruthFoundryV18PartVerseAuthorityStateCorrection.v1',
  'rows':rows,'facets':264,'state_reclassifications':changes,
  'distinct_sources_reclassified':7028,'corrections_by_facet':dict(counts),
  'new_numeric_measurements':0,'source_binary_reads':0,'training_admissions':0,
  'v17_inputs_sha256':V17_SHA,
  'v18_outputs_sha256':{'states':sha(s_out),'values':sha(v_out),'corrections':sha(change_out)},
 }
 (out/'AUTHORITY_CORRECTION_SUMMARY_V18.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--v17',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.v17,a.out),indent=2))
if __name__=='__main__':main()