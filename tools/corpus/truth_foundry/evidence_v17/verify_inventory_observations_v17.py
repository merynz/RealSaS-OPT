#!/usr/bin/env python3
"""Independent readback of V17 against V16 and per-source V11 inventory.

Compares all 4,328,808 state+value cells. No inference from matrix diffs is
accepted without a source field, type-scoped assertion and evidence digest.
"""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,pathlib,re

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=pathlib.Path,required=True)
 p.add_argument('--v16',type=pathlib.Path,required=True);p.add_argument('--v17',type=pathlib.Path,required=True)
 a=p.parse_args()
 summary=json.loads((a.v17/'SUMMARY_V17.json').read_text())
 lock=summary['baseline_v16_sha256']
 assert sha(a.inventory)==lock['inventory']
 source={}
 with gzip.open(a.inventory,'rt',newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   assert row['source_key'] not in source
   source[row['source_key']]=row
 assert len(source)==16397
 paths=[a.v16/'FACET_STATES_EFFECTIVE_V16_16397x264.csv.gz',a.v16/'FACET_VALUES_EFFECTIVE_V16_16397x264.csv.gz',a.v17/'FACET_STATES_EFFECTIVE_V17_16397x264.csv.gz',a.v17/'FACET_VALUES_EFFECTIVE_V17_16397x264.csv.gz']
 for path,k in zip(paths,['states','values','states','values']):
  expected=lock[k] if 'V16_' in path.name else summary['output_sha256'][k]
  assert sha(path)==expected,('INPUT_OR_OUTPUT_SHA_DRIFT',path)
 delta={}
 with gzip.open(a.v17/'FACET_DELTA_INVENTORY_V17.csv.gz','rt',newline='',encoding='utf-8') as f:
  for d in csv.DictReader(f):
   key=d['source_key'];facet=d['facet'];pair=(key,facet)
   assert pair not in delta and key in source
   v=json.loads(d['value_json']);r=source[key]
   assert not v['rights_training_admission'] and not v['model_training_admission'] and not v['semantic_editable_drawable_admission']
   assert v['creator_original_current_rehash'] is False
   fields=v['observed_field_values']
   assert fields and all(r[k]==val and val!='' for k,val in fields.items())
   assert sorted(fields)==sorted([k for k in d['source_field_names'].split('|') if r[k]!=''])
   assert v['source_evidence_domain']==d['domain']
   assert re.fullmatch(r'[0-9a-f]{64}',d['source_evidence_sha256'])
   if d['domain']=='METADATA':
    assert d['source_evidence_sha256']==lock['inventory']
    assert facet in ('original_asset_recovery_status','original_asset_provider_locator')
   elif d['domain']=='MASTER':
    assert r['pool']=='MASTER3993' and r['source_import_observed']=='True' and r['historical_split_hint']=='FIT'
    assert r['master_v11_holdout_protected']!='True'
    assert d['source_evidence_sha256']==r['evidence_master_deep_sha256']
   elif d['domain']=='PARTVERSE':
    assert r['provider']=='PARTVERSE_XL' and r['partverse_v10_processed_numeric_court']=='PASS_PROCESSED_PART_GEOMETRY_NUMERIC_ONLY'
    assert d['source_evidence_sha256']==r['evidence_partverse_processed_sha256']
   elif d['domain']=='CORE':
    assert r['pool']=='CORE370' and r['v5_source_rest_court']=='PASS_SOURCE_REST_INDEPENDENT_COURT'
    assert d['source_evidence_sha256']==r['v5_source_rest_sha256']
   elif d['domain']=='UNIML':
    assert r['provider']=='UNIML3D' and r['canonical_v6_sha_verified']=='True'
    assert d['source_evidence_sha256']==r['canonical_glb_sha256']
   else:raise AssertionError('UNRECOGNIZED_SOURCE_PROVENANCE')
   delta[pair]=d
 assert len(delta)==summary['new_partial_observation_cells']
 counts=collections.Counter();domains=collections.Counter();facets=collections.Counter()
 unknown_before=unknown_after=unchanged=rows=0;seen=set()
 with gzip.open(paths[0],'rt',newline='',encoding='utf-8') as sf, gzip.open(paths[1],'rt',newline='',encoding='utf-8') as vf, gzip.open(paths[2],'rt',newline='',encoding='utf-8') as sn, gzip.open(paths[3],'rt',newline='',encoding='utf-8') as vn:
  src_s,src_v,dst_s,dst_v=map(csv.DictReader,(sf,vf,sn,vn))
  h=src_s.fieldnames
  assert h==src_v.fieldnames==dst_s.fieldnames==dst_v.fieldnames and len(h)==265
  for before_s,before_v,after_s,after_v in zip(src_s,src_v,dst_s,dst_v):
   key=before_s['source_key'];rows+=1
   assert key==before_v['source_key']==after_s['source_key']==after_v['source_key'] and key not in seen and key in source
   seen.add(key)
   for facet in h[1:]:
    old,oldval,new,newval=before_s[facet],before_v[facet],after_s[facet],after_v[facet]
    if old=='UNKNOWN_UNREVIEWED':unknown_before+=1
    if new=='UNKNOWN_UNREVIEWED':unknown_after+=1
    d=delta.get((key,facet))
    if d is None:
     assert old==new and oldval==newval,('UNAUTHORIZED_FACET_CHANGE',key,facet)
     unchanged+=1
    else:
     assert old=='UNKNOWN_UNREVIEWED' and not oldval,('PRIOR_AUTHORITY_COLLISION',key,facet)
     assert new==d['new_state'] and newval==d['value_json']
     assert new.startswith(('RECORDED_INVENTORY_','OBSERVED_PROCESSED_PART_GEOMETRY_ONLY','OBSERVED_HISTORICAL_SOURCE_IMPORT_COUNTS_ONLY','OBSERVED_V5_COMPONENT_COURT_ONLY','OBSERVED_CANONICAL_GLB_DERIVATIVE_ONLY'))
     counts[key]+=1;facets[facet]+=1;domains[d['domain']]+=1
  assert all(next(x,None) is None for x in (src_s,src_v,dst_s,dst_v))
 assert rows==16397 and len(seen)==16397
 assert sum(counts.values())==len(delta)==summary['new_partial_observation_cells']
 assert len(counts)==summary['new_partial_observation_sources']
 assert unchanged==16397*264-len(delta)
 assert dict(facets)==summary['facet_increment_counts']
 assert unknown_after==unknown_before-len(delta)
 readback={'schema':'RealSaS.TruthFoundryV17IndependentSourceFieldReadback.v1',
           'result':'PASS_EXACT_INVENTORY_METADATA_PARTIAL_OVERLAY_ONLY',
           'source_rows':rows,'facet_columns':264,'new_partial_cells':len(delta),'changed_source_records':len(counts),
           'previous_cells_unchanged':unchanged,'unknown_before':unknown_before,'unknown_after':unknown_after,
           'domain_cell_counts':dict(domains),'facet_increment_counts':dict(facets),
           'rights_admissions':0,'training_admissions':0,'creator_original_rehashes':0,'holdout_original_accesses':0,
           'output_sha256':summary['output_sha256']}
 (a.v17/'INDEPENDENT_READBACK_V17.json').write_text(json.dumps(readback,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:readback[k] for k in ('result','new_partial_cells','changed_source_records','previous_cells_unchanged','unknown_before','unknown_after','domain_cell_counts')},indent=2))
if __name__=='__main__':main()