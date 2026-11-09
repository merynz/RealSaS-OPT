#!/usr/bin/env python3
"""Independent readback: V16 changes exactly the 100 authorized UNKNOWN V11 cells."""
import argparse
from pathlib import Path
import csv,gzip,hashlib,json,collections
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
a=parser.parse_args()
B=a.base_dir.resolve();R=a.output_dir.resolve()
source=(B/'FACET_STATES_EFFECTIVE_V11_16397x264.csv.gz',B/'FACET_VALUES_EFFECTIVE_V11_16397x264.csv.gz')
dest=(R/'FACET_STATES_EFFECTIVE_V16_16397x264.csv.gz',R/'FACET_VALUES_EFFECTIVE_V16_16397x264.csv.gz')
manifest=json.loads((R/'SUMMARY_EFFECTIVE_V16.json').read_text())
for kind,path in zip(('states','values'),dest):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['output_sha256'][kind]
with open(R/'PHYSICAL_FACET_DELTA_V16.csv',newline='') as f:
 delta={(r['source_key'],r['facet']):r for r in csv.DictReader(f)}
assert len(delta)==100
changed=collections.Counter(); keys=set(); unchanged=0; prior_state_count=collections.Counter(); after_state_count=collections.Counter()
with gzip.open(source[0],'rt',newline='') as sf,gzip.open(source[1],'rt',newline='') as vf, gzip.open(dest[0],'rt',newline='') as sn,gzip.open(dest[1],'rt',newline='') as vn:
 s0,v0,s1,v1=csv.DictReader(sf),csv.DictReader(vf),csv.DictReader(sn),csv.DictReader(vn)
 assert s0.fieldnames==v0.fieldnames==s1.fieldnames==v1.fieldnames and len(s0.fieldnames)==265
 facets=s0.fieldnames[1:]
 rows=0
 for a,b,c,d in zip(s0,v0,s1,v1):
  key=a['source_key'];assert key==b['source_key']==c['source_key']==d['source_key'] and key not in keys
  keys.add(key);rows+=1
  for facet in facets:
   old_s,old_v,new_s,new_v=a[facet],b[facet],c[facet],d[facet]
   prior_state_count[old_s]+=1;after_state_count[new_s]+=1
   expected=delta.get((key,facet))
   if expected:
    assert old_s=='UNKNOWN_UNREVIEWED' and not old_v
    assert new_s=='OBSERVED_HISTORICAL_NORMALIZED_DERIVATIVE_ONLY'
    assert json.loads(new_v)==json.loads(expected['value_json'])
    changed[facet]+=1
   else:
    assert new_s==old_s and new_v==old_v,('UNAUTHORIZED_V16_CELL_WRITE',key,facet)
    unchanged+=1
 assert not any(next(x,None) is not None for x in (s0,v0,s1,v1))
 assert rows==16397 and len(keys)==16397
assert sum(changed.values())==100 and unchanged==16397*264-100
assert prior_state_count['UNKNOWN_UNREVIEWED']==4202117
assert after_state_count['UNKNOWN_UNREVIEWED']==4202017
assert prior_state_count['OBSERVED_HISTORICAL_NORMALIZED_DERIVATIVE_ONLY']==0
assert after_state_count['OBSERVED_HISTORICAL_NORMALIZED_DERIVATIVE_ONLY']==100
court={'schema':'RealSaS.TruthFoundryV16EffectiveIndependentReadback.v1','result':'PASS_EXACT_100_DERIVATIVE_ONLY_OVERLAY',
 'rows':rows,'facets':len(facets),'modified_cells':sum(changed.values()),'unchanged_cells_exact':unchanged,
 'unknown_before':prior_state_count['UNKNOWN_UNREVIEWED'],'unknown_after':after_state_count['UNKNOWN_UNREVIEWED'],
 'updated_facet_counts':dict(changed),'training_admissions':0,'original_source_parity_admissions':0,
 'facet_states_sha256':manifest['output_sha256']['states'],'facet_values_sha256':manifest['output_sha256']['values']}
(R/'INDEPENDENT_READBACK_EFFECTIVE_V16.json').write_text(json.dumps(court,sort_keys=True,indent=2)+'\n')
print(json.dumps(court,sort_keys=True,indent=2))