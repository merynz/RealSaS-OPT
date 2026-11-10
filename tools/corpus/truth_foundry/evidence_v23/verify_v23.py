#!/usr/bin/env python3
"""Independent full source byte rehash and 16,397x264 V21-to-V23 matrix readback."""
import csv,gzip,hashlib,io,json,pathlib,sys,zipfile
BASE=pathlib.Path('/mnt/data');OUT=BASE/'realsas_truth_v23'
STATE='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2<<20),b''):h.update(b)
 return h.hexdigest()
def zip_reader(z,name):return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(name)),encoding='utf-8',newline=''))
with gzip.open(OUT/'NEW_MASTER726_NPZ_COURT_V23.csv.gz','rt',newline='',encoding='utf-8') as f:receipt=list(csv.DictReader(f))
with gzip.open(OUT/'NPZ_FACET_DELTA_V23.csv.gz','rt',newline='',encoding='utf-8') as f:delta=list(csv.DictReader(f))
assert len(receipt)==32 and len({r['source_key'] for r in receipt})==32 and len({r['folder_name'] for r in receipt})==32
rkey={r['source_key']:r for r in receipt}
for r in receipt:
 name=r['folder_name'];npz=OUT/'sources'/(name+'.npz');struct=OUT/'structures'/(name+'.json')
 assert npz.stat().st_size==int(r['npz_bytes']) and struct.stat().st_size==int(r['structure_bytes'])
 assert digest(npz)==r['npz_sha256'] and digest(struct)==r['structure_sha256']
 with zipfile.ZipFile(npz) as z:assert z.testzip() is None
 assert r['creator_original_byte_parity']=='False' and r['rights_training_admission']=='False'
change={}
for row in delta:
 key=(row['source_key'],row['facet'])
 assert key not in change and row['source_key'] in rkey
 assert row['new_state']==STATE and row['npz_sha256']==rkey[row['source_key']]['npz_sha256']
 val=json.loads(row['value_json'])
 assert val['source_key']==row['source_key'] and val['npz_sha256']==row['npz_sha256']
 assert val['training_admission'] is False and val['creator_original_byte_parity'] is False
 change[key]=row
assert len(delta)==282 and {r['source_key'] for r in delta}==set(rkey)
unchanged=modified=unknown_upgrade=0
with zipfile.ZipFile(BASE/'TruthFoundry_V22_334_NPZ_Evidence.zip') as z,\
 gzip.open(OUT/'FACET_STATES_EFFECTIVE_V23_16397x264.csv.gz','rt',newline='',encoding='utf-8') as state,\
 gzip.open(OUT/'FACET_VALUES_EFFECTIVE_V23_16397x264.csv.gz','rt',newline='',encoding='utf-8') as value:
 old_s=zip_reader(z,'FACET_STATES_EFFECTIVE_V22_16397x264.csv.gz');old_v=zip_reader(z,'FACET_VALUES_EFFECTIVE_V22_16397x264.csv.gz')
 new_s=csv.DictReader(state);new_v=csv.DictReader(value)
 assert old_s.fieldnames==old_v.fieldnames==new_s.fieldnames==new_v.fieldnames and len(new_s.fieldnames)==265
 seen=set()
 for a,b,c,d in zip(old_s,old_v,new_s,new_v):
  name=a['source_key'];assert name==b['source_key']==c['source_key']==d['source_key'] and name not in seen
  seen.add(name)
  for facet in new_s.fieldnames[1:]:
   pair=(name,facet);x=change.get(pair)
   if x:
    assert a[facet]==x['previous_state']=='UNKNOWN_UNREVIEWED'
    assert not b[facet] and x['previous_value_sha256']==''
    assert c[facet]==x['new_state']==STATE and d[facet]==x['value_json']
    modified+=1;unknown_upgrade+=1
   else:
    assert a[facet]==c[facet] and b[facet]==d[facet]
    unchanged+=1
 assert len(seen)==16397
 assert next(old_s,None) is None and next(old_v,None) is None and next(new_s,None) is None and next(new_v,None) is None
assert modified==282 and unchanged==4328526 and modified+unchanged==16397*264
ov=json.loads((OUT/'SUMMARY_OVERLAY_V23.json').read_text());nu=json.loads((OUT/'SUMMARY_NUMERIC_V23.json').read_text())
assert ov['new_unknown_cells']==282 and ov['unchanged_cells']==unchanged
assert nu['combined_numeric_pass']==366 and nu['quarantined']==0
for f in ('FACET_STATES_EFFECTIVE_V23_16397x264.csv.gz','FACET_VALUES_EFFECTIVE_V23_16397x264.csv.gz','NPZ_FACET_DELTA_V23.csv.gz'):
 assert ov[f+'_sha256']==digest(OUT/f)
assert not (OUT/'QUARANTINE_V23.csv').read_text().splitlines()[1:]
report={'schema':'RealSaS.V23.IndependentFullMatrixAndSourceReadback.v1','result':'PASS','source_npz_rehashed':32,'source_structure_rehashed':32,'full_source_keys':16397,'facets_per_source':264,'total_cells':4328808,'changed_unknown_cells':282,'unchanged_cells':4328526,'higher_authority_overwrites':0,'training_admissions':0}
(OUT/'INDEPENDENT_READBACK_V23.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))