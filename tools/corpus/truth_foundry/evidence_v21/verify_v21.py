#!/usr/bin/env python3
"""Independent V20->V21 full 4,328,808-cell audit and 76 original-byte derivative rehash."""
from __future__ import annotations
import csv,gzip,hashlib,io,json,os,pathlib,zipfile
ROOT=pathlib.Path(os.environ.get('REALSAS_TRUTH_ROOT','/mnt/data'))
OUT=ROOT/'realsas_truth_v21'
BASE=ROOT/'TruthFoundry_V20_190_NPZ_Evidence.zip'
SHA='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def zreader(z,p):return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(p)),encoding='utf-8',newline=''))
with gzip.open(OUT/'NEW_MASTER726_NPZ_COURT_V21.csv.gz','rt',newline='',encoding='utf-8') as f:
 rec=list(csv.DictReader(f))
assert len(rec)==76 and len(set(r['source_key'] for r in rec))==76
for r in rec:
 p=OUT/'sources'/(r['folder_name']+'.npz')
 s=OUT/'structures'/(r['folder_name']+'.json')
 assert p.is_file() and s.is_file() and p.stat().st_size==int(r['npz_bytes']) and s.stat().st_size==int(r['structure_bytes'])
 assert digest(p)==r['npz_sha256'] and digest(s)==r['structure_sha256']
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
by_key={r['source_key']:r for r in rec}
with gzip.open(OUT/'NPZ_FACET_DELTA_V21.csv.gz','rt',newline='',encoding='utf-8') as f:
 changes=list(csv.DictReader(f))
seen=set();delta={}
for r in changes:
 pair=(r['source_key'],r['facet'])
 assert pair not in seen;seen.add(pair);delta[pair]=r
 assert r['source_key'] in by_key
 assert r['new_state']==SHA and r['npz_sha256']==by_key[r['source_key']]['npz_sha256']
 assert r['source_scope']
 assert json.loads(r['value_json'])['source_key']==r['source_key']
 assert json.loads(r['value_json'])['training_admission'] is False
unknown=updated=unchanged=0
with zipfile.ZipFile(BASE) as z,gzip.open(OUT/'FACET_STATES_EFFECTIVE_V21_16397x264.csv.gz','rt',newline='',encoding='utf-8') as sf,gzip.open(OUT/'FACET_VALUES_EFFECTIVE_V21_16397x264.csv.gz','rt',newline='',encoding='utf-8') as vf:
 bsi,bvi=zreader(z,'FACET_STATES_EFFECTIVE_V20_16397x264.csv.gz'),zreader(z,'FACET_VALUES_EFFECTIVE_V20_16397x264.csv.gz')
 nsi,nvi=csv.DictReader(sf),csv.DictReader(vf)
 assert bsi.fieldnames==bvi.fieldnames==nsi.fieldnames==nvi.fieldnames and len(nsi.fieldnames)==265
 source_seen=set()
 for old_s,old_v,new_s,new_v in zip(bsi,bvi,nsi,nvi):
  key=old_s['source_key'];assert key==old_v['source_key']==new_s['source_key']==new_v['source_key'] and key not in source_seen
  source_seen.add(key)
  for facet in nsi.fieldnames[1:]:
   previous,previous_value=old_s[facet],old_v[facet]
   now,nv=new_s[facet],new_v[facet]
   c=delta.get((key,facet))
   if c:
    assert c['previous_state']==previous and c['previous_value_sha256']==(hashlib.sha256(previous_value.encode()).hexdigest() if previous_value else '')
    assert c['new_state']==now==SHA and c['value_json']==nv
    assert previous=='UNKNOWN_UNREVIEWED'
    unknown+=1;updated+=1
   else:
    assert previous==now and previous_value==nv
    unchanged+=1
 assert len(source_seen)==16397 and next(bsi,None) is None and next(bvi,None) is None and next(nsi,None) is None and next(nvi,None) is None
assert updated==len(delta)==len(changes)==678 and unknown==678 and unchanged==4328808-678
summary=json.loads((OUT/'SUMMARY_OVERLAY_V21.json').read_text())
assert summary['new_unknown_cells_observed']==unknown and summary['matrix_cells_unchanged']==unchanged
assert summary['state_sha256']==digest(OUT/'FACET_STATES_EFFECTIVE_V21_16397x264.csv.gz')
assert summary['value_sha256']==digest(OUT/'FACET_VALUES_EFFECTIVE_V21_16397x264.csv.gz')
assert summary['delta_sha256']==digest(OUT/'NPZ_FACET_DELTA_V21.csv.gz')
report={'schema':'RealSaS.TruthFoundryV21IndependentRehashAndMatrixReadback.v1', 'verified_new_npz_sha256':len(rec),'verified_new_structure_sha256':len(rec), 'full_matrix_cells':16397*264, 'unchanged_cells':unchanged,'changed_unknown_cells':unknown,'non_unknown_promotions':0,'independent_readback':'PASS','training_admissions':0}
(OUT/'INDEPENDENT_READBACK_V21.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))