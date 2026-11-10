#!/usr/bin/env python3
"""Independent V25 source-byte/readback verifier, fail-closed; never grants training rights."""
from __future__ import annotations
import csv,gzip,hashlib,io,json,pathlib,zipfile,collections
import numpy as np
ROOT=pathlib.Path('/mnt/data');DIR=ROOT/'realsas_v25';OLD=ROOT/'TruthFoundry_V24_386_Source_Court_Evidence.zip'
STATE='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
def hash_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2<<20),b''): h.update(b)
 return h.hexdigest()
def zipcsv(z,n):return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(n)),encoding='utf-8',newline=''))
with gzip.open(DIR/'NEW_MASTER726_NPZ_COURT_V25.csv.gz','rt',newline='') as f: receipts=list(csv.DictReader(f))
with gzip.open(DIR/'NPZ_FACET_DELTA_V25.csv.gz','rt',newline='') as f: changes=list(csv.DictReader(f))
with zipfile.ZipFile(OLD) as z:
 pre=list(zipcsv(z,'NEW_MASTER726_NPZ_COURT_V24.csv.gz'))
assert len(pre)==20 and len(receipts)==36
assert len({r['folder_name'] for r in receipts})==36
assert set(r['folder_name'] for r in pre).isdisjoint(r['folder_name'] for r in receipts)
assert len({r['source_key'] for r in receipts})==36
summary=collections.Counter();index={r['source_key']:r for r in receipts}
for r in receipts:
 n=r['folder_name'];npz=DIR/'sources'/(n+'.npz');structure=DIR/'structures'/(n+'.json')
 assert npz.is_file() and structure.is_file()
 assert int(r['npz_bytes'])==npz.stat().st_size and int(r['structure_bytes'])==structure.stat().st_size
 assert r['npz_sha256']==hash_file(npz) and r['structure_sha256']==hash_file(structure)
 with zipfile.ZipFile(npz) as z:assert z.testzip() is None
 with np.load(npz,allow_pickle=False) as arrays:
  v=arrays['vertices'];faces=arrays['faces'];skin=arrays['skin'];parents=arrays['parents']
  assert np.isfinite(v).all() and np.isfinite(skin).all()
  assert skin.shape==(len(v),len(parents))
  assert faces.shape[1]==3 and np.issubdtype(faces.dtype,np.integer)
  assert np.all(faces>=0) and np.all(faces<len(v))
  assert len(v)==int(r['vertex_count']) and len(faces)==int(r['face_count']) and len(parents)==int(r['joint_count'])
  assert skin.size==int(r['skin_weight_cells'])
  row_sums=skin.astype('float64').sum(axis=1)
  assert int((row_sums<=1e-6).sum())==int(r['zero_skin_vertex_count'])
  assert int((skin<-1e-7).sum())==int(r['negative_weight_count'])
  assert int((skin>1+1e-5).sum())==int(r['overone_weight_count'])
  assert int(((abs(row_sums-1)>1e-3)&(row_sums>1e-6)).sum())==int(r['bad_nonzero_skin_row_sum_count'])
  assert int((skin>1e-7).sum())==int(r['positive_influence_cells'])
 with structure.open() as f: st=json.load(f)
 assert sum(x['vertex_count'] for x in st['mesh_objects'])==int(r['vertex_count'])
 assert sum(x['face_count'] for x in st['mesh_objects'])==int(r['face_count'])
 assert st['bone_count']==len(st['bone_names'])==int(r['joint_count'])
 assert r['creator_original_byte_parity']=='False' and r['rights_training_admission']=='False'
 for key in ['vertex_count','face_count','joint_count','skin_weight_cells','zero_skin_vertex_count','nonmanifold_edge_count']:
  summary[key]+=int(r[key])
D={}
for r in changes:
 k=(r['source_key'],r['facet']);assert k not in D and r['source_key'] in index
 assert r['new_state']==STATE and r['npz_sha256']==index[r['source_key']]['npz_sha256']
 v=json.loads(r['value_json']);assert v['source_key']==r['source_key'] and v['npz_sha256']==r['npz_sha256']
 assert not any(v[x] for x in ('creator_original_byte_parity','rights_admission','training_admission','semantic_truth','2d_front_back_parity','animation_ready_tessa'))
 D[k]=r
assert len(D)==324 and {k[0] for k in D}==set(index)
nochange=changed=0
with zipfile.ZipFile(OLD) as z,gzip.open(DIR/'FACET_STATES_EFFECTIVE_V25_16397x264.csv.gz','rt',newline='') as new_s,gzip.open(DIR/'FACET_VALUES_EFFECTIVE_V25_16397x264.csv.gz','rt',newline='') as new_v:
 s0=zipcsv(z,'FACET_STATES_EFFECTIVE_V24_16397x264.csv.gz');v0=zipcsv(z,'FACET_VALUES_EFFECTIVE_V24_16397x264.csv.gz');s1=csv.DictReader(new_s);v1=csv.DictReader(new_v)
 assert s0.fieldnames==v0.fieldnames==s1.fieldnames==v1.fieldnames and len(s1.fieldnames)==265
 seen=set()
 for a,b,c,d in zip(s0,v0,s1,v1):
  k=a['source_key'];assert k==b['source_key']==c['source_key']==d['source_key'] and k not in seen;seen.add(k)
  for col in s1.fieldnames[1:]:
   delta=D.get((k,col))
   if delta:
    assert a[col]==delta['previous_state']=='UNKNOWN_UNREVIEWED'
    assert b[col]=='' and delta['previous_value_sha256']==''
    assert c[col]==STATE and d[col]==delta['value_json']
    changed+=1
   else:
    assert a[col]==c[col] and b[col]==d[col]
    nochange+=1
 assert len(seen)==16397 and all(next(reader,None) is None for reader in (s0,v0,s1,v1))
assert changed==324 and nochange==4328484
out={'result':'PASS','schema':'RealSaS.V25.IndependentSourceAndFullMatrixReadback.v1',
     'source_npz_rehashed':36,'source_structure_rehashed':36,'combined_normalized_court_sources':422,
     'source_rows':16397,'facets':264,'changed_unknown_cells':changed,'unchanged_cells':nochange,
     'higher_authority_overwrites':0,'training_admissions':0,'independent_numeric_partial_metrics':dict(summary)}
(DIR/'INDEPENDENT_READBACK_V25.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))