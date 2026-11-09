#!/usr/bin/env python3
"""Independent exact-cell, source bytes and SHA readback of V20 against V19."""
from __future__ import annotations
import collections,csv,gzip,hashlib,json,pathlib,zipfile,os
import numpy as np
ROOT=pathlib.Path(os.environ.get('REALSAS_TRUTH_ROOT','/mnt/data')).resolve();BASE=ROOT/'realsas_truth_v19';O=ROOT/'realsas_truth_v20'
S=O/'SUMMARY_V20.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 summary=json.loads(S.read_text())
 names={k:O/v for k,v in {'states':'FACET_STATES_EFFECTIVE_V20_16397x264.csv.gz','values':'FACET_VALUES_EFFECTIVE_V20_16397x264.csv.gz','npz_receipts':'MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz','delta':'NPZ_FACET_DELTA_V20.csv.gz','quarantine':'NPZ_QUARANTINE_V20.csv'}.items()}
 for key,path in names.items():assert sha(path)==summary['output_sha256'][key],('OUTPUT_SHA_MISMATCH',key)
 bfiles=[BASE/'FACET_STATES_EFFECTIVE_V19_16397x264.csv.gz',BASE/'FACET_VALUES_EFFECTIVE_V19_16397x264.csv.gz']
 for p,k in zip(bfiles,['V19_states','V19_values']):assert sha(p)==summary['input_sha256'][k]
 assert sha(ROOT/'realsas_truth_v17/MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz')==summary['input_sha256']['plan']
 allowed=set()
 with gzip.open(ROOT/'realsas_truth_v17/MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   assert r['source_key'] not in allowed
   allowed.add(r['source_key'])
 assert len(allowed)==726
 receipts={}
 with gzip.open(names['npz_receipts'],'rt',newline='') as f:
  for r in csv.DictReader(f):
   k=r['source_key'];assert k in allowed and k not in receipts
   p=O/'sources'/(r['folder_name']+'.npz')
   assert p.is_file() and sha(p)==r['npz_sha256'] and p.stat().st_size==int(r['npz_bytes']),'INDEPENDENT_SOURCE_SHA_OR_LENGTH_MISMATCH'
   with zipfile.ZipFile(p) as z:assert z.testzip() is None,'NPZ_CRC_FAILURE'
   with np.load(p,allow_pickle=False) as z:
    assert len(z['vertices'])==int(r['vertex_count']) and len(z['faces'])==int(r['face_count']) and z['skin'].shape==(int(r['vertex_count']),int(r['joint_count']))
    assert int(z['skin'].size)==int(r['skin_weight_cells'])
    assert np.isfinite(z['skin']).all() and np.isfinite(z['vertices']).all()
    assert np.all(z['faces']>=0) and np.all(z['faces']<len(z['vertices']))
   assert r['creator_original_bytes_courted']=='False' and r['rights_admission']=='False' and r['training_admission']=='False'
   receipts[k]=r
 assert len(receipts)==summary['npz_court_pass']
 delta={}
 with gzip.open(names['delta'],'rt',newline='') as f:
  for r in csv.DictReader(f):
   pair=(r['source_key'],r['facet'])
   assert pair not in delta and r['source_key'] in receipts
   assert r['npz_sha256']==receipts[r['source_key']]['npz_sha256']
   val=json.loads(r['value_json'])
   assert val['source_key']==r['source_key'] and val['npz_sha256']==r['npz_sha256']
   assert val['original_source_byte_parity'] is False and val['rights_training_admission'] is False
   assert val['semantic_truth'] is False and val['motion_ready_tessa_admission'] is False
   delta[pair]=r
 assert len(delta)==summary['source_facet_value_changes']
 src=[gzip.open(b,'rt',newline='') for b in bfiles]
 dst=[gzip.open(names[k],'rt',newline='') for k in ('states','values')]
 old_s,old_v,new_s,new_v=map(csv.DictReader,src+dst)
 hdr=old_s.fieldnames
 assert hdr==old_v.fieldnames==new_s.fieldnames==new_v.fieldnames and len(hdr)==265
 seen=set();unchanged=updated=upgrades=newunknown=before_unknown=after_unknown=0;facets=collections.Counter()
 for so,vo,sn,vn in zip(old_s,old_v,new_s,new_v):
  k=so['source_key'];assert k in allowed or k not in receipts
  assert k==vo['source_key']==sn['source_key']==vn['source_key'] and k not in seen
  seen.add(k)
  for facet in hdr[1:]:
   prev,prevval,now,nowval=so[facet],vo[facet],sn[facet],vn[facet]
   before_unknown+=prev=='UNKNOWN_UNREVIEWED';after_unknown+=now=='UNKNOWN_UNREVIEWED'
   d=delta.get((k,facet))
   if d is None:
    assert prev==now and prevval==nowval,('UNDECLARED_MUTATION',k,facet)
    unchanged+=1
   else:
    assert prev==d['previous_state'] and now==d['new_state']
    assert hashlib.sha256(prevval.encode()).hexdigest()==d['previous_value_sha256'] if prevval else d['previous_value_sha256']==''
    assert nowval==d['value_json']
    assert now=='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
    assert prev in ('UNKNOWN_UNREVIEWED','OBSERVED_HISTORICAL_STRUCTURE_JSON_DERIVATIVE_ONLY')
    if prev!='UNKNOWN_UNREVIEWED':
     assert facet in ('joint_count_vs_anatomical_joint','source_vertex_face_count')
     upgrades+=1
    else:newunknown+=1
    updated+=1;facets[facet]+=1
 assert len(seen)==16397
 assert all(next(x,None) is None for x in (old_s,old_v,new_s,new_v))
 for s in src+dst:s.close()
 assert updated==summary['source_facet_value_changes'] and newunknown==summary['newly_nonunknown_cells']
 assert upgrades==summary['upgraded_historical_structure_cells'] and unchanged==summary['untouched_cells']
 assert before_unknown==summary['unknown_before'] and after_unknown==summary['unknown_after']
 assert dict(facets)==summary['added_facet_counts']
 result={'schema':'RealSaS.TruthFoundryV20IndependentPhysicalAndMatrixReadback.v1','status':'PASS_SHA256_SOURCE_BYTES_AND_EXACT_FACET_DELTA',
  'npz_raw_bytes_rehashed':len(receipts),'candidate_rows_checked':len(seen),'facet_columns':264,
  'npz_crc_tested':len(receipts),'updated_cells':updated,'new_nonunknown_cells':newunknown,
  'prior_structure_cells_upgraded':upgrades,'previous_cells_unchanged':unchanged,
  'quarantined':summary['npz_court_quarantined'],
  'source_sha256_receipt_file':summary['output_sha256']['npz_receipts'],
  'creator_original_courts':0,'rights_admissions':0,'training_admissions':0,'protected_holdout_accesses':0}
 (O/'INDEPENDENT_READBACK_V20.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()