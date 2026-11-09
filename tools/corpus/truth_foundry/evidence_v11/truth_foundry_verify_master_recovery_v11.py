#!/usr/bin/env python3
"""Independent V11 source recovery crosswalk and no-promotion verification.

Read BOTH V10 and V11 matrices and raw historical recovery input; recompute
all intentional changes and reject any unrelated cell change.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,collections
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def rz(p):
 with gzip.open(p,'rt',encoding='utf8',newline='') as f:
  yield from csv.DictReader(f)

def main(a):
 d=Path(a.out);s=json.loads((d/'SUMMARY_MASTER_RECOVERY_V11.json').read_text())
 v10=Path(a.v10_inventory);xstate=Path(a.v10_states);xvalue=Path(a.v10_values)
 ni=d/'SOURCE_INVENTORY_DEEP_16397_V11.csv.gz'; ns=d/'FACET_STATES_EFFECTIVE_V11_16397x264.csv.gz';nv=d/'FACET_VALUES_EFFECTIVE_V11_16397x264.csv.gz'
 availability={}
 for r in csv.DictReader(open(a.availability,newline='',encoding='utf8')):
  key='MASTER3993:'+r['canonical_asset_id']
  assert key not in availability
  availability[key]=r
 assert len(availability)==3993
 assert s['source_input_sha256'][Path(a.availability).name]==sha(a.availability)
 for p in (v10,xstate,xvalue):assert s['source_input_sha256'][p.name]==sha(p)
 for p in (ni,ns,nv):assert s['output_sha256'][p.name]==sha(p)
 rows=0;master=0;other=0;counts=collections.Counter();verified=0
 for before,after in zip(rz(v10),rz(ni)):
  assert before['source_key']==after['source_key'];key=before['source_key']
  assert all(after[k]==v for k,v in before.items()),('SILENT_OLD_COLUMN_MUTATION',key)
  added={k:v for k,v in after.items() if k not in before}
  if key in availability:
   r=availability[key];assert after['candidate_id']==r['candidate_id']
   assert added['master_v11_source_recovery_status']==r['source_availability_status']
   assert added['master_v11_recovery_receipt_source_sha256']==r['sha_verified_in_immutable_recovery_receipt']
   assert added['master_v11_original_byte_currently_rehashed']=='FALSE_HISTORICAL_SOURCE_RECEIPT_ONLY'
   assert added['master_v11_original_creator_rights_court']=='UNKNOWN_RIGHTS_NOT_COURTED'
   assert added['master_v11_holdout_protected']==r['holdout_protected']
   if r['source_availability_status']=='SHA_VERIFIED_RECOVERY_RECEIPT':
    assert r['source_expected_sha256']==r['sha_verified_in_immutable_recovery_receipt']
    verified+=1
   counts[r['source_availability_status']]+=1;master+=1
  else:
   assert not any(added.values()),('CROSS_COHORT_LEAK',key);other+=1
  rows+=1
 assert rows==16397 and master==3993 and other==12404 and verified==3192
 assert counts=={'SHA_VERIFIED_RECOVERY_RECEIPT':3192,'LEGACY_PATH_MATCH_UNREHASHED':644,
                 'PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED':135,'RECOVERY_EXCEPTION':22}

 delta=collections.Counter();changed=0;untouched=0;source_rows=0
 for (s0,v0,s1,v1) in zip(rz(xstate),rz(xvalue),rz(ns),rz(nv)):
  assert s0['source_key']==v0['source_key']==s1['source_key']==v1['source_key']
  key=s0['source_key'];
  for facet in s0:
   if facet=='source_key':continue
   if s0[facet]!=s1[facet] or v0[facet]!=v1[facet]:
    assert key in availability and facet in ('original_asset_recovery_status','source_original_byte_digest_pass'),('UNAUTHORIZED_FACET_MUTATION',key,facet)
    assert s0[facet]=='UNKNOWN_UNREVIEWED' and v0[facet]==''
    if facet=='source_original_byte_digest_pass':
     r=availability[key]
     assert r['source_availability_status']=='SHA_VERIFIED_RECOVERY_RECEIPT'
     assert s1[facet]=='HISTORICAL_RECOVERY_SHA_VERIFIED_RECEIPT__NO_CURRENT_BYTE_REHASH'
     data=json.loads(v1[facet]);assert data['historical_receipt_sha256']==r['sha_verified_in_immutable_recovery_receipt']
     assert data['current_rehash_performed'] is False and data['training_authorized'] is False
    else:
     assert s1[facet]=='HISTORICAL_SOURCE_RECOVERY_AVAILABILITY_OBSERVED__NOT_CURRENT_REHASH'
     data=json.loads(v1[facet]);assert data['historical_recovery_status']==availability[key]['source_availability_status']
     assert data['training_authorized'] is False
    delta[facet]+=1;changed+=1
   else:untouched+=1
  source_rows+=1
 assert source_rows==16397 and delta=={'original_asset_recovery_status':3993,'source_original_byte_digest_pass':3192}
 assert changed==7185 and changed+untouched==16397*264
 assert s['original_source_current_run_rehashed']==s['new_training_admission']==s['new_sdf_signed_pass']==0
 receipt={'schema':'RealSaS.Master3993RecoveryV11IndependentReadback',
 'status':'PASS_HISTORICAL_RECOVERY_BINDING_NO_NEW_RAW_SOURCE_AUTHORITY',
 'source_rows':rows,'master_source_rows':master,'nonmaster_rows_unmodified':other,
 'historical_source_recovery_verified_claims':verified,
 'historical_recovery_status_counts':dict(counts),
 'new_historical_observation_cells':changed,'unchanged_facet_cells':untouched,
 'original_source_current_rehash_count':0,'training_admission':0,'global_sdf_admission':0,
 'license_semantic_2d_art_pass':0,
 'sha256':{'master_inventory':sha(ni),'facet_states':sha(ns),'facet_values':sha(nv)}}
 (d/'INDEPENDENT_READBACK_MASTER_V11.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,indent=2,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('availability','v10_inventory','v10_states','v10_values','out'):
  p.add_argument('--'+name.replace('_','-'),required=True)
 main(p.parse_args())