#!/usr/bin/env python3
"""V11 Master3993 original source-availability evidence reconciliation.

Joins 3993 historical recovery receipts into 16397 source rows and 264 facet
states without equating old SHA verification to a present raw-source court.
No legal/semantic/2D-art/model/geometry global-SDF admission.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,collections
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def gzrows(p):
 with gzip.open(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)

def writer(path,fields):
 f=gzip.open(path,'wt',encoding='utf-8',newline='');w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();return f,w

def main(a):
 ava=Path(a.availability);src=Path(a.v10_inventory);states=Path(a.v10_facet_states);values=Path(a.v10_facet_values)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if a.sha_manifest:
  ref=json.loads(Path(a.sha_manifest).read_text());assert sha(ava)==ref['availability'],('SOURCE_AVAILABILITY_SHA_DRIFT',sha(ava))
 records={};sha_set=set();ct=collections.Counter()
 for r in csv.DictReader(ava.open('r',encoding='utf8',newline='')):
  key='MASTER3993:'+r['canonical_asset_id'];assert key not in records
  assert r['provider'] in ('objaverse_animated_originals','quaternius_cc0','kaykit_cc0')
  assert r['training_authorized']=='False' and r['all_model_heads_admitted']=='False'
  status=r['source_availability_status'];ct[status]+=1
  if status=='SHA_VERIFIED_RECOVERY_RECEIPT':
   h=r['source_expected_sha256']
   assert len(h)==64 and h==r['sha_verified_in_immutable_recovery_receipt']
   assert int(r['reported_verified_byte_count'])>0
   assert h not in sha_set,('RECOVERY_SHA_DUPLICATE_CANDIDATE',key);sha_set.add(h)
  else:
   assert not r['sha_verified_in_immutable_recovery_receipt']
   assert int(r['reported_verified_byte_count'])==0
  assert r['source_character_vs_prop']==r['source_morphology']==r['graphics_stylized_2d_art_court']=='UNKNOWN'
  records[key]=r
 assert len(records)==3993 and ct=={'SHA_VERIFIED_RECOVERY_RECEIPT':3192,'LEGACY_PATH_MATCH_UNREHASHED':644,'PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED':135,'RECOVERY_EXCEPTION':22},ct

 extra=['master_v11_source_recovery_status','master_v11_recovery_receipt_source_sha256','master_v11_recovery_source_bytes_reported','master_v11_recovery_source_sha_claim','master_v11_original_byte_currently_rehashed','master_v11_original_creator_rights_court','master_v11_holdout_protected','master_v11_recovery_exception','master_v11_recovery_next_action','master_v11_source_import_remains_missing']
 invp=out/'SOURCE_INVENTORY_DEEP_16397_V11.csv.gz'
 invcount=0;enriched=0;deep=0;historical_pending=0;seen=set()
 with gzip.open(src,'rt',encoding='utf8',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames+extra
  assert 'master_v9_import_receipt_internal_court' in fields
  fp,w=writer(invp,fields)
  for r in reader:
   key=r['source_key'];assert key not in seen;seen.add(key)
   if key in records:
    x=records[key];assert r['candidate_id']==x['candidate_id'],('CANDIDATE_ID_DRIFT',key)
    assert r['provider']==x['provider'],('PROVIDER_DRIFT',key)
    st=x['source_availability_status'];h=x['sha_verified_in_immutable_recovery_receipt'];isdeep=bool(r['master_v9_import_receipt_internal_court']);
    if isdeep:
     assert st=='SHA_VERIFIED_RECOVERY_RECEIPT'
     assert r['master_deep_import_source_sha256']==h
     deep+=1
    elif st=='SHA_VERIFIED_RECOVERY_RECEIPT':historical_pending+=1
    r.update({
     extra[0]:st,extra[1]:h,extra[2]:x['reported_verified_byte_count'],
     extra[3]:x['source_expected_sha256'],extra[4]:'FALSE_HISTORICAL_SOURCE_RECEIPT_ONLY',
     extra[5]:'UNKNOWN_RIGHTS_NOT_COURTED',extra[6]:x['holdout_protected'],
     extra[7]:x['recovery_exception'],
     extra[8]:('REIMPORT_ORIGINAL_SCENE_AND_COURT' if st=='SHA_VERIFIED_RECOVERY_RECEIPT' and not isdeep
                else 'RECOVER_AND_REHASH_ORIGINAL' if st=='LEGACY_PATH_MATCH_UNREHASHED'
                else 'PRESERVE_HOLDOUT' if st=='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED'
                else 'DIAGNOSE_RECOVERY_EXCEPTION' if st=='RECOVERY_EXCEPTION'
                else 'ALREADY_DEEP_IMPORT_RECEIPT__OTHER_COURTS_MISSING'),
     extra[9]:'FALSE' if isdeep else 'TRUE'})
    enriched+=1
   else:
    for e in extra:r[e]=''
   w.writerow(r);invcount+=1
  fp.close()
 assert invcount==16397 and enriched==3993 and deep==2466 and historical_pending==726

 old_cols=['original_asset_recovery_status','source_original_byte_digest_pass','original_source_full_byte_sha']
 facet_delta=[];stat_count=collections.Counter();source_count=0
 sp=out/'FACET_STATES_EFFECTIVE_V11_16397x264.csv.gz'
 vp=out/'FACET_VALUES_EFFECTIVE_V11_16397x264.csv.gz'
 with gzip.open(states,'rt',encoding='utf8') as fs,gzip.open(values,'rt',encoding='utf8') as fv:
  sr=csv.DictReader(fs);vr=csv.DictReader(fv)
  assert sr.fieldnames==vr.fieldnames and len(sr.fieldnames)==265 and all(k in sr.fieldnames for k in old_cols)
  sf,sw=writer(sp,sr.fieldnames);vf,vw=writer(vp,vr.fieldnames)
  for s,v in zip(sr,vr):
   assert s['source_key']==v['source_key'];key=s['source_key']
   if key in records:
    r=records[key];st=r['source_availability_status'];h=r['sha_verified_in_immutable_recovery_receipt']
    field='original_asset_recovery_status'
    assert s[field]=='UNKNOWN_UNREVIEWED' and v[field]=='',('OVERWRITE_EXISTING_EVIDENCE',key,field)
    s[field]='HISTORICAL_SOURCE_RECOVERY_AVAILABILITY_OBSERVED__NOT_CURRENT_REHASH'
    v[field]=json.dumps({'historical_recovery_status':st,'holdout_protected':r['holdout_protected']=='True',
       'source_expected_sha_claim':r['source_expected_sha256'] or None,
       'recovery_exception':r['recovery_exception'] or None,
       'original_creator_license':'UNKNOWN','training_authorized':False},sort_keys=True,separators=(',',':'))
    facet_delta.append((key,field,s[field],v[field]));stat_count[field]+=1
    if h:
     assert s['original_source_full_byte_sha']=='RECORDED_SOURCE_IDENTITY_NOT_REHASHED_BY_THIS_BUILDER'
     assert json.loads(v['original_source_full_byte_sha'])==h,('IDENTITY_SHA_NOT_EQUAL_RECOVERY_SHA',key)
     field='source_original_byte_digest_pass'
     assert s[field]=='UNKNOWN_UNREVIEWED' and v[field]=='',('OVERWRITE_EXISTING_EVIDENCE',key,field)
     s[field]='HISTORICAL_RECOVERY_SHA_VERIFIED_RECEIPT__NO_CURRENT_BYTE_REHASH'
     v[field]=json.dumps({'historical_receipt_sha256':h,'historical_verified_byte_count':int(r['reported_verified_byte_count']),
       'current_rehash_performed':False,'current_source_sha_pass':False,'training_authorized':False},sort_keys=True,separators=(',',':'))
     facet_delta.append((key,field,s[field],v[field]));stat_count[field]+=1
   sw.writerow(s);vw.writerow(v);source_count+=1
  sf.close();vf.close()
 assert source_count==16397 and stat_count=={'original_asset_recovery_status':3993,'source_original_byte_digest_pass':3192},stat_count
 d=out/'FACET_V11_RECOVERY_EVIDENCE_DELTA.csv.gz'
 with gzip.open(d,'wt',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['source_key','facet','observation_state','observed_value']);w.writerows(facet_delta)
 summary={'schema':'RealSaS.Master3993HistoricalRecoveryReconciliation.v11',
  'source_rows_preserved':16397,'master_source_rows_reconciled':enriched,
  'master_with_prior_deep_import_receipt':deep,
  'master_sha_recovery_historical_but_no_deep_import':historical_pending,
  'source_availability_status_breakdown':dict(ct),
  'original_source_current_run_rehashed':0,'current_original_source_rights_pass':0,
  'new_sdf_signed_pass':0,'new_semantic_or_2d_art_pass':0,'new_training_admission':0,
  'facet_observations_new':dict(stat_count),'facet_observations_new_total':len(facet_delta),
  'source_input_sha256':{p.name:sha(p) for p in [ava,src,states,values]},
  'output_sha256':{p.name:sha(p) for p in [invp,sp,vp,d]}}
 (out/'SUMMARY_MASTER_RECOVERY_V11.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('availability','v10_inventory','v10_facet_states','v10_facet_values','out'):
  p.add_argument('--'+name.replace('_','-'),required=True)
 p.add_argument('--sha-manifest');main(p.parse_args())