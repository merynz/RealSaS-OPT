#!/usr/bin/env python3
"""Build a source-key-complete 16,397 candidate evidence queue (no admission)."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, pathlib, zipfile, collections

ROOT=pathlib.Path('/mnt/data'); OUT=ROOT/'realsas_v26'
N=16397

def iter_zip_csv(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        with gzip.open(z.open(member), 'rt', encoding='utf8', newline='') as f:
            yield from csv.DictReader(f)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

v17=ROOT/'TruthFoundry_V17_Bulk_16397_Evidence.zip'
original=list(iter_zip_csv(v17,'BASE_V11_SOURCE_INVENTORY.csv.gz'))
assert len(original)==N and len({r['source_key'] for r in original})==N
master_allow={r['source_key']:r for r in iter_zip_csv(v17,'MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz')}
assert len(master_allow)==726
receipts={}
for v,archive,member in [
 ('V20',ROOT/'TruthFoundry_V20_190_NPZ_Evidence.zip','MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz'),
 ('V21',ROOT/'TruthFoundry_V21_266_NPZ_Evidence.zip','NEW_MASTER726_NPZ_COURT_V21.csv.gz'),
 ('V22',ROOT/'TruthFoundry_V22_334_NPZ_Evidence.zip','NEW_MASTER726_NPZ_COURT_V22.csv.gz'),
 ('V23',ROOT/'TruthFoundry_V23_366_NPZ_Evidence.zip','NEW_MASTER726_NPZ_COURT_V23.csv.gz')]:
 for row in iter_zip_csv(archive,member):
    k=row['source_key']
    assert k in master_allow and k not in receipts and row.get('rights_training_admission',row.get('training_admission'))=='False'
    receipts[k]=(v,row)
for row in iter_zip_csv(ROOT/'TruthFoundry_V24_386_Source_Court_Evidence.zip','NEW_MASTER726_NPZ_COURT_V24.csv.gz'):
    k=row['source_key']
    assert k in master_allow and k not in receipts and row.get('rights_training_admission')=='False'
    receipts[k]=('V24',row)
for row in iter_zip_csv(ROOT/'TruthFoundry_V25_422_Source_Court_Evidence.zip','NEW_MASTER726_NPZ_COURT_V25.csv.gz'):
    k=row['source_key']
    assert k in master_allow and k not in receipts and row.get('rights_training_admission')=='False'
    receipts[k]=('V25',row)
with gzip.open(OUT/'NEW_MASTER726_NPZ_COURT_V26.csv.gz','rt',encoding='utf8',newline='') as f:
 for row in csv.DictReader(f):
    k=row['source_key']
    assert k in master_allow and k not in receipts and row.get('rights_training_admission',row.get('training_admission'))=='False'
    receipts[k]=('V26',row)
assert len(receipts)==504
fields=['source_key','pool','provider','candidate_id','source_format_claim','original_source_path_claim',
 'original_sha_claim','rights_license_class_claim','source_recovery_status_historical',
 'protected_holdout','allowlisted_master726_normalized','normalized_numeric_source_court',
 'normalized_numeric_source_court_version','normalized_npz_current_sha256','normalized_structure_current_sha256',
 'original_creator_bytes_currently_courted','verified_real_character','verified_morphology',
 'verified_anatomical_front','verified_front_back_pair','lineage_dedup_court',
 'rights_training_admission','training_authorized','materialization_authorized',
 'source_evidence_next_action','candidate_authority_boundary']
count=collections.Counter();provider=collections.Counter();pstage=collections.defaultdict(collections.Counter)
with gzip.open(OUT/'SOURCE_UNIVERSE_V26_16397.csv.gz','wt',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in original:
    key=r['source_key']; el=master_allow.get(key); court=receipts.get(key); historic=r['master_v11_source_recovery_status']
    protected=historic=='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED'
    if court: stage='NORMALIZED_DERIVATIVE_NUMERIC_PASS_ORIGINAL_AND_RIGHTS_PENDING'
    elif el: stage='NORMALIZED_DERIVATIVE_COURT_QUEUE'
    elif protected: stage='PROTECTED_HOLDOUT_NO_SOURCE_COURT'
    elif historic=='LEGACY_PATH_MATCH_UNREHASHED':stage='MASTER_PATH_ONLY_NEEDS_SOURCE_RECOVERY'
    elif historic=='RECOVERY_EXCEPTION':stage='MASTER_RECOVERY_EXCEPTION'
    elif r['pool']=='MASTER3993':stage='MASTER_ORIGINAL_RECOVERY_AND_RIGHTS_COURT_REQUIRED'
    else:stage='PROVIDER_SPECIFIC_ORIGINAL_OR_PROCESSED_SOURCE_COURT_REQUIRED'
    count[stage]+=1;provider[r['provider']]+=1;pstage[r['provider']][stage]+=1
    if court:
      ver,e=court; npz_sha=e['npz_sha256']; structure_sha=e.get('structure_sha256','')
    else:ver=npz_sha=structure_sha=''
    # Source rows explicitly remain non-admitted until separate original/rights/character/2-view courts.
    row={
      'source_key':key,'pool':r['pool'],'provider':r['provider'],'candidate_id':r['candidate_id'],
      'source_format_claim':r['source_format'],'original_source_path_claim':r['source_relative_path'],
      'original_sha_claim':r['original_sha_claim'],'rights_license_class_claim':r['rights_license_class'],
      'source_recovery_status_historical':historic,'protected_holdout':str(protected),
      'allowlisted_master726_normalized':str(el is not None),'normalized_numeric_source_court':str(court is not None),
      'normalized_numeric_source_court_version':ver,'normalized_npz_current_sha256':npz_sha,
      'normalized_structure_current_sha256':structure_sha,
      'original_creator_bytes_currently_courted':'False','verified_real_character':'False',
      'verified_morphology':'False','verified_anatomical_front':'False','verified_front_back_pair':'False',
      'lineage_dedup_court':'False','rights_training_admission':'False','training_authorized':'False',
      'materialization_authorized':'False','source_evidence_next_action':stage,
      'candidate_authority_boundary':'Scoped normalized derivative measurements only; no source-original parity or admission' if court else 'Historical/metadata only; no new original court'
    }
    w.writerow(row)
assert sum(count.values())==N
assert count['NORMALIZED_DERIVATIVE_NUMERIC_PASS_ORIGINAL_AND_RIGHTS_PENDING']==504
assert count['NORMALIZED_DERIVATIVE_COURT_QUEUE']==222
summary={'schema':'RealSaS.V26.FullSourceUniverseQueue.v1','source_universe':N,
 'candidate_rows_with_actual_normalized_npz_numeric_evidence':504,'eligible_master726':726,
 'remaining_master726_normalized_numeric_courts':222,
 'normalized_numeric_percent_of_master726':round(504/726*100,3),
 'normalized_numeric_percent_of_all_candidate_rows':round(504/N*100,3),
 'original_creator_byte_parity_currently_proven':0,'real_characters_currently_verified':0,
 'morphology_currently_verified':0,'two_view_front_back_currently_verified':0,
 'lineage_dedup_final_pass':0,'rights_training_admissions':0,
 'source_next_action_counts':dict(sorted(count.items())),
 'provider_candidate_counts':dict(sorted(provider.items())),
 'provider_next_actions':{p:dict(c) for p,c in sorted(pstage.items())},
 'v17_source_sha256':digest(v17),'v23_evidence_zip_sha256':digest(ROOT/'TruthFoundry_V23_366_NPZ_Evidence.zip'),
 'v24_parent_sha256':digest(ROOT/'TruthFoundry_V24_386_Source_Court_Evidence.zip'),
 'v25_parent_sha256':digest(ROOT/'TruthFoundry_V25_422_Source_Court_Evidence.zip'),
 'v26_receipt_sha256':digest(OUT/'NEW_MASTER726_NPZ_COURT_V26.csv.gz'),
 'full_universe_csv_sha256':digest(OUT/'SOURCE_UNIVERSE_V26_16397.csv.gz')}
(OUT/'SOURCE_UNIVERSE_SUMMARY_V26.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))