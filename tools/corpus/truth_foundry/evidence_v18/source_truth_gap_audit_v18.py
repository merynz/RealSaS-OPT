#!/usr/bin/env python3
"""Source-truth gap audit over immutable V11 inventory and corrected V18 states.

This does not create source truth or training admission. It stratifies evidence
claims and identifies missing *qualified* gates for scalable Truth Foundry work.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, pathlib

EXPECTED_ROWS=16397
EXPECTED_FACETS=264
SOURCE_LOCK='6436a6696a55c7643818914b6ebab5a5728d4eae23409d67e6e6d210c2b17ae4'
STATE_LOCK='6e79d839ef38b2f210e9e77bf1a89291f014330e38cafef6b742114bd15c0e65'
GATES={
 'original_source_bytes_courted': 'True',
 'rights_training_eligible':'True',
 'lineage_dedup_court':'PASS',
 'lineage_safe_split_court':'PASS',
 'subject_is_character_court':'PASS',
 'morphology_verified':'PASS',
 'native_pose_court':'PASS',
 'anatomical_front_court':'PASS',
 'camera_court':'PASS',
 'two_view_front_back_pair_court':'PASS',
 'attachment_relation_court':'PASS',
 'native_rgba_parity_court':'PASS',
 'training_authorized':'True',
}
# A provider claim, recording of historical recovery, or an unreviewed
# segmentation suggestion must NOT be treated as numeric physical truth.
NON_PHYSICAL_PREFIXES=(
 'RECORDED_', 'PROVIDER_', 'HISTORICAL_', 'SEGMENTATION_PROPOSAL_',
)

def sha256(path: pathlib.Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):
   h.update(block)
 return h.hexdigest()

def authority(state: str):
 if state=='UNKNOWN_UNREVIEWED':return 'UNKNOWN'
 if state.startswith(NON_PHYSICAL_PREFIXES):return 'METADATA_OR_HISTORICAL_RECEIPT_ONLY'
 if state.startswith(('OBSERVED_', 'IMPORTED_NUMERIC_', 'PROCESSED_MESH_', 'CANONICAL_GLB_')):
  return 'SCOPED_NUMERIC_OR_DERIVED_OBSERVATION_NOT_ADMISSION'
 raise ValueError('UNCLASSIFIED_STATE:'+state)

def gate_ok(value: str, expected: str):
 return value==expected

def run(inventory: pathlib.Path,states: pathlib.Path,out: pathlib.Path):
 if sha256(inventory)!=SOURCE_LOCK:raise ValueError('INVENTORY_SHA_DRIFT')
 if sha256(states)!=STATE_LOCK:raise ValueError('STATE_SHA_DRIFT')
 out.mkdir(parents=True,exist_ok=True)
 gate_values={k:collections.Counter() for k in GATES}
 provider=collections.defaultdict(collections.Counter)
 states_seen=collections.Counter();family=collections.Counter();facets=collections.defaultdict(collections.Counter)
 source_rows=0;row_keys=set()
 queue=out/'SOURCE_TRUTH_GAPS_V18.csv.gz'
 cols=['source_key','pool','provider','candidate_id','structure_evidence_class','source_import_observed',
  'mesh_object_count','mesh_connected_island_count','bone_count','skin_rows_sampled',
  'observed_action_count','texture_image_count','partverse_v10_processed_numeric_court',
  'canonical_v6_sha_verified','v5_source_rest_court','master_v11_source_recovery_status',
  'unreviewed_facets','metadata_or_receipt_facets','scoped_numeric_or_derivative_facets',
  'qualified_source_truth_gates_passed','missing_source_truth_gates','source_truth_ready_for_training']
 with gzip.open(inventory,'rt',newline='',encoding='utf-8') as fi,gzip.open(states,'rt',newline='',encoding='utf-8') as fs,gzip.open(queue,'wt',newline='',encoding='utf-8') as fo:
  ir,sr=csv.DictReader(fi),csv.DictReader(fs)
  facet_names=sr.fieldnames[1:]
  if len(facet_names)!=EXPECTED_FACETS:raise ValueError('FACET_SCHEMA_DRIFT')
  w=csv.DictWriter(fo,fieldnames=cols);w.writeheader()
  for inv,ss in zip(ir,sr):
   key=inv['source_key']
   if key!=ss['source_key'] or key in row_keys:raise ValueError('SOURCE_IDENTITY_DRIFT')
   row_keys.add(key);source_rows+=1
   domain=inv['provider'];cnt=provider[domain];cnt['records']+=1
   observed=collections.Counter()
   for facet in facet_names:
    state=ss[facet];category=authority(state)
    states_seen[state]+=1;family[category]+=1;facets[facet][category]+=1
    observed[category]+=1
   for category,value in observed.items():cnt[category]+=value
   passes=[];missing=[]
   for gate,expected in GATES.items():
    value=inv.get(gate,'')
    gate_values[gate][value]+=1
    if gate_ok(value,expected):passes.append(gate)
    else:missing.append(gate)
   if inv.get('training_authorized')=='True' and missing:raise ValueError('AUTHORITY_INCONSISTENCY')
   row={k:inv.get(k,'') for k in cols if k in inv}
   row.update(unreviewed_facets=observed['UNKNOWN'],
      metadata_or_receipt_facets=observed['METADATA_OR_HISTORICAL_RECEIPT_ONLY'],
      scoped_numeric_or_derivative_facets=observed['SCOPED_NUMERIC_OR_DERIVED_OBSERVATION_NOT_ADMISSION'],
      qualified_source_truth_gates_passed=len(passes),
      missing_source_truth_gates='|'.join(missing),
      source_truth_ready_for_training='False' if missing else 'GATES_ONLY_NOT_AUTHORIZATION')
   w.writerow(row)
  if next(ir,None) is not None or next(sr,None) is not None:raise ValueError('ROW_COUNT_MISMATCH')
 if source_rows!=EXPECTED_ROWS or len(row_keys)!=EXPECTED_ROWS:raise ValueError('SOURCE_COUNT_DRIFT')
 total=EXPECTED_ROWS*EXPECTED_FACETS
 if sum(family.values())!=total:raise ValueError('FACET_TOTAL_DRIFT')
 with (out/'FACET_AUTHORITY_GAPS_V18.csv').open('w',newline='',encoding='utf-8') as fo:
  w=csv.writer(fo);w.writerow(['facet','total','unknown','metadata_historical','scoped_derived','qualified_training_admission'])
  for facet,cc in facets.items():
   w.writerow([facet,EXPECTED_ROWS,cc['UNKNOWN'],cc['METADATA_OR_HISTORICAL_RECEIPT_ONLY'],cc['SCOPED_NUMERIC_OR_DERIVED_OBSERVATION_NOT_ADMISSION'],0])
 with (out/'PROVIDER_AUTHORITY_GAPS_V18.csv').open('w',newline='',encoding='utf-8') as fo:
  w=csv.writer(fo);w.writerow(['provider','candidate_records','unknown_facets','metadata_historical_facets','scoped_derived_facets','training_admissions'])
  for name,cc in sorted(provider.items()):w.writerow([name,cc['records'],cc['UNKNOWN'],cc['METADATA_OR_HISTORICAL_RECEIPT_ONLY'],cc['SCOPED_NUMERIC_OR_DERIVED_OBSERVATION_NOT_ADMISSION'],0])
 with (out/'ADMISSION_GATE_GAPS_V18.csv').open('w',newline='',encoding='utf-8') as fo:
  w=csv.writer(fo);w.writerow(['gate','target_value','passed','not_passed','observed_value_distribution_json'])
  for k,expected in GATES.items():
   cc=gate_values[k];passed=cc[expected]
   w.writerow([k,expected,passed,EXPECTED_ROWS-passed,json.dumps(dict(cc),sort_keys=True)])
 summary={
  'schema':'RealSaS.TruthFoundryV18EvidenceAuthorityGapAudit.v1',
  'scope':'EXISTING_INVENTORY_AND_CORRECTED_V18_FACET_STATES_ONLY_NO_NEW_SOURCE_COURT',
  'source_records':EXPECTED_ROWS,'facet_columns':EXPECTED_FACETS,'total_facets':total,
  'authority_counts':dict(family),'distinct_states':dict(states_seen),
  'true_training_admissions':0,'original_source_rehash_admissions':0,
  'sources_with_training_gate_pass':0,
  'input_sha256':{'inventory':SOURCE_LOCK,'states':STATE_LOCK},
  'output_sha256':{p.name:sha256(p) for p in [out/'SOURCE_TRUTH_GAPS_V18.csv.gz',out/'FACET_AUTHORITY_GAPS_V18.csv',out/'PROVIDER_AUTHORITY_GAPS_V18.csv',out/'ADMISSION_GATE_GAPS_V18.csv']},
 }
 (out/'SUMMARY_V18.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf-8')
 return summary

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--inventory',required=True,type=pathlib.Path)
 p.add_argument('--states',required=True,type=pathlib.Path)
 p.add_argument('--out',required=True,type=pathlib.Path)
 a=p.parse_args();s=run(a.inventory,a.states,a.out)
 print(json.dumps({k:s[k] for k in ('source_records','total_facets','authority_counts','distinct_states')},sort_keys=True,indent=2))
if __name__=='__main__':main()