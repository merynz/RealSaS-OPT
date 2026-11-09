#!/usr/bin/env python3
"""Generate provider+facet coverage and protected 726-source transfer plan from V11/V17."""
import argparse,csv,gzip,collections,json,hashlib,pathlib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory',required=True,type=pathlib.Path)
p.add_argument('--queue',required=True,type=pathlib.Path)
p.add_argument('--physical-pilot',required=True,type=pathlib.Path)
p.add_argument('--out',required=True,type=pathlib.Path)
args=p.parse_args()
O=args.out;O.mkdir(parents=True,exist_ok=True);I=args.inventory
source={}
with gzip.open(I,'rt',newline='') as f:
 for r in csv.DictReader(f):source[r['source_key']]=r
assert len(source)==16397
providers=collections.defaultdict(lambda:collections.Counter())
with gzip.open(O/'SOURCE_READINESS_V17_16397.csv.gz','rt',newline='') as f:
 for r in csv.DictReader(f):
  p=r['provider'];providers[p]['candidate_records']+=1
  providers[p]['new_partial_cells']+=int(r['new_partial_cells'])
  providers[p]['candidate_records_with_new_partial_cells']+=int(int(r['new_partial_cells'])>0)
  providers[p]['prior_observed_cells']+=int(r['previous_observed_facet_cells'])
  providers[p]['source_court_records']+=int(r['evidence_route'] in ('MASTER','PARTVERSE','CORE','UNIML'))
  providers[p]['metadata_only_records']+=int(r['evidence_route']=='METADATA')
  providers[p]['no_mapped_court_records']+=int(r['evidence_route']=='METADATA')
with (O/'PROVIDER_READINESS_V17.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['provider','candidate_records','source_court_records','metadata_only_records','prior_observed_facets','new_partial_facets','candidates_with_new_observations','current_original_rehash','training_admissions'])
 for p,c in sorted(providers.items()):w.writerow([p,c['candidate_records'],c['source_court_records'],c['metadata_only_records'],c['prior_observed_cells'],c['new_partial_cells'],c['candidate_records_with_new_partial_cells'],0,0])
facet_stats=collections.defaultdict(lambda:collections.Counter())
with gzip.open(O/'FACET_STATES_EFFECTIVE_V17_16397x264.csv.gz','rt',newline='') as f:
 reader=csv.DictReader(f);facets=reader.fieldnames[1:]
 for row in reader:
  for facet in facets:facet_stats[facet][row[facet]]+=1
with (O/'FACET_COVERAGE_V17.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['facet','candidate_rows','unknown_unreviewed','observed_or_recorded','observed_percent','state_count','state_distribution_json'])
 for facet in facets:
  c=facet_stats[facet];unknown=c['UNKNOWN_UNREVIEWED'];known=16397-unknown
  w.writerow([facet,16397,unknown,known,round(known/16397*100,2),len(c),json.dumps(dict(c),sort_keys=True)])
with args.queue.open(newline='',encoding='utf-8') as qfile:
 queue=list(csv.DictReader(qfile))
assert len(queue)==1527
eligible=[];holdouts=[];other=collections.Counter()
with args.physical_pilot.open(encoding='utf-8') as f:
 actual={r['source_key']:r for r in map(json.loads,f)}
assert len(actual)==10
for r in queue:
 src=source[r['source_key']]
 status=src['master_v11_source_recovery_status']
 if status=='SHA_VERIFIED_RECOVERY_RECEIPT':eligible.append((r,src))
 elif status=='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED':holdouts.append(r)
 else:other[status]+=1
assert (len(eligible),len(holdouts))==(726,135)
with gzip.open(O/'MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz','wt',newline='') as f:
 w=csv.writer(f);w.writerow(['source_key','candidate_id','provider','relative_folder_under_variants','normalized_npz_relative_path','source_structure_relative_path','historical_recovery_sha256','pilot_v16_npz_rehashed','pilot_v16_npz_sha256','pilot_v16_structure_sha256','creator_original_byte_parity','rights_training_admission'])
 for r,src in eligible:
  folder=r['candidate_id'].replace(':','_',1)
  v=actual.get(r['source_key'])
  w.writerow([r['source_key'],r['candidate_id'],r['provider'],folder,folder+'/normalized.npz',folder+'/source_structure.json',src['master_v11_recovery_receipt_source_sha256'],bool(v),v['normalized_npz_sha256'] if v else '',v['source_structure_sha256'] if v else '',False,False])
print('provider_count',len(providers),'total',sum(x['candidate_records'] for x in providers.values()))
print('recovery 726, holdouts 135, other',other,'pilot_10_verified',sum(k in {r['source_key'] for r,s in eligible} for k in actual))
print('sources',len(source),'facets',len(facet_stats))