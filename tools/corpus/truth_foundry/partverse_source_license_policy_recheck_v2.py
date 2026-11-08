#!/usr/bin/env python3
"""Reclassify official original license declarations without re-downloading Parquet.

Uses exactly the SHA-sealed 7034 source crosswalk from Objaverse-XL; original
license text remains untouched. This is *policy triage*, not legal clearance.
"""
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from tools.corpus.truth_foundry.partverse_objaverse_original_license_crosswalk_v1 import license_class

INPUT_SHA='e24758b8c3e88e04ed9714d82d83a2542ff1b24acacb5762dbe96a85bb52c986'
CLASS_MAP={
 'ATTRIBUTION_REVIEW':('ATTRIBUTION_REQUIRED','PERMISSIVE_LICENSE_CANDIDATE__RIGHTS_AND_SOURCE_BYTES_UNVERIFIED'),
 'ATTRIBUTION_SHAREALIKE_REVIEW':('ATTRIBUTION_SHAREALIKE_REVIEW_REQUIRED','LICENSE_COPyleft_SCOPE_NOT_COURTED'),
 'CC0_OR_PUBLIC_DOMAIN_CLAIM':('PUBLIC_DOMAIN_CLAIM_NEEDS_SOURCE_VERIFICATION','PERMISSIVE_LICENSE_CANDIDATE__RIGHTS_AND_SOURCE_BYTES_UNVERIFIED'),
 'NONCOMMERCIAL_OR_RESTRICTED':('COMMERCIAL_ML_EXCLUDE_PENDING_INDEPENDENT_PERMISSION','NOT_COMMERCIALLY_ELIGIBLE_FROM_CLAIM'),
 'NO_DERIVATIVES_OR_RESTRICTED':('DERIVATIVE_USE_UNSUPPORTED_PENDING_INDEPENDENT_PERMISSION','NOT_COMMERCIALLY_ELIGIBLE_FROM_CLAIM'),
 'OTHER_UNREVIEWED_LICENSE':('UNKNOWN_REVIEW_REQUIRED','NOT_LEGALLY_CLEARED'),
 'UNSPECIFIED':('MISSING_LICENSE','NOT_LEGALLY_CLEARED'),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def run(source,out,source_sha=INPUT_SHA):
 source,out=Path(source),Path(out)
 if sha(source)!=source_sha:raise ValueError('FROZEN_SOURCE_SHA_MISMATCH')
 with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=7034 or len({r['uid'] for r in rows})!=7034:raise ValueError('SOURCE_COUNT_OR_DUPLICATE')
 cc=Counter();by_morph=Counter();outrows=[]
 for row in rows:
  if row['source_metadata_match_state']!='EXACT_SHA_AND_SOURCE_UID_METADATA_CLAIM':
   raise ValueError('SOURCE_CROSSWALK_NOT_EXACT')
  c=license_class(row['objaverse_license_claim'])
  policy,training=CLASS_MAP[c]
  record={**row,'corrected_license_class_v2':c,'source_license_policy_triage':policy,
   'source_training_candidate_triage':training,
   'original_source_bytes_verified':'False',
   'independent_original_creator_permission_court':'NOT_RUN',
   'license_terms_applicability_court':'NOT_RUN',
   'attribution_plan_complete':'False','head_training_eligible':'False'}
  outrows.append(record)
  cc[c]+=1
  by_morph[(row['provider_weak_morphology'],c)]+=1
 out.mkdir(parents=True,exist_ok=True)
 a=out/'PARTVERSE_7034_ORIGINAL_SOURCE_LICENSE_TRIAGE_V2.csv'
 with a.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
 b=out/'PARTVERSE_LICENSE_BY_WEAK_MORPHOLOGY_V2.csv'
 with b.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['weak_morphology','source_license_class','source_record_count']);w.writeheader()
  for (m,cls),n in sorted(by_morph.items()):w.writerow({'weak_morphology':m,'source_license_class':cls,'source_record_count':n})
 possible=cc['ATTRIBUTION_REVIEW']+cc['ATTRIBUTION_SHAREALIKE_REVIEW']+cc['CC0_OR_PUBLIC_DOMAIN_CLAIM']
 s={'schema':'RealSaS.PartVerseSourceOriginalLicenseTriage.v2','input_sha256':sha(source),
    'source_count':len(rows),'provider_identity_exact_match_count':7034,
    'license_class_counts':dict(cc), 'source_license_permissive_candidate_not_training_admitted':possible,
    'commercial_restriction_or_no_derivative_claim_count':cc['NONCOMMERCIAL_OR_RESTRICTED']+cc['NO_DERIVATIVES_OR_RESTRICTED'],
    'original_bytes_rehashed':0,'source_rights_independent_pass':0,'training_admission':0,
    'source_2d_art_approved':0,'outputs_sha256':{p.name:sha(p) for p in (a,b)}}
 (out/'SUMMARY_V2.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 return s

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.source,a.out),indent=2))