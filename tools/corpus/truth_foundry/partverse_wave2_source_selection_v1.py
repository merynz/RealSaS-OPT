#!/usr/bin/env python3
"""Select disjoint rare/character-focused PartVerse second wave from pinned metadata.

The first 2,500 source face-map/GLB bundles remain frozen in Drive. This tool
never rewrites them, never treats a VLM-caption tag as morphology truth, and
never admits a PartVerse part as a semantic concept or drawable.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,tempfile
from collections import Counter
from pathlib import Path
from tools.corpus.truth_foundry.partverse_morphology_triage_v1 import run as triage_run

EXPECTED_INITIAL_COUNT=2500
EXPECTED_WAVE2_COUNT=4799
EXPECTED_UNION_COUNT=7299
EXPECTED_OLD_QUEUE_SHA='24fcfd745e0142feed734ce42b605ebd42dc2e5a64e45b7939964ca8780023c1'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def read(path):
 with Path(path).open('r',newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def run(metadata,train,val,source_roster,old_queue,out):
 out.mkdir(parents=True,exist_ok=True)
 if sha(old_queue)!=EXPECTED_OLD_QUEUE_SHA:raise ValueError('FIRST_WAVE_QUEUE_SHA_CHANGED')
 old=read(old_queue)
 if len(old)!=EXPECTED_INITIAL_COUNT:raise ValueError('FIRST_WAVE_COUNT_CHANGED')
 old_ids={r['partverse_uid'] for r in old}
 if len(old_ids)!=len(old):raise ValueError('FIRST_WAVE_DUPLICATE_UID')
 with tempfile.TemporaryDirectory() as temp:
  work=Path(temp)
  report=triage_run(metadata,work,source_roster=source_roster,max_selected=7500,
                    provider_train=train,provider_val=val)
  full=read(work/'PARTVERSE_CANDIDATE_DOWNLOAD_QUEUE_V1.csv')
 if len(full)!=EXPECTED_UNION_COUNT:raise ValueError('FULL_WAVE_QUOTA_DRIFT')
 ids={r['partverse_uid'] for r in full}
 if not old_ids.issubset(ids):raise ValueError('INITIAL_SELECTION_NOT_PRESERVED')
 new=[r for r in full if r['partverse_uid'] not in old_ids]
 if len(new)!=EXPECTED_WAVE2_COUNT:raise ValueError('WAVE2_COUNT_DRIFT')
 if any(r['provider_split']!='PROVIDER_TRAIN_CANDIDATE' for r in new):raise ValueError('VAL_LEAK')
 if any(r['head_training_admitted']!='False' or r['source_original_license_court']!='UNKNOWN' for r in new):
  raise ValueError('TRAINING_FALSE_AUTHORITY')
 if any(r['scene_source_original_sha_matches_current_roster']=='True' for r in new):
  raise ValueError('EXACT_SOURCE_SHA_OVERLAP_REVIEW_REQUIRED')
 shas=[r['source_sha256_provider_claim'] for r in new]
 if len(set(shas))!=len(shas):raise ValueError('DUPLICATE_ORIGINAL_PROVIDER_SHA')
 p=out/'PARTVERSE_WAVE2_4799_SOURCE_QUEUE_V1.csv'
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=full[0].keys());w.writeheader();w.writerows(new)
 summary={
  'schema':'RealSaS.PartVerseWave2SourceSelection.v1',
  'provider_total_metadata':report['record_count'],
  'old_source_uids_protected':len(old),'wave2_selected_uids':len(new),
  'combined_unique_provider_uids':len(ids),
  'wave2_provider_morphology_weak_counts':dict(Counter(r['object_type_candidate'] for r in new)),
  'wave2_style_weak_counts':dict(Counter(r['style_candidate'] for r in new)),
  'metadata_sha256':sha(metadata),'train_sha256':sha(train),'val_sha256':sha(val),
  'original_census_sha256':sha(source_roster),'first_wave_queue_sha256':sha(old_queue),
  'second_wave_queue_sha256':sha(p),
  'source_original_rights_qualified':0,
  'independent_face_semantics_qualified':0,
  'independent_2d_art_qualified':0,
  'drawable_necessity_qualified':0,
  'training_authorized':False,
  'source_raw_bytes_downloaded_by_this_selection_builder':0,
  'policy':'NON_BIPEDAL/CREATURE_FIRST_WEAK_VLM_TAGS_ONLY; SEE_PER_ASSET_APPROVAL_COURTS',
 }
 (out/'SUMMARY_WAVE2_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for arg in ('metadata','train','val','source-roster','old-queue','out'):
  p.add_argument('--'+arg,type=Path,required=True)
 a=p.parse_args()
 print(json.dumps(run(a.metadata,a.train,a.val,a.source_roster,a.old_queue,a.out),indent=2))