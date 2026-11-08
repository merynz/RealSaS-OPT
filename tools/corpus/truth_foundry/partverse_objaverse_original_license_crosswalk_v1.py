#!/usr/bin/env python3
"""PartVerse-XL to original Objaverse-XL Sketchfab SHA/license evidence join.

Only the official source metadata .parquet is downloaded (79.6MB), never 3D assets.
Even exact fileIdentifier + provider SHA + license is a *metadata assertion*, NOT
original source byte parity, creator permission, artwork quality, source UV truth,
or legal clearance for commercial ML training.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from pathlib import Path
from collections import Counter,defaultdict

SHA=re.compile(r'^[0-9a-f]{64}$')
UID=re.compile(r'^[0-9a-f]{32}$')
SCHEMA='RealSaS.PartVerseXLObjaverseXLOriginalMetadataMatch.v1'
HEADS=('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
FILE_SHA_EXPECTED='6462ef2ecf39069e210f6df1416e8a28ca931039e3c7fe6aa3dca593115c64e3'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def license_class(text):
 s=str(text or '').lower().strip()
 if not s or s in ('nan','none','null','unknown'):return 'UNSPECIFIED'
 if 'noncommercial' in s or 'non-commercial' in s or re.search(r'\bnc\b',s):return 'NONCOMMERCIAL_OR_RESTRICTED'
 if 'no derivatives' in s or 'no-derivatives' in s or re.search(r'\bnd\b',s):return 'NO_DERIVATIVES_OR_RESTRICTED'
 if 'cc0' in s or 'public domain' in s or 'creative commons zero' in s:return 'CC0_OR_PUBLIC_DOMAIN_CLAIM'
 if 'by-sa' in s or 'sharealike' in s:return 'ATTRIBUTION_SHAREALIKE_REVIEW'
 if 'cc-by' in s or 'creative commons attribution' in s:return 'ATTRIBUTION_REVIEW'
 return 'OTHER_UNREVIEWED_LICENSE'


def sketch_uid(identifier):
 s=str(identifier or '').strip()
 m=re.search(r'(?i)(?:https?://)?(?:www\.)?sketchfab\.com/3d-models/([a-f0-9]{32})(?:[/\?#]|$)',s)
 return m.group(1).lower() if m else ''


def partverse_inputs(paths):
 rows=[];seen=set()
 for path in paths:
  with Path(path).open(newline='',encoding='utf-8-sig') as f:
   for row in csv.DictReader(f):
    uid=row.get('uid','').lower();sh=row.get('source_sha_provider_claim','').lower()
    if not UID.fullmatch(uid) or not SHA.fullmatch(sh) or uid in seen:
     raise ValueError('INVALID_OR_DUPLICATE_SOURCE:'+uid)
    if row.get('provider_split')!='PROVIDER_TRAIN_CANDIDATE':raise ValueError('HELDOUT_OR_UNKNOWN_SPLIT')
    seen.add(uid);rows.append({'uid':uid,'provider_original_sha_claim':sh,
                               'provider_weak_morphology':row.get('weak_subject_candidate','UNKNOWN'),
                               'provider_weak_style':row.get('weak_style_candidate','UNKNOWN')})
 if len(rows)!=7034:raise ValueError('EXPECTED_7034_PARTVERSE_SOURCE_ROWS')
 return rows


def read_parquet_matches(path,needed_sha,needed_uid):
 """Stream small batches with schema projection. Never materialize all source rows."""
 import pyarrow.parquet as pq
 needed={'sha256','fileIdentifier','license','source','fileType'}
 p=pq.ParquetFile(path)
 if not needed.issubset(p.schema_arrow.names):raise ValueError('UNEXPECTED_OBJAVERSE_SOURCE_SCHEMA')
 found=[];scanned=0
 for batch in p.iter_batches(batch_size=10000,columns=sorted(needed)):
  columns=batch.to_pydict()
  for i in range(batch.num_rows):
   scanned+=1
   h=str(columns['sha256'][i] or '').lower()
   ident=str(columns['fileIdentifier'][i] or '')
   uid=sketch_uid(ident)
   if h not in needed_sha and uid not in needed_uid:continue
   found.append({'sha256':h,'uid':uid,'fileIdentifier':ident,
                 'license':str(columns['license'][i] or ''),
                 'source':str(columns['source'][i] or ''),
                 'fileType':str(columns['fileType'][i] or '')})
 return found,scanned


def correlate(part_records,metadata_records):
 by_sha=defaultdict(list);by_uid=defaultdict(list)
 for m in metadata_records:
  if m['source'].lower()!='sketchfab':continue
  by_sha[m['sha256']].append(m)
  if m['uid']:by_uid[m['uid']].append(m)
 for p in part_records:
  uid=p['uid'];sha_claim=p['provider_original_sha_claim']
  by_exact=[m for m in by_sha.get(sha_claim,[]) if m['uid']==uid]
  by_bhash=by_sha.get(sha_claim,[]);by_buid=by_uid.get(uid,[])
  if len(by_exact)==1:
   m=by_exact[0];match='EXACT_SHA_AND_SOURCE_UID_METADATA_CLAIM'
  elif len(by_exact)>1:
   m=by_exact[0];match='DUPLICATE_OR_AMBIGUOUS_METADATA_EXACT_MATCH'
  elif by_bhash and by_buid:
   m=by_bhash[0];match='CONFLICT_SHA_AND_UID_MAP_TO_DIFFERENT_SOURCE_RECORDS'
  elif by_bhash:
   m=by_bhash[0];match='SHA_MATCH_UID_MISMATCH_OR_UNAVAILABLE'
  elif by_buid:
   m=by_buid[0];match='UID_MATCH_SHA_DIFFERENT'
  else:
   m=None;match='ORIGINAL_PROVIDER_SOURCE_METADATA_UNFOUND'
  declared_license=m['license'] if m else ''
  lic=license_class(declared_license)
  yield {**p,'source_metadata_match_state':match,
         'objaverse_license_claim':declared_license,
         'objaverse_license_class':lic,
         'objaverse_fileIdentifier':m['fileIdentifier'] if m else '',
         'objaverse_fileType':m['fileType'] if m else '',
         'objaverse_sha256_recorded':m['sha256'] if m else '',
         'exact_metadata_match_count':len(by_exact),
         'original_source_bytes_rehashed':False,
         'source_license_commercial_ml_rights_court':'NOT_RUN__INDIVIDUAL_ASSET_REVIEW_REQUIRED',
         'source_identity_lineage_court':'NOT_RUN',
         'source_appearance_teacher_court':'NOT_RUN',
         'source_semantic_review_court':'NOT_RUN',
         'head_training_authorized':False}


def run(parquet:Path,source_csvs,out:Path,expected_file_sha=FILE_SHA_EXPECTED):
 parquet,out=Path(parquet),Path(out)
 if sha(parquet)!=expected_file_sha:raise ValueError('PINNED_PROVIDER_PARQUET_SHA_MISMATCH')
 source_rows=partverse_inputs(source_csvs)
 meta,scanned=read_parquet_matches(parquet,{x['provider_original_sha_claim'] for x in source_rows},
                                  {x['uid'] for x in source_rows})
 results=list(correlate(source_rows,meta))
 if len(results)!=7034:raise ValueError('OUTPUT_COUNT')
 out.mkdir(parents=True,exist_ok=True)
 p1=out/'PARTVERSE_7034_ORIGINAL_PROVIDER_SHA_LICENSE_CROSSWALK_V1.csv'
 with p1.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 p2=out/'PARTVERSE_7034_SIX_HEAD_SOURCE_RIGHTS_EVIDENCE_V1.csv'
 with p2.open('w',encoding='utf-8',newline='') as f:
  fields=['source_key','head','source_sha_claim','original_metadata_match','license_class',
          'source_byte_verified','rights_court','train_allowed']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in results:
   for head in HEADS:
    w.writerow({'source_key':'PARTVERSE_XL:'+r['uid'],'head':head,
                'source_sha_claim':r['provider_original_sha_claim'],
                'original_metadata_match':r['source_metadata_match_state'],
                'license_class':r['objaverse_license_class'],
                'source_byte_verified':False,'rights_court':'NOT_RUN','train_allowed':False})
 status=Counter(x['source_metadata_match_state'] for x in results)
 license_status=Counter(x['objaverse_license_class'] for x in results)
 summary={'schema':SCHEMA,'source_rows':len(results),'head_route_rows':len(results)*len(HEADS),
          'official_sketchfab_parquet_scanned_records':scanned,
          'official_parquet_sha256':sha(parquet),
          'source_queue_shas':{Path(p).name:sha(Path(p)) for p in source_csvs},
          'match_state_counts':dict(status),'source_license_claim_class_counts':dict(license_status),
          'actual_original_source_bytes_rehashed':0,
          'independently_licensed_for_commercial_ml':0,
          'source_face_geometry_parity_approved':0,'source_semantic_approved':0,
          'source_graphics_2d_art_approved':0,'source_six_head_truth_approved':{h:0 for h in HEADS},
          'outputs_sha256':{p.name:sha(p) for p in [p1,p2]},'training_authorized':False}
 (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 return summary


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--parquet',type=Path,required=True)
 p.add_argument('--wave1',type=Path,required=True)
 p.add_argument('--wave2',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.parquet,[a.wave1,a.wave2],a.out),indent=2))

if __name__=='__main__':main()