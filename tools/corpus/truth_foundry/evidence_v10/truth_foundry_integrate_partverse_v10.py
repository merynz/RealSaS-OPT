#!/usr/bin/env python3
"""V9 -> V10 source-bound PartVerse physical graph overlay.

Not model-target admission. Does not touch the existing 264 facet cells: factual
processed part geometry becomes new independently named inventory columns.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from collections import Counter
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def rows(path):
 with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
  yield from csv.DictReader(f)

PFX='partverse_v10_'
KEEP=['processed_parts_available','processed_numeric_court','processed_face_count','processed_part_label_count',
 'processed_part_area_fraction_sum','processed_part_area_largest_fraction','processed_part_area_smallest_fraction',
 'processed_part_area_gini','processed_intralabel_piece_count','processed_labels_with_multiple_islands',
 'processed_interpart_edges','processed_external_adjacent_face_pairs','processed_graph_components',
 'processed_graph_isolated_labels','processed_graph_cycle_rank','processed_graph_max_degree',
 'processed_graph_max_component_size','processed_first_face_id','processed_last_face_id',
 'processed_face_partition_gap_count','processed_face_partition_overlap_count',
 'processed_part_neighbor_parity','processed_part_face_pair_parity','processed_region_exception']

def main(a):
 src=Path(a.v9_inventory);court=Path(a.partverse_court);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 c={r['source_key']:r for r in rows(court)}
 assert len(c)==7034
 output=out/'SOURCE_INVENTORY_DEEP_16397_V10.csv.gz'
 with gzip.open(src,'rt',encoding='utf8',newline='') as f:
  rdr=csv.DictReader(f)
  original_columns=list(rdr.fieldnames)
  for k in KEEP:
   assert k not in original_columns
  assert 'source_key' in original_columns and 'original_sha_claim' in original_columns
  fieldnames=original_columns+[PFX+k for k in KEEP]+[PFX+'provider_license_class_claim',PFX+'provider_rights_not_courted']
  source_count=0;updated=0;nonpart=0;seen=set();states=Counter()
  with gzip.open(output,'wt',encoding='utf8',newline='') as g:
   writer=csv.DictWriter(g,fieldnames=fieldnames);writer.writeheader()
   for r in rdr:
    key=r['source_key'];assert key not in seen;seen.add(key)
    if key in c:
     x=c[key];assert key.startswith('PARTVERSE_XL:')
     assert r['original_sha_claim']==x['provider_original_sha_claim'],('SOURCE_SHA_CLAIM_CONFLICT',key)
     for k in KEEP:r[PFX+k]=x.get(k,'')
     r[PFX+'provider_license_class_claim']=x['license_class_claim']
     r[PFX+'provider_rights_not_courted']='ORIGINAL_BYTE_RIGHTS_REVIEW_REQUIRED'
     updated+=1;states[x['processed_numeric_court']]+=1
    else:
     assert not key.startswith('PARTVERSE_XL:'),('MISSING_PARTVERSE_UID',key)
     for k in KEEP:r[PFX+k]=''
     r[PFX+'provider_license_class_claim']=''
     r[PFX+'provider_rights_not_courted']=''
     nonpart+=1
    writer.writerow(r);source_count+=1
 assert source_count==16397 and updated==7034 and nonpart==9363
 summary={'schema':'RealSaS.TruthFoundryInventoryPartVerseV10',
 'source_rows':source_count,'partverse_rows_joined_by_source_uid_and_original_sha_claim':updated,
 'other_source_rows_untouched':nonpart,'original_v9_columns_preserved':len(original_columns),
 'additional_processed_part_columns':len(fieldnames)-len(original_columns),
 'processed_numeric_status_counts':dict(states),
 'source_original_sha_rehash_pass':0,'global_sdf_pass':0,'semantic_ontology_pass':0,
 'head_training_admission':0,
 'inputs_sha256':{src.name:sha(src),court.name:sha(court)},
 'output_sha256':{output.name:sha(output)}}
 (out/'INVENTORY_PARTVERSE_V10_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['v9_inventory','partverse_court','out']:p.add_argument('--'+name.replace('_','-'),required=True)
 main(p.parse_args())