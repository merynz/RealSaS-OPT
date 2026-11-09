#!/usr/bin/env python3
"""Independent readback of the hosted PartVerse source-part court and V10 inventory.

Counts physical processed receipts and validates source identity and immutable V9
columns. This is deliberately separate from the source graph builder.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,collections
from pathlib import Path

def iterz(p):
 with gzip.open(p,'rt',encoding='utf8',newline='') as f:
  yield from csv.DictReader(f)

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def run(a):
 d=Path(a.directory)
 summary=json.loads((d/'SUMMARY_PARTVERSE_V10.json').read_text())
 integrated=json.loads((d/'INVENTORY_PARTVERSE_V10_SUMMARY.json').read_text())
 court=d/'PARTVERSE_SOURCE_BOUND_PART_GRAPH_V10_7034.csv.gz'
 parts=d/'PARTVERSE_SOURCE_BOUND_PART_DETAIL_V10_69742.csv.gz'
 matrix=d/'SOURCE_INVENTORY_DEEP_16397_V10.csv.gz'
 old=Path(a.v9_inventory)
 assert summary['output_sha256'][court.name]==digest(court)
 assert summary['output_sha256'][parts.name]==digest(parts)
 assert integrated['output_sha256'][matrix.name]==digest(matrix)
 assert integrated['inputs_sha256'][old.name]==digest(old)
 assert integrated['inputs_sha256'][court.name]==digest(court)
 assert summary['source_rows']==integrated['partverse_rows_joined_by_source_uid_and_original_sha_claim']==7034
 assert summary['processed_part_rows']==69742 and summary['processed_adjacency_rows']==9881
 assert summary['processed_source_rows']==7028 and summary['processed_triangles']==128076897
 assert summary['training_admission']==integrated['head_training_admission']==0
 assert summary['global_sdf_sign_pass']==integrated['global_sdf_pass']==0
 assert summary['semantic_ontology_pass']==integrated['semantic_ontology_pass']==0
 assert summary['original_creator_bytes_rehashed']==integrated['source_original_sha_rehash_pass']==0

 bysource=collections.defaultdict(lambda:{'parts':0,'triangles':0,'min':10**100,'max':-1,'fraction':0.,'pieces':0,'fragmented':0})
 n=0
 for r in iterz(parts):
  assert r['training_authorized']=='FALSE' and r['part_role']=='UNKNOWN'
  assert r['original_source_part_parity']=='NOT_COURTED'
  src=r['source_key'];assert src.startswith('PARTVERSE_XL:')
  p=bysource[src];t=int(r['triangles']);lo=int(r['face_id_start']);hi=int(r['face_id_end'])
  assert hi-lo+1==t and t>0
  p['parts']+=1;p['triangles']+=t;p['min']=min(p['min'],lo);p['max']=max(p['max'],hi)
  p['fraction']+=float(r['area_fraction']);p['pieces']+=int(r['connected_pieces'])
  p['fragmented']+=int(int(r['connected_pieces'])>1)
  n+=1
 assert n==69742 and len(bysource)==7028
 courtrows={}
 num=collections.Counter()
 for r in iterz(court):
  key=r['source_key'];assert key not in courtrows;courtrows[key]=r
  assert r['training_authorized']=='FALSE' and r['global_sdf_sign_truth']=='UNKNOWN'
  assert r['original_creator_bytes_rehashed']=='FALSE' and r['original_mesh_vs_processed_parity']=='UNKNOWN'
  assert r['anatomical_semantic_truth']==r['drawable_layer_truth']=='UNKNOWN'
  assert r['provider_original_sha_claim'] and len(r['provider_original_sha_claim'])==64
  if key in bysource:
   p=bysource[key]
   assert r['processed_numeric_court']=='PASS_PROCESSED_PART_GEOMETRY_NUMERIC_ONLY'
   assert int(r['processed_part_label_count'])==p['parts']
   assert int(r['processed_face_count'])==p['triangles']
   assert int(r['processed_first_face_id'])==p['min']==0
   assert int(r['processed_last_face_id'])==p['max']==p['triangles']-1
   assert int(r['processed_intralabel_piece_count'])==p['pieces']
   assert int(r['processed_labels_with_multiple_islands'])==p['fragmented']
   assert abs(float(r['processed_part_area_fraction_sum'])-p['fraction'])<1e-7
   assert r['processed_part_neighbor_parity']=='PASS' and r['processed_part_face_pair_parity']=='PASS'
   assert int(r['processed_face_partition_gap_count'])==0
   assert int(r['processed_face_partition_overlap_count'])==0
   num['processed_pass']+=1
  else:
   assert r['processed_numeric_court']=='NO_PROCESSED_PART_SOURCE_RECEIPT'
   num['processed_missing']+=1
 assert len(courtrows)==7034 and num['processed_pass']==7028 and num['processed_missing']==6

 # Compare all original V9 row values exactly and confirm only the 7034 PartVerse
 # records gain typed additional source-bound fields. Never permit silent edits.
 base=iterz(old);new=iterz(matrix);rows=0;matched=0
 for arow,brow in zip(base,new):
  assert arow['source_key']==brow['source_key']
  assert all(brow[k]==v for k,v in arow.items()),('BASELINE_COLUMN_OVERWRITE',arow['source_key'])
  key=arow['source_key']
  if key in courtrows:
   assert arow['original_sha_claim']==courtrows[key]['provider_original_sha_claim']
   assert brow['partverse_v10_processed_numeric_court']==courtrows[key]['processed_numeric_court']
   assert brow['partverse_v10_provider_rights_not_courted']=='ORIGINAL_BYTE_RIGHTS_REVIEW_REQUIRED'
   matched+=1
  else:
   assert not any(v for k,v in brow.items() if k.startswith('partverse_v10_')),('CROSS_COHORT_LEAK',key)
  rows+=1
 assert next(base,None) is None and next(new,None) is None
 assert rows==16397 and matched==7034
 receipt={'schema':'RealSaS.PartVerseV10IndependentReadback',
  'status':'PASS_PROCESSED_SOURCE_LEDGER_AND_UNMODIFIED_V9_BASELINE_ONLY',
  'source_rows':rows,'processed_source_rows':num['processed_pass'],'unprocessed_source_rows':num['processed_missing'],
  'parts_crosschecked_independently':n,'partverse_joined_inventory_rows':matched,
  'nonpartverse_inventory_rows_unchanged':rows-matched,
  'input_sha256':{old.name:digest(old)},
  'court_sha256':{court.name:digest(court),parts.name:digest(parts),matrix.name:digest(matrix)},
  'original_source_creator_sha_rehashed':0,'model_training_admission':0,
  'generic_semantic_truth_admitted':0,'two_view_2d_art_approved':0,'global_sdf_sign_admitted':0}
 (d/'INDEPENDENT_READBACK_PARTVERSE_V10.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,indent=2,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--v9-inventory',required=True)
 run(p.parse_args())