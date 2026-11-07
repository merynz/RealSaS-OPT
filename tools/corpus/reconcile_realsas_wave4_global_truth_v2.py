#!/usr/bin/env python3
"""Read-only 32-shard Wave4 observations -> Master3993 V2 family/head matrix.

Trusted inputs: SHA-bound frozen SourceAvailability summary, the V1 inventory
(which already includes 1,024 imports), 32 disjoint Wave4 source plans,
batch statuses, imported results and per-asset numeric source-scene summaries.

This does not download or import 3D source bytes, create model labels, alter
historical splits, add native IK, infer morphology, or admit training data.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

SCHEMA='RealSaS.GlobalExistingTruthEvidenceInventoryWave4Join.v2'
WAVEPLAN='RealSaS.Corpus4008Wave4DisjointOriginalSourcePlan.v1'
WAVEBATCH='RealSaS.Corpus4008Wave1ImportBatchSummary.v1'
WAVESUMMARY='RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1'
WAVESTATUS='IMPORTED_SOURCE_OBSERVATIONS_ONLY'
HEADS=('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
FAMILIES=('SOURCE_IMPORT','CAMERA_OBSERVATION','SURFACE_GEOMETRY','SURFACE_TOPOLOGY',
 'SIGNED_DISTANCE_OCCUPANCY','ORIENTED_NORMALS','APPEARANCE_CAA',
 'VISIBILITY_OCCLUSION','RAY_COMPOSITING','CAUSAL_RESPONSE',
 'RIG_HIERARCHY','SKIN_WEIGHTS','ANIMATION_KINEMATICS')
OBSERVED={
 'SURFACE_GEOMETRY':'IMPORTED_MESH_NUMERIC_ONLY',
 'SURFACE_TOPOLOGY':'IMPORTED_INDEX_INCIDENCE_ONLY',
 'RIG_HIERARCHY':'IMPORTED_JOINT_FOREST_NOT_NATIVE_CONTROL_TRUTH',
 'SKIN_WEIGHTS':'IMPORTED_SPARSE_WEIGHT_CENSUS_NOT_BIND_PARITY',
 'ANIMATION_KINEMATICS':'IMPORTED_ACTION_CURVE_INVENTORY_NOT_MOTION_SEMANTICS',
}

def require(ok,code):
 if not ok:raise ValueError('C4008_WAVE4_TRUTH_'+code)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):
 require(path.is_file(),'REQUIRED_FILE_MISSING:'+str(path))
 return json.loads(path.read_text())
def csvread(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def dump(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 return sha(path)
def integer(x,name):
 require(type(x) is int and x>=0,'NONNEGATIVE_'+name)
 return x


def run(v1_dir:Path,source_matrix:Path,source_summary:Path,wave4_dir:Path,out:Path,
        *,expect_master=3993,expect_baseline=1024,expect_shards=32):
 a=read(source_summary)
 require(a['schema']=='RealSaS.Corpus4008SourceAvailabilityMatrix.v1'
         and a['historical_master_source_selected']==expect_master
         and a['training_authorized'] is False
         and a['reported_recovered_sha_receipt_count']==3192,
         'SOURCE_AVAILABILITY_SUMMARY_OR_RECOVERY')
 av=csvread(source_matrix)
 require(len(av)==expect_master,'SOURCE_AVAILABILITY_MATRIX_COUNT')
 avmap={r['canonical_asset_id']:r for r in av}
 require(len(avmap)==len(av),'SOURCE_AVAILABILITY_DUPLICATE')
 v1=read(v1_dir/'GLOBAL_INVENTORY_SUMMARY_V1.json')
 require(v1['schema']=='RealSaS.GlobalExistingTruthEvidenceInventory.v1'
         and v1['master3993_canonical_subjects']==expect_master
         and v1['master3993_imported_scene_observed']==expect_baseline
         and v1['global_training_authorized'] is False
         and v1['input_sha256']['availability']==sha(source_matrix)
         and v1['input_sha256']['availability_summary']==sha(source_summary),
         'V1_PROVENANCE')
 for fn,detail in v1['output_files'].items():
  require(sha(v1_dir/fn)==detail['sha256'],'V1_OUTPUT_SHA:'+fn)
 originals=csvread(v1_dir/'MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv')
 cells=csvread(v1_dir/'MASTER3993_51909_TRUTH_FAMILY_CELLS_V1.csv')
 heads=csvread(v1_dir/'MASTER3993_23958_SIX_HEAD_ROUTES_V1.csv')
 origmap={x['canonical_asset_id']:x for x in originals}
 require(len(origmap)==expect_master and len(cells)==expect_master*13
         and len(heads)==expect_master*6,'V1_ROW_COUNTS')
 initial={x['canonical_asset_id']:x for x in originals if x['source_import_observed']=='True'}
 require(len(initial)==expect_baseline,'INITIAL_IMPORT_COUNTER')
 status_counter=Counter(); wave4ids=set();wave4shas=set()
 wave4_rows={};shard_summaries=[]
 for shard in range(expect_shards):
  base=wave4_dir/f'shard-{shard}'
  p=read(base/'PLAN.json');b=read(base/'BATCH_STATUS.json')
  require(p['schema']==WAVEPLAN and p['shard']==shard
          and p['previous_wave_shard_reserved_count']==32
          and p['training_authorized'] is False
          and p['materialization_authorized'] is False
          and len(p['entries'])==p['new_selected_count']
          and b['schema']==WAVEBATCH and b['shard']==shard
          and b['selected_source_count']==len(p['entries'])
          and b['truth_admitted']==0
          and b['training_authorized'] is False,
          'PLAN_OR_BATCH:'+str(shard))
  with (base/'RESULTS.jsonl').open() as f:
   result=[json.loads(line) for line in f if line.strip()]
  expected={r['canonical_asset_id']:r for r in p['entries']}
  observed={r['asset_id']:r for r in result}
  require(len(expected)==len(result)==len(observed)
          and set(expected)==set(observed),
          'BATCH_SOURCE_RESULT_UNIQUENESS:'+str(shard))
  require(dict(sorted(Counter(x['status'] for x in result).items()))==b['status_counts'],
          'BATCH_STATUS_COUNTS:'+str(shard))
  prior={*p['previous_wave_shard_8_sha256'],*p['previous_wave_shard_24_sha256']}
  require(len(prior)==32,'PREVIOUS32_SOURCE_SHA')
  for aid,e in expected.items():
   h=e['source_sha256'];r=observed[aid]
   require(aid in avmap and aid in origmap and aid not in wave4ids
           and h not in wave4shas and aid not in initial,
           'UNEXPECTED_DUPLICATE_OR_SOURCE_OVERLAP')
   wave4ids.add(aid);wave4shas.add(h)
   m=avmap[aid]
   require(m['source_availability_status']=='SHA_VERIFIED_RECOVERY_RECEIPT'
           and m['historical_split']=='FIT'
           and e['split']=='FIT' and e['source_shard']==shard
           and m['source_expected_sha256']==h
           and m['sha_verified_in_immutable_recovery_receipt']==h
           and m['candidate_id']==e['candidate_id']
           and r['source_sha256']==h
           and r['training_authorized'] is False
           and r['truth_admitted'] is False
           and e['training_authorized'] is False
           and re.fullmatch('[a-f0-9]{64}',h),
           'RECOVERY_SPLIT_OR_SOURCE_LINEAGE')
   require(h not in prior,'WAVE4_REIMPORTED_RESERVED_SHA')
   status=r['status'];status_counter[status]+=1
   census=None;summary_sha=''
   if status==WAVESTATUS:
    sp=base/'receipts'/(aid+'.summary.json')
    content=read(sp);summary_sha=sha(sp)
    require(content['schema']==WAVESUMMARY
            and content['status']=='FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS'
            and content['asset_id']==aid
            and content['source_sha256']==h
            and content['candidate_id']==e['candidate_id']
            and content['source_format']==e['source_ext']
            and content['training_authorized'] is False
            and content['commercial_ml_rights_courted'] is False
            and content['source_character_vs_prop']=='UNKNOWN'
            and content['functional_native_ik_fk']=='UNKNOWN'
            and re.fullmatch('[a-f0-9]{64}',content['receipt_gz_sha256']),
            'FULL_SOURCE_RECEIPT_SUMMARY')
    census=content['numeric_census']
    for k in ('vertex_count','mesh_object_count','mesh_boundary_edges',
              'mesh_nonmanifold_edges','skin_rows_sampled','skin_rows_bad_sum',
              'skin_rows_zero_weights','observed_bone_count',
              'observed_action_curve_count','observed_action_count',
              'material_count','image_count','image_dependency_missing_count',
              'observed_ik_constraint_count'):
     integer(census[k],k)
   else:
    require(status in ('SOURCE_FETCH_ERROR','SOURCE_SHA_OR_SIZE_MISMATCH',
                      'SOURCE_IMPORT_EXCEPTION'), 'UNEXPECTED_SOURCE_EXCEPTION')
   wave4_rows[aid]={'status':status,'summary':census,'summary_sha256':summary_sha,'source_sha256':h,
                    'shard':shard,'source_format':e['source_ext']}
  shard_summaries.append({'shard':shard,'plan_sha256':sha(base/'PLAN.json'),
                          'batch_status_sha256':sha(base/'BATCH_STATUS.json'),
                          'results_sha256':sha(base/'RESULTS.jsonl'),
                          'selected_count':len(expected)})
 require(len(wave4ids)==sum(x['selected_count'] for x in shard_summaries),
         'WAVE4_TOTAL_DUPLICATE')
 require(not wave4ids.intersection(initial),'WAVE4_BASELINE_OVERLAP')
 require(not wave4shas.intersection(x['source_import_sha256'] for x in initial.values()),
         'WAVE4_GLOBAL_SHA_COLLISION')
 updated=[];new_success=0
 for row in originals:
  a_id=row['canonical_asset_id'];w=wave4_rows.get(a_id)
  item=dict(row)
  item.update({
   'wave4_import_status':'NOT_SELECTED' if not w else w['status'],
   'wave4_receipt_summary_sha256':w['summary_sha256'] if w else '',
   'wave4_import_shard':w['shard'] if w else '',
   'source_import_wave':'3' if row['source_import_observed']=='True' else '',
   'existing_source_import_receipt_count':1 if row['source_import_observed']=='True' else 0,
  })
  if w and w['summary']:
   c=w['summary'];new_success+=1
   item['source_import_observed']=True
   item['source_import_wave']='4'
   item['source_import_sha256']=w['source_sha256']
   item['source_import_format']=w['source_format']
   item['source_mesh_observation']='IMPORTED_GEOMETRY_OBSERVATION' if c['mesh_object_count'] and c['vertex_count'] else 'NO_MESH_OBSERVED_SOURCE_UNKNOWN'
   item['source_topology_observation']='IMPORTED_INDEX_INCIDENCE_OBSERVATION_NOT_SOLID_TRUTH' if c['mesh_object_count'] else 'NO_TOPOLOGY_OBSERVED_SOURCE_UNKNOWN'
   item['source_joint_observation']='IMPORTED_JOINT_FOREST' if c['observed_bone_count'] else 'NO_JOINTS_OBSERVED_SOURCE_UNKNOWN'
   item['source_skin_observation']='IMPORTED_SPARSE_WEIGHT_HINT' if c['skin_rows_sampled'] else 'NO_WEIGHTS_OBSERVED_SOURCE_UNKNOWN'
   item['source_motion_observation']='IMPORTED_ACTION_CURVE_INVENTORY' if c['observed_action_curve_count'] else 'NO_ACTION_CURVES_OBSERVED_SOURCE_UNKNOWN'
   item['source_material_observation']='IMPORTED_MATERIAL_SLOT' if c['material_count'] else 'NO_MATERIAL_OBSERVED_SOURCE_UNKNOWN'
   item['native_IK_truth']='NOT_OBSERVED_BY_IMPORT_SOURCE_UNKNOWN'
   flags=item['pending_source_cleanup'].split(';')
   flags=[f for f in flags if f!='SOURCE_SCENE_NOT_CLASSIFIED']
   if c['skin_rows_bad_sum']:flags.append('REVIEW_IMPORTED_WEIGHT_SUM')
   if c['skin_rows_zero_weights']:flags.append('REVIEW_UNWEIGHTED_VERTICES')
   if c['mesh_boundary_edges'] or c['mesh_nonmanifold_edges']:
    flags.append('REVIEW_MODULAR_OR_OPEN_MESH_NOT_AUTO_REJECT')
   if c['image_dependency_missing_count']:flags.append('RELINK_EXTERNAL_TEXTURES')
   if not c['observed_ik_constraint_count']:
    flags.append('IK_NOT_OBSERVED_SOURCE_UNKNOWN_NOT_NEGATIVE_LABEL')
   item['pending_source_cleanup']=';'.join(sorted(set(flags)))
   item['existing_source_import_receipt_count']=1
  updated.append(item)
 source_states={r['canonical_asset_id']:r for r in updated}
 family_out=[]
 for row in cells:
  updated_row=dict(row)
  r=wave4_rows.get(row['asset_id'])
  if r and r['summary']:
   c=r['summary']; f=row['family'];state=row['evidence_state']
   if f=='SURFACE_GEOMETRY' and c['vertex_count'] and c['mesh_object_count']:
    state=OBSERVED[f]
   elif f=='SURFACE_TOPOLOGY' and c['vertex_count'] and c['mesh_object_count']:
    state=OBSERVED[f]
   elif f=='RIG_HIERARCHY' and c['observed_bone_count']:
    state=OBSERVED[f]
   elif f=='SKIN_WEIGHTS' and c['skin_rows_sampled']:
    state=OBSERVED[f]
   elif f=='ANIMATION_KINEMATICS' and c['observed_action_curve_count']:
    state=OBSERVED[f]
   updated_row['evidence_state']=state
   updated_row['producer_source']='WAVE4_PRIVATE_SOURCE_NUMERIC_OBSERVATION_ONLY'
  require(updated_row['authority']=='PENDING_INDEPENDENT_FAMILY_COURT'
          and updated_row['training_authorized']=='False', 'NO_TRUTH_PROMOTION')
  family_out.append(updated_row)
 grouped={}
 for row in family_out:grouped.setdefault(row['asset_id'],{})[row['family']]=row['evidence_state']
 head_out=[]
 for row in heads:
  item=dict(row)
  fns=grouped[row['asset_id']]
  needed=row['required_family_courts'].split(';')
  item['observation_hint_families']=';'.join(sorted(f for f in needed
     if fns.get(f,'').startswith('IMPORTED_')))
  require(item['head_truth_state']=='PENDING_INDEPENDENT_FAMILY_COURT'
          and item['head_training_admitted']=='False', 'NO_HEAD_PASS')
  head_out.append(item)
 require(len(updated)==expect_master and len(family_out)==expect_master*13
         and len(head_out)==expect_master*6, 'GLOBAL_OUTPUT_COUNTS')
 require(sum(str(x['source_import_observed']).lower()=='true' for x in updated)==expect_baseline+new_success,
         'UPDATED_IMPORT_COUNT')
 dest=out
 dest.mkdir(parents=True,exist_ok=True)
 files={
  'MASTER3993_SOURCE_TRUTH_INVENTORY_V2.csv':dump(dest/'MASTER3993_SOURCE_TRUTH_INVENTORY_V2.csv',updated),
  'MASTER3993_51909_TRUTH_FAMILY_CELLS_V2.csv':dump(dest/'MASTER3993_51909_TRUTH_FAMILY_CELLS_V2.csv',family_out),
  'MASTER3993_23958_SIX_HEAD_ROUTES_V2.csv':dump(dest/'MASTER3993_23958_SIX_HEAD_ROUTES_V2.csv',head_out),
 }
 report={
  'schema':SCHEMA,'source_count':expect_master,'baseline_import_success':expect_baseline,
  'wave4_selected_source_count':len(wave4ids),'wave4_import_success':new_success,
  'wave4_status_counts':dict(sorted(status_counter.items())),
  'total_existing_import_observations':expect_baseline+new_success,
  'remaining_without_import_observation':expect_master-expect_baseline-new_success,
  'source_status_counts':a['source_stage_counts'],
  'source_2d_game_art_truth_courted':False,
  'source_native_ik_truth_courted':False,
  'source_license_rights_courted':False,
  'source_character_morphology_courted':False,
  'model_training_admitted':{head:0 for head in HEADS},
  'family_truth_courts_closed_by_this_ledger':0,
  'source_original_source_sha_rehashed_in_this_join':False,
  'source_scene_full_gz_rehashed_in_this_join':False,
  'new_solver_or_renderer_experiment':False,
  'master3993_family_cell_count':len(family_out),
  'six_head_route_count':len(head_out),
  'v1_summary_sha256':sha(v1_dir/'GLOBAL_INVENTORY_SUMMARY_V1.json'),
  'source_availability_matrix_sha256':sha(source_matrix),
  'source_availability_summary_sha256':sha(source_summary),
  'wave4_original_batch_receipts':shard_summaries,
  'output_file_sha256':files,
  'materialization_authorized':False,'training_authorized':False,
 }
 (dest/'MASTER3993_WAVE4_TRUTH_RECONCILIATION_V2.json').write_text(
    json.dumps(report,indent=2,sort_keys=True)+'\n')
 return report


def main():
 p=argparse.ArgumentParser()
 for x in ('v1','source-availability','source-summary','wave4','output'):
  p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args()
 r=run(a.v1,a.source_availability,a.source_summary,a.wave4,a.output)
 print(json.dumps({k:v for k,v in r.items() if k in (
 'source_count','wave4_selected_source_count','wave4_import_success',
 'wave4_status_counts','total_existing_import_observations',
 'remaining_without_import_observation','training_authorized')},sort_keys=True))

if __name__=='__main__':main()
