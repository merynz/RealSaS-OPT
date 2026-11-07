"""Synthetic cold-join smoke; all asset/source references originate from existing frozen ledgers."""
from pathlib import Path
import sys,json,csv,tempfile,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconcile_realsas_wave4_global_truth_v2 import run
ROOT=Path(os.environ.get('REALSAS_AUDIT_INPUT_ROOT','/mnt/data'))
AV=list(csv.DictReader((ROOT/'SOURCE_AVAILABILITY_MATRIX_V1.csv').open()))
imported={r['canonical_asset_id'] for r in csv.DictReader((ROOT/'SOURCE_EXISTING_EVIDENCE_CLASSIFICATION_V1.csv').open())}
unused=[x for x in AV if x['source_availability_status']=='SHA_VERIFIED_RECOVERY_RECEIPT' and x['historical_split']=='FIT' and x['canonical_asset_id'] not in imported]
assert len(unused)>=32
with tempfile.TemporaryDirectory() as t:
 base=Path(t);wave=base/'wave4';out=base/'out'
 for shard in range(32):
  source=unused[shard];aid=source['canonical_asset_id'];h=source['source_expected_sha256']
  d=wave/f'shard-{shard}';(d/'receipts').mkdir(parents=True)
  p={
   'schema':'RealSaS.Corpus4008Wave4DisjointOriginalSourcePlan.v1',
   'shard':shard,'previous_wave_shard_reserved_count':32,
   'previous_wave_shard_8_sha256':[f'{10000+i+shard*32:064x}' for i in range(8)],
   'previous_wave_shard_24_sha256':[f'{20000+i+shard*32:064x}' for i in range(24)],
   'training_authorized':False,'materialization_authorized':False,
   'new_selected_count':1,
   'entries':[{'canonical_asset_id':aid,'source_sha256':h,'source_ext':'.fbx',
               'source_shard':shard,'candidate_id':source['candidate_id'],
               'training_authorized':False,'split':'FIT'}],
  }
  (d/'PLAN.json').write_text(json.dumps(p))
  b={'schema':'RealSaS.Corpus4008Wave1ImportBatchSummary.v1','shard':shard,
     'selected_source_count':1,'status_counts':{'IMPORTED_SOURCE_OBSERVATIONS_ONLY':1},
     'truth_admitted':0,'training_authorized':False}
  (d/'BATCH_STATUS.json').write_text(json.dumps(b))
  (d/'RESULTS.jsonl').write_text(json.dumps({'asset_id':aid,'source_sha256':h,
    'status':'IMPORTED_SOURCE_OBSERVATIONS_ONLY','truth_admitted':False,
    'training_authorized':False})+'\n')
  c={k:0 for k in ('vertex_count','mesh_object_count','mesh_boundary_edges',
    'mesh_nonmanifold_edges','skin_rows_sampled','skin_rows_bad_sum',
    'skin_rows_zero_weights','observed_bone_count','observed_action_curve_count',
    'observed_action_count','material_count','image_count',
    'image_dependency_missing_count','observed_ik_constraint_count')}
  c.update(vertex_count=50,mesh_object_count=1,observed_bone_count=4,
           skin_rows_sampled=50,observed_action_curve_count=8)
  s={'schema':'RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1',
     'status':'FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS',
     'asset_id':aid,'source_sha256':h,'candidate_id':source['candidate_id'],
     'source_format':'.fbx','training_authorized':False,
     'commercial_ml_rights_courted':False,'source_character_vs_prop':'UNKNOWN',
     'functional_native_ik_fk':'UNKNOWN','numeric_census':c,'receipt_gz_sha256':'f'*64}
  (d/'receipts'/f'{aid}.summary.json').write_text(json.dumps(s))
 r=run(ROOT/'realsas_global_truth_inventory_v1',ROOT/'SOURCE_AVAILABILITY_MATRIX_V1.csv',
       ROOT/'SOURCE_AVAILABILITY_SUMMARY_V1.json',wave,out)
 assert r['wave4_selected_source_count']==32
 assert r['wave4_import_success']==32
 assert r['total_existing_import_observations']==1056
 assert r['family_truth_courts_closed_by_this_ledger']==0
 assert all(x==0 for x in r['model_training_admitted'].values())
 # Case 1: original source SHA swap must fail after all baseline hashes are correct.
 d=wave/'shard-8'/'RESULTS.jsonl'
 x=json.loads(d.read_text());x['source_sha256']='1'*64;d.write_text(json.dumps(x)+'\n')
 try:run(ROOT/'realsas_global_truth_inventory_v1',ROOT/'SOURCE_AVAILABILITY_MATRIX_V1.csv',
       ROOT/'SOURCE_AVAILABILITY_SUMMARY_V1.json',wave,out)
 except ValueError as e:assert 'RECOVERY_SPLIT_OR_SOURCE_LINEAGE' in str(e)
 else:raise AssertionError('forged source SHA mistakenly accepted')
 print('PASS: all 32 source shards closed; 3993 rows/51909 cells/23958 routes; false SHA fails')
