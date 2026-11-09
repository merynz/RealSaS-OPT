#!/usr/bin/env python3
"""Build explicit work queues from observed missing evidence, with no automatic upgrades."""
from __future__ import annotations
import csv,gzip,json,sqlite3,zipfile,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path('/mnt/data'); V8=ROOT/'RealSaS_Truth_Foundry_Deep_Inspection_V6_20261109';OUT=ROOT/'RealSaS_Truth_Foundry_Source_Courts_V9_20261109';OUT.mkdir(exist_ok=True)
DB=OUT/'MASTER_CORE_INDEPENDENT_RECEIPT_COURTS_V9.sqlite'
partial=json.load(open(ROOT/'41835338185_CORPUS370_PARTIAL_AUTHORITY_MATRIX_V1.json'))
with zipfile.ZipFile(ROOT/'github_artifact_13310446434.zip') as z:v5=json.loads(z.read('CORPUS370_PREPARED_OBSERVATION_PACKETS_V5.json'))
with zipfile.ZipFile(ROOT/'github_artifact_13315890950.zip') as z:mask=json.loads(z.read('CORPUS370_V5_18_MASK_DIAGNOSTIC.json'))
with zipfile.ZipFile(ROOT/'github_artifact_13318448943.zip') as z:geom=json.loads(z.read('COMPONENT_GEOMETRY_COURTS_INDEX.json'))
assert len(partial['subjects'])==370 and len(v5['subjects'])==352 and len(v5['failures'])==18
v5_pass={s['subject_id']:s for s in v5['subjects']};v5_fail={s['subject_id']:s for s in v5['failures']};masks={s['subject_id']:s for s in mask['per_subject']}
assert len(v5_pass)+len(v5_fail)==370 and len(masks)==18 and len(geom['subject_results'])==352

cols=['source_key','source_sha256','provider','source_format','v5_observation_status','v5_failure_reason','v5_failure_mask_gap_px','v5_failure_reference_only_pixels','v5_failure_raster_only_pixels',
 'partial_import_status','partial_camera_status','partial_rig_status','partial_skin_status','partial_sdf_status','partial_mech_topology_status','partial_tessa_teacher_status',
 'partial_geometry_udf_status','v5_geom_candidate','v5_sign_candidate','cross_stage_reconciliation','model_IRIS_partial_diagnostic','model_TESSA_partial_diagnostic',
 'model_AXIS_partial_diagnostic','model_MIRA_partial_diagnostic','next_required_independent_courts','source_parent_evidence','approval_scope']
counts=Counter()
with open(OUT/'CORE370_V5_VS_PARTIAL_FAMILY_RECONCILIATION_370.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for s in partial['subjects']:
  sid=s['subject_id'];key='CORE370:'+sid
  assert sid in v5_pass or sid in v5_fail
  if sid in v5_pass:
   obs='V5_SOURCE_OBSERVATION_PREPARED';reason='';gap='';refpx='';rasterpx=''
   p=v5_pass[sid]
   psha=p.get('source_sha256') or p.get('source_sha') or p.get('input',{}).get('source_sha256') or p.get('source',{}).get('sha256')
   # v5 packet source identity may be in nested fields. Cross-check only if recorded.
   if psha:assert psha==s['source_sha256']
  else:
   obs='V5_SOURCE_OBSERVATION_MASK_FAIL';p=v5_fail[sid];reason=p.get('reason','');m=masks[sid]
   gap=m['failure_gap_px'];refpx=m['reference_only_pixels'];rasterpx=m['raster_only_pixels']
   assert p['source_sha256']==s['source_sha256']
  g=geom['subject_results'].get(sid)
  if g:assert g['input']['source_sha256']==s['source_sha256']
  status=s['families']['SOURCE_IMPORT']['status']
  stage='SEPARATE_COURT_TYPES__DO_NOT_RECONCILE_TO_SINGLE_PASS'
  model={k:('DIAGNOSTIC_ELIGIBLE_NOT_TRAINING' if s['model_eligibility'][k]['corpus_eligible'] else 'BLOCKED') for k in ('IRIS','TESSA','AXIS','MIRA')}
  tasks=['ORIGINAL_SOURCE_SHA_BYTES_AND_LICENSE', 'GLOBAL_LINEAGE_SPLIT', 'ANATOMICAL_FRONT_BACK_2D_ART', 'ALL_MODEL_HEAD_TARGET_COURTS']
  if obs=='V5_SOURCE_OBSERVATION_MASK_FAIL':tasks.insert(0,'REVALIDATE_8VIEW_MASK_SOURCE_PROJECTION_DELTA_WITHOUT_THRESHOLD_RELAXATION')
  if s['families']['GLOBAL_SOLID_SDF']['status']!='PASS':tasks.append('GLOBAL_SOLID_SIGN_VALIDITY_COURT_OR_ABSTAIN')
  if s['families']['ANIMATION_KINEMATICS']['status']!='PASS':tasks.append('ANIMATION_KINEMATICS_INTERVENTION_COURT')
  if s['families']['VISIBILITY_OCCLUSION']['status']!='PASS':tasks.append('MULTIHIT_VISIBILITY_OCCLUSION_COURT')
  if s['families']['APPEARANCE_CAA']['status']!='PASS':tasks.append('SOURCE_BOUND_GRAPHICS_CAA_AND_ART_COURT')
  if s['families']['TESSA_TEACHER']['status']!='PASS':tasks.append('MECHANICALLY_ADMISSIBLE_TOPOLOGY_TEACHER_COURT')
  w.writerow({'source_key':key,'source_sha256':s['source_sha256'],'provider':s['source_id'],'source_format':s['source_format'],
    'v5_observation_status':obs,'v5_failure_reason':reason,'v5_failure_mask_gap_px':gap,
    'v5_failure_reference_only_pixels':refpx,'v5_failure_raster_only_pixels':rasterpx,
    'partial_import_status':status,'partial_camera_status':s['families']['CAMERA_OBSERVATION']['status'],
    'partial_rig_status':s['families']['RIG_HIERARCHY']['status'],
    'partial_skin_status':s['families']['SKIN_WEIGHTS']['status'],
    'partial_sdf_status':s['families']['GLOBAL_SOLID_SDF']['status'],
    'partial_mech_topology_status':s['families']['MECHANICAL_SURFACE_TOPOLOGY']['status'],
    'partial_tessa_teacher_status':s['families']['TESSA_TEACHER']['status'],
    'partial_geometry_udf_status':s['families']['SURFACE_GEOMETRY_UDF']['status'],
    'v5_geom_candidate':g['family_candidate_outcomes']['SURFACE_GEOMETRY'] if g else 'NOT_COURTED_OBSERVATION_FAIL',
    'v5_sign_candidate':g['family_candidate_outcomes']['SIGNED_DISTANCE_OCCUPANCY'] if g else 'NOT_COURTED_OBSERVATION_FAIL',
    'cross_stage_reconciliation':stage,'model_IRIS_partial_diagnostic':model['IRIS'],
    'model_TESSA_partial_diagnostic':model['TESSA'],'model_AXIS_partial_diagnostic':model['AXIS'],
    'model_MIRA_partial_diagnostic':model['MIRA'],
    'next_required_independent_courts':';'.join(tasks),'source_parent_evidence':'GitHub:41802949504;41822566858;Drive:41835338185',
    'approval_scope':'PREREADINESS_ONLY_NEVER_TRAINING_OR_SPLIT'})
  counts[(status,obs)]+=1

with gzip.open(V8/'SOURCE_INVENTORY_DEEP_16397_V8.csv.gz','rt',newline='') as f:
 master_missing=[]
 for row in csv.DictReader(f):
  if row['pool']!='MASTER3993':continue
  if row.get('master_deep_import_source_sha256'):continue
  master_missing.append(row)
assert len(master_missing)==1527,len(master_missing)
allfields=['source_key','candidate_id','provider','provider_locator','original_sha_claim','source_format','historical_split_hint',
 'source_import_observed','original_source_bytes_courted','rights_policy_triage','graphics_census_graphics_priority_bucket','graphics_census_source_recovery_next_action',
 'next_evidence_action','missing_court_reason','training_authorized']
with open(OUT/'MASTER1527_MISSING_FULL_IMPORT_RECEIPT_QUEUE.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=allfields);w.writeheader()
 for row in master_missing:
  w.writerow({**{k:row.get(k,'') for k in allfields},
   'next_evidence_action':'LOCATE_ORIGINAL_BYTES_AND_IMPORT_RECEIPT;HASH_BIND_TO_SOURCE_ID;REPLAY_IMPORTER_UNDER_FIXED_VERSION;COURT_SCENE_NODE_MESH_MATERIAL_RIG_ACTION',
   'missing_court_reason':'NO_FULL_SCENE_IMPORT_RECEIPT_IN_AVAILABLE_MASTER2466_DATASET',
   'training_authorized':'FALSE'})

summary={'schema':'RealSaS.Corpus370AndMasterGapAnalysis.v9','core370_subjects':370,
  'core370_v5_prepared':352,'core370_v5_mask_fail':18,'core_cross_stage_comparison':{str(k):v for k,v in counts.items()},
  'master3993_missing_detailed_scene_receipt':len(master_missing),
  'rights_and_split_admission_minted':0,
  'notes':['V5 observation is an 8-view mask/import packet gate, not identical to partial SOURCE_IMPORT family.','A partial family PASS is not full teacher/corpus training admission.','Master missing source is a missing detailed import receipt, not proof raw bytes do not exist.']}
(OUT/'V9_GAP_RECONCILIATION_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))