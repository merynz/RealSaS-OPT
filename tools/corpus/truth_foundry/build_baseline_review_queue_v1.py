#!/usr/bin/env python3
"""Existing source evidence -> 4,363 row preliminary detailed audit, NEVER a court PASS."""
import argparse,csv,collections,json,hashlib,zipfile
from pathlib import Path
from tools.corpus.truth_foundry.deep_source_audit_v1 import STAGES
parser=argparse.ArgumentParser()
parser.add_argument('--physical',required=True,type=Path)
parser.add_argument('--master-v2',required=True,type=Path)
parser.add_argument('--core',required=True,type=Path)
parser.add_argument('--out',required=True,type=Path)
a=parser.parse_args()
out=a.out
out.mkdir(parents=True,exist_ok=True)
with a.physical.open(newline='',encoding='utf-8') as f: phys=list(csv.DictReader(f))
with a.master_v2.open(newline='',encoding='utf-8') as f: v2={r['canonical_asset_id']:r for r in csv.DictReader(f)}
with a.core.open(newline='',encoding='utf-8') as f: core=list(csv.DictReader(f))
assert len(phys)==len(v2)==3993 and len(core)==370
rows=[];tasks=[];cov=collections.Counter();positive=collections.Counter();stage_count=collections.Counter()
count_fields=['source_scene_node_count','mesh_object_count','mesh_disconnected_components','mesh_boundary_edges','mesh_nonmanifold_edges','polygon_count','vertex_count','armature_count','observed_bone_count','bone_parented_mesh_candidates','skin_rows_sampled','skin_rows_bad_sum','skin_rows_zero_weights','material_count','image_count','image_dependency_missing_count','observed_action_count','observed_action_curve_count','observed_driver_channel_count','observed_ik_constraint_count','observed_nla_strip_count','observed_nondeform_bone_count']
cols=['corpus','asset_id','source_sha256','provider','source_format','historical_split','source_import_state','source_sha_verified_recovery','source_rights_state','source_appearance_state','source_import_sha256','geometry_mesh_observation','skin_observation','motion_observation','texture_observation','morphology','character_type','anatomical_pose','anatomical_front_up','drawable_layerability','2d_art_suitability','native_IK_functionality','independent_court_passed','training_authorized']+count_fields
for x in phys:
 aid=x['asset_id'];v=v2[aid];observed=x['census_state']=='IMPORTED_NUMERIC_OBSERVATION_ONLY';cov['MASTER3993']+=1;cov['MASTER_OBSERVED']+=observed
 r={'corpus':'MASTER3993','asset_id':aid,'source_sha256':x['source_sha256'],'provider':x['provider'],'source_format':x['source_format'],'historical_split':x['historical_split'],'source_import_state':x['census_state'],'source_sha_verified_recovery':v['source_recovery_sha_verified'],'source_rights_state':v['source_license_ml_rights'],'source_appearance_state':v['source_appearance_authority_truth'],'source_import_sha256':v['source_import_sha256'],'geometry_mesh_observation':v['source_mesh_observation'],'skin_observation':v['source_skin_observation'],'motion_observation':v['source_motion_observation'],'texture_observation':v['source_material_observation'],'morphology':'UNKNOWN','character_type':'UNKNOWN','anatomical_pose':'UNKNOWN','anatomical_front_up':'UNKNOWN','drawable_layerability':'UNKNOWN','2d_art_suitability':'UNKNOWN','native_IK_functionality':'UNKNOWN','independent_court_passed':'False','training_authorized':'False'}
 r.update({k:x.get(k,'') if observed else '' for k in count_fields});rows.append(r)
 for k in ['image_count','image_dependency_missing_count','observed_action_count','observed_ik_constraint_count','observed_driver_channel_count','skin_rows_bad_sum','skin_rows_zero_weights']:
  if observed and int(x[k])>0:positive[k]+=1
 for stage in STAGES:
  tasks.append({'corpus':'MASTER3993','asset_id':aid,'source_sha256':x['source_sha256'],'stage':stage,'observed_evidence':'IMPORTED_NUMERIC_ONLY' if observed else 'SOURCE_IMPORT_NOT_OBSERVED','independent_court':'UNKNOWN','review_receipt_sha256':'','training_authorized':False});stage_count[stage]+=1
for x in core:
 aid=x['subject_id'];cov['CORE370']+=1
 r={k:'' for k in cols}
 r.update({'corpus':'CORE370','asset_id':aid,'source_sha256':x['source_sha256'],'provider':'CORE370_'+x['source_origin'],'source_format':x['source_format'],'historical_split':'CORE370_SPLIT_TO_JOIN','source_import_state':'CORE370_HISTORICAL_OBSERVATION_NOT_THIS_MASTER_CENSUS','source_sha_verified_recovery':'HISTORICAL_MACHINE_SHA_PR327_REVALIDATION_PENDING','source_rights_state':'UNKNOWN','source_appearance_state':'UNKNOWN','source_import_sha256':'','geometry_mesh_observation':'CORE370_HISTORICAL_TRUTH_OUTSIDE_THIS_CENSUS','skin_observation':'CORE370_HISTORICAL_TRUTH_OUTSIDE_THIS_CENSUS','motion_observation':'CORE370_HISTORICAL_TRUTH_OUTSIDE_THIS_CENSUS','texture_observation':'UNKNOWN','morphology':'UNKNOWN','character_type':'UNKNOWN','anatomical_pose':'UNKNOWN','anatomical_front_up':'UNKNOWN','drawable_layerability':'UNKNOWN','2d_art_suitability':'UNKNOWN','native_IK_functionality':'UNKNOWN','independent_court_passed':'False','training_authorized':'False'})
 rows.append(r)
 for stage in STAGES:
  tasks.append({'corpus':'CORE370','asset_id':aid,'source_sha256':x['source_sha256'],'stage':stage,'observed_evidence':'CORE370_HISTORICAL_EVIDENCE_UNJOINED','independent_court':'UNKNOWN','review_receipt_sha256':'','training_authorized':False});stage_count[stage]+=1
assert len(rows)==4363 and len(tasks)==4363*len(STAGES)
assert len({(r['corpus'],r['asset_id']) for r in rows})==4363
for name,data in [('ALL_4363_DEEP_AUDIT_BASELINE_V1.csv',rows),('ALL_4363_INDEPENDENT_REVIEW_QUEUE_V1.csv',tasks)]:
 with (out/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={'schema':'RealSaS.TruthFoundryFullCorpusDeepAuditBaseline.v1','total_source_records':4363,'source_identity_cohorts':dict(cov),'stages_per_source':len(STAGES),'review_tasks':len(tasks),'observed_master_facets_positive_asset_counts':dict(positive),'source_class_reviewed':0,'morphology_reviewed':0,'pose_orientation_reviewed':0,'drawable_layerability_reviewed':0,'2d_style_qualified':0,'independent_head_target_admission':0,'core370_current_parity_joined':False,'master_full_scene_compressed_receipts_sha_validated_in_this_baseline':False,'source_sha_lineage_crosswalk_53_overlaid_in_this_baseline':False,'training_authorized':False,'materialization_authorized':False}
summary['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
zip_path=out.parent/'Truth_Foundry_4363_Deep_Audit_Baseline_20261108.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
print(json.dumps(summary,indent=2))
print('ZIP',zip_path,'bytes',zip_path.stat().st_size)