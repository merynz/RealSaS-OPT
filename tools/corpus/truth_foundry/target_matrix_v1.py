#!/usr/bin/env python3
"""RealSaS Truth Foundry: source-bound six-head target inventory, never admission.

Run without Blender/network; accepts frozen Master3993 V2, Core370 machine census,
first-1024 dense numeric court, and optional full physical summary census.
Makes 37 target slots per original source *record*. It does not merge the 53
verified cross-cohort duplicates without their individual identity receipt.

Teacher name hints, reviewed semantics, drawable claims, and graphics 2D art do
not arise from source import or asset name. All materialization/training gates
stay closed. Source tri-state UNKNOWN != false/zero, N_A != UNKNOWN.
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from collections import Counter,defaultdict
from pathlib import Path

COLUMNS=(
('IRIS_GEOMETRY','mesh_surface_xyz','SOURCE_MESH_VERTICES','SOURCE_GEOMETRY_PARITY_COURT'),
('IRIS_GEOMETRY','surface_topology_faces','SOURCE_TOPOLOGY_INCIDENCE','SOURCE_TOPOLOGY_PARITY_COURT'),
('IRIS_GEOMETRY','camera_depth','SOURCE_CAMERA_PROJECTION','SOURCE_BOUND_MULTIVIEW_CAMERA_COURT'),
('IRIS_GEOMETRY','camera_normals','SOURCE_NORMAL_ORIENTATION','SOURCE_NORMAL_PARITY_COURT'),
('IRIS_GEOMETRY','silhouette_coverage','MULTIVIEW_RENDER_AND_SILHOUETTE','SOURCE_BOUND_MULTIVIEW_CAMERA_COURT'),
('IRIS_GEOMETRY','udf_occupancy','GEOMETRY_OCCUPANCY_UDF_TSDF_SIGN','SOURCE_GEOMETRY_FIELD_COURT'),
('IRIS_GEOMETRY','solid_sdf_sign_conditional','GEOMETRY_OCCUPANCY_UDF_TSDF_SIGN','SOURCE_SOLID_SIGN_COURT'),
('IRIS_SEMANTIC','source_face_region_support','SEMANTIC_REGIONS_INSTANCES','SOURCE_FACE_SEMANTIC_COURT'),
('IRIS_SEMANTIC','instance_component_membership','SCENE_NODE_COMPONENT_TREE','SOURCE_SCENE_GRAPH_COURT'),
('IRIS_SEMANTIC','generic_morphological_roles','MORPHOLOGY','SOURCE_MORPHOLOGY_COURT'),
('IRIS_SEMANTIC','drawable_necessity_three_way','DRAWABLE_OWNERSHIP_EDIT_CONSEQUENCE','SOURCE_DRAWABLE_EDIT_CONSEQUENCE_COURT'),
('IRIS_SEMANTIC','layer_source_visibility_relation','VISIBILITY_OCCLUSION_DRAW_ORDER','SOURCE_VISIBILITY_COMPOSITING_COURT'),
('IRIS_SEMANTIC','semantic_multiview_consistency','MULTIVIEW_RENDER_AND_SILHOUETTE','SOURCE_BOUND_MULTIVIEW_CAMERA_COURT'),
('IRIS_GRAPHICS','source_native_rgba_evidence','SOURCE_NATIVE_RGBA','SOURCE_NATIVE_RGBA_COURT'),
('IRIS_GRAPHICS','approved_stylized_2d_rgba','SOURCE_BOUND_2D_ART','STYLIZED_2D_GAME_ART_IDENTITY_QUALITY_COURT'),
('IRIS_GRAPHICS','alpha_edge_and_bleed','SOURCE_BOUND_2D_ART','STYLIZED_2D_GAME_ART_IDENTITY_QUALITY_COURT'),
('IRIS_GRAPHICS','visibility_occlusion_order','VISIBILITY_OCCLUSION_DRAW_ORDER','SOURCE_VISIBILITY_COMPOSITING_COURT'),
('IRIS_GRAPHICS','ray_depth_compositing','RAY_DEPTH_COMPOSITING','SOURCE_RAY_COMPOSITING_COURT'),
('IRIS_GRAPHICS','edit_reveal_presentation','DRAWABLE_OWNERSHIP_EDIT_CONSEQUENCE','SOURCE_DRAWABLE_EDIT_CONSEQUENCE_COURT'),
('IRIS_GRAPHICS','face_layer_optional','SEMANTIC_REGIONS_INSTANCES','SOURCE_FACE_SEMANTIC_COURT'),
('TESSA','source_face_component_graph','SOURCE_TOPOLOGY_INCIDENCE','SOURCE_TOPOLOGY_PARITY_COURT'),
('TESSA','mesh_connectivity_and_charts','MESH_MANIFOLD_MODULAR_ROLE','SOURCE_MESH_TOPOLOGY_ROLE_COURT'),
('TESSA','motion_surface_correspondence','DEFORMATION_PARITY','SOURCE_DEFORMATION_RESPONSE_COURT'),
('TESSA','rig_aware_deformability','JOINT_REST_BIND_TRANSFORMS','SOURCE_REST_BIND_PARITY_COURT'),
('AXIS','skeleton_graph','JOINT_HIERARCHY','SOURCE_JOINT_HIERARCHY_COURT'),
('AXIS','rest_bind_reference','JOINT_REST_BIND_TRANSFORMS','SOURCE_REST_BIND_PARITY_COURT'),
('AXIS','motion_joint_trajectories','ACTION_CURVE_INVENTORY','SOURCE_MOTION_CURVES_COURT'),
('AXIS','typed_rigprogram_controllers','NATIVE_IK_FK_CONSTRAINT_GRAPH','SOURCE_NATIVE_RIG_CONTROL_COURT'),
('AXIS','end_effector_contact_limits','MOTION_SEMANTICS_CONTACT_LIMITS','SOURCE_MOTION_SEMANTICS_COURT'),
('AXIS','control_attachment_consequence','ATTACHMENT_DEPENDENT_CONTROL','SOURCE_ATTACHMENT_CONTROL_COURT'),
('AXIS','derived_ik_separate_channel','JOINT_HIERARCHY','SOURCE_DERIVED_IK_BEHAVIOR_COURT'),
('MIRA','sparse_vertex_joint_influences','SOURCE_SKIN_INFLUENCES','SOURCE_SKIN_WEIGHT_PARITY_COURT'),
('MIRA','bind_skin_joint_alignment','JOINT_REST_BIND_TRANSFORMS','SOURCE_REST_BIND_PARITY_COURT'),
('MIRA','skin_token_optional','SOURCE_SKIN_INFLUENCES','SOURCE_SKIN_WEIGHT_PARITY_COURT'),
('MIRA','deformed_vertex_response','DEFORMATION_PARITY','SOURCE_DEFORMATION_RESPONSE_COURT'),
('MIRA','unaffected_negative_regions','DEFORMATION_PARITY','SOURCE_DEFORMATION_RESPONSE_COURT'),
('MIRA','attachment_influence_relation','ATTACHMENT_DEPENDENT_CONTROL','SOURCE_ATTACHMENT_CONTROL_COURT'),
)
# PR392 proposal includes two semantic targets that 8 Nov source-architecture audit
# assigns to Compiler.Graphics *qualification*, not learned semantic classes.
QUALIFICATION_REASSIGN={
 'drawable_necessity_three_way':'COMPILER_GRAPHICS_QUALIFICATION',
 'layer_source_visibility_relation':'COMPILER_GRAPHICS_SOURCE_VISIBILITY_COURT',
 'edit_reveal_presentation':'COMPILER_GRAPHICS_EDIT_CONSEQUENCE_QUALIFICATION',
 'visibility_occlusion_order':'IRIS_GRAPHICS_EVIDENCE_AND_COMPILER_GRAPHICS_QUALIFICATION',
 'ray_depth_compositing':'IRIS_GRAPHICS_EVIDENCE_AND_COMPILER_GRAPHICS_QUALIFICATION',
}
# FACETS that have some data observed in current Master3993 V2; not truth.
SOURCE_HINTS={
 'SOURCE_MESH_VERTICES':'source_mesh_observation',
 'SOURCE_TOPOLOGY_INCIDENCE':'source_topology_observation',
 'SCENE_NODE_COMPONENT_TREE':'source_import_observed',
 'MESH_MANIFOLD_MODULAR_ROLE':'source_topology_observation',
 'JOINT_HIERARCHY':'source_joint_observation',
 'JOINT_REST_BIND_TRANSFORMS':'source_joint_observation',
 'SOURCE_SKIN_INFLUENCES':'source_skin_observation',
 'ACTION_CURVE_INVENTORY':'source_motion_observation',
}
# Core machine partial family mapping: every historical state remains
# non-current and non-training-authoritative.
CORE_FAMILY={
 'SOURCE_MESH_VERTICES':'SURFACE_GEOMETRY_UDF',
 'SOURCE_TOPOLOGY_INCIDENCE':'MECHANICAL_SURFACE_TOPOLOGY',
 'MESH_MANIFOLD_MODULAR_ROLE':'MECHANICAL_SURFACE_TOPOLOGY',
 'JOINT_HIERARCHY':'RIG_HIERARCHY',
 'JOINT_REST_BIND_TRANSFORMS':'RIG_HIERARCHY',
 'SOURCE_SKIN_INFLUENCES':'SKIN_WEIGHTS',
 'ACTION_CURVE_INVENTORY':'ANIMATION_KINEMATICS',
 'SOURCE_NORMAL_ORIENTATION':'SURFACE_NORMALS',
 'GEOMETRY_OCCUPANCY_UDF_TSDF_SIGN':'SURFACE_GEOMETRY_UDF',
 'SOURCE_NATIVE_RGBA':'APPEARANCE_CAA',
 'VISIBILITY_OCCLUSION_DRAW_ORDER':'VISIBILITY_OCCLUSION',
 'RAY_DEPTH_COMPOSITING':'RAY_COMPOSITING',
 'DEFORMATION_PARITY':'CAUSAL_RESPONSE',
}
PHYSICAL_FIELDS=('source_scene_node_count','mesh_object_count','mesh_disconnected_components',
 'mesh_boundary_edges','mesh_nonmanifold_edges','polygon_count','vertex_count',
 'armature_count','observed_bone_count','bone_parented_mesh_candidates',
 'skin_rows_sampled','skin_rows_bad_sum','skin_rows_zero_weights',
 'material_count','image_count','image_dependency_missing_count',
 'observed_action_count','observed_action_curve_count',
 'observed_driver_channel_count','observed_ik_constraint_count',
 'observed_nla_strip_count','observed_nondeform_bone_count')

def require(ok,reason):
 if not ok:raise ValueError('TRUTH_FOUNDRY_V1_'+str(reason))

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as fp:
  for block in iter(lambda:fp.read(1<<20),b''): h.update(block)
 return h.hexdigest()

def read(path):
 with open(path,newline='',encoding='utf-8-sig') as fp: return list(csv.DictReader(fp))

def writer(path,headers):
 fp=open(path,'w',newline='',encoding='utf-8')
 w=csv.DictWriter(fp,fieldnames=headers,extrasaction='ignore');w.writeheader()
 return fp,w

def signal_master(row,facet):
 if facet not in SOURCE_HINTS:return 'UNKNOWN_NO_QUALIFIED_SIGNAL'
 observed=str(row['source_import_observed'])=='True'
 if not observed:return 'SOURCE_IMPORT_NOT_OBSERVED'
 field=SOURCE_HINTS[facet]
 if field=='source_import_observed':return 'IMPORTED_SCENE_OBSERVATION_ONLY'
 value=row[field]
 if value.startswith('IMPORTED_'):return 'IMPORTED_OBSERVATION_ONLY'
 return 'NOT_OBSERVED_IN_IMPORT__SOURCE_UNKNOWN'

def priority(h,t):
 if h=='IRIS_SEMANTIC':return 'P0_SEMANTIC_REGION_AND_INDEPENDENT_REVIEW' if t not in QUALIFICATION_REASSIGN else 'P0_GRAPHICS_CAUSAL_QUALIFICATION'
 if h=='IRIS_GRAPHICS':return 'P0_SOURCE_APPEARANCE_2D_LAYER_ART_QUALIFICATION'
 if h=='TESSA':return 'P1_ARTICULATION_READY_TOPOLOGY'
 if h=='AXIS':return 'P1_RIGPROGRAM_AND_MOTION'
 if h=='MIRA':return 'P1_SKIN_AND_DEFORMATION'
 return 'P0_SOURCE_GEOMETRY_AND_CAMERA'

def run(master_path,core_path,dense_path,core_families_path,output,physical=None):
 require(len(COLUMNS)==37 and len({(x[0],x[1]) for x in COLUMNS})==37,'TARGET_CATALOG_UNIQUE')
 m=read(master_path); c=read(core_path);d=read(dense_path);cf=read(core_families_path)
 require(len(m)==3993 and len(c)==370 and len(d)==1024 and len(cf)==5180,'SOURCE_CENSUS')
 ids={x['canonical_asset_id'] for x in m}; cores={x['subject_id'] for x in c}
 require(len(ids)==3993 and len(cores)==370,'SOURCE_KEY_UNIQUE')
 require(all(x['training_authorized']=='False' for x in m+c),'TRAINING_ALREADY_GRANTED')
 dense={x['canonical_asset_id']:x for x in d}
 states={(x['subject_id'],x['family']):x['historical_machine_state'] for x in cf}
 require(len(states)==5180,'CORE_FAMILY_KEYS')
 phys={}
 if physical:
  p=read(physical)
  require(len(p)==3993 and len({x['asset_id'] for x in p})==3993,'PHYSICAL_UNIVERSE')
  phys={x['asset_id']:x for x in p}
 output.mkdir(parents=True,exist_ok=True)
 catalog=[{'proposed_head':h,'target':t,'source_facet':facet,'required_independent_court':court,
  'resolved_consumer_or_court':QUALIFICATION_REASSIGN.get(t,h),
  'training_head_target':False if t in QUALIFICATION_REASSIGN and h=='IRIS_SEMANTIC' else True,
  'priority':priority(h,t), 'ontology_label_inferred_from_object_name':False,
  'teacher_status':'PROPOSED_TARGET_ONLY', 'loss_mask_valid':False} for h,t,facet,court in COLUMNS]
 fp,w=writer(output/'TARGET_CATALOG_37.csv',list(catalog[0]));w.writerows(catalog);fp.close()

 assetheaders=['corpus','asset_id','candidate_id','provider','historical_split','source_sha256',
  'source_identity_source','import_status','import_wave','dense_numeric_status',
  'dense_vertices','dense_bone_rest','dense_influence_entries',
  'rights_state','character_vs_prop','morphology','anatomical_front_up','native_ik','native_skin_deform',
  'semantic_region_truth','stylized_2d_art','drawable_truth','physical_census_loaded',
  *PHYSICAL_FIELDS,'cleanup_flags','cross_corpus_cluster_overlay','training_authorized','materialization_authorized']
 targetheaders=['corpus','asset_id','candidate_id','provider','historical_split','source_sha256',
  'proposed_head','target','resolved_consumer_or_court','source_facet','required_independent_court',
  'observed_signal','historical_core_family_state','head_target_teacher_status',
  'review_priority','training_mask_valid','independent_court_passed','training_authorized',
  'source_record_not_unique_character','reason_codes']
 afp,aw=writer(output/'SOURCE_RECORD_CENSUS_4363.csv',assetheaders)
 tfp,tw=writer(output/'HEAD_TARGET_MATRIX_161431.csv',targetheaders)
 roll=Counter();obs=Counter();sums=defaultdict(Counter); phys_n=0; numeric_n=0
 for cohort,rows in (('MASTER3993',m),('CORE370',c)):
  for row in rows:
   master=cohort=='MASTER3993'; aid=row['canonical_asset_id'] if master else row['subject_id']
   dense_row=dense.get(aid) if master else None
   good_dense=bool(dense_row and dense_row['status']=='IMPORTED_FULL_VERTEX_BONE_WEIGHT_NUMERIC_CONSISTENCY_ONLY')
   physical_row=phys.get(aid) if master else None
   if physical_row and physical_row.get('census_state')=='IMPORTED_NUMERIC_OBSERVATION_ONLY': phys_n+=1
   if good_dense:numeric_n+=1
   source_hash=(row['source_import_sha256'] or row['expected_source_sha256']) if master else row['source_sha256']
   common={'corpus':cohort,'asset_id':aid,'candidate_id':row['candidate_id'] if master else row['source_id'],
    'provider':row['provider'] if master else row['source_origin'],
    'historical_split':row['historical_split'] if master else 'CORE370_SEPARATE_UNRECONCILED_SPLIT',
    'source_sha256':source_hash}
   arecord={**common,
    'source_identity_source':'MASTER_V2_IMPORTED_OR_EXPECTED' if master else 'CORE370_HISTORICAL_MACHINE_SHA',
    'import_status':row['source_import_observed'] if master else 'CORE370_CURRENT_V5_COHORT_PASS_352_FAIL_18__ROW_UNJOINED',
    'import_wave':row['source_import_wave'] if master else '',
    'dense_numeric_status':dense_row['status'] if dense_row else 'NOT_IN_MASTER1024_DENSE_COURT',
    'dense_vertices':int(dense_row['imported_vertices']) if good_dense else '',
    'dense_bone_rest':int(dense_row['imported_bone_rest_matrices']) if good_dense else '',
    'dense_influence_entries':int(dense_row['sparse_influence_entries']) if good_dense else '',
    'rights_state':row['source_license_ml_rights'] if master else 'UNKNOWN',
    'character_vs_prop':row['source_character_vs_prop'] if master else 'UNKNOWN',
    'morphology':row['morphology'] if master else 'UNKNOWN',
    'anatomical_front_up':row['anatomical_frame'] if master else 'UNKNOWN',
    'native_ik':row['native_IK_truth'] if master else 'UNKNOWN',
    'native_skin_deform':row['native_skin_deform_truth'] if master else 'UNKNOWN',
    'semantic_region_truth':row['source_semantic_regions_truth'] if master else 'UNKNOWN',
    'stylized_2d_art':row['source_2d_style_graphics_truth'] if master else 'UNKNOWN',
    'drawable_truth':row['source_drawable_ownership_truth'] if master else 'UNKNOWN',
    'physical_census_loaded':bool(physical_row and physical_row.get('census_state')=='IMPORTED_NUMERIC_OBSERVATION_ONLY'),
    'cleanup_flags':row['pending_source_cleanup'] if master else 'CORE370_V5_ROW_BINDING_REQUIRED',
    'cross_corpus_cluster_overlay':'PENDING_PR393_53_FULL_SHA_EDGES_PER_RECORD_BINDING',
    'training_authorized':False,'materialization_authorized':False}
   for f in PHYSICAL_FIELDS: arecord[f]=physical_row.get(f,'') if physical_row else ''
   aw.writerow(arecord)
   roll[(cohort,'source_records')]+=1
   for h,t,facet,court in COLUMNS:
    historic=''
    if master: signal=signal_master(row,facet)
    else:
     fam=CORE_FAMILY.get(facet)
     historic=states.get((aid,fam),'') if fam else ''
     signal='HISTORICAL_MACHINE_FAMILY_ONLY__NOT_CURRENT_COURT' if historic and historic not in ('UNKNOWN','MASKED') else 'UNKNOWN_CURRENT_CORE_V5_ROW_UNBOUND'
    reason='INDEPENDENT_TARGET_COURT_NOT_BOUND'
    if facet in ('SEMANTIC_REGIONS_INSTANCES','MORPHOLOGY','SCENE_NODE_COMPONENT_TREE'):
     reason+=';NO_SEMANTIC_LABEL_FROM_OBJECT_MATERIAL_OR_BONE_NAME'
    if facet=='SOURCE_BOUND_2D_ART':reason+=';SOURCE_NATIVE_RENDER_NOT_2D_STYLIZED_ART'
    if t in QUALIFICATION_REASSIGN:reason+=';COMPILER_GRAPHICS_QUALIFICATION_NOT_SEMANTIC_CLASS'
    if t in ('solid_sdf_sign_conditional',):reason+=';SIGN_REQUIRES_INDEPENDENT_SOLID_COURT_NONWATERTIGHT_UNKNOWN'
    if facet=='NATIVE_IK_FK_CONSTRAINT_GRAPH':reason+=';MISSING_IMPORTED_IK_NOT_NEGATIVE_LABEL'
    if not source_hash:reason+=';NO_SOURCE_SHA_AVAILABLE'
    tw.writerow({**common,'proposed_head':h,'target':t,'resolved_consumer_or_court':QUALIFICATION_REASSIGN.get(t,h),
     'source_facet':facet,'required_independent_court':court,'observed_signal':signal,
     'historical_core_family_state':historic,'head_target_teacher_status':'UNKNOWN_OR_OBSERVATION_ONLY__NO_TARGET_ADMISSION',
     'review_priority':priority(h,t),'training_mask_valid':False,'independent_court_passed':False,
     'training_authorized':False,'source_record_not_unique_character':True,'reason_codes':reason})
    obs[(cohort,h,t,signal)]+=1
    sums[(cohort,h)]['total_target_slots']+=1
    if signal in ('IMPORTED_OBSERVATION_ONLY','IMPORTED_SCENE_OBSERVATION_ONLY'):sums[(cohort,h)]['unqualified_observation_hints']+=1
  # one family of records per cohort, not asserting dedup
 afp.close();tfp.close()
 summary_rows=[]
 for (cohort,h),counter in sorted(sums.items()):
  summary_rows.append({'corpus':cohort,'head':h,'possible_target_slots':counter['total_target_slots'],
   'observation_only_hints':counter['unqualified_observation_hints'],
   'independent_target_court_PASS':0,'training_eligible_target_slots':0})
 fp,w=writer(output/'HEAD_COVERAGE_SUMMARY.csv',list(summary_rows[0]));w.writerows(summary_rows);fp.close()
 obsrows=[{'corpus':cohort,'head':h,'target':t,'observed_signal':sig,'record_count':count}
  for (cohort,h,t,sig),count in sorted(obs.items())]
 fp,w=writer(output/'TARGET_OBSERVATION_DISTRIBUTION.csv',list(obsrows[0]));w.writerows(obsrows);fp.close()
 report={
  'schema':'RealSaS.TruthFoundryDetailedSourceTargetMatrix.v1','status':'EXACT_SOURCE_RECORD_ROSTER__PARTIAL_PHYSICAL_CENSUS__NO_LEARNED_TRUTH_ADMISSION',
  'source_record_count':4363,'master3993_count':3993,'core370_count':370,'target_catalog_count':37,
  'head_target_route_count':sum(v['total_target_slots'] for v in sums.values()),
  'known_cross_corpus_sha_edges_PR393_not_matched_here':53,'provisional_unique_file_identities_after_only_known_53_edges':4310,
  'unique_character_count':'UNKNOWN','verified_independent_target_pass_count':0,
  'master_import_observed':sum(r['source_import_observed']=='True' for r in m),
  'master_dense_numeric_success':numeric_n,'master_per_asset_physical_census_loaded':phys_n,
  'master_rest_not_import_observed':sum(r['source_import_observed']=='False' for r in m),
  'core370_current_V5_cohort':{'source_observation_pass':352,'source_observation_fail':18,'per_asset_status_joined':False},
  'source_face_semantic_annotations_reviewed_in_this_builder':0,
  'source_layer_full_off_only_reveal_qualified_in_this_builder':0,
  'native_pose_orientation_verified_in_this_builder':0,
  'training_authorized':False,'materialization_authorized':False,
  'lineage_heldout_quarantine_released':False,
  'input_sha256':{'master':sha(master_path),'core370':sha(core_path),'dense':sha(dense_path),'core_families':sha(core_families_path),**({'physical':sha(physical)} if physical else {})},
  'output_sha256':{p.name:sha(p) for p in output.glob('*.csv')},
  'owner_policy':'PR392_37_TARGET_PROPOSAL_WITH_20261108_COMPILER_GRAPHICS_QUALIFICATION_ROUTING__NOT_TRAINER_CHANGE',
  'restrictions':['no source-name student leakage','no unknown-as-negative','no source import as independent semantics','no native render as approved 2d art','no mesh component as semantic instance','no closed mesh assumption for solid signed distance','no train split dedup release without verified per-edge crosswalk']
 }
 require(report['head_target_route_count']==4363*37,'ROUTE_COUNT')
 (output/'SUMMARY_V1.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 return report

def main():
 p=argparse.ArgumentParser()
 for name in ('master','core','dense','core-families','output','physical'):p.add_argument('--'+name,type=Path,required=name!='physical')
 a=p.parse_args()
 print(json.dumps(run(a.master,a.core,a.dense,a.core_families,a.output,a.physical),indent=2,ensure_ascii=False))
if __name__=='__main__':main()