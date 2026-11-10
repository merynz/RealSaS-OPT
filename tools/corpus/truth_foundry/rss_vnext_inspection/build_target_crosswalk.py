#!/usr/bin/env python3
"""RSS.vNext 264-facet -> provisional target crosswalk, no inferred training PASS.

The 14 historical Corpus370 authority families are frozen as existing source
policy; this registry is a versioned non-normative design review for vNext.
"""
from __future__ import annotations
import csv,gzip,collections,json,pathlib,argparse

LEGACY={
 'SOURCE_IMPORT','CAMERA_OBSERVATION','SURFACE_GEOMETRY_UDF','GLOBAL_SOLID_SDF',
 'SURFACE_NORMALS','APPEARANCE_CAA','VISIBILITY_OCCLUSION','RAY_COMPOSITING',
 'MECHANICAL_SURFACE_TOPOLOGY','CAUSAL_RESPONSE','RIG_HIERARCHY','SKIN_WEIGHTS',
 'ANIMATION_KINEMATICS','TESSA_TEACHER'}
# Registry row: (identifier, consumer, route, legacy family, required source-facet signals,
# target kind). REQUIRED facets define **evidence routing**, never sufficient proof.
TARGETS=[
 ('surface_udf','IRIS.Geometry','neural','SURFACE_GEOMETRY_UDF','udf_occupancy_and_surface_sampling|source_vertex_face_count|source_surface_coverage_fraction','continuous_field'),
 ('validity_masked_global_sdf','IRIS.Geometry','neural','GLOBAL_SOLID_SDF','source_sdf_sign_court_or_mask|watertightness_and_connected_components','continuous_masked_signed_field'),
 ('qualified_global_occupancy','IRIS.Geometry','neural','GLOBAL_SOLID_SDF','source_sdf_sign_court_or_mask|interior_exterior_shells_nested_assembly','masked_binary_query'),
 ('oriented_normals','IRIS.Geometry','neural','SURFACE_NORMALS','normal_orientation_consistency|source_normal_world_camera_xyz','continuous_surface_normal'),
 ('surface_support_confidence','IRIS.Geometry','neural','SURFACE_GEOMETRY_UDF','source_surface_coverage_fraction|source_face_id_pixel_pass','masked_confidence'),
 ('view_source_geometry_consistency','IRIS.Geometry','neural','CAMERA_OBSERVATION','camera_extrinsics_and_matrix|projection_face_pixel_parity|source_vertex_face_count','cross_view_geometry'),
 ('semantic_region_identity','IRIS.Semantic','neural','NEW_VNEXT','semantic_source_face_polyg_corner_lineage|source_face_concept_review|semantic_ontology_review_receipt','surface_region_categorical'),
 ('semantic_instance_cardinality','IRIS.Semantic','neural','NEW_VNEXT','source_instance_id_pass|same_concept_multi_instance_id','instance_id_set'),
 ('semantic_affordance_roles','IRIS.Semantic','neural','NEW_VNEXT','morphological_role_evidence|semantic_morphological_role_instances','typed_role_affordance'),
 ('cross_view_semantic_consistency','IRIS.Semantic','neural','NEW_VNEXT','semantic_cross_view_hidden_region|visible_mask_semantic_view_consistency','cross_view_region_correspondence'),
 ('morphology_multilabel','IRIS.Semantic','neural','NEW_VNEXT','species_taxonomy_multilabel|radial_serpentine_arachnid_forms','variable_morphology_labels'),
 ('semantic_part_material_disentangle','IRIS.Semantic','neural','NEW_VNEXT','semantic_part_material_conflict_resolution|source_material_boundary_not_semantic','typed_disagreement_mask'),
 ('source_rgba_appearance','IRIS.Graphics','neural','APPEARANCE_CAA','source_full_rgba_render|source_native_rgba_pixel_parity','rgba_with_authority_mask'),
 ('alpha_boundary_support','IRIS.Graphics','neural','APPEARANCE_CAA','source_silhouette_and_alpha_pass|alpha_edges_bleed_halos_caa','alpha_boundary'),
 ('surface_pixel_address','IRIS.Graphics','neural','APPEARANCE_CAA','source_face_id_pixel_pass|source_uv_barycentric_pixel_pass','surface_to_image_address'),
 ('visibility_first_hit','IRIS.Graphics','neural','VISIBILITY_OCCLUSION','first_hit_vs_second_hidden_surface|self_occlusion_and_occluder_identity','ray_hit_visibility'),
 ('depth_ray_stack','IRIS.Graphics','neural','RAY_COMPOSITING','all_hit_rays_depth_stack|source_depth_linear_camera_z','ray_depth_stack'),
 ('source_color_appearance_consistency','IRIS.Graphics','neural','APPEARANCE_CAA','source_texture_vs_atlas_identity|material_color_to_style_identity_parity','source_color_consistency'),
 ('projected_silhouette_boundary','IRIS.Graphics','neural','CAMERA_OBSERVATION','source_silhouette_and_alpha_pass|projection_face_pixel_parity','silhouette_contour'),
 ('hidden_reveal_uncertainty','IRIS.Graphics','neural','VISIBILITY_OCCLUSION','newly_revealed_hidden_source_faces|hidden_unseen_area_confidence','masked_reveal_region'),
 ('triangle_connectivity','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','source_vertex_face_count|degenerate_sliver_zero_area_faces|nonmanifold_and_self_intersection','discrete_mesh_topology'),
 ('component_part_graph','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','mesh_object_vs_connected_island|part_to_component_many_to_many','typed_component_graph'),
 ('chart_and_seam_structure','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','uv_chart_seam_and_island|layer_chart_overlap_seam_stability','chart_seam_graph'),
 ('source_surface_correspondence','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','triangulation_source_face_barycentric_map|semantic_source_face_polyg_corner_lineage','surface_address_mapping'),
 ('structural_attachment_sites','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','attachment_direction_control_deps|rigid_bone_parented_components','structural_attachment_candidates'),
 ('deformation_sensitive_topology','TESSA','neural','TESSA_TEACHER','joint_neighborhood_edge_flow|deform_twist_bend_collapse|tessa_topology_deform_compatibility','mechanical_deformation'),
 ('multipose_mesh_integrity','TESSA','neural','TESSA_TEACHER','pose_induced_mesh_intersections|tessa_animation_ready_face_chart','posed_quality_regression'),
 ('nonweld_independence_relations','TESSA','neural','MECHANICAL_SURFACE_TOPOLOGY','modular_vs_welded_vs_multimesh|rigid_plate_flexible_region_boundary','structural_relation_graph'),
 ('joint_positions','AXIS','neural','RIG_HIERARCHY','joint_count_vs_anatomical_joint|joint_rest_and_bind_world_local_matrices','joint_coordinates'),
 ('joint_parent_structure','AXIS','neural','RIG_HIERARCHY','joint_parent_hierarchy_and_cycles|hierarchy_kinematic_reachability','joint_parent_graph'),
 ('joint_rest_frames','AXIS','neural','RIG_HIERARCHY','joint_rest_and_bind_world_local_matrices|bone_orientation_roll_and_axis','rest_frames'),
 ('joint_dof_and_limits','AXIS','neural','NEW_VNEXT','joint_limits_rotation_translation|axis_control_target_chain_constraint','dof_limits'),
 ('kinematic_chain_targets','AXIS','neural','ANIMATION_KINEMATICS','ik_pole_target_end_effector|hierarchy_kinematic_reachability','chain_effector'),
 ('rigprogram_control_graph','AXIS','neural','NEW_VNEXT','axis_rigprogram_grammar_validity|controller_input_output_program','discrete_program_events'),
 ('socket_anchor_attachments','AXIS','neural','NEW_VNEXT','attachment_socket_frame_parent|grip_weapon_hand_constraint','socket_frame_graph'),
 ('contact_support_policy','AXIS','neural','ANIMATION_KINEMATICS','motion_contact_phase_foot_ground|functional_rig_source_intervention','contact_target'),
 ('secondary_control_policy','AXIS','neural','NEW_VNEXT','secondary_motion_spring_eligible|secondary_tail_wing_hair_dynamics','secondary_control'),
 ('dense_skin_weight_distribution','MIRA','neural','SKIN_WEIGHTS','skin_sparse_influence_index_weight|skin_weight_normalization','vertex_joint_weight_matrix'),
 ('sparse_joint_influence_set','MIRA','neural','SKIN_WEIGHTS','skin_sparse_influence_index_weight|bone_joint_cap_and_influence_distribution','sparse_joint_weight_set'),
 ('skintokens_codec_reconstruction','MIRA','neural','NEW_VNEXT','mira_vertex_skin_codec_fidelity|skin_bind_pose_rest_parity','learned_skin_code'),
 ('vertex_to_control_influence','MIRA','neural','NEW_VNEXT','deform_vs_control_bones|deform_intervention_positive_region','control_influence_field'),
 ('exact_mesh_skin_lineage','MIRA','neural','SKIN_WEIGHTS','mira_skin_retarget_mesh_correspondence|skin_bind_pose_rest_parity','surface_skin_binding'),
 ('posed_deformation_consistency','MIRA','neural','ANIMATION_KINEMATICS','joint_pose_trajectory_validity|skin_deformation_negative_court','posed_vertex_trajectory'),
 ('drawable_split_necessity','Compiler.Graphics','qualification_only','NEW_VNEXT','drawable_split_necessity_three_way|edit_consequence_renderer_repro','edit_consequence_oracle'),
 ('dynamic_occlusion_draw_order','Compiler.Graphics','qualification_only','VISIBILITY_OCCLUSION','draw_order_at_view_change|occlusion_edit_consequence_under_pose','depth_compositor_oracle'),
 ('source_preserving_edit_roundtrip','Compiler.Graphics','qualification_only','APPEARANCE_CAA','edit_replay_texture_unchanged|puppet_edit_save_reopen_identity','product_runtime_oracle'),
 ('full_body_ik_behavioral_proof','Compiler.Mechanics','qualification_only','NEW_VNEXT','axis_ik_sword_tip_behavioral_test|compiler_45_degree_character_rotation','interaction_oracle')
]
FROZEN=['IRIS.Geometry','IRIS.Semantic','IRIS.Graphics','TESSA','AXIS','MIRA','Compiler.Graphics','Compiler.Mechanics']
STATES_SCOPED={'IMPORTED_NUMERIC_OBSERVATION_ONLY','PROCESSED_MESH_OBSERVATION__NOT_ORIGINAL_PARITY',
 'OBSERVED_PROCESSED_PART_GEOMETRY_ONLY','OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY',
 'OBSERVED_CANONICAL_GLTF_DERIVATIVE_PARTIAL__NO_SOURCE_ADMISSION',
 'OBSERVED_CANONICAL_GLTF_SYNTHETIC_POSE_DIAGNOSTIC__NO_AUTHORED_TEACHER_OR_ADMISSION',
 'OBSERVED_V5_COMPONENT_COURT_ONLY','OBSERVED_HISTORICAL_SOURCE_IMPORT_COUNTS_ONLY',
 'OBSERVED_HISTORICAL_STRUCTURE_JSON_DERIVATIVE_ONLY',
 'OBSERVED_HISTORICAL_NORMALIZED_DERIVATIVE_ONLY','OBSERVED_IMPORT_STRUCTURE_ONLY__NOT_SEMANTIC_MODULE',
 'OBSERVED_MESH_OBJECT_SCENE_NODE_INDEXES__NOT_FACE_LEVEL_ADDRESS',
 'OBSERVED_AGGREGATE_ARMATURE_MODIFIERS__NOT_EXACT_JOINT_BINDING',
 'OBSERVED_PROCESSED_PART_GRAPH_STRUCTURE_ONLY__NO_ANATOMICAL_SOURCE_OR_DRAWABLE_AUTHORITY',
 'CANONICAL_GLB_STRUCTURE__NOT_ORIGINAL_PARITY','CANONICAL_GLB_SKIN_JOINT_COUNT__HIERARCHY_NOT_COURTED',
 'OBSERVED_CANONICAL_GLB_DERIVATIVE_ONLY'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=pathlib.Path,required=True);p.add_argument('--universe',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 a.out.mkdir(exist_ok=True,parents=True)
 with gzip.open(a.matrix,'rt') as f:fieldnames=csv.DictReader(f).fieldnames
 facets=set(fieldnames)-{'source_key'}
 regs=[]
 for name,owner,route,legacy,fstr,typ in TARGETS:
  req=fstr.split('|')
  if set(req)-facets:raise ValueError(name+': missing '+str(set(req)-facets))
  if legacy!='NEW_VNEXT' and legacy not in LEGACY:raise ValueError('UNKNOWN_LEGACY_'+legacy)
  if owner not in FROZEN:raise ValueError('UNKNOWN_OWNER_'+owner)
  regs.append({'target_id':name,'owner':owner,'route':route,'legacy_family':legacy,
      'source_facet_evidence_requirements':req,'target_semantics':typ,
      'admission_requires':'INDEPENDENT_SOURCE_ORIGINAL_AND_FAMILY_COURT_WITH_VALIDITY_MASKS_AND_LINEAGE_SEAL',
      'source_facet_states_are_not_training_admission':True})
 keys=set();provider={}
 with gzip.open(a.universe,'rt') as f:
  for r in csv.DictReader(f):
   key=r['source_key'];assert key not in keys;keys.add(key);provider[key]=r['provider']
 assert len(keys)==16397
 counters={t['target_id']:collections.Counter() for t in regs}
 by_facet={k:collections.Counter() for k in facets};by_provider=collections.defaultdict(lambda:collections.Counter())
 hard_gates=['character_vs_vehicle_prop_environment','source_rights_independent_pass','original_source_full_byte_sha','lineage_split_safe_holdout','anatomical_front_axis','front_back_side_eight_direction_views','two_d_art_independent_visual_review','original_vs_transcoded_parity']
 gates={k:collections.Counter() for k in hard_gates}
 seen=set()
 outpath=a.out/'TARGET_EVIDENCE_SCREEN_16397x47_V1.csv.gz'
 with gzip.open(a.matrix,'rt') as inp,gzip.open(outpath,'wt',newline='') as outf:
  rdr=csv.DictReader(inp);w=csv.writer(outf);w.writerow(['source_key','provider']+[x['target_id'] for x in regs])
  for source in rdr:
   sid=source['source_key'];assert sid in provider and sid not in seen;seen.add(sid)
   p=provider[sid];by_provider[p]['candidate_sources']+=1
   for f in facets:
    st=source[f];by_facet[f][st]+=1
    if f in gates:gates[f][st]+=1
   scores=[]
   for t in regs:
    states=[source[k] for k in t['source_facet_evidence_requirements']]
    if all(x=='UNKNOWN_UNREVIEWED' for x in states):status='NO_RECORDED_EVIDENCE'
    elif all(x in STATES_SCOPED for x in states):status='ALL_REQUIRED_SCOPED_DERIVATIVE_EVIDENCE_NOT_ADMITTED'
    elif any(x in STATES_SCOPED for x in states):status='PARTIAL_SCOPED_DERIVATIVE_EVIDENCE_NOT_ADMITTED'
    else:status='ONLY_CLAIMS_HISTORICAL_OR_METADATA'
    scores.append(status);counters[t['target_id']][status]+=1
   w.writerow([sid,p,*scores])
 assert len(seen)==16397
 (a.out/'TARGET_CATALOG_PROVISIONAL_V1.json').write_text(json.dumps({'schema':'RealSaS.RSSvNext.ProposedSupervisionTargetCrosswalk.v1',
  'normative_architecture':'canonical/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_V1_20261109.json',
  'current_legacy_corpus370_family_count':14,'provisional_target_count':len(regs),
  'target_count_is_not_frozen':True,'training_targets':sum(t['route']=='neural' for t in regs),
  'compiler_qualification_targets':sum(t['route']!='neural' for t in regs),
  'targets':regs},indent=2,ensure_ascii=False)+'\n')
 with (a.out/'FACET_COVERAGE_264_V1.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['facet','candidate_count','unknown_unreviewed','scoped_derived_observation_rows','claim_historical_receipt_or_metadata_rows','state_distribution_json'])
  for key in sorted(facets):
   c=by_facet[key];w.writerow([key,16397,c['UNKNOWN_UNREVIEWED'],sum(v for k,v in c.items() if k in STATES_SCOPED),sum(v for k,v in c.items() if k not in STATES_SCOPED and k!='UNKNOWN_UNREVIEWED'),json.dumps(c,sort_keys=True)])
 with (a.out/'SUPERVISION_TARGET_EVIDENCE_COVERAGE_V1.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['target_id','owner','route','legacy_family','required_facets','no_recorded_evidence','only_claims_or_historical','partial_scoped','all_required_scoped_not_admitted','final_admitted_count'])
  for t in regs:
   c=counters[t['target_id']];w.writerow([t['target_id'],t['owner'],t['route'],t['legacy_family'],'|'.join(t['source_facet_evidence_requirements']),c['NO_RECORDED_EVIDENCE'],c['ONLY_CLAIMS_HISTORICAL_OR_METADATA'],c['PARTIAL_SCOPED_DERIVATIVE_EVIDENCE_NOT_ADMITTED'],c['ALL_REQUIRED_SCOPED_DERIVATIVE_EVIDENCE_NOT_ADMITTED'],'NOT_DETERMINABLE_FROM_FACETS'])
 receipt={'schema':'RealSaS.RSSvNext.16397TargetEvidenceScreen.v1','candidates':len(seen),'facet_count':len(facets),
  'legacy_source_truth_families':len(LEGACY),'provisional_target_count':len(regs),
  'training_targets':sum(t['route']=='neural' for t in regs),
  'compiler_qualification_targets':sum(t['route']=='qualification_only' for t in regs),
  'source_status_aggregation_policy':'Scoped derivative observation is NOT original source, owner, semantic review, rights, two-view or training admission',
  'qualified_subject_counts':'NOT_DETERMINABLE_FROM_264_FACET_STATES',
  'hard_gates':{k:dict(v) for k,v in gates.items()},
  'provider_rows':{k:dict(v) for k,v in by_provider.items()},
  'training_authorized':False}
 (a.out/'TARGET_COVERAGE_SUMMARY_V1.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:receipt[k] for k in ['candidates','facet_count','legacy_source_truth_families','provisional_target_count','training_targets','compiler_qualification_targets']},sort_keys=True))

if __name__=='__main__':main()