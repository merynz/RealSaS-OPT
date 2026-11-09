#!/usr/bin/env python3
"""V17 inventory-bound partial observations into V16's 16397x264 facet matrix.

No downloads. No creator-original parity, rights, training, 2D art, semantic
ownership, drawable, motion-ready TESSA or global SDF truth is promoted. Every
cell carries its actual scope, source fields and evidence receipt SHA. Source
candidate metadata is NEVER silently equated to physical original authority.
"""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,io,json,os,pathlib,re,tempfile

BASE_SHA={
 'inventory':'6436a6696a55c7643818914b6ebab5a5728d4eae23409d67e6e6d210c2b17ae4',
 'states':'eb9376bc9cc317deb7aee902294d107c1e6638a73432b1a6ecb678ca508061a7',
 'values':'13bf437bfb67de62ffb2f580ca2972b63a1444ddaae795034924821af7f3411c',
}
# Source fields are recorded literally. A partial observation only means the
# narrower listed measurement is known; it does NOT settle its facet contract.
# Tuple: facet, canonical source field(s), narrower observed fact.
PARTVERSE=[
 ('source_vertex_face_count','partverse_v10_processed_face_count','processed_partverse_faces_only__vertex_count_unknown'),
 ('source_surface_coverage_fraction','partverse_v10_processed_part_area_fraction_sum','fraction_of_processed_partverse_faces_only__not_creator_original'),
 ('semantic_open_mesh_component_multiplicity','partverse_v10_processed_part_label_count|partverse_v10_processed_intralabel_piece_count|partverse_v10_processed_labels_with_multiple_islands','processed_part_label_piece_multiplicity__not_anatomical_semantic'),
 ('part_to_component_many_to_many','partverse_v10_processed_part_label_count|partverse_v10_processed_intralabel_piece_count|partverse_v10_processed_graph_components','processed_label_and_connected_piece_counts_only__no_editable_component_claim'),
 ('watertightness_and_connected_components','partverse_v10_processed_graph_components|partverse_v10_processed_interpart_edges','processed_part_adjacency_graph_only__watertightness_unknown'),
 ('original_asset_recovery_status','partverse_v10_processed_numeric_court','processed_part_numeric_receipt_only__original_uncourted'),
 ('original_asset_provider_locator','evidence_partverse_license_source','source_provider_link_claim_only__not_source_rights'),
 ('attribution_obligations','partverse_v10_provider_license_class_claim','provider_license_class_claim_only__obligations_not_courted'),
]
MASTER=[
 ('source_vertex_face_count','mesh_total_vertex_count|mesh_total_face_count','source_import_counts__not_current_original_rehash'),
 ('mesh_object_vs_connected_island','mesh_object_count|mesh_connected_island_count','imported_object_and_index_island_counts_only__not_drawables'),
 ('modular_vs_welded_vs_multimesh','mesh_object_count|mesh_connected_island_count|mesh_total_vertex_count','imported_multimesh_count_only__welding_policy_unknown'),
 ('original_object_collection_hierarchy','scene_node_count|mesh_object_count','imported_scene_node_counts_only__hierarchy_parity_unknown'),
 ('source_action_curve_inventory','observed_action_count|observed_action_curve_count|observed_action_keyframe_count','source_import_action_curve_counts_only__clip_quality_unknown'),
 ('animated_bone_and_curve_channels','observed_animated_bone_name_count_teacher_only|observed_action_curve_count','source_import_bone_and_curve_counts_only__no_motion_skill_proof'),
 ('mesh_armature_bind_association','armature_count|armature_modifier_mesh_objects|bone_parented_mesh_objects','imported_mesh_rig_binding_counts_only__no_ik_proof'),
 ('skin_weight_normalization','skin_rows_sampled|skin_rows_bad_sum','imported_skin_sample_row_sum_counts_only__not_all_source_weights'),
 ('zero_weight_vertices_and_unbound_mesh','skin_rows_sampled|skin_rows_zero_weights','imported_skin_sample_zero_weight_count_only'),
 ('joint_count_vs_anatomical_joint','bone_count','imported_bone_count_only__anatomical_joint_identity_unknown'),
 ('native_texture_image_references','texture_image_count','imported_image_reference_count_only__original_pixels_uncourted'),
 ('source_material_slot_inventory','material_count','imported_material_count_only__slot_to_face_uncourted'),
 ('original_asset_recovery_status','master_v11_source_recovery_status','historical_source_recovery_status_only__current_original_rehash_zero'),
]
CORE=[
 ('source_vertex_face_count','v5_source_rest_vertices|v5_source_rest_faces','v5_independent_source_rest_counts__not_original_rehash'),
 ('source_sdf_sign_court_or_mask','v5_geometry_signed_query_fraction|v5_geometry_sign_valid_components','v5_component_local_sign_mask_only__global_sign_abstain'),
 ('udf_occupancy_and_surface_sampling','v5_geometry_queries|v5_geometry_distance_courts','v5_component_geometry_query_counts_only__model_training_not_authorized'),
]
UNIML=[
 ('source_vertex_face_count','canonical_v6_vertex_records|canonical_v6_triangle_faces','canonical_derivative_glb_counts_only__not_creator_original'),
 ('mesh_object_vs_connected_island','canonical_v6_primitive_count','canonical_glb_primitive_count_only__mesh_object_unknown'),
 ('joint_count_vs_anatomical_joint','canonical_glb_joint_count','canonical_glb_joint_count_only__anatomical_identity_unknown'),
 ('skin_weight_normalization','canonical_v6_skin_vertices|canonical_v6_skin_vertices_weight_sum1','canonical_glb_numeric_skin_sum1_vertex_counts_only'),
 ('boundary_loops_and_open_surfaces','canonical_v6_mesh_boundary_edges','canonical_glb_boundary_edges_only__loops_unreviewed'),
 ('nonmanifold_and_self_intersection','canonical_v6_mesh_nonmanifold_edges','canonical_glb_nonmanifold_edges_only__self_intersection_unknown'),
 ('degenerate_sliver_zero_area_faces','canonical_v6_degenerate_faces','canonical_glb_degenerate_face_count_only__slivers_unknown'),
 ('native_texture_image_references','canonical_v6_embedded_image_count','canonical_glb_embedded_image_count_only__creator_source_not_checked'),
 ('deform_intervention_positive_region','canonical_v7_positive_control_moved_poses','canonical_glb_tested_moved_pose_count_only__region_truth_unverified'),
 ('deform_intervention_negative_region','canonical_v7_negative_control_immobile_poses','canonical_glb_tested_immobile_pose_count_only__source_negative_region_unverified'),
 ('front_back_side_eight_direction_views','canonical_v6_source_relative_8_views|canonical_v7_relative_two_vs_eight_area_recall','canonical_glb_source_relative_view_geometry_coverage_only__not_rgb_render_pair'),
]
SHA=re.compile(r'^[0-9a-f]{64}$')
METADATA=[
 ('original_asset_recovery_status','structure_evidence_class|graphics_census_source_recovery_next_action','inventory_candidate_status_only__no_physical_original_source_bytes'),
 ('original_asset_provider_locator','provider_locator','recorded_provider_locator_only__not_validated_source'),
]
STATES={'METADATA':'RECORDED_INVENTORY_CANDIDATE_STATUS_ONLY','PARTVERSE':'OBSERVED_PROCESSED_PART_GEOMETRY_ONLY',
        'MASTER':'OBSERVED_HISTORICAL_SOURCE_IMPORT_COUNTS_ONLY',
        'CORE':'OBSERVED_V5_COMPONENT_COURT_ONLY',
        'UNIML':'OBSERVED_CANONICAL_GLB_DERIVATIVE_ONLY'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()

def source_class(r):
 if r['provider']=='PARTVERSE_XL' and r['partverse_v10_processed_numeric_court']=='PASS_PROCESSED_PART_GEOMETRY_NUMERIC_ONLY' and r['partverse_v10_processed_parts_available']=='True':
  return 'PARTVERSE','evidence_partverse_processed_sha256',PARTVERSE
 if r['pool']=='MASTER3993' and r['source_import_observed']=='True' and r['historical_split_hint']=='FIT' and r['master_v11_holdout_protected']!='True':
  return 'MASTER','evidence_master_deep_sha256',MASTER
 if r['pool']=='CORE370' and r['v5_source_rest_court']=='PASS_SOURCE_REST_INDEPENDENT_COURT':
  return 'CORE','v5_source_rest_sha256',CORE
 if r['provider']=='UNIML3D' and r['canonical_v6_sha_verified']=='True':
  return 'UNIML','canonical_glb_sha256',UNIML
 # No source proof? Only the already SHA-locked inventory's candidate status
 # may be published; do not create physical geometry or training evidence.
 return 'METADATA','__BASELINE_INVENTORY_SHA__',METADATA

def value_for(r,columns,domain,scope,evidence):
 fields={k:r[k] for k in columns if r.get(k,'')!=''}
 # No inferred zeros for empty fields; even a missing one remains unknown.
 if not fields:return None
 return json.dumps({'observed_field_values':fields, 'observed_scope':scope,
  'source_observation_provenance':'V11_SNAPSHOT_ALREADY_CAPTURED',
  'source_evidence_sha256':evidence,'source_evidence_domain':domain,
  'creator_original_current_rehash':False,'rights_training_admission':False,
  'model_training_admission':False,'semantic_editable_drawable_admission':False},
  sort_keys=True,separators=(',',':'),ensure_ascii=False)

def write_gzip(path):
 fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='.v17-tmp-')
 raw=os.fdopen(fd,'wb')
 z=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='')
 txt=io.TextIOWrapper(z,encoding='utf-8',newline='')
 return tmp,raw,txt

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=pathlib.Path,required=True)
 p.add_argument('--states',type=pathlib.Path,required=True);p.add_argument('--values',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,required=True)
 a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
 for kind,path in [('inventory',a.inventory),('states',a.states),('values',a.values)]:
  if sha(path)!=BASE_SHA[kind]:raise ValueError('BASE_SHA_DRIFT:'+kind)
 inventory={};scopes=collections.Counter()
 with gzip.open(a.inventory,'rt',newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   key=r['source_key'];assert key not in inventory
   inventory[key]=r
 assert len(inventory)==16397
 states_out=a.out/'FACET_STATES_EFFECTIVE_V17_16397x264.csv.gz'
 values_out=a.out/'FACET_VALUES_EFFECTIVE_V17_16397x264.csv.gz'
 delta_out=a.out/'FACET_DELTA_INVENTORY_V17.csv.gz'
 readiness_out=a.out/'SOURCE_READINESS_V17_16397.csv.gz'
 counters=collections.Counter();facets=collections.Counter();domains=collections.Counter();changes={}
 opened=[];success=False
 try:
  for path in (states_out,values_out,delta_out,readiness_out):opened.append(write_gzip(path))
  streams=[x[2] for x in opened]
  ds,vs,dt,rs=streams
  with gzip.open(a.states,'rt',newline='',encoding='utf-8') as sf,gzip.open(a.values,'rt',newline='',encoding='utf-8') as vf:
   sr,vr=csv.DictReader(sf),csv.DictReader(vf);h=sr.fieldnames
   assert h==vr.fieldnames and len(h)==265
   map_names={facet for m in (PARTVERSE,MASTER,CORE,UNIML,METADATA) for facet,_,_ in m}
   assert map_names.issubset(set(h))
   sw,vw=csv.DictWriter(ds,fieldnames=h),csv.DictWriter(vs,fieldnames=h)
   dw=csv.DictWriter(dt,fieldnames=['source_key','facet','new_state','value_json','source_field_names','source_evidence_sha256','domain'])
   rw=csv.DictWriter(rs,fieldnames=['source_key','pool','provider','candidate_id','historical_split_hint','evidence_route','new_partial_cells','previous_observed_facet_cells','original_source_bytes_courted','rights_training_eligible','training_authorized','next_action'])
   for w in (sw,vw,dw,rw):w.writeheader()
   seen=set()
   for st,va in zip(sr,vr):
    key=st['source_key'];assert key==va['source_key'] and key in inventory and key not in seen
    seen.add(key);r=inventory[key];route=source_class(r);modified=0
    previous=sum(x!='UNKNOWN_UNREVIEWED' for k,x in st.items() if k!='source_key')
    if route:
     domain,sha_field,mappings=route
     evidence=BASE_SHA['inventory'] if sha_field=='__BASELINE_INVENTORY_SHA__' else r[sha_field]
     assert SHA.fullmatch(evidence),'MISSING_SOURCE_EVIDENCE_SHA:'+key+':'+sha_field
     domains[domain]+=1
     for facet,fields,scope in mappings:
      if st[facet]!='UNKNOWN_UNREVIEWED':
       counters['preexisting_cells_untouched']+=1;continue
      if va[facet]:raise ValueError('UNKNOWN_WITH_EXISTING_VALUE:'+key+':'+facet)
      names=fields.split('|');new_value=value_for(r,names,domain,scope,evidence)
      if new_value is None:continue
      new_state=STATES[domain]
      st[facet]=new_state;va[facet]=new_value;modified+=1;facets[facet]+=1
      dw.writerow({'source_key':key,'facet':facet,'new_state':new_state,'value_json':new_value,
       'source_field_names':fields,'source_evidence_sha256':evidence,'domain':domain})
    counters['rows']+=1;counters['new_cells']+=modified
    if modified:counters['changed_sources']+=1
    # Readiness is source truth, never a training approval.
    observed_court='ORIGINAL_SOURCE_NOT_CURRENTLY_REHASHED'
    rw.writerow({'source_key':key,'pool':r['pool'],'provider':r['provider'],'candidate_id':r['candidate_id'],
      'historical_split_hint':r['historical_split_hint'],'evidence_route':route[0] if route else 'NO_MAPPED_PHYSICAL_COURT',
      'new_partial_cells':modified,'previous_observed_facet_cells':previous,
      'original_source_bytes_courted':r['original_source_bytes_courted'],
      'rights_training_eligible':r['rights_training_eligible'],'training_authorized':r['training_authorized'],
      'next_action':r.get('graphics_census_source_recovery_next_action','')})
    sw.writerow(st);vw.writerow(va)
   assert len(seen)==16397 and next(sr,None) is None and next(vr,None) is None
  for tmp,raw,stream in opened:
   stream.close();raw.flush();os.fsync(raw.fileno());raw.close()
  for (tmp,raw,stream),path in zip(opened,(states_out,values_out,delta_out,readiness_out)):
   os.replace(tmp,path)
  success=True
 finally:
  if not success:
   for tmp,raw,stream in opened:
    try:stream.close();raw.close()
    except Exception:pass
    if pathlib.Path(tmp).exists():pathlib.Path(tmp).unlink()
 assert counters['rows']==16397 and counters['new_cells']>0
 summary={'schema':'RealSaS.TruthFoundryV17HistoricalInventoryPartialOverlay.v1',
  'baseline_v16_sha256':BASE_SHA,'source_rows':16397,'facet_columns':264,
  'new_partial_observation_cells':counters['new_cells'],'new_partial_observation_sources':counters['changed_sources'],
  'routed_source_counts':dict(domains),'facet_increment_counts':dict(facets),
  'untouched_existing_evidence_encounters':counters['preexisting_cells_untouched'],
  'new_private_source_downloads':0,'new_original_source_byte_rehashes':0,
  'training_admissions':0,'rights_admissions':0,'original_source_parity_admissions':0,
  'semantic_admissions':0,'graphics_art_admissions':0,'tessa_animation_ready_admissions':0,
  'output_sha256':{name:sha(path) for name,path in [('states',states_out),('values',values_out),('delta',delta_out),('readiness',readiness_out)]}}
 (a.out/'SUMMARY_V17.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:summary[k] for k in ('new_partial_observation_cells','new_partial_observation_sources','routed_source_counts','facet_increment_counts')},indent=2))

if __name__=='__main__':main()