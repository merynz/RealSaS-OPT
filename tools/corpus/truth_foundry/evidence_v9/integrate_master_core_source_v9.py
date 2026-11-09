#!/usr/bin/env python3
"""V8 -> V9 evidence-only source matrix overlay, no admission or inherited status upgrade.
Every non-UNKNOWN facet change is traceable to a local raw importer-derived receipt
that passed independent arithmetic and index-domain checks (not original raw SHA).
Core370 partial family snapshot is a dedicated namespace, never the current court.
"""
from __future__ import annotations
import csv,gzip,sqlite3,json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path('/mnt/data')
V8=ROOT/'RealSaS_Truth_Foundry_Deep_Inspection_V6_20261109'
OUT=ROOT/'RealSaS_Truth_Foundry_Source_Courts_V9_20261109'
DB=OUT/'MASTER_CORE_INDEPENDENT_RECEIPT_COURTS_V9.sqlite' 

def xrows(p):
 with gzip.open(p,'rt',encoding='utf8',newline='') as f:
  yield from csv.DictReader(f)

def xwriter(name, fields):
 f=gzip.open(OUT/name,'wt',encoding='utf8',newline='',compresslevel=6)
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader()
 return f,w

c=sqlite3.connect(DB);c.row_factory=sqlite3.Row
masters={r['source_key']:r for r in c.execute('SELECT * FROM master_asset')}
cores={r['source_key']:r for r in c.execute('SELECT * FROM core_identity')}
families={}
for r in c.execute('SELECT source_key,family,snapshot_status FROM core_partial_family'):
 families.setdefault(r['source_key'],{})[r['family']]=r['snapshot_status']
models={}
for r in c.execute('SELECT source_key,model,corpus_eligible_diagnostic FROM core_partial_model'):
 models.setdefault(r['source_key'],{})[r['model']]=bool(r['corpus_eligible_diagnostic'])
namecounts={}
for r in c.execute('SELECT source_key,name_kind,COUNT(*) n FROM master_teacher_name GROUP BY source_key,name_kind'):
 namecounts.setdefault(r['source_key'],{})[r['name_kind']]=r['n']
assert len(masters)==2466 and len(cores)==370

with gzip.open(V8/'SOURCE_INVENTORY_DEEP_16397_V8.csv.gz','rt') as f: original_fields=next(csv.reader(f))
master_fields=['master_v9_import_receipt_internal_court','master_v9_import_receipt_gz_sha256','master_v9_mesh_record_count',
 'master_v9_mesh_multiconnected_object_count','master_v9_mesh_with_vertex_groups_count','master_v9_mesh_weighted_rows_candidate_objects',
 'master_v9_fully_unweighted_mesh_object_count','master_v9_armature_modifier_mesh_object_count','master_v9_object_bone_parented_count',
 'master_v9_object_parented_count','master_v9_scene_node_indices_unique','master_v9_action_name_teacher_only_count',
 'master_v9_scene_node_name_teacher_only_count','master_v9_material_image_name_teacher_only_count',
 'master_v9_original_source_rehashed','master_v9_source_receipt_manifest_sha256',
 'master_v9_possible_global_sdf_not_certified','master_v9_2d_art_not_certified']
core_fields=['core_v9_partial_snapshot_sha256','core_v9_partial_snapshot_scope', 'core_v9_partial_family_known_count',
 'core_v9_partial_family_pass_count','core_v9_partial_family_fail_count','core_v9_partial_family_masked_count',
 'core_v9_partial_model_eligibility_is_diagnostic_only','core_v9_current_v5_observation_vs_partial_import','core_v9_original_source_not_rehashed']
core_family_fields=['core_partial_family_'+fam.lower() for fam in sorted(next(iter(families.values())))]
core_model_fields=['core_partial_model_'+mdl.lower()+'_diagnostic' for mdl in ('IRIS','TESSA','AXIS','MIRA')]
fields=original_fields+master_fields+core_fields+core_family_fields+core_model_fields
f,w=xwriter('SOURCE_INVENTORY_DEEP_16397_V9.csv.gz',fields)
candidate_counts=Counter();candidate_seen=set();v5_vs_partial=Counter()

for row in xrows(V8/'SOURCE_INVENTORY_DEEP_16397_V8.csv.gz'):
  key=row['source_key'];assert key not in candidate_seen;candidate_seen.add(key)
  for k in fields[len(original_fields):]:row[k]=''
  if key.startswith('MASTER3993:') and key in masters:
    s=masters[key];d=json.loads(s['diagnostic_json']);assert s['receipt_integrity_state']=='PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX'
    assert row['master_deep_import_source_sha256']==s['source_sha_claim'],(key,'MASTER_SHA_MISMATCH')
    row.update({
      'master_v9_import_receipt_internal_court':s['receipt_integrity_state'],
      'master_v9_import_receipt_gz_sha256':s['imported_receipt_sha'],
      'master_v9_mesh_record_count':d['mesh_object_count'],
      'master_v9_mesh_multiconnected_object_count':d['multi_island_mesh_object_count'],
      'master_v9_mesh_with_vertex_groups_count':d['vertex_group_mesh_object_count'],
      'master_v9_mesh_weighted_rows_candidate_objects':d['weighted_candidate_mesh_object_count'],
      'master_v9_fully_unweighted_mesh_object_count':d['fully_unweighted_mesh_object_count'],
      'master_v9_armature_modifier_mesh_object_count':d['armature_modifier_mesh_object_count'],
      'master_v9_object_bone_parented_count':d['bone_parented_mesh_object_count'],
      'master_v9_object_parented_count':d['object_parented_mesh_object_count'],
      'master_v9_scene_node_indices_unique':'TRUE_IN_IMPORT_RECEIPT',
      'master_v9_action_name_teacher_only_count':namecounts.get(key,{}).get('action',0),
      'master_v9_scene_node_name_teacher_only_count':namecounts.get(key,{}).get('node',0),
      'master_v9_material_image_name_teacher_only_count':namecounts.get(key,{}).get('material_image',0),
      'master_v9_original_source_rehashed':'FALSE_IMPORT_RECEIPT_ONLY',
      'master_v9_source_receipt_manifest_sha256':d['source_scene_receipt_sha256'],
      'master_v9_possible_global_sdf_not_certified':'NEEDS_SIGN_VALIDITY_COURT_NOT_INFERRED_FROM_BOUNDARY',
      'master_v9_2d_art_not_certified':'NO_SOURCE_2D_ART_APPROVAL',
    })
    candidate_counts['master_matched']+=1
  if key.startswith('CORE370:'):
    assert key in cores,(key,'CORE_ID_MISSING')
    assert row['original_sha_claim']==cores[key]['source_sha256'],(key,'CORE_SHA_MISMATCH')
    st=families[key];p=sum(v=='PASS' for v in st.values());fails=sum(v=='FAIL' for v in st.values());masked=sum(v=='MASKED' for v in st.values());known=p+fails+masked
    obs='V5_OBSERVATION_QUALIFIED' if row.get('v5_observation_state')=='PREPARED_OBSERVATION_V5' else 'V5_OBSERVATION_FAILED'
    i=st['SOURCE_IMPORT']
    v5_vs_partial[(i,obs)]+=1
    row.update({
      'core_v9_partial_snapshot_sha256':'2d77f0ed30c80ca9edf69e32797fbfe522be87e243aadc6490a61a2cdeea16c0',
      'core_v9_partial_snapshot_scope':'PARTIAL_DIAGNOSTIC__NOT_V5_FAMILY_ADMISSION',
      'core_v9_partial_family_known_count':known,
      'core_v9_partial_family_pass_count':p,
      'core_v9_partial_family_fail_count':fails,
      'core_v9_partial_family_masked_count':masked,
      'core_v9_partial_model_eligibility_is_diagnostic_only':'TRUE_NEVER_TRAINING_READY',
      'core_v9_current_v5_observation_vs_partial_import':f'{obs}__PARTIAL_IMPORT_{i}',
      'core_v9_original_source_not_rehashed':'TRUE_SOURCE_SHA_REFERENCED_IN_COURT',
    })
    for fam,status in st.items():row['core_partial_family_'+fam.lower()]=status
    for model,status in models[key].items():row['core_partial_model_'+model.lower()+'_diagnostic']='DIAGNOSTIC_ELIGIBLE_NOT_ADMITTED' if status else 'DIAGNOSTIC_BLOCKED'
    candidate_counts['core_matched']+=1
  w.writerow(row)
  candidate_counts['all']+=1
f.close();assert candidate_counts['all']==16397 and candidate_counts['master_matched']==2466 and candidate_counts['core_matched']==370

facet_cols=['modular_vs_welded_vs_multimesh','primitive_face_to_scene_node','mesh_armature_bind_association']
with gzip.open(V8/'FACET_STATES_EFFECTIVE_V8_16397x264.csv.gz','rt') as f: facet_fields=next(csv.reader(f))
assert all(k in facet_fields for k in facet_cols) and len(facet_fields)==265
state_f,state_w=xwriter('FACET_STATES_EFFECTIVE_V9_16397x264.csv.gz',facet_fields)
value_f,value_w=xwriter('FACET_VALUES_EFFECTIVE_V9_16397x264.csv.gz',facet_fields)
delta_f,delta_w=xwriter('FACET_DELTA_V9_MASTER_RECEIPT.csv.gz', ['source_key','facet','prior_state','new_state','evidence_value','import_receipt_sha256','evidence_policy'])
statuses=xrows(V8/'FACET_STATES_EFFECTIVE_V8_16397x264.csv.gz')
values=xrows(V8/'FACET_VALUES_EFFECTIVE_V8_16397x264.csv.gz')
changes=Counter();unchanged=0
for state,val in zip(statuses,values,strict=True):
  key=state['source_key'];assert key==val['source_key']
  src=masters.get(key)
  if src and src['receipt_integrity_state']=='PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX':
    d=json.loads(src['diagnostic_json']);n=d['mesh_object_count'];islands=d['mesh_connected_islands'];weighted=d['weighted_candidate_mesh_object_count'];mods=d['armature_modifier_mesh_object_count'];sign=d['source_scene_receipt_sha256']
    observations={
      'modular_vs_welded_vs_multimesh':('OBSERVED_IMPORT_STRUCTURE_ONLY__NOT_SEMANTIC_MODULE','mesh_object_count='+str(n)+';disconnected_islands='+str(islands)+';multi_island_mesh_objects='+str(d['multi_island_mesh_object_count'])),
      'primitive_face_to_scene_node':('OBSERVED_MESH_OBJECT_SCENE_NODE_INDEXES__NOT_FACE_LEVEL_ADDRESS', 'mesh_scene_node_indices='+','.join(map(str,d['mesh_nodes']))),
      'mesh_armature_bind_association':('OBSERVED_AGGREGATE_ARMATURE_MODIFIERS__NOT_EXACT_JOINT_BINDING', 'armature_modifier_mesh_objects='+str(mods)+';candidate_nonzero_weight_mesh_objects='+str(weighted)),
    }
    for facet,(new_state,new_val) in observations.items():
      if state[facet].startswith('UNKNOWN'):
        delta_w.writerow({'source_key':key,'facet':facet,'prior_state':state[facet],
         'new_state':new_state,'evidence_value':new_val,'import_receipt_sha256':sign,
         'evidence_policy':'DERIVED_IMPORTER_RECEIPT_ONLY__NO_SOURCE_BYTE_PARITY_OR_HEAD_ADMISSION'})
        state[facet]=new_state;val[facet]=new_val;changes[facet]+=1
      else:unchanged+=1
  state_w.writerow(state);value_w.writerow(val)
state_f.close();value_f.close();delta_f.close()
assert sum(changes.values())==2466*3 and len(facet_fields)==265,(changes,unchanged)

# Stage comparison: distinct tests, never promote import PASS to raster-mask PASS.
summary={
 'schema':'RealSaS.DeepTruthMatrixOverlay.v9',
 'candidate_count':16397,'facet_count':264,'new_import_receipt_observations':sum(changes.values()),
 'facet_observation_counts':dict(changes),'existing_facet_data_modified':0,
 'master_source_receipt_crosswalk_count':candidate_counts['master_matched'],
 'core_partial_source_sha_crosswalk_count':candidate_counts['core_matched'],
 'core_partial_family_diagnostic_rows':370*14,
 'core_v5_observation_vs_partial_import':{f'{a}__{b}':n for (a,b),n in v5_vs_partial.items()},
 'core_import_snapshot_is_not_same_court_as_v5_observation':True,
 'original_source_bytes_rehashed':0,'new_train_admission':0,
 'global_sdf_validity_admitted':0,'semantic_ontology_admitted':0,
 '2d_art_review_admitted':0,'lineage_safe_split_admitted':0,
 'evidence_databases': [DB.name],
}
(OUT/'INVENTORY_MATRIX_V9_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2));c.close()