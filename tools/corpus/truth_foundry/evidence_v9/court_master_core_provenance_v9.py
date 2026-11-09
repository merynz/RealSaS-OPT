#!/usr/bin/env python3
"""RealSaS Truth Foundry V9: independent receipt-level census + parallel historical court crosswalk.

Two intentional evidence boundaries:
- MASTER3993 full-scene receipts are *importer derivatives*, not source original bytes.
- CORE370 family readback is a *partial diagnostic snapshot*, not contemporary V5
  observation court, global split, materialization or trainer authority.

No classification, split/admission, semantic interpretation or SDF global sign is minted.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path('/mnt/data')
V8 = ROOT/'RealSaS_Truth_Foundry_Deep_Inspection_V6_20261109'
OUT = ROOT/'RealSaS_Truth_Foundry_Source_Courts_V9_20261109'
OUT.mkdir(exist_ok=True)
DB = OUT/'MASTER_CORE_INDEPENDENT_RECEIPT_COURTS_V9.sqlite'
CENSUS = ROOT/'MASTER2466_FULL_SOURCE_DEEP_AUDIT_V1.jsonl.bin'
CORE = ROOT/'41835338185_CORPUS370_PARTIAL_AUTHORITY_MATRIX_V1.json'

parity_fields = {
    'mesh_object_count': None,
    'vertex_count': 'vertex_count',
    'polygon_count': 'polygon_count',
    'mesh_disconnected_components': 'connected_islands',
    'mesh_boundary_edges': 'boundary_edges',
    'mesh_nonmanifold_edges': 'nonmanifold_edges',
    'skin_rows_sampled': 'skin_row_count',
}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))

if DB.exists():DB.unlink()
c=sqlite3.connect(DB)
c.execute('''CREATE TABLE master_asset(
 source_key TEXT PRIMARY KEY, source_sha_claim TEXT NOT NULL, imported_receipt_sha TEXT,
 receipt_integrity_state TEXT NOT NULL, diagnostic_json TEXT NOT NULL,
 source_raw_rehashed INTEGER NOT NULL DEFAULT 0, training_authorized INTEGER NOT NULL DEFAULT 0)''')
c.execute('''CREATE TABLE master_mesh(
 source_key TEXT NOT NULL, mesh_record_index INTEGER NOT NULL, scene_node_index INTEGER NOT NULL,
 node_name_teacher_only TEXT, vertex_count INTEGER NOT NULL, polygon_count INTEGER NOT NULL,
 connected_islands INTEGER NOT NULL, boundary_edges INTEGER NOT NULL,
 nonmanifold_edges INTEGER NOT NULL, vertex_group_count INTEGER NOT NULL,
 skin_row_count INTEGER NOT NULL, unweighted_row_count INTEGER NOT NULL,
 material_slots INTEGER NOT NULL, row_sha256 TEXT NOT NULL,
 PRIMARY KEY(source_key,mesh_record_index))''')
c.execute('''CREATE TABLE master_teacher_name(
 source_key TEXT NOT NULL, name_kind TEXT NOT NULL, ordinal INTEGER NOT NULL,
 source_name TEXT, authority TEXT NOT NULL, PRIMARY KEY(source_key,name_kind,ordinal))''')
c.execute('''CREATE TABLE core_partial_family(
 source_key TEXT NOT NULL, family TEXT NOT NULL, snapshot_status TEXT NOT NULL,
 reason_codes_json TEXT NOT NULL, court_refs_json TEXT NOT NULL,
 evidence_refs_json TEXT NOT NULL, producer_refs_json TEXT NOT NULL,
 claim_metadata_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
 PRIMARY KEY(source_key,family))''')
c.execute('''CREATE TABLE core_partial_model(
 source_key TEXT NOT NULL, model TEXT NOT NULL, corpus_eligible_diagnostic INTEGER NOT NULL,
 blocking_families_json TEXT NOT NULL, required_pass_json TEXT NOT NULL,
 post_corpus_dependencies_json TEXT NOT NULL, PRIMARY KEY(source_key,model))''')

asset_stats=Counter();court_count=Counter();mesh_stats=Counter();asset_counts=Counter()
with open(CENSUS,'rt') as f:
  for ln,line in enumerate(f,1):
    a=json.loads(line)
    key='MASTER3993:'+a['asset_id']; mesh=a['mesh_records']; counts=a['counts']; md=a['privileged_source_metadata_teacher_only']; errors=[]
    if a['schema']!='RealSaS.TruthFoundryDeepSourceAudit.v1':errors.append('SOURCE_RECEIPT_SCHEMA_DRIFT')
    if len(md.get('scene_node_names',[]))!=counts.get('source_scene_node_count'):errors.append('NAME_NODE_CARDINALITY')
    if len(md.get('action_names',[]))!=counts.get('observed_action_count'):errors.append('NAME_ACTION_CARDINALITY')
    indices=[m['scene_node_index'] for m in mesh]
    if len(set(indices))!=len(indices):errors.append('MESH_NODE_INDEX_DUPLICATE')
    if any(i<0 or i>=counts['source_scene_node_count'] for i in indices):errors.append('MESH_NODE_INDEX_RANGE')
    for field,mesh_field in parity_fields.items():
      actual=len(mesh) if mesh_field is None else sum(int(m.get(mesh_field,0)) for m in mesh)
      if actual!=int(counts[field]):errors.append('SUM_PARITY_'+field)
    if any(any(m.get(k,0)<0 for k in ('vertex_count','polygon_count','connected_islands','boundary_edges','nonmanifold_edges','skin_row_count','unweighted_row_count','material_slots','vertex_group_count')) for m in mesh):errors.append('MESH_NEGATIVE_COUNT')
    if any(m['unweighted_row_count']>m['skin_row_count'] or m['skin_row_count']>m['vertex_count'] for m in mesh):errors.append('MESH_SKIN_DOMAIN_DRIFT')
    if not a['source_sha256'] or len(a['source_sha256'])!=64:errors.append('INVALID_IMPORT_SOURCE_SHA_CLAIM')
    statuses='PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX' if not errors else 'FAIL_'+','.join(sorted(set(errors)))
    asset_stats[statuses]+=1
    fully_unweighted=sum(int(m['skin_row_count']>0 and m['skin_row_count']==m['unweighted_row_count']) for m in mesh)
    skin_group_meshes=sum(int(m['vertex_group_count']>0) for m in mesh)
    weighted_candidate_meshes=sum(int(m['skin_row_count']>m['unweighted_row_count']) for m in mesh)
    multi_island_meshes=sum(int(m['connected_islands']>1) for m in mesh)
    record={
       'asset_id':a['asset_id'], 'source_sha256':a['source_sha256'],
       'mesh_object_count':len(mesh),'mesh_connected_islands':sum(m['connected_islands'] for m in mesh),
       'mesh_nodes':indices,'armature_modifier_mesh_object_count':a['armature_modifier_mesh_object_count'],
       'bone_parented_mesh_object_count':a['bone_parented_mesh_object_count'],
       'object_parented_mesh_object_count':a['object_parented_mesh_object_count'],
       'weighted_candidate_mesh_object_count':weighted_candidate_meshes,
       'vertex_group_mesh_object_count':skin_group_meshes,
       'fully_unweighted_mesh_object_count':fully_unweighted,
       'multi_island_mesh_object_count':multi_island_meshes,
       'any_source_action_count':counts['observed_action_count'],
       'source_action_curve_count':counts['observed_action_curve_count'],
       'observed_animated_bone_count':a['observed_animated_bone_name_count_teacher_only'],
       'scene_node_count':counts['source_scene_node_count'],
       'name_hints_teacher_only_not_ontology':True,
       'source_scene_receipt_sha256':a['full_receipt_gz_sha256'],
       'source_import_summary_sha256':a['original_import_summary_sha256'],
       'raw_source_bytes_rehashed_in_this_court':False,
       'status':statuses,
       'errors':errors,
    }
    c.execute('INSERT INTO master_asset VALUES(?,?,?,?,?,0,0)',(key,a['source_sha256'],a['full_receipt_gz_sha256'],statuses,canon(record)))
    for mi,m in enumerate(mesh):
      row={**m,'mesh_record_index':mi,'source_key':key}
      c.execute('INSERT INTO master_mesh VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(
        key,mi,m['scene_node_index'],md['scene_node_names'][m['scene_node_index']] if not errors else None,
        m['vertex_count'],m['polygon_count'],m['connected_islands'],m['boundary_edges'],
        m['nonmanifold_edges'],m['vertex_group_count'],m['skin_row_count'],m['unweighted_row_count'],m['material_slots'],
        hashlib.sha256(canon(row).encode()).hexdigest()))
      mesh_stats['all']+=1
      if m['unweighted_row_count']>0:mesh_stats['has_unweighted']+=1
      if m['skin_row_count']>0 and m['skin_row_count']==m['unweighted_row_count']:mesh_stats['fully_unweighted']+=1
      if m['connected_islands']>1:mesh_stats['multi_island']+=1
    for k,entries in [('action',md.get('action_names',[])),('node',md.get('scene_node_names',[])),('material_image',md.get('material_or_image_names',[]))]:
      for i,v in enumerate(entries):
        c.execute('INSERT INTO master_teacher_name VALUES(?,?,?,?,?)',(key,k,i,v,'TEACHER_ONLY_NAME_NOT_SEMANTIC_TARGET'))
        court_count[k]+=1
    asset_counts['total']+=1
    if fully_unweighted:asset_counts['has_fully_unweighted_mesh']+=1
    if multi_island_meshes:asset_counts['has_multi_island_mesh']+=1
    if counts['observed_action_count']:asset_counts['has_action_names']+=1
    if counts.get('image_dependency_missing_count',0):asset_counts['has_missing_images']+=1

snap=json.load(open(CORE));snap_sha=sha(CORE)
assert snap['schema']=='RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1'
assert snap['readback_scope']=='PARTIAL_DIAGNOSTIC_PROJECTION__NOT_FINAL_ADMISSION_OR_READINESS'
assert snap['status']=='OPEN' and not snap['training_authorized']
core_stats=Counter()
for a in snap['subjects']:
 key='CORE370:'+a['subject_id']
 for family,v in a['families'].items():
   c.execute('INSERT INTO core_partial_family VALUES(?,?,?,?,?,?,?,?,?)',(
     key,family,v['status'],canon(v.get('reason_codes',[])),canon(v.get('court_refs',[])),
     canon(v.get('evidence_refs',[])),canon(v.get('producer_refs',[])),canon(v.get('claim_metadata',[])),snap_sha))
   core_stats[(family,v['status'])]+=1
 for model,v in a['model_eligibility'].items():
   c.execute('INSERT INTO core_partial_model VALUES(?,?,?,?,?,?)',(
     key,model,int(v['corpus_eligible']),canon(v.get('blocking_families',[])),
     canon(v.get('required_pass',[])),canon(v.get('post_corpus_dependencies',[]))))
 c.execute('INSERT OR IGNORE INTO core_source_shadow VALUES(?,?,?)',(key,a['source_sha256'],a['source_id'])) if False else None
 c.execute('CREATE TABLE IF NOT EXISTS core_identity (source_key TEXT PRIMARY KEY, source_sha256 TEXT, provider_source_id TEXT)')
 c.execute('INSERT INTO core_identity VALUES(?,?,?)',(key,a['source_sha256'],a['source_id']))

c.commit()
assert c.execute('SELECT COUNT(*) FROM master_asset').fetchone()[0]==2466
assert c.execute('SELECT COUNT(*) FROM master_mesh').fetchone()[0]==14351
assert c.execute('SELECT COUNT(*) FROM core_partial_family').fetchone()[0]==370*14
assert c.execute('SELECT COUNT(*) FROM core_partial_model').fetchone()[0]==370*4
assert c.execute('SELECT COUNT(*) FROM core_identity').fetchone()[0]==370
assert asset_stats['PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX']==2466,asset_stats

summary={
 'schema':'RealSaS.MasterCoreIndependentReceiptCourt.v9',
 'master_full_importer_receipt_candidates':2466,
 'master_mesh_objects':14351,
 'master_receipt_internal_parity_pass':asset_stats['PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX'],
 'master_receipt_internal_parity_fail':sum(v for k,v in asset_stats.items() if k!='PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX'),
 'master_mesh_diagnostics':dict(mesh_stats),
 'master_asset_diagnostics':dict(asset_counts),
 'master_source_name_records_teacher_only':dict(court_count),
 'core_partial_snapshot_candidate_count':370,
 'core_partial_family_status_rows':c.execute('SELECT COUNT(*) FROM core_partial_family').fetchone()[0],
 'core_partial_model_diagnostics':c.execute('SELECT COUNT(*) FROM core_partial_model').fetchone()[0],
 'core_partial_status_breakdown':{fam:{status:n for (f,status),n in core_stats.items() if f==fam} for fam in sorted(set(f for f,_ in core_stats))},
 'evidence_sources':[
   {'path':str(CENSUS),'sha256':sha(CENSUS),'scope':'MASTER_IMPORTER_DERIVATIVE_RECEIPTS_NOT_ORIGINAL_SOURCE'},
   {'path':str(CORE),'sha256':snap_sha,'scope':'CORE_PARTIAL_DIAGNOSTIC_SNAPSHOT_NOT_CURRENT_ADMISSION'},
 ],
 'original_source_bytes_rehashed':0,
 'new_semantic_or_graphics_pass':0,
 'new_training_admission':0,
 'new_split_authority':0,
 'new_global_sdf_authority':0
}
(OUT/'MASTER_CORE_SOURCE_RECEIPT_COURT_SUMMARY_V9.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
c.close()
print(json.dumps(summary,indent=2)[:7000])