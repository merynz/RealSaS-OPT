#!/usr/bin/env python3
"""Independent fail-closed V8->V9 forensic overlay verifier.
Recounts inputs independently; rejects altered pre-existing values, unjustified PASS,
asset ID/SHA drift and model/admission promotion. Does not trust V9 summary counters.
"""
import csv,gzip,json,hashlib,sqlite3,zipfile
from pathlib import Path
from collections import Counter
root=Path('/mnt/data');src=root/'RealSaS_Truth_Foundry_Deep_Inspection_V6_20261109';out=root/'RealSaS_Truth_Foundry_Source_Courts_V9_20261109'
conn=sqlite3.connect(out/'MASTER_CORE_INDEPENDENT_RECEIPT_COURTS_V9.sqlite');conn.row_factory=sqlite3.Row

masters={r['source_key']:r for r in conn.execute('SELECT * FROM master_asset')};cores={r['source_key']:r for r in conn.execute('SELECT * FROM core_identity')};assert len(masters)==2466 and len(cores)==370
assert conn.execute('SELECT COUNT(*) FROM master_mesh').fetchone()[0]==14351
assert conn.execute('SELECT COUNT(*) FROM core_partial_family').fetchone()[0]==5180
assert conn.execute('SELECT COUNT(*) FROM core_partial_model').fetchone()[0]==1480
assert conn.execute('SELECT COUNT(*) FROM master_teacher_name').fetchone()[0]==13291+28190+6582

def read(p):
 with gzip.open(p,'rt',encoding='utf8',newline='') as f:
  yield from csv.DictReader(f)

def header(p):
 with gzip.open(p,'rt',encoding='utf8',newline='') as f:return next(csv.reader(f))

v8src=src/'SOURCE_INVENTORY_DEEP_16397_V8.csv.gz';v9src=out/'SOURCE_INVENTORY_DEEP_16397_V9.csv.gz'
h8=header(v8src);h9=header(v9src);assert h9[:len(h8)]==h8
counts=Counter();seen=set()
for a,b in zip(read(v8src),read(v9src),strict=True):
 k=a['source_key'];assert k not in seen;seen.add(k);assert a=={f:b[f] for f in h8},('OLD_SOURCE_CHANGED',k)
 if k in masters:
  assert b['master_v9_import_receipt_internal_court']=='PASS_RECEIPT_INTERNAL_ARITHMETIC_AND_NODE_INDEX'
  assert b['master_v9_original_source_rehashed']=='FALSE_IMPORT_RECEIPT_ONLY'
  assert b['master_v9_mesh_record_count']==str(conn.execute('SELECT COUNT(*) FROM master_mesh WHERE source_key=?',(k,)).fetchone()[0])
  assert b['master_v9_material_image_name_teacher_only_count']==str(conn.execute("SELECT COUNT(*) FROM master_teacher_name WHERE source_key=? AND name_kind='material_image'",(k,)).fetchone()[0])
  assert b['master_v9_action_name_teacher_only_count']==str(conn.execute("SELECT COUNT(*) FROM master_teacher_name WHERE source_key=? AND name_kind='action'",(k,)).fetchone()[0])
  assert b['master_v9_scene_node_name_teacher_only_count']==str(conn.execute("SELECT COUNT(*) FROM master_teacher_name WHERE source_key=? AND name_kind='node'",(k,)).fetchone()[0])
  assert b['master_v9_source_receipt_manifest_sha256']==masters[k]['imported_receipt_sha']
  counts['master']+=1
 elif k in cores:
  assert b['original_sha_claim']==cores[k]['source_sha256']
  assert b['core_v9_partial_snapshot_scope']=='PARTIAL_DIAGNOSTIC__NOT_V5_FAMILY_ADMISSION'
  assert b['core_v9_partial_model_eligibility_is_diagnostic_only']=='TRUE_NEVER_TRAINING_READY'
  fam={r['family']:r['snapshot_status'] for r in conn.execute('SELECT family,snapshot_status FROM core_partial_family WHERE source_key=?',(k,))}
  for family,status in fam.items():assert b['core_partial_family_'+family.lower()]==status
  assert len(fam)==14
  for mdl,eligible in conn.execute('SELECT model,corpus_eligible_diagnostic FROM core_partial_model WHERE source_key=?',(k,)):
   assert b['core_partial_model_'+mdl.lower()+'_diagnostic']==('DIAGNOSTIC_ELIGIBLE_NOT_ADMITTED' if eligible else 'DIAGNOSTIC_BLOCKED')
  counts['core']+=1
 else:
  assert all(b[f]=='' for f in h9[len(h8):]),('NON_TARGET_UPGRADED',k)
  counts['unmodified_other']+=1
 counts['sources']+=1
assert counts['sources']==16397 and counts['master']==2466 and counts['core']==370 and counts['unmodified_other']==13561

allowed={'modular_vs_welded_vs_multimesh':'OBSERVED_IMPORT_STRUCTURE_ONLY__NOT_SEMANTIC_MODULE',
 'primitive_face_to_scene_node':'OBSERVED_MESH_OBJECT_SCENE_NODE_INDEXES__NOT_FACE_LEVEL_ADDRESS',
 'mesh_armature_bind_association':'OBSERVED_AGGREGATE_ARMATURE_MODIFIERS__NOT_EXACT_JOINT_BINDING'}
state8=src/'FACET_STATES_EFFECTIVE_V8_16397x264.csv.gz';state9=out/'FACET_STATES_EFFECTIVE_V9_16397x264.csv.gz'
val8=src/'FACET_VALUES_EFFECTIVE_V8_16397x264.csv.gz';val9=out/'FACET_VALUES_EFFECTIVE_V9_16397x264.csv.gz'
assert header(state8)==header(state9)==header(val8)==header(val9)
assert len(header(state8))==265
changed=Counter();original_preserved=0;rowcount=0
for a,b,va,vb in zip(read(state8),read(state9),read(val8),read(val9),strict=True):
 k=a['source_key'];assert k==b['source_key']==va['source_key']==vb['source_key'];rowcount+=1
 for facet in header(state8)[1:]:
  if a[facet]!=b[facet] or va[facet]!=vb[facet]:
   assert k in masters and facet in allowed
   assert a[facet].startswith('UNKNOWN') and b[facet]==allowed[facet]
   assert va[facet] in ('','null','None')
   assert 'PASS' not in b[facet] and 'CANDIDATE' not in b[facet]
   assert vb[facet]
   changed[facet]+=1
  else:original_preserved+=1
assert rowcount==16397
assert changed==Counter({f:2466 for f in allowed}),changed
assert original_preserved==16397*264-7398,(original_preserved,16397*264-7398)
with gzip.open(out/'FACET_DELTA_V9_MASTER_RECEIPT.csv.gz','rt') as f:
 delta=list(csv.DictReader(f))
assert len(delta)==7398
assert all(x['evidence_policy']=='DERIVED_IMPORTER_RECEIPT_ONLY__NO_SOURCE_BYTE_PARITY_OR_HEAD_ADMISSION' for x in delta)
for d in delta:
 assert d['source_key'] in masters and d['import_receipt_sha256']==masters[d['source_key']]['imported_receipt_sha']

with open(out/'CORE370_V5_VS_PARTIAL_FAMILY_RECONCILIATION_370.csv') as f:
 reconciliation=list(csv.DictReader(f))
assert len(reconciliation)==370 and len({r['source_key'] for r in reconciliation})==370
with open(out/'MASTER1527_MISSING_FULL_IMPORT_RECEIPT_QUEUE.csv') as f:
 missing=list(csv.DictReader(f))
assert len(missing)==1527 and len({r['source_key'] for r in missing})==1527
assert not set(r['source_key'] for r in missing)&set(masters)

artifacts=[v9src,state9,val9,out/'FACET_DELTA_V9_MASTER_RECEIPT.csv.gz',out/'MASTER_CORE_INDEPENDENT_RECEIPT_COURTS_V9.sqlite',out/'CORE370_V5_VS_PARTIAL_FAMILY_RECONCILIATION_370.csv',out/'MASTER1527_MISSING_FULL_IMPORT_RECEIPT_QUEUE.csv']
sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
result={'schema':'RealSaS.TruthFoundryIndependentReadback.v9',
 'source_rows_validated':counts['sources'],'facet_columns':264,'original_facet_cells_preserved':original_preserved,
 'new_observed_cells':sum(changed.values()),'new_observations_by_facet':dict(changed),
 'master_full_import_receipt_sha_crosswalk':counts['master'], 'core_partial_receipt_sha_crosswalk':counts['core'],
 'untouched_other_source_rows':counts['unmodified_other'],'master_mesh_objects_recount':14351,
 'core_family_rows_recount':5180,'core_diagnostic_model_rows_recount':1480,
 'core_reconciliation_rows':len(reconciliation), 'master_missing_deep_receipt_queue_rows':len(missing),
 'source_original_sha_rehash_pass_minted':0,'new_training_admission':0,'new_anatomical_or_2d_art_court_pass':0,
 'new_lineage_split_authority':0,'global_sdf_pass_minted':0,
 'integrity':'PASS_ALL_INDEPENDENT_RECOUNTS__EVIDENCE_ONLY', 'artifact_sha256':sha}
(out/'INDEPENDENT_READBACK_COURT_V9.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2)[:6000]);conn.close()