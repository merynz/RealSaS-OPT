#!/usr/bin/env python3
"""RealSaS Truth Foundry V20: actual SHA-256, geometry and skin array court.

Source namespace: strict V17 Master726 derivative-only allowlist.
Inputs: locally materialized NPZ bytes and historical source_structure JSON
receipt index. No creator-original bytes, permissions, render or training approval.
Existing V19 state/value pairs are immutable inputs; changes are explicit
scoped upgrades in *new* V20 outputs with exact per-cell delta provenance.
"""
from __future__ import annotations
import collections, csv, gzip, hashlib, io, json, os, pathlib, re, zipfile
import numpy as np

ROOT=pathlib.Path(os.environ.get('REALSAS_TRUTH_ROOT','/mnt/data')).resolve()
OUT=ROOT/'realsas_truth_v20';OUT.mkdir(exist_ok=True)
SRC=OUT/'sources'
DOC=ROOT/'RealSaS Truth Foundry | V19 726-source structure and NPZ inventory courts.txt'
PLAN=ROOT/'realsas_truth_v17'/'MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz'
PRE=ROOT/'realsas_truth_v19'
INPUT_S=PRE/'FACET_STATES_EFFECTIVE_V19_16397x264.csv.gz'
INPUT_V=PRE/'FACET_VALUES_EFFECTIVE_V19_16397x264.csv.gz'
EXPECTED_PRE={
  'states':'28db3c5dd8e4d65540c791dcc30f07b4af2b6d7404c4a458b6cc5365f31c6db3',
  'values':'823ac80beab9fd74866251daf88adeae744d72c50d154a6461304f716b244c6f',
  'plan':'f9c3a1a98bbf794f9514ae2ad547f8051b503ccea4b058a73e6f1823884cd0bc',
}
STATE='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
OLD_WEAKER='OBSERVED_HISTORICAL_STRUCTURE_JSON_DERIVATIVE_ONLY'
FACC={
  'source_vertex_face_count':'NPZ canonicalized vertex and triangle counts; NOT creator-original polygon census',
  'joint_count_vs_anatomical_joint':'NPZ joint/parent array count; anatomical identity unknown',
  'skin_weight_normalization':'NPZ complete vertex weight-row sum court; creator-original skin not checked',
  'zero_weight_vertices_and_unbound_mesh':'NPZ zero-weight vertex count; other original unbound meshes unknown',
  'overweight_negative_nonfinite_influences':'NPZ finite / negative / over-one full-array metrics; original influences unknown',
  'skin_sparse_influence_index_weight':'NPZ count of positive influences per vertex; no anatomical semantics',
  'boundary_loops_and_open_surfaces':'NPZ boundary edge count; loops and editable drawables unknown',
  'nonmanifold_and_self_intersection':'NPZ >2-incident-face edge count; self-intersection not tested',
  'degenerate_sliver_zero_area_faces':'NPZ zero-edge and area-threshold degenerates; sliver quality not measured',
}

def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def parse_doc(p):
 rows=[re.split(r' {8,}',s.lstrip('\ufeff')) for s in p.read_text(encoding='utf-8-sig').splitlines()]
 meta={};last={};counts=collections.Counter()
 for r in rows:
  if len(r)>=10 and r[0]=='META':
   if r[1] in meta:raise ValueError('DUPLICATE_META:'+r[1])
   meta[r[1]]=r
  if len(r)>=9 and r[0]=='STRUCT':
   last[r[1]]=r;counts[r[2]]+=1
 return meta,last,dict(counts)

def strict_plan():
 p={}
 with gzip.open(PLAN,'rt',newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   name=r['relative_folder_under_variants']
   if name in p:raise ValueError('DUPLICATE_SOURCE_PLAN')
   if not re.fullmatch(r'objaverse_animated_originals_[0-9a-f]{64}',name):raise ValueError('PATH_NOT_ALLOWED')
   if not re.fullmatch(r'[0-9a-f]{64}',r['historical_recovery_sha256']):raise ValueError('INVALID_PLAN_SHA')
   if r['creator_original_byte_parity']!='False' or r['rights_training_admission']!='False': raise ValueError('UNEXPECTED_SOURCE_AUTHORITY')
   p[name]=r
 if len(p)!=726:raise ValueError('PLAN_SIZE_DRIFT')
 return p

def parent_contract(parents):
 if parents.ndim!=1 or not np.issubdtype(parents.dtype,np.integer):return False,0
 J=len(parents);p=parents.astype(np.int64)
 if np.any((p<-1)|(p>=J)):return False,int(np.sum((p<-1)|(p>=J)))
 bad=0
 for i in range(J):
  seen=set();x=i
  while x!=-1:
   if x in seen:bad+=1;break
   seen.add(x);x=int(p[x])
 return bad==0,bad

def numeric_court(path,structure):
 # Avoid pickle and prohibit large unexpected archive members (zip bombs).
 if path.stat().st_size>400*1024*1024:raise ValueError('NPZ_COMPRESSED_BYTES_LIMIT')
 with zipfile.ZipFile(path,'r') as z:
  items=z.infolist()
  if not items or len(items)>80 or len({x.filename for x in items})!=len(items):raise ValueError('ZIP_ENTRIES')
  if any(not x.filename.endswith('.npy') or '/' in x.filename or x.file_size>1024*1024*1024 for x in items):raise ValueError('ZIP_NON_NPY_OR_OVERSIZED')
  if sum(x.file_size for x in items)>2*1024*1024*1024:raise ValueError('ZIP_EXPANSION_LIMIT')
 with np.load(path,allow_pickle=False) as npz:
  req=('vertices','vertices_source','faces','vertex_normals','parents','skin','bone_heads','bone_tails','rest_local_source','rest_world_source')
  if not all(k in npz.files for k in req):raise ValueError('MISSING_TYPED_ARRAY')
  v,vs,fac,norm,parent,skin=npz['vertices'],npz['vertices_source'],npz['faces'],npz['vertex_normals'],npz['parents'],npz['skin']
  j=len(parent)
  if not (v.ndim==2 and v.shape[1]==3 and vs.shape==v.shape and norm.shape==v.shape):raise ValueError('VERTEX_NORMAL_SHAPE')
  if not (fac.ndim==2 and fac.shape[1]==3 and np.issubdtype(fac.dtype,np.integer)):raise ValueError('FACE_SHAPE')
  if not (skin.shape==(len(v),j) and j>0):raise ValueError('SKIN_SHAPE')
  if not all(npz[k].shape==(j,3) for k in ('bone_heads','bone_tails')):raise ValueError('JOINT_POINT_SHAPE')
  if not all(npz[k].shape==(j,4,4) for k in ('rest_local_source','rest_world_source')):raise ValueError('REST_TRANSFORM_SHAPE')
  if len(v)<3 or len(fac)<1:raise ValueError('EMPTY_TRIANGLE_MODEL')
  if not all(np.isfinite(arr).all() for arr in (v,vs,norm,skin,npz['bone_heads'],npz['bone_tails'],npz['rest_local_source'],npz['rest_world_source'])):raise ValueError('NONFINITE_TYPED_ARRAY')
  if np.any(fac<0) or np.any(fac>=len(v)):raise ValueError('TRIANGLE_INDEX_OUT_OF_RANGE')
  parent_ok,parent_errors=parent_contract(parent)
  if not parent_ok:raise ValueError('PARENT_CYCLE_OR_OUT_OF_RANGE:'+str(parent_errors))
  nobj=int(structure[3]);expected=(int(structure[4]),int(structure[5]),int(structure[6]));actual=(len(v),len(fac),j)
  if actual!=expected:raise ValueError('SOURCE_STRUCTURE_COUNT_PARITY')
  f=fac.astype(np.int64,copy=False)
  tri=v[f].astype(np.float64,copy=False)
  tri_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*0.5
  repeated=(f[:,0]==f[:,1]) | (f[:,1]==f[:,2]) | (f[:,2]==f[:,0]);zero_area=(tri_area<=1e-12)
  # Edges counted in canonicalized-index NPZ, not original mesh ownership.
  edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges.sort(axis=1)
  e=np.ascontiguousarray(edges,dtype=np.int64).view(np.dtype([('u','<i8'),('v','<i8')])).ravel()
  _,multiplicity=np.unique(e,return_counts=True)
  boundary=int(np.count_nonzero(multiplicity==1));nonmanifold=int(np.count_nonzero(multiplicity>2));zero_edges=int(np.count_nonzero(edges[:,0]==edges[:,1]))
  row_sum=np.sum(skin.astype(np.float64),axis=1);positive=(skin>1e-7).sum(axis=1)
  negative=int(np.count_nonzero(skin<-1e-7));overone=int(np.count_nonzero(skin>1+1e-5))
  bad_sum=int(np.count_nonzero((np.abs(row_sum-1)>1e-3)&(row_sum>1e-6)))
  zero_skin=int(np.count_nonzero(row_sum<=1e-6));nonzero=int(np.count_nonzero(skin>1e-7))
  return {
   'status':'PASS_NUMERIC_DERIVATIVE_COURT_WITH_QUALITY_COUNTS',
   'mesh_objects_reported_only':nobj,
   'vertex_count':len(v),'face_count':len(fac),'joint_count':j,'skin_weight_cells':int(skin.size),
   'parent_graph_valid':True,
   'negative_weight_count':negative,'overone_weight_count':overone,'zero_skin_vertex_count':zero_skin,
   'bad_nonzero_skin_row_sum_count':bad_sum,'max_skin_row_sum_error_nonzero':float(np.max(np.abs(row_sum[row_sum>1e-6]-1),initial=0)),
   'positive_influence_cells':nonzero,'max_positive_influences_per_vertex':int(positive.max()),
   'boundary_edge_count':boundary,'nonmanifold_edge_count':nonmanifold,
   'same_index_triangle_count':int(repeated.sum()),'area_below_1e_minus_12_triangle_count':int(zero_area.sum()),
   'zero_length_index_edge_count':zero_edges,
   'quality_has_irregularities':bool(negative or overone or zero_skin or bad_sum or nonmanifold or zero_area.any()),
  }

def main():
 for p,k in [(INPUT_S,'states'),(INPUT_V,'values'),(PLAN,'plan')]:
  if sha_file(p)!=EXPECTED_PRE[k]:raise RuntimeError('BASELINE_SHA_DRIFT:'+k)
 plan=strict_plan();meta,structure,attempts=parse_doc(DOC)
 receipts={};errors={};names=set()
 for p in sorted(SRC.glob('objaverse_animated_originals_*.npz')):
  name=p.stem
  if name not in plan:raise RuntimeError('UNAUTHORIZED_NON_ALLOWLISTED_SOURCE:'+name)
  if name in names:raise RuntimeError('DUPLICATE_SOURCE')
  names.add(name)
  m,st=meta.get(name),structure.get(name)
  if not m or not st or st[2]!='READ_STRUCTURE':errors[name]='SOURCE_STRUCTURE_METADATA_MISSING';continue
  if m[3]!='1' or m[4]!='1' or m[5]=='' or m[7]=='':errors[name]='REQUIRED_DRIVE_FILES_MISSING';continue
  if p.stat().st_size!=int(m[6]):errors[name]='DRIVE_MANIFEST_NPZ_SIZE_MISMATCH';continue
  if len(st)<9 or int(st[6])!=int(st[7]):errors[name]='SOURCE_STRUCTURE_BONE_NAMES_MISMATCH';continue
  h=sha_file(p)
  if plan[name]['pilot_v16_npz_rehashed']=='True' and plan[name]['pilot_v16_npz_sha256'] and h!=plan[name]['pilot_v16_npz_sha256']:
   errors[name]='V16_PILOT_SHA_MISMATCH';continue
  try:
   n=numeric_court(p,st)
  except Exception as e:
   errors[name]='NUMERIC_COURT_BLOCKED:'+str(e);continue
  r=plan[name]
  receipts[r['source_key']]={**n,'source_key':r['source_key'],'candidate_id':r['candidate_id'],
   'folder_name':name,'npz_sha256':h,'npz_bytes':p.stat().st_size,
   'npz_drive_file_id':m[5],'source_structure_drive_file_id':m[7],
   'historical_receipt_source_sha256':r['historical_recovery_sha256'],
   'creator_original_bytes_courted':False,'rights_admission':False,'training_admission':False,
   'evidence_authority':STATE}
 if not receipts:raise RuntimeError('NO_REAL_NUMERIC_SOURCE_RECEIPTS')
 field=['source_key','candidate_id','folder_name','npz_sha256','npz_bytes','npz_drive_file_id','source_structure_drive_file_id','historical_receipt_source_sha256',
  'status','vertex_count','face_count','joint_count','skin_weight_cells','parent_graph_valid','negative_weight_count','overone_weight_count',
  'zero_skin_vertex_count','bad_nonzero_skin_row_sum_count','max_skin_row_sum_error_nonzero','positive_influence_cells',
  'max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count','same_index_triangle_count',
  'area_below_1e_minus_12_triangle_count','zero_length_index_edge_count','quality_has_irregularities',
  'mesh_objects_reported_only','creator_original_bytes_courted','rights_admission','training_admission','evidence_authority']
 with gzip.open(OUT/'MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz','wt',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=field);w.writeheader();w.writerows(sorted(receipts.values(),key=lambda x:x['source_key']))
 with (OUT/'NPZ_QUARANTINE_V20.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['folder_name','error_reason']);w.writerows(sorted(errors.items()))
 delta=[];unchanged=0;before_unknown=after_unknown=0;upgrade=0;newcells=0;byfacet=collections.Counter();skipped=collections.Counter()
 with gzip.open(INPUT_S,'rt',newline='',encoding='utf-8') as sf,gzip.open(INPUT_V,'rt',newline='',encoding='utf-8') as vf,\
   gzip.open(OUT/'FACET_STATES_EFFECTIVE_V20_16397x264.csv.gz','wt',newline='',encoding='utf-8') as ns,\
   gzip.open(OUT/'FACET_VALUES_EFFECTIVE_V20_16397x264.csv.gz','wt',newline='',encoding='utf-8') as nv,\
   gzip.open(OUT/'NPZ_FACET_DELTA_V20.csv.gz','wt',newline='',encoding='utf-8') as df:
  si,vi=csv.DictReader(sf),csv.DictReader(vf)
  assert si.fieldnames==vi.fieldnames and len(si.fieldnames)==265
  so,vo=csv.DictWriter(ns,fieldnames=si.fieldnames),csv.DictWriter(nv,fieldnames=vi.fieldnames)
  cols=['source_key','facet','previous_state','previous_value_sha256','new_state','value_json','npz_sha256','npz_bytes','source_scope']
  dw=csv.DictWriter(df,fieldnames=cols)
  for w in [so,vo,dw]:w.writeheader()
  seen=set()
  for s,v in zip(si,vi):
   k=s['source_key'];assert k==v['source_key'] and k not in seen;seen.add(k)
   r=receipts.get(k)
   for facet in si.fieldnames[1:]:
    old=s[facet];ov=v[facet];before_unknown+=(old=='UNKNOWN_UNREVIEWED')
    should_change=bool(r and facet in FACC and (old=='UNKNOWN_UNREVIEWED' or (old==OLD_WEAKER and facet in ('source_vertex_face_count','joint_count_vs_anatomical_joint'))))
    if should_change:
     if old=='UNKNOWN_UNREVIEWED' and ov:raise ValueError('UNKNOWN_STATE_WITH_NONEMPTY_VALUE')
     core={x:r[x] for x in ('vertex_count','face_count','joint_count','skin_weight_cells','negative_weight_count','overone_weight_count',
      'zero_skin_vertex_count','bad_nonzero_skin_row_sum_count','max_skin_row_sum_error_nonzero','positive_influence_cells',
      'max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count',
      'same_index_triangle_count','area_below_1e_minus_12_triangle_count','zero_length_index_edge_count','parent_graph_valid')}
     value=json.dumps({'npz_sha256':r['npz_sha256'],'npz_drive_file_id':r['npz_drive_file_id'],
      'npz_bytes':r['npz_bytes'],'source_key':k,'observed_scope':FACC[facet],
      'npz_numeric_metrics':core,'previous_state':old,
      'previous_value_sha256':hashlib.sha256(ov.encode()).hexdigest() if ov else '',
      'original_source_byte_parity':False,'semantic_truth':False,
      'rights_training_admission':False,'motion_ready_tessa_admission':False},sort_keys=True,separators=(',',':'),ensure_ascii=False)
     delta.append({'source_key':k,'facet':facet,'previous_state':old,
      'previous_value_sha256':hashlib.sha256(ov.encode()).hexdigest() if ov else '',
      'new_state':STATE,'value_json':value,'npz_sha256':r['npz_sha256'],
      'npz_bytes':r['npz_bytes'],'source_scope':FACC[facet]})
     s[facet]=STATE;v[facet]=value;byfacet[facet]+=1
     if old=='UNKNOWN_UNREVIEWED':newcells+=1
     else:upgrade+=1
    else:
     unchanged+=1
     if r and facet in FACC:skipped[old]+=1
    after_unknown+=(s[facet]=='UNKNOWN_UNREVIEWED')
   so.writerow(s);vo.writerow(v)
  if len(seen)!=16397 or next(si,None)!=None or next(vi,None)!=None:raise AssertionError('MATRIX_ROWS_OR_ALIGNMENT_DRIFT')
  dw.writerows(delta)
 assert len(delta)==newcells+upgrade and unchanged==16397*264-len(delta)
 assert before_unknown-after_unknown==newcells
 o={
  'schema':'RealSaS.TruthFoundryV20RehashedNormalizedDerivativeNPZ.v1',
  'source_candidate_universe':16397,'eligible_726_count':726,'materialized_npz_count':len(names),
  'npz_court_pass':len(receipts),'npz_court_quarantined':len(errors),
  'npz_court_eligible_percent':round(100*len(receipts)/726,3),
  'npz_court_universe_percent':round(100*len(receipts)/16397,3),
  'structure_json_parity_pass':len(receipts),
  'read_vertex_count':sum(x['vertex_count'] for x in receipts.values()),
  'read_triangle_count':sum(x['face_count'] for x in receipts.values()),
  'read_skin_weight_cells':sum(x['skin_weight_cells'] for x in receipts.values()),
  'sources_with_nonmanifold_edges':sum(x['nonmanifold_edge_count']>0 for x in receipts.values()),
  'sources_with_zero_weight_vertices':sum(x['zero_skin_vertex_count']>0 for x in receipts.values()),
  'source_facet_value_changes':len(delta),'newly_nonunknown_cells':newcells,
  'upgraded_historical_structure_cells':upgrade,'untouched_cells':unchanged,
  'unknown_before':before_unknown,'unknown_after':after_unknown,
  'added_facet_counts':dict(byfacet),'not_promoted_old_states':dict(skipped),
  'npz_evidence_sha256_included':True,
  'creator_original_court':0,'training_admissions':0,'rights_admissions':0,
  'protected_holdout_accesses':0,'semantic_morphology_court':0,'front_back_art_court':0,
  'source_doc_sha256':sha_file(DOC),
  'input_sha256':{x:sha_file(p) for x,p in [('V19_states',INPUT_S),('V19_values',INPUT_V),('plan',PLAN)]},
  'output_sha256':{x:sha_file(OUT/y) for x,y in [('states','FACET_STATES_EFFECTIVE_V20_16397x264.csv.gz'),('values','FACET_VALUES_EFFECTIVE_V20_16397x264.csv.gz'),
    ('npz_receipts','MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz'),('delta','NPZ_FACET_DELTA_V20.csv.gz'),('quarantine','NPZ_QUARANTINE_V20.csv')]},
  'notes':'This is rehashed normalized derivative NPZ truth, not creator-original or training/rights/semantics/render admission.'
 }
 (OUT/'SUMMARY_V20.json').write_text(json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:o[k] for k in ('npz_court_pass','npz_court_quarantined','npz_court_eligible_percent','read_vertex_count','read_triangle_count','read_skin_weight_cells','source_facet_value_changes','newly_nonunknown_cells','upgraded_historical_structure_cells','untouched_cells','unknown_before','unknown_after')},indent=2))

if __name__=='__main__':main()