#!/usr/bin/env python3
"""Evidence-authority-preserving 16,397x264 V20->V21 sparse matrix overlay.

Adds independently source-locked normalized derivative numeric courts ONLY. Excludes
original rights, anatomical semantics, motion and training acceptance.
"""
from __future__ import annotations
import csv,gzip,hashlib,io,json,os,pathlib,zipfile,collections
ROOT=pathlib.Path(os.environ.get('REALSAS_TRUTH_ROOT','/mnt/data'))
OUT=ROOT/'realsas_truth_v21'
BASE=ROOT/'TruthFoundry_V20_190_NPZ_Evidence.zip'
STATE='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
FACC={
  'source_vertex_face_count':'NPZ canonicalized vertex+face counts, not original polygon census',
  'joint_count_vs_anatomical_joint':'NPZ typed joint and parent array count only; anatomical meaning unverified',
  'skin_weight_normalization':'NPZ sum over typed dense skin weight rows, not original skin',
  'zero_weight_vertices_and_unbound_mesh':'NPZ zero-weight vertices, not original unbound mesh verification',
  'overweight_negative_nonfinite_influences':'NPZ finite, negative and >1 influences; not original skin parity',
  'skin_sparse_influence_index_weight':'NPZ positive influence cell counts only; semantic influence unknown',
  'boundary_loops_and_open_surfaces':'NPZ boundary edge count, not full boundary-loop truth',
  'nonmanifold_and_self_intersection':'NPZ nonmanifold edge count; does NOT check self-intersection',
  'degenerate_sliver_zero_area_faces':'NPZ same-index and area-threshold degenerates; sliver policy not determined'
}
WEAKER='OBSERVED_HISTORICAL_STRUCTURE_JSON_DERIVATIVE_ONLY'

# Source receipts are local reproducible outcomes; no bytes from user originals are in this zip.
with gzip.open(OUT/'NEW_MASTER726_NPZ_COURT_V21.csv.gz','rt',encoding='utf-8',newline='') as f:
 source_rows=list(csv.DictReader(f))
receipts={r['source_key']:r for r in source_rows}
assert len(receipts)==len(source_rows)==76
assert all(r['evidence_authority']==STATE and r['original_source_byte_parity']=='False' and r['rights_training_admission']=='False' for r in source_rows)

def sha(b):return hashlib.sha256(b).hexdigest()

def open_csv_zip(z,name):return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(name)),encoding='utf-8',newline=''))

statefile=OUT/'FACET_STATES_EFFECTIVE_V21_16397x264.csv.gz'
valuefile=OUT/'FACET_VALUES_EFFECTIVE_V21_16397x264.csv.gz'
deltafile=OUT/'NPZ_FACET_DELTA_V21.csv.gz'
columns=['source_key','facet','previous_state','previous_value_sha256','new_state','value_json','npz_sha256','source_scope']
updated={};new=upgrade=unchanged=0; before_unknown=after_unknown=0; byfacet=collections.Counter();old_state=collections.Counter()
with zipfile.ZipFile(BASE) as z:
 s_iter=open_csv_zip(z,'FACET_STATES_EFFECTIVE_V20_16397x264.csv.gz')
 v_iter=open_csv_zip(z,'FACET_VALUES_EFFECTIVE_V20_16397x264.csv.gz')
 assert s_iter.fieldnames==v_iter.fieldnames and len(s_iter.fieldnames)==265
 with gzip.open(statefile,'wt',encoding='utf-8',newline='') as s_out,gzip.open(valuefile,'wt',encoding='utf-8',newline='') as v_out,gzip.open(deltafile,'wt',encoding='utf-8',newline='') as d_out:
  sw=csv.DictWriter(s_out,fieldnames=s_iter.fieldnames);vw=csv.DictWriter(v_out,fieldnames=v_iter.fieldnames);dw=csv.DictWriter(d_out,fieldnames=columns)
  sw.writeheader();vw.writeheader();dw.writeheader();seen=set()
  for s,v in zip(s_iter,v_iter):
   key=s['source_key'];assert key==v['source_key'] and key not in seen
   seen.add(key); r=receipts.get(key)
   for facet in s_iter.fieldnames[1:]:
    old=s[facet];prev=v[facet];before_unknown+=(old=='UNKNOWN_UNREVIEWED')
    permit=bool(r and facet in FACC and (old=='UNKNOWN_UNREVIEWED' or (old==WEAKER and facet in ('source_vertex_face_count','joint_count_vs_anatomical_joint'))))
    if permit:
     if old=='UNKNOWN_UNREVIEWED' and prev:raise ValueError('PREV_UNKNOWN_HAS_VALUE')
     actual={k:int(r[k]) for k in ('vertex_count','face_count','joint_count','skin_weight_cells','negative_weight_count','overone_weight_count','zero_skin_vertex_count','bad_nonzero_skin_row_sum_count','positive_influence_cells','max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count','same_index_triangle_count','area_below_1e_minus_12_triangle_count','zero_length_index_edge_count')}
     val=json.dumps({'source_key':key,'npz_sha256':r['npz_sha256'],'npz_drive_file_id':r['npz_drive_file_id'],'source_structure_sha256':r['structure_sha256'],'observed_scope':FACC[facet],'numeric_metrics':actual,'previous_state':old,'previous_value_sha256':sha(prev.encode()) if prev else '', 'creator_original_byte_parity':False,'semantic_truth':False,'training_admission':False,'2d_front_back_parity':False,'animation_ready_tessa':False},sort_keys=True,ensure_ascii=False,separators=(',',':'))
     dw.writerow({'source_key':key,'facet':facet,'previous_state':old,'previous_value_sha256':sha(prev.encode()) if prev else '','new_state':STATE,'value_json':val,'npz_sha256':r['npz_sha256'],'source_scope':FACC[facet]})
     s[facet]=STATE;v[facet]=val;byfacet[facet]+=1;old_state[old]+=1;updated[key]=updated.get(key,0)+1
     if old=='UNKNOWN_UNREVIEWED':new+=1
     else:upgrade+=1
    else:unchanged+=1
    after_unknown+=(s[facet]=='UNKNOWN_UNREVIEWED')
   sw.writerow(s);vw.writerow(v)
  if len(seen)!=16397 or next(s_iter,None) is not None or next(v_iter,None) is not None:raise RuntimeError('BASE_ROW_ALIGNMENT_DRIFT')
assert len(updated)==76 and new+upgrade==sum(updated.values())
assert unchanged+new+upgrade==16397*264 and before_unknown-after_unknown==new
summary={'schema':'RealSaS.TruthFoundryV21NumericDerivativeOverlay.v1','source_rows':16397,'facet_count':264,'baseline_numeric_courts':190,'new_npz_courts':76,'total_numeric_courts':266,'master726_numeric_percent':round(266/726*100,3),'whole_corpus_numeric_percent':round(266/16397*100,3),'new_unknown_cells_observed':new,'historical_structure_cells_upgraded':upgrade,'matrix_cells_unchanged':unchanged,'baseline_unknown':before_unknown,'effective_unknown':after_unknown,'by_facet':dict(byfacet),'old_states_upgraded':dict(old_state),'new_source_keys':len(updated),'training_admissions':0,'source_original_parity':0,'v20_evidence_zip_sha256':sha(BASE.read_bytes()),'state_sha256':sha(statefile.read_bytes()),'value_sha256':sha(valuefile.read_bytes()),'delta_sha256':sha(deltafile.read_bytes())}
(OUT/'SUMMARY_OVERLAY_V21.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))