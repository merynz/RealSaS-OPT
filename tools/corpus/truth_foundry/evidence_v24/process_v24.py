#!/usr/bin/env python3
"""Truth Foundry V24: fail-closed bulk normalized-derivative source court + sparse 264-facet overlay.

Private raw NPZ files remain outside GitHub. No source-original, semantic, rights,
front/back camera, motion, editability or training authority is granted here.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, importlib.util, io, json, pathlib, zipfile

METRIC_FIELDS=('vertex_count','face_count','joint_count','skin_weight_cells',
 'negative_weight_count','overone_weight_count','zero_skin_vertex_count',
 'bad_nonzero_skin_row_sum_count','positive_influence_cells',
 'max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count',
 'same_index_triangle_count','area_below_1e_minus_12_triangle_count',
 'zero_length_index_edge_count')
FACC={
 'source_vertex_face_count':'Normalized NPZ vertex and triangle counts, not original mesh parity',
 'joint_count_vs_anatomical_joint':'Normalized parent array and joint count, no anatomy verification',
 'skin_weight_normalization':'Normalized NPZ skin-weight row sums only',
 'zero_weight_vertices_and_unbound_mesh':'Normalized NPZ zero-sum skin rows; no original unbound mesh verification',
 'overweight_negative_nonfinite_influences':'Normalized NPZ finite/range skin-weight tests',
 'skin_sparse_influence_index_weight':'Normalized NPZ positive influence counts, no semantic ownership',
 'boundary_loops_and_open_surfaces':'Normalized NPZ boundary-edge count; not boundary-loop supervision',
 'nonmanifold_and_self_intersection':'Normalized NPZ nonmanifold-edge count; self-intersection not tested',
 'degenerate_sliver_zero_area_faces':'Normalized NPZ coincident indices and tiny triangles; sliver policy not established',
}
STATE='OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY'
WEAKER='OBSERVED_HISTORICAL_STRUCTURE_JSON_DERIVATIVE_ONLY'

def digest(path: pathlib.Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def ziprows(z,member):
 return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(member)),encoding='utf-8',newline=''))

def load_allowlist(source: pathlib.Path):
 with zipfile.ZipFile(source) as z:
  rows=list(ziprows(z,'MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz'))
 if len(rows)!=726:raise ValueError('ALLOWLIST_CARDINALITY')
 from re import fullmatch
 p={r['relative_folder_under_variants']:r for r in rows}
 if len(p)!=726 or any(not fullmatch('objaverse_animated_originals_[0-9a-f]{64}',k) for k in p):raise ValueError('ALLOWLIST_IDENTITY')
 for r in rows:
  if r['creator_original_byte_parity']!='False' or r['rights_training_admission']!='False':raise ValueError('ILLEGAL_ADMISSION_PROMOTION')
 return p

def load_previously_courted(v20,v21,v22,v23):
 with zipfile.ZipFile(v20) as z: p20=list(ziprows(z,'MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz'))
 with zipfile.ZipFile(v21) as z: p21=list(ziprows(z,'NEW_MASTER726_NPZ_COURT_V21.csv.gz'))
 with zipfile.ZipFile(v22) as z: p22=list(ziprows(z,'NEW_MASTER726_NPZ_COURT_V22.csv.gz'))
 with zipfile.ZipFile(v23) as z: p23=list(ziprows(z,'NEW_MASTER726_NPZ_COURT_V23.csv.gz'))
 previous={r['folder_name'] for r in (p20+p21+p22+p23)}
 if len(p20)!=190 or len(p21)!=76 or len(p22)!=68 or len(p23)!=32 or len(previous)!=366:raise ValueError('PREVIOUS_COURT_CARDINALITY_DRIFT')
 if any(r.get('training_admission')=='True' or r.get('rights_training_admission')=='True' for r in p20+p21+p22):raise ValueError('PREVIOUS_AUTHORITY_DRIFT')
 return previous

def read_manifest(path,allow,previous):
 with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 out={}
 for r in rows:
  k=r['source_key']
  if k not in allow or k in previous or k in out:raise ValueError('SOURCE_NOT_ELIGIBLE_OR_DUPLICATE:'+k)
  if not r.get('npz_id') or not r.get('structure_id') or int(r['npz_size'])<=0 or int(r['structure_size'])<=0:raise ValueError('BAD_FILE_METADATA')
  out[k]=r
 if not out:raise ValueError('EMPTY_MANIFEST')
 return out

def run_court(args):
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 allow=load_allowlist(args.v17);previous=load_previously_courted(args.v20,args.v21,args.v22,args.v23)
 manifest=read_manifest(args.manifest,allow,previous)
 spec=importlib.util.spec_from_file_location('realsas_v20_numeric',args.numeric_module)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 actual={p.stem for p in args.source_dir.glob('objaverse_animated_originals_*.npz')}
 if actual!=set(manifest):raise ValueError('SOURCE_CACHE_MANIFEST_DISAGREEMENT:'+str((len(actual),len(manifest))))
 records=[];rejected=[]
 for name,r in sorted(manifest.items()):
  npz=args.source_dir/(name+'.npz'); stp=args.structure_dir/(name+'.json')
  try:
   if not stp.is_file() or npz.stat().st_size!=int(r['npz_size']) or stp.stat().st_size!=int(r['structure_size']):raise ValueError('MISSING_OR_WRONG_SIZE')
   with stp.open(encoding='utf-8') as f:st=json.load(f)
   mesh=st.get('mesh_objects');bone=st.get('bone_names')
   if not isinstance(mesh,list) or not mesh or not isinstance(bone,list) or len(bone)<1 or st.get('bone_count')!=len(bone):raise ValueError('BAD_STRUCTURE')
   start=0; nf=0
   for obj in mesh:
    if not all(isinstance(obj.get(k),int) and obj[k]>=0 for k in ('vertex_start','vertex_count','face_count')):raise ValueError('BAD_MESH_OBJECT')
    if obj['vertex_start']!=start:raise ValueError('SOURCE_MESH_OFFSETS_DISJOINT_OR_NONCONTIGUOUS')
    start+=obj['vertex_count'];nf+=obj['face_count']
   if start<=0 or nf<=0 or name.rsplit('_',1)[-1] not in st.get('source',''):raise ValueError('STRUCTURE_ORIGINAL_ID_OR_COUNTS')
   with zipfile.ZipFile(npz) as z:
    if z.testzip() is not None:raise ValueError('ZIP_CRC_FAILURE')
   struct=['STRUCT',name,'READ_STRUCTURE',str(len(mesh)),str(start),str(nf),str(len(bone)),str(len(bone)),'readback']
   metrics=mod.numeric_court(npz,struct)
   row={**metrics,'source_key':allow[name]['source_key'],'candidate_id':allow[name]['candidate_id'],
    'folder_name':name,'npz_sha256':digest(npz),'structure_sha256':digest(stp),
    'npz_bytes':npz.stat().st_size,'structure_bytes':stp.stat().st_size,
    'npz_drive_file_id':r['npz_id'],'structure_drive_file_id':r['structure_id'],
    'historical_receipt_source_sha256':allow[name]['historical_recovery_sha256'],
    'evidence_authority':STATE,'creator_original_byte_parity':False,'rights_training_admission':False}
   records.append(row)
  except Exception as e:rejected.append({'folder_name':name,'error_reason':str(e)[:250]})
 with (out/'QUARANTINE_V24.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['folder_name','error_reason']);w.writeheader();w.writerows(rejected)
 if rejected or len(records)!=len(manifest):raise RuntimeError('PHYSICAL_COURT_INCOMPLETE:'+str((len(records),len(rejected))))
 field=['source_key','candidate_id','folder_name','npz_sha256','structure_sha256','npz_bytes','structure_bytes','npz_drive_file_id','structure_drive_file_id','historical_receipt_source_sha256','status','vertex_count','face_count','joint_count','skin_weight_cells','parent_graph_valid','negative_weight_count','overone_weight_count','zero_skin_vertex_count','bad_nonzero_skin_row_sum_count','max_skin_row_sum_error_nonzero','positive_influence_cells','max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count','same_index_triangle_count','area_below_1e_minus_12_triangle_count','zero_length_index_edge_count','mesh_objects_reported_only','quality_has_irregularities','evidence_authority','creator_original_byte_parity','rights_training_admission']
 with gzip.open(out/'NEW_MASTER726_NPZ_COURT_V24.csv.gz','wt',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=field,extrasaction='ignore');w.writeheader();w.writerows(sorted(records,key=lambda r:r['source_key']))
 stats=collections.Counter()
 for r in records:
  for k in METRIC_FIELDS:
   if k=='max_positive_influences_per_vertex':stats[k]=max(stats[k],int(r[k]))
   else:stats[k]+=int(r[k])
 summary={'schema':'RealSaS.V24.NormalizedDerivativeCourt.v1','prior_sources':len(previous),'new_numeric_pass':len(records),'quarantined':len(rejected), 'combined_numeric_pass':len(previous)+len(records),'eligible_master':726,'candidate_universe':16397,'eligible_percent':round(100*(len(previous)+len(records))/726,3),'all_candidates_percent':round(100*(len(previous)+len(records))/16397,3),'numeric_totals_new':dict(stats),'training_admissions':0,'creator_original_parity':0,'v17_sha256':digest(args.v17),'v23_sha256':digest(args.v23),'manifest_sha256':digest(args.manifest),'court_receipt_sha256':digest(out/'NEW_MASTER726_NPZ_COURT_V24.csv.gz')}
 (out/'SUMMARY_NUMERIC_V24.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 return records,summary

def run_overlay(args,records,summary):
 out=args.output;delta=[];updated=set();counts=collections.Counter()
 receipt={r['source_key']:r for r in records}
 with zipfile.ZipFile(args.v23) as z:
  states=ziprows(z,'FACET_STATES_EFFECTIVE_V23_16397x264.csv.gz')
  values=ziprows(z,'FACET_VALUES_EFFECTIVE_V23_16397x264.csv.gz')
  if states.fieldnames!=values.fieldnames or len(states.fieldnames)!=265:raise ValueError('BASE_FACET_CONTRACT')
  with gzip.open(out/'FACET_STATES_EFFECTIVE_V24_16397x264.csv.gz','wt',newline='',encoding='utf-8') as sf,gzip.open(out/'FACET_VALUES_EFFECTIVE_V24_16397x264.csv.gz','wt',newline='',encoding='utf-8') as vf,gzip.open(out/'NPZ_FACET_DELTA_V24.csv.gz','wt',newline='',encoding='utf-8') as df:
   sw=csv.DictWriter(sf,fieldnames=states.fieldnames);vw=csv.DictWriter(vf,fieldnames=states.fieldnames)
   dfcols=['source_key','facet','previous_state','previous_value_sha256','new_state','value_json','npz_sha256','source_scope']
   dw=csv.DictWriter(df,fieldnames=dfcols);sw.writeheader();vw.writeheader();dw.writeheader()
   seen=set()
   for ss,vv in zip(states,values):
    key=ss['source_key']
    if key!=vv['source_key'] or key in seen:raise ValueError('BASE_SOURCE_ORDER_OR_DUPLICATE')
    seen.add(key);r=receipt.get(key)
    for facet in states.fieldnames[1:]:
     old=ss[facet];prev=vv[facet]
     allow=bool(r and facet in FACC and (old=='UNKNOWN_UNREVIEWED' or (old==WEAKER and facet in ('source_vertex_face_count','joint_count_vs_anatomical_joint'))))
     if allow:
      if old=='UNKNOWN_UNREVIEWED' and prev:raise ValueError('NONEMPTY_UNKNOWN_VALUE')
      val=json.dumps({'source_key':key,'npz_sha256':r['npz_sha256'],'npz_drive_file_id':r['npz_drive_file_id'],'structure_sha256':r['structure_sha256'],'observed_scope':FACC[facet], 'numeric_metrics':{name:int(r[name]) for name in METRIC_FIELDS},'previous_state':old,'previous_value_sha256':hashlib.sha256(prev.encode()).hexdigest() if prev else '','creator_original_byte_parity':False,'training_admission':False,'rights_admission':False,'semantic_truth':False,'2d_front_back_parity':False,'animation_ready_tessa':False},ensure_ascii=False,separators=(',',':'),sort_keys=True)
      dw.writerow({'source_key':key,'facet':facet,'previous_state':old,'previous_value_sha256':hashlib.sha256(prev.encode()).hexdigest() if prev else '','new_state':STATE,'value_json':val,'npz_sha256':r['npz_sha256'],'source_scope':FACC[facet]})
      ss[facet]=STATE;vv[facet]=val;updated.add(key);counts['modified']+=1
      if old=='UNKNOWN_UNREVIEWED':counts['new_unknown']+=1
      else:counts['upgraded_weaker']+=1
     else:counts['unchanged']+=1
    sw.writerow(ss);vw.writerow(vv)
   if len(seen)!=16397 or next(states,None) is not None or next(values,None) is not None:raise ValueError('SOURCE_ROW_COUNT_DRIFT')
 if len(updated)!=len(records):raise ValueError('EVERY_NEW_SOURCE_MUST_HAVE_EVIDENCE')
 if counts['unchanged']+counts['modified']!=16397*264:raise ValueError('FULL_MATRIX_ACCOUNTING')
 overlay={'schema':'RealSaS.V24.SparseNumericOverlay.v1','source_rows':16397,'facet_count':264,'new_sources':len(updated),'new_unknown_cells':counts['new_unknown'],'upgraded_historic_structure_cells':counts['upgraded_weaker'],'unchanged_cells':counts['unchanged'],'total_cells':16397*264,'training_admissions':0,'source_original_parity':0,'v23_sha256':digest(args.v23)}
 for name in ('FACET_STATES_EFFECTIVE_V24_16397x264.csv.gz','FACET_VALUES_EFFECTIVE_V24_16397x264.csv.gz','NPZ_FACET_DELTA_V24.csv.gz'):overlay[name+'_sha256']=digest(out/name)
 (out/'SUMMARY_OVERLAY_V24.json').write_text(json.dumps(overlay,indent=2,sort_keys=True)+'\n')
 return overlay

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/mnt/data'));p.add_argument('--expected-new',type=int,default=2);args=p.parse_args()
 rt=args.root;out=rt/'realsas_v24'
 from types import SimpleNamespace
 opts=SimpleNamespace(output=out,source_dir=out/'sources',structure_dir=out/'structures',manifest=out/'transfer_manifest_v24.csv',v17=rt/'TruthFoundry_V17_Bulk_16397_Evidence.zip',v20=rt/'TruthFoundry_V20_190_NPZ_Evidence.zip',v21=rt/'TruthFoundry_V21_266_NPZ_Evidence.zip',v22=rt/'TruthFoundry_V22_334_NPZ_Evidence.zip',v23=rt/'TruthFoundry_V23_366_NPZ_Evidence.zip',numeric_module=rt/'realsas_v24/numeric_v20.py')
 records,summary=run_court(opts)
 if len(records)!=args.expected_new:raise ValueError('UNEXPECTED_NEW_COUNT')
 overlay=run_overlay(opts,records,summary)
 print(json.dumps({'court':summary,'overlay':overlay},indent=2,ensure_ascii=False))
if __name__=='__main__':main()