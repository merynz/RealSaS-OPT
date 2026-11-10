#!/usr/bin/env python3
"""RealSaS V21 independent derivative-only numeric court for new mounted Master726 sources.

Requires the strict SHA-bound V17 recovery queue and V19 Drive metadata manifest.
Does not promote creator-original, legality, morphology, articulation, graphics, or training claims.
"""
from __future__ import annotations
import csv, gzip, hashlib, importlib.util, json, os, pathlib, re, zipfile
from collections import Counter

ROOT=pathlib.Path(os.environ.get('REALSAS_TRUTH_ROOT','/mnt/data')).resolve()
OUT=ROOT/'realsas_truth_v21'
SOURCE=OUT/'sources'
STRUCT=OUT/'structures'
V19=ROOT/'RealSaS Truth Foundry | V19 726-source structure and NPZ inventory courts.txt'
V17=ROOT/'TruthFoundry_V17_Bulk_16397_Evidence.zip'
V20=ROOT/'TruthFoundry_V20_190_NPZ_Evidence.zip'
V20_COURT=ROOT/'realsas_truth_v20/code/build_v20.py'
COLS=['source_key','candidate_id','folder_name','npz_sha256','structure_sha256','npz_bytes','structure_bytes','npz_drive_file_id','structure_drive_file_id','historical_receipt_source_sha256','status','vertex_count','face_count','joint_count','skin_weight_cells','parent_graph_valid','negative_weight_count','overone_weight_count','zero_skin_vertex_count','bad_nonzero_skin_row_sum_count','positive_influence_cells','max_positive_influences_per_vertex','boundary_edge_count','nonmanifold_edge_count','same_index_triangle_count','area_below_1e_minus_12_triangle_count','zero_length_index_edge_count','mesh_objects_reported_only','quality_has_irregularities','evidence_authority','original_source_byte_parity','rights_training_admission']

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()

def load_plan():
 with zipfile.ZipFile(V17) as z:
  with gzip.open(z.open('MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz'),'rt',newline='') as f:
   rows=list(csv.DictReader(f))
 assert len(rows)==726
 plan={r['relative_folder_under_variants']:r for r in rows}
 assert len(plan)==726 and all(re.fullmatch(r'objaverse_animated_originals_[0-9a-f]{64}',k) for k in plan)
 assert all(r['creator_original_byte_parity']=='False' and r['rights_training_admission']=='False' for r in rows)
 return plan

def load_meta():
 result={}
 for line in V19.read_text(encoding='utf-8-sig').splitlines():
  t=re.split(r' {8,}',line)
  if len(t)>=10 and t[0]=='META':
   if t[1] in result:raise RuntimeError('DUPLICATE_META:'+t[1])
   result[t[1]]=t
 return result

def load_existing():
 with zipfile.ZipFile(V20) as z:
  with gzip.open(z.open('MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz'),'rt',newline='') as f:
   rows=list(csv.DictReader(f))
 assert len(rows)==190 and len(set(x['source_key'] for x in rows))==190
 return {r['folder_name'] for r in rows}

def main():
 plan=load_plan();meta=load_meta();existing=load_existing()
 try:
  from tools.corpus.truth_foundry.evidence_v20 import build_v20 as mod
 except ImportError:
  spec=importlib.util.spec_from_file_location('v20court',V20_COURT)
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 records=[];quarantine=[]
 files=sorted(SOURCE.glob('objaverse_animated_originals_*.npz'))
 for source in files:
  k=source.stem
  try:
   if k not in plan:raise RuntimeError('SOURCE_NOT_IN_MASTER726')
   if k in existing:raise RuntimeError('SOURCE_ALREADY_COURTED_IN_V20')
   sfile=STRUCT/(k+'.json')
   if not sfile.is_file():raise RuntimeError('MISSING_SOURCE_STRUCTURE')
   smeta=meta.get(k)
   if not smeta or smeta[3]!='1' or smeta[4]!='1':raise RuntimeError('DRIVE_ID_METADATA_MISSING')
   if source.stat().st_size!=int(smeta[6]) or sfile.stat().st_size!=int(smeta[8]):raise RuntimeError('FILE_SIZE_NOT_MATCH_DRIVE_METADATA')
   with sfile.open('r',encoding='utf-8') as f:st=json.load(f)
   mesh=st.get('mesh_objects')
   bones=st.get('bone_names')
   if not isinstance(mesh,list) or not mesh or not isinstance(bones,list) or not bones or st.get('bone_count')!=len(bones):raise RuntimeError('SOURCE_STRUCTURE_MALFORMED')
   nv=nf=0
   for x in mesh:
    if not all(isinstance(x.get(p),int) and x[p]>=0 for p in ('vertex_start','vertex_count','face_count')):raise RuntimeError('INVALID_MESH_STRUCTURE')
    if x['vertex_start']!=nv:raise RuntimeError('MESH_VERTEX_OFFSET_DRIFT')
    nv+=x['vertex_count'];nf+=x['face_count']
   if nv<3 or nf<1:raise RuntimeError('EMPTY_STRUCTURE')
   if k.split('_')[-1] not in st.get('source',''):raise RuntimeError('SOURCE_ORIGINAL_ID_MISMATCH')
   structure=['STRUCT',k,'READ_STRUCTURE',str(len(mesh)),str(nv),str(nf),str(len(bones)),str(len(bones)),'readback']
   with zipfile.ZipFile(source) as zp:
    bad=zp.testzip()
    if bad is not None:raise RuntimeError('ZIP_CRC_BAD:'+str(bad))
   metrics=mod.numeric_court(source,structure)
   p=plan[k]
   n={**metrics,'source_key':p['source_key'],'candidate_id':p['candidate_id'],'folder_name':k,
      'npz_sha256':sha(source),'structure_sha256':sha(sfile),'npz_bytes':source.stat().st_size,'structure_bytes':sfile.stat().st_size,
      'npz_drive_file_id':smeta[5],'structure_drive_file_id':smeta[7],'historical_receipt_source_sha256':p['historical_recovery_sha256'],
      'evidence_authority':'OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY',
      'original_source_byte_parity':False,'rights_training_admission':False}
   records.append(n)
  except Exception as e:
   quarantine.append({'folder_name':k,'error_reason':str(e)[:180]})
 if len(files)!=len(records)+len(quarantine):raise RuntimeError('SOURCE_COURT_ACCOUNTING_MISMATCH')
 with gzip.open(OUT/'NEW_MASTER726_NPZ_COURT_V21.csv.gz','wt',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,COLS,extrasaction='ignore');w.writeheader();w.writerows(sorted(records,key=lambda r:r['source_key']))
 with (OUT/'QUARANTINE_V21.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,['folder_name','error_reason']);w.writeheader();w.writerows(quarantine)
 c=Counter()
 for r in records:
  for field in ['vertex_count','face_count','skin_weight_cells','zero_skin_vertex_count','nonmanifold_edge_count','boundary_edge_count']:
   c[field]+=int(r[field])
 summary={'schema':'RealSaS.V21.Master726NewNormalizedDerivativeCourt.v1','input_file_count':len(files),'numeric_pass':len(records),
          'quarantined':len(quarantine),'prior_v20_numeric_pass':len(existing),
          'combined_numeric_pass':len(existing)+len(records),'eligible_cohort':726,'total_candidate_universe':16397,
          'cohort_percent':round(100*(len(existing)+len(records))/726,3),
          'total_universe_percent':round(100*(len(existing)+len(records))/16397,3),
          'metrics_new':dict(c), 'training_admissions':0,'creator_original_byte_parity_pass':0,
          'input_v20_evidence_sha256':sha(V20),'input_v17_evidence_sha256':sha(V17),
          'new_court_receipts_sha256':sha(OUT/'NEW_MASTER726_NPZ_COURT_V21.csv.gz'),
          'authority':'NORMALIZED_DERIVATIVE_ONLY'}
 (OUT/'SUMMARY_NUMERIC_V21.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(summary,indent=2))
 if quarantine:print('QUARANTINE_FIRST',quarantine[:5])
 return 0 if not quarantine else 4
if __name__=='__main__':raise SystemExit(main())