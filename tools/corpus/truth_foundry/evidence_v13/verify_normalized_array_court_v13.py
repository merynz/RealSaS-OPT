#!/usr/bin/env python3
"""Independent V13 artifact readback: source SHA and per-joint/vertex statistics.
Re-reads all original downloaded normalized derivative NPZ bytes, but does not
promote original creator bytes or legal/semantic teacher authority.
"""
import argparse,csv,gzip,hashlib,json,pathlib,collections
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--variant-dir',required=True);p.add_argument('--queue',required=True);p.add_argument('--baseline-inventory',required=True)
 a=p.parse_args();o=pathlib.Path(a.out)
 with open(a.queue,newline='') as f:qr=list(csv.DictReader(f))
 assert len(qr)==1527
 valid={}
 with gzip.open(a.baseline_inventory,'rt',newline='') as f:
  for row in csv.DictReader(f):
   if row['pool']=='MASTER3993':valid[row['source_key']]=row['master_v11_source_recovery_status']
 expected={row['source_key']:row for row in qr if valid[row['source_key']]=='SHA_VERIFIED_RECOVERY_RECEIPT'}
 assert len(expected)==726
 metrics=[]
 with gzip.open(o/'MASTER726_NORMALIZED_DERIVATIVE_PER_ASSET_V13.jsonl.gz','rt') as f:
  for line in f:metrics.append(json.loads(line))
 assert len(metrics)==726 and len({r['source_key'] for r in metrics})==726
 assert {r['source_key'] for r in metrics}==set(expected)
 seen=collections.Counter();joint_by=collections.Counter();num=[];passed=set()
 with gzip.open(o/'MASTER726_NORMALIZED_DERIVATIVE_JOINTS_V13.csv.gz','rt',newline='') as f:
  for row in csv.DictReader(f):
   assert row['source_key'] in expected
   joint_by[row['source_key']]+=1
   assert int(row['parent_index'])>=-1
 for rec in metrics:
  key=rec['source_key'];state=rec['court'];seen[state]+=1
  assert state in ('NUMERIC_DERIVATIVE_OBSERVATION_ONLY','MISSING_HISTORICAL_NORMALIZED_DERIVATIVE','QUARANTINED_BAD_HISTORICAL_NPZ')
  if state!='NUMERIC_DERIVATIVE_OBSERVATION_ONLY':
   assert joint_by[key]==0;continue
  assert rec['source_original_bytes_rehashed'] is False if 'source_original_bytes_rehashed' in rec else True
  assert rec['teacher_training_admission'] is False and rec['global_sdf_sign_admission'] is False and rec['2d_art_approval'] is False
  path=pathlib.Path(a.variant_dir)/rec['candidate_id'].replace(':','_',1)/'normalized.npz'
  assert path.is_file() and sha(path)==rec['normalized_npz_sha256']
  with np.load(path,allow_pickle=False) as npz:
   v=npz['vertices'];f=npz['faces'];w=npz['skin'];p=npz['parents']
   assert tuple(v.shape)==(rec['vertex_count'],3)
   assert tuple(f.shape)==(rec['triangle_count'],3)
   assert tuple(w.shape)==(rec['vertex_count'],rec['joint_count'])
   assert len(p)==rec['joint_count']==joint_by[key]
   assert int(np.count_nonzero(w.astype(np.float64).sum(axis=1)<=1e-8))==rec['skin_zero_row_count']
   assert np.all(f>=0) and np.all(f<len(v))
  passed.add(key)
 assert sum(seen.values())==726 and sum(joint_by.values())==sum(rec['joint_count'] for rec in metrics if rec['court']=='NUMERIC_DERIVATIVE_OBSERVATION_ONLY')
 summ=json.loads((o/'SUMMARY_NORMALIZED_V13.json').read_text())
 assert summ['results']==dict(seen)
 assert summ['per_joint_rows']==sum(joint_by.values())
 assert summ['protected_holdout_folders_visited']==summ['training_authorized']==summ['original_source_byte_rehash_admission']==0
 assert summ['global_sdf_sign_admission']==summ['mechanically_qualified_tessa_teacher_admission']==summ['rights_admission']==0
 report={'schema':'RealSaS.HistoricalNormalizedDerivativeIndependentV13','status':'PASS_DERIVATIVE_ONLY',
         'candidate_rows':726,'reverified_npz_bytes':len(passed),'per_joint_rows':sum(joint_by.values()),
         'court_states':dict(seen),'protected_holdout_folders_visited':0,'original_source_rehash_admissions':0,
         'graphics_semantic_mechanics_training_admissions':0}
 (o/'INDEPENDENT_READBACK_NORMALIZED_V13.json').write_text(json.dumps(report,indent=2,sort_keys=True))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()