#!/usr/bin/env python3
"""V16 local/Drive-mounted resumable, source-locked historical derivative replay.

Uses the established V13 physical court in this repository. Never requires a
GitHub-hosted bulk downloader or touches protected holdouts. Does not mint
creator-original source, signed SDF, rights or model training authority.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,json,os,pathlib,tempfile,sys,collections

INPUT_LOCKS={
 'queue':'18234396bf86fa0e9c156c6e3eddb73aa3963870459491a479ffb56974349ab7',
 'inventory':'6436a6696a55c7643818914b6ebab5a5728d4eae23409d67e6e6d210c2b17ae4',
}
ELIGIBLE='SHA_VERIFIED_RECOVERY_RECEIPT'
HOLDOUT='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED'
SCHEMA='RealSaS.TruthFoundryMountedRecoveryV16.v1'

def sha(path:pathlib.Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def jsonwrite(path, obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='.pending-v15-')
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:
   json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def select(queue,inventory,shard_index,shard_count):
 if not 0<=shard_index<shard_count<=64:raise ValueError('BAD_SHARD')
 if sha(queue)!=INPUT_LOCKS['queue'] or sha(inventory)!=INPUT_LOCKS['inventory']:
  raise ValueError('SOURCE_SHA_LOCK_DRIFT')
 with gzip.open(inventory,'rt',newline='',encoding='utf-8') as f:
  statuses={r['source_key']:r['master_v11_source_recovery_status'] for r in csv.DictReader(f) if r['pool']=='MASTER3993'}
 with queue.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=1527 or len({x['source_key'] for x in rows})!=1527:raise ValueError('QUEUE_IDENTITY_DRIFT')
 counts=collections.Counter(statuses.get(x['source_key'],'NOT_IN_V11') for x in rows)
 if counts[ELIGIBLE]!=726 or counts[HOLDOUT]!=135:raise ValueError('ELIGIBILITY_OR_HOLDOUT_DRIFT')
 eligible=[x for x in rows if statuses[x['source_key']]==ELIGIBLE]
 # Validate path tokens strictly against original V14 allowlist: provider:hex source ID.
 import re
 for r in eligible:
  if not re.fullmatch('[a-z0-9_]+:[a-f0-9]{24,64}',r['candidate_id']):
   raise ValueError('FORBIDDEN_SOURCE_ID')
 selected=[x for x in eligible if int(hashlib.sha256(x['source_key'].encode()).hexdigest()[:16],16)%shard_count==shard_index]
 assert all(statuses[x['source_key']]==ELIGIBLE for x in selected)
 return selected,counts

def import_court(path):
 spec=importlib.util.spec_from_file_location('truth_foundry_v13_court',path)
 if spec is None or spec.loader is None:raise ValueError('NO_V13_COURT')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if not callable(mod.court):raise ValueError('V13_COURT_NO_FUNCTION')
 return mod

def process(row,root,out,court,court_sha,plan_sha):
 key=row['source_key'];stem=hashlib.sha256(key.encode()).hexdigest()
 folder=(root/row['candidate_id'].replace(':','_',1)).resolve()
 if not folder.is_relative_to(root.resolve()):raise ValueError('PATH_ESCAPE')
 src=folder/'normalized.npz';struct=folder/'source_structure.json'
 receipt=out/'receipts'/(stem+'.json')
 if not src.is_file():return {'source_key':key,'state':'SOURCE_FILE_NOT_MOUNTED','receipt':None}
 if not struct.is_file():return {'source_key':key,'state':'STRUCTURE_FILE_NOT_MOUNTED','receipt':None}
 source_sha=sha(src);struct_sha=sha(struct)
 if receipt.is_file():
  previous=json.loads(receipt.read_text(encoding='utf-8'))
  if (previous.get('source_key')!=key or previous.get('source_npz_sha256')!=source_sha
      or previous.get('structure_sha256')!=struct_sha or previous.get('court_module_sha256')!=court_sha
      or previous.get('plan_sha256')!=plan_sha or previous.get('scope')!='HISTORICAL_DERIVATIVE_ONLY'):
   raise ValueError('CHECKPOINT_IDENTITY_OR_COURT_DRIFT:'+stem[:14])
  prior_joint=out/'joints'/(stem+'.json.gz')
  if (not prior_joint.is_file() or sha(prior_joint)!=previous.get('joint_receipt_sha256')):
   raise ValueError('CACHED_JOINT_RECEIPT_MISSING_OR_DRIFT')
  # Re-read file bytes on restart before trusting cached numerical receipt.
  return {'source_key':key,'state':'REHASHED_CHECKPOINT_REUSED','receipt':str(receipt)}
 structure=json.loads(struct.read_text(encoding='utf-8'))
 numeric,joints=court.court(src,structure)
 if numeric.get('normalized_npz_sha256')!=source_sha:
  raise ValueError('V13_COURT_SOURCE_HASH_DRIFT')
 with __import__('numpy').load(src,allow_pickle=False) as z:
  n=len(z['vertices']);f=len(z['faces']);j=len(z['parents'])
  if (n!=numeric['vertex_count'] or f!=numeric['triangle_count'] or j!=numeric['joint_count']
    or z['skin'].shape!=(n,j)):
   raise ValueError('INDEPENDENT_ARRAY_READBACK_MISMATCH')
 if sha(src)!=source_sha or sha(struct)!=struct_sha:
  raise ValueError('SOURCE_MUTATED_DURING_COURT')
 joint_path=out/'joints'/(stem+'.json.gz')
 joint_path.parent.mkdir(parents=True,exist_ok=True)
 fd,temp=tempfile.mkstemp(dir=str(joint_path.parent),prefix='.pending-v15-')
 try:
  with os.fdopen(fd,'wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
    z.write((json.dumps(joints,sort_keys=True,separators=(',',':'))+'\n').encode())
   f.flush();os.fsync(f.fileno())
  os.replace(temp,joint_path)
 finally:
  if os.path.exists(temp):os.unlink(temp)
 obj={'schema':SCHEMA,'source_key':key,'candidate_id':row['candidate_id'],'scope':'HISTORICAL_DERIVATIVE_ONLY',
  'source_npz_sha256':source_sha,'structure_sha256':struct_sha,'court_module_sha256':court_sha,
  'plan_sha256':plan_sha,'numeric':numeric,'joint_rows':len(joints),
  'joint_receipt_sha256':sha(joint_path),'created_by':'V16_MOUNTED_REPLAY',
  'creator_original_source_byte_rehash':False,'rights_admission':False,'training_admission':False,
  'global_sdf_sign_admission':False,'tessa_animation_ready_admission':False}
 jsonwrite(receipt,obj)
 return {'source_key':key,'state':'COURTED_DERIVATIVE_ONLY','receipt':str(receipt)}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--queue',type=pathlib.Path,required=True)
 p.add_argument('--inventory',type=pathlib.Path,required=True)
 p.add_argument('--variants-root',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--v13-court',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'evidence_v13'/'normalized_array_court_v13.py')
 p.add_argument('--shard-index',type=int,default=0)
 p.add_argument('--shard-count',type=int,default=8)
 p.add_argument('--max-sources',type=int,default=0,help='0 = all assigned sources, otherwise bounded first N')
 p.add_argument('--preflight-only',action='store_true')
 a=p.parse_args(argv)
 if not 0<=a.max_sources<=726:raise ValueError('BAD_MAX_SOURCES')
 rows,counts=select(a.queue,a.inventory,a.shard_index,a.shard_count)
 a.output.mkdir(parents=True,exist_ok=True)
 planned=len(rows)
 if a.max_sources:rows=rows[:a.max_sources]
 plan={'schema':SCHEMA,'source_queue_sha256':INPUT_LOCKS['queue'],'inventory_sha256':INPUT_LOCKS['inventory'],
  'shard_index':a.shard_index,'shard_count':a.shard_count,'assigned_sources':planned,
  'selected_sources':len(rows),'holdouts_excluded':counts[HOLDOUT],
  'selected_source_ids_sha256':hashlib.sha256('\n'.join(sorted(r['source_key'] for r in rows)).encode()).hexdigest(),
  'no_source_reads_in_preflight':True,'training_admissions':0, 'expected_eligible_total':726}
 jsonwrite(a.output/'PLAN_V16.json',plan)
 print('V16_PLAN',json.dumps(plan,sort_keys=True),flush=True)
 if a.preflight_only:return 0
 if not a.v13_court.is_file():raise FileNotFoundError('V13_COURT_MODULE_UNAVAILABLE')
 court=import_court(a.v13_court);court_sha=sha(a.v13_court)
 plan_sha=sha(a.output/'PLAN_V16.json')
 results=[]
 for i,row in enumerate(rows,1):
  try:r=process(row,a.variants_root,a.output,court,court_sha,plan_sha)
  except OSError as e:
   r={'source_key':row['source_key'],'state':'SOURCE_READ_IO_BLOCKER','exception_type':type(e).__name__}
   results.append(r)
   print('V16_RATE_OR_IO_CIRCUIT_STOP',type(e).__name__,flush=True)
   break
  except Exception as e:
   r={'source_key':row['source_key'],'state':'QUARANTINED_COURT_EXCEPTION','exception_type':type(e).__name__,
      'detail':str(e)[:160]}
  results.append(r)
  if i%10==0 or i==len(rows):print('V16_PROGRESS',i,'/',len(rows),collections.Counter(x['state'] for x in results),flush=True)
 summary={'schema':SCHEMA,'plan_sha256':plan_sha,'attempted':len(results),
  'states':dict(collections.Counter(x['state'] for x in results)),'results':results,
  'training_admissions':0,'original_source_authority':0,'holdouts_touched':0}
 jsonwrite(a.output/'PROGRESS_V16.json',summary)
 return 4 if any(x['state'] in ('SOURCE_READ_IO_BLOCKER','QUARANTINED_COURT_EXCEPTION') for x in results) else 0

if __name__=='__main__':sys.exit(main())