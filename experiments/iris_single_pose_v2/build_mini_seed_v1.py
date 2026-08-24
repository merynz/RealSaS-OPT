from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
EXPECTED_REPRESENTATION_SEED_SHA256='f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af';EXPECTED_PANEL_DIGEST='366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961';EXPECTED_MEMBERSHIP_CANONICAL_SHA256='4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055';ROLES=('FIT_TRAIN','FIT_SELECT','TUNE_FINAL')
def sha256_file(path,chunk=8<<20):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(chunk),b''):h.update(b)
 return h.hexdigest()
def canonical_json_sha(path):
 obj=json.load(open(path,encoding='utf-8'));return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest(),obj
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def digest_ids(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description='Freeze learner mini seed from CI104 panel');ap.add_argument('--representation-seed',required=True);ap.add_argument('--membership',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 if sha256_file(a.representation_seed)!=EXPECTED_REPRESENTATION_SEED_SHA256:raise RuntimeError('representation seed SHA drift')
 msh,mem=canonical_json_sha(a.membership)
 if msh!=EXPECTED_MEMBERSHIP_CANONICAL_SHA256:raise RuntimeError(f'mini membership canonical SHA drift {msh}')
 rep=json.load(open(a.representation_seed,encoding='utf-8'))
 if rep.get('record_count')!=256 or rep.get('sealed_splits_opened') is not False:raise RuntimeError('CI104 representation seed contract drift')
 if mem.get('source_panel_asset_id_digest')!=EXPECTED_PANEL_DIGEST or mem.get('sealed_splits_opened') is not False:raise RuntimeError('membership authority drift')
 rep_map={r['asset_id']:r for r in rep['records']};panel=[r['asset_id'] for r in rep['records']]
 if digest_ids(panel)!=EXPECTED_PANEL_DIGEST:raise RuntimeError('representation panel identity drift')
 rs={role:list(mem['roles'][role]) for role in ROLES}
 if [len(rs[x]) for x in ROLES]!=[128,32,26]:raise RuntimeError('frozen mini role counts drift')
 if len(set().union(*(set(v) for v in rs.values())))!=sum(map(len,rs.values())):raise RuntimeError('mini roles overlap')
 for aid in rs['FIT_TRAIN']+rs['FIT_SELECT']:
  if aid not in rep_map or rep_map[aid]['split']!='FIT':raise RuntimeError(f'FIT role mismatch {aid}')
 for aid in rs['TUNE_FINAL']:
  if aid not in rep_map or rep_map[aid]['split']!='TUNE':raise RuntimeError(f'TUNE role mismatch {aid}')
 records=[]
 for role in ROLES:
  for aid in rs[role]:src=dict(rep_map[aid]);src['mini_role']=role;records.append(src)
 out={'schema':'RealSaS.IRISSinglePoseV2.MiniExtractabilitySeed.v1','status':'FROZEN_OPEN_ONLY_MINI_SEED','source_representation_seed_sha256':EXPECTED_REPRESENTATION_SEED_SHA256,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA256,'panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'record_count':len(records),'role_counts':{k:len(v) for k,v in rs.items()},'records':records,'sealed_splits_opened':False,'optimizer_steps':0};atomic_json(a.out,out);print(json.dumps({k:out[k] for k in ('status','record_count','role_counts','sealed_splits_opened')},indent=2))
if __name__=='__main__':main()
