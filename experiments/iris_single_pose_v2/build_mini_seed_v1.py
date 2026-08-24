from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
EXPECTED_REPRESENTATION_SEED_SHA256='f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af';EXPECTED_PANEL_DIGEST='366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961';EXPECTED_MEMBERSHIP_CANONICAL_SHA256='4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055';ROLES=('FIT_TRAIN','FIT_SELECT','TUNE_FINAL')
def canonical_json_sha(path):
 obj=json.load(open(path,encoding='utf-8'));return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest(),obj
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def main():
 ap=argparse.ArgumentParser(description='Build minimal learner runtime seed from frozen membership only; CI104 seed is provenance, never runtime input.');ap.add_argument('--membership',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 msh,mem=canonical_json_sha(a.membership)
 if msh!=EXPECTED_MEMBERSHIP_CANONICAL_SHA256:raise RuntimeError(f'mini membership canonical SHA drift {msh}')
 if mem.get('source_panel_asset_id_digest')!=EXPECTED_PANEL_DIGEST or mem.get('source_representation_seed_sha256')!=EXPECTED_REPRESENTATION_SEED_SHA256 or mem.get('sealed_splits_opened') is not False:raise RuntimeError('membership authority drift')
 rs={role:list(mem['roles'][role]) for role in ROLES}
 if [len(rs[x]) for x in ROLES]!=[128,32,26]:raise RuntimeError('frozen mini role counts drift')
 if len(set().union(*(set(v) for v in rs.values())))!=sum(map(len,rs.values())):raise RuntimeError('mini roles overlap')
 records=[]
 for role in ROLES:
  split='TUNE' if role=='TUNE_FINAL' else 'FIT'
  for aid in rs[role]:records.append({'asset_id':aid,'split':split,'mini_role':role})
 out={'schema':'RealSaS.IRISSinglePoseV2.MiniExtractabilityRuntimeSeed.v2','status':'FROZEN_MINIMAL_MEMBERSHIP_ONLY_RUNTIME_SEED','source_representation_seed_sha256_provenance_only':EXPECTED_REPRESENTATION_SEED_SHA256,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA256,'panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'record_count':len(records),'role_counts':{k:len(v) for k,v in rs.items()},'records':records,'runtime_tune_fields':['asset_id','split','mini_role'],'representation_seed_consumed_at_runtime':False,'sealed_splits_opened':False,'optimizer_steps':0};atomic_json(a.out,out);print(json.dumps({k:out[k] for k in ('status','record_count','role_counts','representation_seed_consumed_at_runtime','sealed_splits_opened')},indent=2))
if __name__=='__main__':main()
