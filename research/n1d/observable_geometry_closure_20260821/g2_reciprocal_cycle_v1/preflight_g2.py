from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

SCHEMA='RealSaS.N1D.Rank2G.G2.Preflight.v1'
RANK_TAG='REALSAS_V1_2P_INTERNAL_SPLIT|'


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def canonical_content_sha(d:dict)->str:
    x=dict(d);x.pop('content_sha256',None)
    b=json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(b).hexdigest()


def ranked_train(index:dict):
    f=[int(k) for k,v in index['families'].items() if v['split']=='train']
    return sorted(f,key=lambda x:hashlib.sha256(f'{RANK_TAG}{x}'.encode()).digest())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--science-dir',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--corpus-root',type=Path,required=True)
    ap.add_argument('--episode-index',type=Path,required=True)
    ap.add_argument('--family-audit',type=Path,required=True)
    ap.add_argument('--source-manifest',type=Path,required=True)
    ap.add_argument('--input-manifest',type=Path,required=True)
    ap.add_argument('--prereg',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise RuntimeError('refusing overwrite')
    src=json.loads(a.source_manifest.read_text());inp=json.loads(a.input_manifest.read_text());pre=json.loads(a.prereg.read_text())
    tests={}
    tests['source_manifest_sha']=sha256_file(a.source_manifest)==pre['authorities']['source_manifest_sha256']
    tests['input_manifest_sha']=sha256_file(a.input_manifest)==pre['authorities']['input_manifest_sha256']
    tests['checkpoint_sha']=sha256_file(a.checkpoint)==src['parent_observable']['checkpoint_sha256']
    for rel,want in src['repo_files_sha256'].items():
        tests[f'repo:{rel}']=sha256_file(a.repo_root/rel)==want
    tests['observable_phase_sha']=sha256_file(a.science_dir/'observable_phase.py')==src['parent_observable']['observable_phase_sha256']
    tests['evaluation_phase_sha']=sha256_file(a.science_dir/'evaluation_phase.py')==src['parent_observable']['evaluation_phase_sha256']
    idx=json.loads(a.episode_index.read_text());audit=json.loads(a.family_audit.read_text())
    tests['episode_index_raw_sha']=sha256_file(a.episode_index)==inp['corpus_authority']['episode_index_raw_sha256']
    tests['episode_index_content_sha']=canonical_content_sha(idx)==inp['corpus_authority']['episode_index_content_sha256']==idx['content_sha256']
    tests['family_audit_raw_sha']=sha256_file(a.family_audit)==inp['corpus_authority']['family_audit_raw_sha256']
    expected=[int(x) for x in inp['surface']['families']]
    tests['ranked_surface_exact']=ranked_train(idx)[48:64]==expected
    audit_by={int(x['family_id']):x for x in audit['rows']}
    family=[]
    for row in inp['family_contracts']:
        fam=int(row['family']);mfile=a.corpus_root/row['family_manifest'];m=json.loads(mfile.read_text())
        fr={'family':fam,'tests':{}}
        fr['tests']['manifest_content_sha']=canonical_content_sha(m)==row['family_manifest_content_sha256']==m['content_sha256']
        fr['tests']['split_train']=m.get('split')=='train' and idx['families'][str(fam)]['split']=='train'
        fr['tests']['episode_e01']=any(e.get('episode_id')=='e01' for e in m['episodes'])
        ar=audit_by[fam]
        fr['tests']['poseA_manifest_audit_parity']=m['poseA_hashes']==ar['A_hashes']
        for name,want in m['poseA_hashes'].items():
            fr['tests'][f'A:{name}']=sha256_file(mfile.parent/'poseA'/name)==want
        e=next(x for x in m['episodes'] if x['episode_id']=='e01')
        for name,want in e['poseB_hashes'].items():
            fr['tests'][f'B:{name}']=sha256_file(mfile.parent/e['poseB_dir']/name)==want
        fr['tests']['sidecar_opaque_sha']=sha256_file(mfile.parent/e['sidecar'])==e['observation_sidecar_sha256']
        fr['pass']=all(fr['tests'].values());family.append(fr)
    result={'schema':SCHEMA,'status':'PASS' if all(tests.values()) and all(x['pass'] for x in family) else 'FAIL','truth_access':'NONE__SIDECAR_BYTES_HASHED_ONLY','tests':tests,'families':family,'sealed21':'CLOSED','external10':'CLOSED'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'family_pass_n':sum(x['pass'] for x in family),'family_n':len(family)},indent=2))
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
