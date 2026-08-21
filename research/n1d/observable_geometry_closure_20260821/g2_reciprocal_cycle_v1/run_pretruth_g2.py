from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from pathlib import Path

SCHEMA='RealSaS.N1D.Rank2G.G2.PretruthFreeze.v1'


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def run(cmd):subprocess.run([str(x) for x in cmd],check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--science-dir',type=Path,required=True)
    ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--input-manifest',type=Path,required=True)
    ap.add_argument('--source-manifest',type=Path,required=True)
    ap.add_argument('--prereg',type=Path,required=True)
    ap.add_argument('--preflight',type=Path,required=True)
    ap.add_argument('--out-root',type=Path,required=True)
    a=ap.parse_args()
    preflight=json.loads(a.preflight.read_text())
    if preflight.get('status')!='PASS':raise RuntimeError('preflight not PASS')
    inp=json.loads(a.input_manifest.read_text());pre=json.loads(a.prereg.read_text())
    if sha256_file(a.input_manifest)!=pre['authorities']['input_manifest_sha256']:raise RuntimeError('input manifest SHA mismatch')
    if sha256_file(a.source_manifest)!=pre['authorities']['source_manifest_sha256']:raise RuntimeError('source manifest SHA mismatch')
    g2dir=Path(__file__).resolve().parent
    evidence_py=g2dir/'src/rank2_g2_reciprocal_cycle_evidence.py';g2_py=g2dir/'src/rank2_g2_reciprocal_cycle_proposal.py'
    g1_py=a.repo_root/'research/n1d/observable_geometry_closure_20260820/rank2_g1_geometry_grounded_proposal.py'
    v5_py=a.repo_root/'research/n1d/rank2_global_relational_world_20260820/rank2_r_v5_full_pairwise_proposal.py'
    out=a.out_root
    if (out/'G2_PRETRUTH_FREEZE.json').exists():raise RuntimeError('refusing overwrite freeze')
    rows=[]
    for fam in map(int,inp['surface']['families']):
        state=a.panel/str(fam)/f'{fam}_e01_OBSERVABLE_STATE.npz';meta=state.with_suffix('.json')
        sm=json.loads(meta.read_text())
        if sm.get('truth_access')!='NONE':raise RuntimeError(('state truth contract',fam))
        ep=out/'evidence'/f'{fam}_G2_RECIPROCAL_CYCLE_EVIDENCE.npz';em=out/'evidence'/f'{fam}_G2_RECIPROCAL_CYCLE_EVIDENCE.json'
        g1=out/'g1'/f'{fam}_G1_SELECTION.json';g2=out/'g2'/f'{fam}_G2_SELECTION.json'
        for p in (ep,em,g1,g2):
            if p.exists():raise RuntimeError(('refusing overwrite',p))
        run([sys.executable,evidence_py,'--root',a.root,'--science-dir',a.science_dir,'--state',state,'--family',fam,'--episode','e01','--out-evidence',ep,'--out-meta',em])
        e=json.loads(em.read_text());base=e['base_proposal_evidence_content_sha256'];full=e['g2_evidence_content_sha256']
        run([sys.executable,g1_py,'--evidence',ep,'--family',fam,'--episode','e01','--expected-content-sha',base,'--out',g1])
        run([sys.executable,g2_py,'--v5-source',v5_py,'--evidence',ep,'--family',fam,'--episode','e01','--expected-base-content-sha',base,'--expected-g2-content-sha',full,'--out',g2])
        s1=json.loads(g1.read_text());s2=json.loads(g2.read_text())
        if s1.get('truth_access')!='NONE' or s2.get('truth_access')!='NONE':raise RuntimeError(('selection truth contract',fam))
        rows.append({'family':fam,'observable_state_sha256':sha256_file(state),'observable_state_meta_sha256':sha256_file(meta),'evidence_sha256':sha256_file(ep),'evidence_meta_sha256':sha256_file(em),'base_evidence_content_sha256':base,'g2_evidence_content_sha256':full,'g1_selection_sha256':sha256_file(g1),'g2_selection_sha256':sha256_file(g2)})
    freeze={'schema':SCHEMA,'status':'PRETRUTH_FROZEN__TRUTH_STILL_CLOSED','truth_access':'NONE','source_manifest_sha256':sha256_file(a.source_manifest),'input_manifest_sha256':sha256_file(a.input_manifest),'prereg_sha256':sha256_file(a.prereg),'preflight_sha256':sha256_file(a.preflight),'families':rows,'required_next_gate':'PERSIST_THIS_FREEZE_AND_ALL_REFERENCED_OUTPUTS_BEFORE_TRUTH_OPEN','sealed21':'CLOSED','external10':'CLOSED'}
    p=out/'G2_PRETRUTH_FREEZE.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':freeze['status'],'family_n':len(rows),'freeze_sha256':sha256_file(p)},indent=2))

if __name__=='__main__':main()
