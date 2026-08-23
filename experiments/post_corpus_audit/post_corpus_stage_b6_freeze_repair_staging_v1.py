#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from pathlib import Path

ROOT_DEFAULT='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3'
LOCAL_B5_DEFAULT='/content/realsas_stageb5_dryrun'
SCHEMA='RealSaS.PostCorpus.StageB6FrozenRepairStaging.v1'
BLANK_ASSET='asset_7da4136e89f94d4ce0d18337'

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
    os.replace(tmp,path)

def atomic_copy(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name(dst.name+'.tmp')
    tmp.unlink(missing_ok=True)
    shutil.copy2(src,tmp)
    os.replace(tmp,dst)

def load_selected(root:Path):
    raw=json.loads((root/'metadata'/'CANONICAL_VARIANT_SELECTION.json').read_text())
    x=raw.get('selected',raw) if isinstance(raw,dict) else raw
    if isinstance(x,dict): x=list(x.values())
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=ROOT_DEFAULT)
    ap.add_argument('--local-b5',default=LOCAL_B5_DEFAULT)
    args=ap.parse_args()
    root=Path(args.root); local=Path(args.local_b5)
    outdir=root/'reports'/'post_corpus_audit'
    b5_path=outdir/'POST_CORPUS_STAGE_B5_REPAIR_DRYRUN_RESULT_V1.json'
    if not b5_path.exists(): raise RuntimeError('B5 result missing: '+str(b5_path))
    b5=json.loads(b5_path.read_text())
    if b5.get('schema')!='RealSaS.PostCorpus.StageB5RepairDryRun.v1': raise RuntimeError('unexpected B5 schema')
    if b5.get('corpus_mutation') is not False: raise RuntimeError('B5 must be read-only authority')
    if int(b5.get('ok',-1))!=int(b5.get('auto_repair_requested',-2)) or int(b5.get('failures',-1))!=0:
        raise RuntimeError('B5 not all-pass')
    if b5.get('iris_capability_failures_after_repair') or b5.get('geppetto_capability_failures_after_repair'):
        raise RuntimeError('B5 IRIS/Geppetto failure present')
    results=sorted(b5['results'],key=lambda x:x['asset'])
    if len(results)!=56: raise RuntimeError(f'expected 56 B5 results, got {len(results)}')
    staging=outdir/'B6_REPAIR_STAGING_V1'
    staging.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,r in enumerate(results,1):
        aid=r['asset']; srcdir=local/aid
        norm=srcdir/'normalized.npz'; struct=srcdir/'source_structure.json'
        if not norm.exists() or not struct.exists():
            raise RuntimeError(f'B5 local staging missing for {aid}; do not silently re-extract')
        got=sha256_file(norm); exp=str(r['staged_npz_sha256'])
        if got!=exp: raise RuntimeError(f'B5 staged SHA mismatch {aid}: expected={exp} got={got}')
        od=staging/aid; od.mkdir(parents=True,exist_ok=True)
        atomic_copy(norm,od/'normalized.npz')
        atomic_copy(struct,od/'source_structure.json')
        rec={
            'schema':'RealSaS.PostCorpus.RepairArtifact.v1',
            'asset':aid,
            'source_raw_path':r.get('raw_path'),
            'source_sha256':(r.get('candidate') or {}).get('source_sha256'),
            'normalized_sha256':exp,
            'old_capabilities':r.get('old_capabilities'),
            'new_audit':r.get('new_audit'),
            'counts':r.get('counts'),
            'normalization':r.get('normalization'),
            'repair_reason':{
                'eval_surface_p95_rel':(r.get('candidate') or {}).get('eval_surface_p95_rel'),
                'tri_surface_p95_rel':(r.get('candidate') or {}).get('tri_surface_p95_rel'),
                'non_armature_modifiers':(r.get('candidate') or {}).get('non_armature_modifiers'),
            },
            'source_structure_sha256':sha256_file(od/'source_structure.json'),
        }
        atomic_json(od/'repair_record.json',rec)
        rows.append({
            'asset':aid,
            'normalized_path':str((od/'normalized.npz').relative_to(root)),
            'normalized_sha256':exp,
            'source_structure_path':str((od/'source_structure.json').relative_to(root)),
            'source_structure_sha256':rec['source_structure_sha256'],
            'old_capabilities':r.get('old_capabilities'),
            'new_capabilities':{k:bool((r.get('new_audit') or {}).get(k+'_capable')) for k in ['iris','geppetto','arachne']},
        })
        if i%10==0 or i==len(results): print(f'[stageB6] frozen {i}/{len(results)}',flush=True)

    shape_q=sorted(b5.get('shape_key_quarantine_assets',[]),key=lambda x:x['asset'])
    selected=load_selected(root); selected_ids=sorted(x['canonical_asset_id'] for x in selected)
    excluded={BLANK_ASSET:'ALL8_BLANK_OBSERVATION'}
    for x in shape_q: excluded[x['asset']]='ACTIVE_NONBASIS_SHAPE_KEY_TARGET_AUTHORITY'
    missing_excludes=sorted(set(excluded)-set(selected_ids))
    if missing_excludes: raise RuntimeError('quarantine asset absent from selected: '+repr(missing_excludes))
    eligible=[aid for aid in selected_ids if aid not in excluded]
    quarantine={
        'schema':'RealSaS.IRISTrainingEligibility.v1',
        'selected_count':len(selected_ids),
        'eligible_count':len(eligible),
        'excluded_count':len(excluded),
        'excluded':[{'asset':aid,'reason':excluded[aid]} for aid in sorted(excluded)],
        'eligible_assets':eligible,
        'policy':'IRIS training/evaluation must consume only eligible_assets until an exclusion is explicitly requalified.',
    }
    atomic_json(outdir/'IRIS_TRAINING_ELIGIBILITY_V1.json',quarantine)
    manifest={
        'schema':SCHEMA,
        'repair_artifact_count':len(rows),
        'repair_artifacts':rows,
        'shape_key_quarantine_count':len(shape_q),
        'shape_key_quarantine_assets':shape_q,
        'all8_blank_quarantine_asset':BLANK_ASSET,
        'iris_training_eligible_count':len(eligible),
        'arachne_drops_after_repair':b5.get('arachne_drops_after_repair',[]),
        'canonical_corpus_mutation':False,
        'source_b5_result_sha256':sha256_file(b5_path),
    }
    atomic_json(outdir/'POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_RESULT_V1.json',manifest)
    lines=[
        '# Stage-B6 Frozen Repair Staging', '',
        f'- frozen repair artifacts: **{len(rows)}/56**',
        f'- shape-key training quarantine: **{len(shape_q)}**',
        '- all-8 blank training quarantine: **1**',
        f'- IRIS eligible selected assets: **{len(eligible)}/{len(selected_ids)}**',
        f'- anticipated Arachne drops after repair: **{len(b5.get("arachne_drops_after_repair",[]))}**',
        '', 'No canonical variant, asset, render, export, ledger or shard was mutated.',
        'The frozen NPZ SHA-256 values are the only geometry artifacts authorized for Stage-B7 publication.'
    ]
    (outdir/'POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_REPORT_V1.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='repair_artifacts'},indent=2),flush=True)

if __name__=='__main__': main()
