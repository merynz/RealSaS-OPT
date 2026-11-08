"""Independent forensic ledger coverage and non-promotion checks (no teacher PASS)."""
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT=Path(os.environ.get('REALSAS_FORENSIC_MATRIX_OUT','/mnt/data/truth_foundry_forensic_v3'))

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()

def verify(root=ROOT):
    summary=json.load(open(root/'SUMMARY_V3.json',encoding='utf-8'))
    assert summary['source_record_candidates']==11863
    assert summary['head_target_slots']==438931
    assert summary['facet_slots']==3131832
    assert summary['review_group_tasks']==166082
    assert summary['facet_non_authoritative_observation_slots']+summary['facet_unknown_slots']==3131832
    assert summary['per_head_qualified_sources']=={h:0 for h in ('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')}
    assert not summary['training_authorized']
    assert all(digest(root/n)==h for n,h in summary['output_sha256'].items())
    counts=Counter();seen=set();observed=0;special=Counter()
    with gzip.open(root/'FORENSIC_ASSET_RECORDS_11863x264_V3.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line)
            key=d['source_key'];assert key not in seen;seen.add(key)
            assert len(d['source_facet_evidence'])==264
            assert len(d['head_route_observations'])==6
            assert d['training_authorized'] is False
            assert not d['source_original_bytes_sha_courted']
            for ev in d['source_facet_evidence'].values():
                assert ev['independent_court_state']=='UNKNOWN'
                assert ev['supervision_status']=='NOT_ADMITTED'
                assert ev['training_mask_valid'] is False
                if ev['observation_state']!='UNKNOWN_UNREVIEWED': observed+=1
            for h in d['head_route_observations'].values():assert not h['valid_supervision_loss_mask']
            counts[d['provider'] if d['corpus']=='EXTERNAL_CANDIDATE' else 'ORIGINAL']+=1
            if d['provider']=='UNIML3D' and d['glb_canonical_observation_fields'] and d['glb_canonical_observation_fields']['download_status']=='DOWNLOADED':special['glb']+=1
    assert len(seen)==11863
    assert counts['ORIGINAL']==4363 and counts['PARTVERSE_XL']==2500 and counts['ARTICULATION_XL2']==2500 and counts['UNIML3D']==2500
    assert observed==summary['facet_non_authoritative_observation_slots']
    assert special['glb']==84
    with open(root/'HEAD_TARGET_EVIDENCE_11863x37_V3.csv',newline='',encoding='utf-8') as f:
        rows=csv.DictReader(f)
        target_counts=Counter();n=0
        for r in rows:
            n+=1;target_counts[r['head']]+=1
            assert r['training_mask_valid']=='False'
            assert r['reviewed_truth_status']=='NOT_ADMITTED'
            assert r['source_original_sha_verified']=='False'
            assert r['source_rights_court']=='UNKNOWN'
        assert n==438931
        assert sum(target_counts.values())==438931
        assert len(target_counts)==6
        # The seven/other target counts differ by head; each is evaluated per original source.
        for head,n_head in target_counts.items():
            assert n_head%11863==0
    with open(root/'GROUP_COURT_REVIEW_QUEUE_11863x14_V3.csv',newline='',encoding='utf-8') as f:
        n=0
        for r in csv.DictReader(f):
            n+=1
            assert r['task_status']=='OPEN__NOT_INDEPENDENTLY_COURTED'
            assert int(r['independent_pass'])==0
        assert n==166082
    return {'asset_records':len(seen),'head_target_slots':438931,
            'facet_observations_not_courted':observed,'canonical_glb_84':special['glb'],
            'source_type_counts':dict(counts),'head_target_rows':dict(target_counts)}

if __name__=='__main__':print(json.dumps(verify(),indent=2))