#!/usr/bin/env python3
"""Independent V12 byte-for-byte prior-court immutability and provenance readback."""
import argparse, csv, gzip, hashlib, json, pathlib, collections
ALLOWED={'source_vertex_face_count','skin_weight_normalization','zero_weight_vertices_and_unbound_mesh','canonical_frame_transform','source_name_only_teacher_hint_isolation','original_asset_provider_locator','attribution_obligations'}
STATE='OBSERVED_LEGACY_VARIANT_NOT_INDEPENDENTLY_COURTED'
def records(p):
    op=gzip.open if str(p).endswith('.gz') else open
    h=op(p,'rt',encoding='utf-8',newline='')
    return h,csv.DictReader(h)
def sha(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):d.update(b)
    return d.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-inventory',required=True);p.add_argument('--baseline-states',required=True);p.add_argument('--baseline-values',required=True);p.add_argument('--queue',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=pathlib.Path(a.out);q={}
    h,reader=records(a.queue)
    with h:
        for r in reader:
            assert r['source_key'] not in q
            q[r['source_key']]=r
    assert len(q)==1527
    details={};state_counts=collections.Counter();reported_sha=set()
    with gzip.open(out/'LEGACY_VARIANT_SOURCE_RECEIPTS_1527.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);key=r['source_key']
            assert key not in details and key in q and r['candidate_id']==q[key]['candidate_id']
            assert r['training_authorized'] is False and r['source_original_bytes_rehashed'] is False
            assert set(r['facet_candidates'])<=ALLOWED
            if r['inspection_state']=='SKIPPED_PROTECTED_HOLDOUT':
                assert r['facet_candidates']=={} and not r.get('receipt_sha256')
            details[key]=r;state_counts[r['inspection_state']]+=1
    assert len(details)==1527
    assert state_counts['SKIPPED_PROTECTED_HOLDOUT']==135, state_counts
    assert state_counts['FOUR_DOCUMENTS_OBSERVED']+state_counts['PARTIAL_RECEIPT_OBSERVED']+state_counts['CONFLICT_QUARANTINED']+state_counts['NO_VARIANT_RECEIPT_FOUND']<=1392
    h1,r1=records(a.baseline_inventory);h2,r2=records(out/'SOURCE_INVENTORY_DEEP_16397_V12.csv.gz')
    with h1,h2:
        old_fields=r1.fieldnames;new_fields=r2.fieldnames
        assert new_fields[:len(old_fields)]==old_fields
        assert len(new_fields)==len(old_fields)+14
        n=0;active=0
        for original,new in zip(r1,r2):
            n+=1;assert {k:new[k] for k in old_fields}==original
            if original['source_key'] in details:
                active+=1;assert new['variant_v12_inspection_state']==details[original['source_key']]['inspection_state']
                assert new['variant_v12_not_original_bytes_rehash']=='TRUE'
            else:
                assert not any(new[k] for k in new_fields[len(old_fields):])
        assert n==16397 and active==1527
    bs,oldst=records(a.baseline_states);ns,newst=records(out/'FACET_STATES_EFFECTIVE_V12_16397x264.csv.gz')
    bv,oldval=records(a.baseline_values);nv,newval=records(out/'FACET_VALUES_EFFECTIVE_V12_16397x264.csv.gz')
    counter=collections.Counter();changed={};rows=0;preserved=0
    with bs,ns,bv,nv:
        assert oldst.fieldnames==newst.fieldnames==oldval.fieldnames==newval.fieldnames
        assert len(oldst.fieldnames)==265
        for x,y,a,b in zip(oldst,newst,oldval,newval):
            rows+=1;key=x['source_key'];assert key==y['source_key']==a['source_key']==b['source_key']
            if key not in q:
                assert x==y and a==b
                preserved+=264;continue
            rec=details[key]
            for f in x:
                if f=='source_key':continue
                if x[f]!=y[f] or a[f]!=b[f]:
                    assert f in ALLOWED and f in rec['facet_candidates']
                    assert x[f]=='UNKNOWN_UNREVIEWED' and not a[f]
                    assert y[f]==STATE
                    entry=json.loads(b[f]);assert entry['observation']==rec['facet_candidates'][f]
                    assert entry['source_record_sha256']==rec['record_sha256']
                    assert entry['authority']=='HISTORICAL_SOURCE_DERIVATIVE_NOT_CURRENT_COURT'
                    changed[(key,f)]=b[f];counter[f]+=1
                else:preserved+=1
        assert rows==16397
    drows=0
    with gzip.open(out/'FACET_OBSERVATION_DELTA_V12.csv.gz','rt') as f:
        for rec in csv.DictReader(f):
            drows+=1;key=(rec['source_key'],rec['facet'])
            assert key in changed and changed.pop(key)==rec['value']
            assert rec['status']==STATE and rec['source_record_sha256']==details[key[0]]['record_sha256']
    assert not changed and drows==sum(counter.values())
    summary=json.loads((out/'SUMMARY_VARIANTS_V12.json').read_text())
    assert summary['new_observations']==drows and summary['new_observations_by_facet']==dict(counter)
    assert summary['holdout_source_variants_read']==0
    assert summary['full_scene_original_byte_rehash_count']==summary['current_rights_court_pass']==summary['new_training_admissions']==0
    assert preserved+drows==16397*264
    for name,digest in summary['output_sha256'].items():assert sha(out/name)==digest,name
    verification={'schema':'RealSaS.V12LegacyVariantIndependentReadback','status':'PASS_NO_EXISTING_CELLS_OVERWRITTEN_LEGACY_ONLY',
                  'candidate_rows':rows,'facet_columns':264,'legacy_candidate_rows':active,
                  'historical_variant_states':dict(state_counts),'new_observation_cells':drows,
                  'new_by_facet':dict(counter),'existing_cells_preserved':preserved,
                  'holdout_folders_visited':0,'training_admissions':0,'original_source_byte_rehash_courts':0,
                  'independent_detail_bytes_sha256':sha(out/'LEGACY_VARIANT_SOURCE_RECEIPTS_1527.jsonl.gz')}
    (out/'INDEPENDENT_READBACK_VARIANTS_V12.json').write_text(json.dumps(verification,sort_keys=True,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()