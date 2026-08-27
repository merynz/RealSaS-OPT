from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
M=json.load(open(ROOT/'E0_MEMBERSHIP_V1.json',encoding='utf-8'))

def test_frozen_membership_counts_and_disjointness():
    ca=M['calibration_anchor8']; qu=[x['asset_id'] for x in M['qualification_proxy32']]
    assert M['probe_train_512_authority']['count']==512
    assert len(ca)==8 and len(set(ca))==8
    assert len(qu)==32 and len(set(qu))==32
    assert M['family_disjoint_proxy_vs_train512'] is True
    assert M['anchor8_disjoint_from_train512'] is True

def test_source_membership_authority_and_sealed_firewall():
    assert M['source_membership_canonical_json_sha256']=='8531360f1c61dc4cdb699c095790c35ab3af0da5a7a290b420e5777bb4e9169b'
    assert M['selection_used_image_geometry_or_difficulty'] is False
    assert all(M['sealed'].values())
    assert M['all_splits']=='FIT_ONLY'

def test_internal_canonical_digest():
    d=dict(M); expected=d.pop('canonical_json_sha256'); raw=json.dumps(d,sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(raw).hexdigest()==expected
