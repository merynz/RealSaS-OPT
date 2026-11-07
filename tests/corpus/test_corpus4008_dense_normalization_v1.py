"""Source sparse skin and full imported-scene reversible transform tests."""
import csv
import hashlib
import json

import pytest

from tools.corpus.blender_corpus4008_dense_source_consistency_v1 import similarity, matrix_parity, weight_record, digest_weight_row
from tools.corpus.build_corpus4008_normalization_plan_v1 import point
from tools.corpus.plan_corpus4008_dense_normalization_shard_v1 import plan


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(tmp, shard=2):
    rows=[]
    for i in range(32):
        s=f"{100*shard+i+1:064x}"
        rows.append(dict(canonical_asset_id=f"asset_{100*shard+i+1:024x}",
                         candidate_id="objaverse_animated_originals:"+s,
                         source_sha256=s,
                         source_format=(".glb" if i%2 else ".fbx"),
                         source_shard=str(shard),split="FIT",
                         source_import_observation_status="IMPORTED_SOURCE_OBSERVATIONS_ONLY",
                         head_admission_count="0",training_authorized="False",
                         functional_native_ik_fk="UNKNOWN"))
    a=tmp/"SHARD_TYPED_OBSERVATIONS.csv"
    with a.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    b=tmp/"SHARD_READBACK_SUMMARY.json"
    b.write_text(json.dumps(dict(schema="RealSaS.Corpus4008ParallelShardReadback.v1",
                                 shard=shard, full_import_receipts_sha_verified=True,
                                 training_authorized=False,per_head_training_admitted_count=0,
                                 typed_observations_csv_sha256=sha(a),
                                 total_unique_source_count=32)))
    return b,a,rows


def test_selection_exact_and_no_training(tmp_path):
    s,c,r=make(tmp_path)
    result=plan(s,c,2)
    assert result==plan(s,c,2)
    assert result["source_count"]==32
    assert len(set(x["original_source_sha256"] for x in result["source_candidates"]))==32
    assert not result["training_authorized"]


def test_holdout_rejected_even_if_report_rehashed(tmp_path):
    s,c,rows=make(tmp_path)
    rows[0]["split"]="EXTERNAL_HOLDOUT"
    with c.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    data=json.loads(s.read_text())
    data["typed_observations_csv_sha256"]=sha(c)
    s.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="SOURCE_IDENTITY_OR_STATUS"):
        plan(s,c,2)


def test_original_readback_sha_tamper_rejected(tmp_path):
    s,c,_=make(tmp_path)
    c.write_text(c.read_text().replace(".glb",".gltf",1))
    with pytest.raises(ValueError,match="SOURCE_READBACK_PROOF"):
        plan(s,c,2)


def test_nonhumanoid_global_similarity_and_bone_rest():
    n,inv,span=similarity({"source_world_min":[-.001,-20.,4.],
                           "source_world_max":[.001,80.,4.008]})
    assert span==100.
    assert n[0]==n[5]==n[10]
    for p in ((-.001,-20,4),(.001,80,4.008),(0.,3.,4.005)):
        q=point(n,p)
        r=point(inv,q)
        assert max(abs(a-b) for a,b in zip(p,r))/span<1e-12
    src=[1.,0.,0.,.5,0.,0.,-1.,-1.2,0.,1.,0.,3.,0.,0.,0.,1.]
    assert matrix_parity(n,inv,src)/span<1e-12


def test_sparse_weight_vertex_and_bone_id_invariant():
    import hashlib
    a=weight_record([(9,.35),(2,.65)])
    b=weight_record([(2,.65),(9,.35)])
    assert a==b
    x=hashlib.sha256()
    y=hashlib.sha256()
    digest_weight_row(x,1,17,a)
    digest_weight_row(y,1,17,b)
    assert x.digest()==y.digest()
    z=hashlib.sha256()
    digest_weight_row(z,1,18,a)
    assert z.digest()!=x.digest()
    with pytest.raises(ValueError,match="INVALID_IMPORTED_SOURCE_INFLUENCE"):
        weight_record([(2,float("nan"))])
