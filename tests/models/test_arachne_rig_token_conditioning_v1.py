from __future__ import annotations

import torch
import pytest

from compiler.realsas_compiler_core.canonical_rig_tokens_v1 import (
    build_canonical_rig_tokens_v1,
    canonical_rig_token_features_v1,
    canonical_rig_token_indices_for_joint_ids_v1,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
)
from models.arachne.v4.rig_token_conditioning_v1 import (
    RigTokenConditioningConfigV1,
    RigTokenConditioningV1,
)


def _skeleton(ids=("root","a","b")):
    r,a,b=ids
    return QualifiedSkeletonIR(
        (
            QualifiedJoint(r,(0.0,0.0,0.0),None,(),"src-root"),
            QualifiedJoint(a,(0.0,1.0,0.0),r,(),"src-a"),
            QualifiedJoint(b,(1.0,2.0,0.0),a,(),"src-b"),
        ),
        r,{"status":"PASS"},"lineage",
    )


def _packed(skeleton, joint_ids):
    ts=build_canonical_rig_tokens_v1(skeleton)
    features=torch.tensor([canonical_rig_token_features_v1(ts)],dtype=torch.float32)
    indices=torch.tensor(
        [canonical_rig_token_indices_for_joint_ids_v1(skeleton,joint_ids)],
        dtype=torch.long,
    )
    mask=torch.ones((1,len(ts.tokens)),dtype=torch.bool)
    joint_mask=torch.ones((1,len(joint_ids)),dtype=torch.bool)
    return ts,features,mask,indices,joint_mask


def test_rig_token_features_and_alignment_are_id_rename_invariant():
    a=_skeleton(("root","a","b"))
    b=_skeleton(("X","Y","Z"))
    ta,fa,ma,ia,ja=_packed(a,("root","a","b"))
    tb,fb,mb,ib,jb=_packed(b,("X","Y","Z"))
    assert ta.token_set_hash==tb.token_set_hash
    torch.testing.assert_close(fa,fb,rtol=0.0,atol=0.0)
    assert torch.equal(ia,ib)
    assert fa.shape[-1]==12


def test_rig_token_conditioner_is_additive_identity_at_initialization():
    torch.manual_seed(3)
    s=_skeleton()
    _t,f,m,idx,jm=_packed(s,("root","a","b"))
    model=RigTokenConditioningV1(
        RigTokenConditioningConfigV1(model_dim=32,token_dim=20,attention_heads=5)
    )
    out=model(
        token_features=f,
        token_mask=m,
        joint_token_indices=idx,
        joint_mask=jm,
    )
    assert out.shape==(1,3,32)
    assert torch.equal(out,torch.zeros_like(out))


def test_rig_token_conditioner_uses_alignment_after_projection_is_enabled():
    torch.manual_seed(4)
    s=_skeleton()
    _t,f,m,idx,jm=_packed(s,("root","a","b"))
    model=RigTokenConditioningV1(
        RigTokenConditioningConfigV1(model_dim=16,token_dim=20,attention_heads=5)
    )
    with torch.no_grad():
        model.local_to_model.weight.copy_(
            torch.randn_like(model.local_to_model.weight) * 0.1
        )
        model.global_to_model.weight.zero_()
    out_a=model(
        token_features=f,token_mask=m,joint_token_indices=idx,joint_mask=jm
    )
    out_b=model(
        token_features=f,
        token_mask=m,
        joint_token_indices=idx.flip(1),
        joint_mask=jm,
    )
    assert not torch.allclose(out_a,out_b)


def test_rig_token_conditioner_fails_closed_on_bad_alignment():
    s=_skeleton()
    _t,f,m,idx,jm=_packed(s,("root","a","b"))
    model=RigTokenConditioningV1(
        RigTokenConditioningConfigV1(model_dim=16,token_dim=20,attention_heads=5)
    )
    bad=idx.clone()
    bad[0,1]=99
    with pytest.raises(ValueError,match="outside token axis"):
        model(
            token_features=f,
            token_mask=m,
            joint_token_indices=bad,
            joint_mask=jm,
        )
