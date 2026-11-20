from __future__ import annotations

"""Upstream MV-Adapter classifier-free guidance is not an optional toy prior."""

import torch
import pytest

from models.rss_vnext.realsas_mvadapter_dual_reference_v2 import (
    PairedReferenceCacheV2,
)
from models.rss_vnext.realsas_mvadapter_guidance_policy_v1 import (
    OfficialMVAdapterGuidancePolicyV1,
    MVAdapterSubjectDropoutV1,
    draw_official_mvadapter_dropout_v1,
    select_official_mvadapter_conditions_v1,
    zero_native_mvadapter_reference_cache_v1,
)


def _cache(b=2):
    # B subjects, two modalities and SIX native row-attention views per
    # modality. The two canonical images are not reduced to one.
    hidden=torch.arange(b*2*6*4*3,dtype=torch.float32).reshape(b*12,4,3)+1
    return PairedReferenceCacheV2(
        features={"ref.block.0":hidden,"ref.block.1":hidden*2},
        batch_size=b,modalities=2,num_views=6,
    )


def test_official_sd_xl_i2mv_guidance_policy_exact():
    policy=OfficialMVAdapterGuidancePolicyV1()
    assert policy.prompt_drop_prob==.1
    assert policy.reference_drop_prob==.1
    assert policy.combined_drop_prob==.1
    assert policy.inference_guidance_scale==3
    assert len(policy.digest)==64
    with pytest.raises(ValueError,match="GUIDANCE_POLICY_INVALID"):
        OfficialMVAdapterGuidancePolicyV1(
            prompt_drop_prob=-.2,
        ).validate()
    with pytest.raises(ValueError,match="GUIDANCE_POLICY_INVALID"):
        OfficialMVAdapterGuidancePolicyV1(
            inference_guidance_scale=float("nan"),
        ).validate()


def test_subject_level_dropout_preserves_native_grouping():
    cache=_cache(2)
    positive=torch.ones(2,3,2048)
    negative=torch.full_like(positive,-5)
    pooled=torch.ones(2,1280)
    empty_pooled=torch.full_like(pooled,-10)
    flags=MVAdapterSubjectDropoutV1(
        prompt=torch.tensor([False,True]),
        reference=torch.tensor([True,False]),
        joint=torch.tensor([False,False]),
    )
    text,pool,sub=select_official_mvadapter_conditions_v1(
        conditional_prompt=positive,conditional_pooled=pooled,
        empty_prompt=negative,empty_pooled=empty_pooled,
        reference_cache=cache,dropout=flags,
    )
    assert torch.all(text[0]==1) and torch.all(text[1]==-5)
    assert torch.all(pool[0]==1) and torch.all(pool[1]==-10)
    for name,feature in sub.features.items():
        assert feature.shape==(24,4,3)
        assert feature[:12].abs().sum()==0
        assert torch.equal(feature[12:],cache.features[name][12:])
        assert not torch.equal(cache.features[name][:12],feature[:12])


def test_joint_dropout_always_drops_text_and_both_canonical_images():
    cache=_cache(2)
    policy=OfficialMVAdapterGuidancePolicyV1()
    # All masks generated on source IDs and never drawn separately for
    # the six views or RGB/normal twins.
    masks=draw_official_mvadapter_dropout_v1(
        2,device=torch.device("cpu"),policy=policy,
        generator=torch.Generator().manual_seed(1907),
    )
    masks.validate(2)
    assert masks.prompt.shape==(2,)
    assert bool((~masks.joint | (masks.prompt&masks.reference)).all())
    invalid=MVAdapterSubjectDropoutV1(
        prompt=torch.tensor([False,False]),
        reference=torch.tensor([False,True]),
        joint=torch.tensor([True,False]),
    )
    with pytest.raises(ValueError,match="SUBJECT_DROPOUT_INVALID"):
        invalid.validate(2)
    unconditional=zero_native_mvadapter_reference_cache_v1(cache)
    for name,val in unconditional.features.items():
        assert torch.equal(val,torch.zeros_like(val))
        assert torch.equal(cache.features[name],_cache(2).features[name])


def test_native_classifier_free_guidance_formula_uses_real_empty_prompt():
    epsilon_negative=torch.tensor([[-1.,0.,1.]])
    epsilon_conditional=torch.tensor([[2.,3.,4.]])
    guidance=OfficialMVAdapterGuidancePolicyV1()
    guided=epsilon_negative+guidance.inference_guidance_scale*(
        epsilon_conditional-epsilon_negative
    )
    assert torch.equal(guided,torch.tensor([[8.,9.,10.]]))
    assert not torch.equal(guided,epsilon_conditional)
