from __future__ import annotations

"""Token-by-token exact-weight causal KV parity on AXIS V6 event grammar."""
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    AXISRigProgramModelV6,
    AXISSequenceV6,
    RigProgramEventV6 as E,
    RigTokenKindV6 as K,
)
from models.rss_vnext.axis_causal_kv_cache_v1 import AXISV6CausalKVCacheV1


def _events():
    return (
        E(K.BOS),
        E(K.JOINT,pointer_a=-1,coord_indices=(1,2,3)),
        E(K.JOINT,pointer_a=0,coord_indices=(3,4,5)),
        E(K.FRAMES_START),
        E(K.FRAME,pointer_a=0,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.FRAME,pointer_a=1,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.CONTROLS_START),
        E(K.CHAINS_START),
        E(K.CHAIN,pointer_a=0,pointer_b=1),
        E(K.CONSTRAINTS_START),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
        E(K.SKIN_CODE,skin_code_id=0),
        E(K.SKIN_CODE,skin_code_id=1),
        E(K.SKIN_CODE,skin_code_id=2),
        E(K.SKIN_CODE,skin_code_id=3),
        E(K.EOS),
    )


@pytest.mark.parametrize("layers",[1,2])
def test_kv_cache_matches_full_prefix_all_heads_and_hidden(layers):
    torch.manual_seed(20261119)
    torch.set_num_threads(1)
    model=AXISRigProgramModelV6(AXISRigProgramConfigV6(
        scene_dim=32,hidden_dim=32,layers=layers,heads=4,ff_mult=2,
        coord_bins=16,skin_vocab_size=16,max_sequence_length=64,
        max_pointer_entities=32,dropout=0.,
    )).eval()
    scene=torch.randn(1,5,32)
    events=_events()
    with torch.no_grad():
        cache=AXISV6CausalKVCacheV1(model,scene)
        for i,event in enumerate(events):
            full=model(scene,AXISSequenceV6.from_events(events[:i+1],device=scene.device))
            cached=cache.consume(event)
            assert full.keys()==cached.keys()
            for key in full:
                torch.testing.assert_close(
                    cached[key],full[key][:,-1:,:] if key!="coord_logits"
                    else full[key][:,-1:,:,:],
                    atol=4.e-5,rtol=2.e-4,
                    msg=f"prefix={i+1},head={key}",
                )
            assert cache.count==i+1
            for key in cache.keys:
                assert key.shape==(1,4,i+1,8)
            for value in cache.values:
                assert value.shape==(1,4,i+1,8)


def test_generation_cache_is_eval_only_and_resource_bounded():
    torch.manual_seed(1)
    model=AXISRigProgramModelV6(AXISRigProgramConfigV6(
        scene_dim=16,hidden_dim=16,layers=1,heads=4,
        max_sequence_length=2,max_pointer_entities=16,
    ))
    scene=torch.zeros(1,2,16)
    with torch.no_grad():
        with pytest.raises(ValueError,match="EVAL_BATCH_ONE_REQUIRED"):
            AXISV6CausalKVCacheV1(model,scene)
        model.eval()
        cache=AXISV6CausalKVCacheV1(model,scene)
        cache.consume(E(K.BOS))
        cache.consume(E(K.JOINT,pointer_a=-1))
        with pytest.raises(ValueError,match="EVENT_RESOURCE_GUARD"):
            cache.consume(E(K.JOINT,pointer_a=0))
    with pytest.raises(ValueError,match="EVAL_BATCH_ONE_REQUIRED"):
        AXISV6CausalKVCacheV1(model,scene)
