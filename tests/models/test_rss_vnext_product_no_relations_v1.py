from __future__ import annotations

"""Production has no learned Drawable/Cut/Directed-Attachment relation truth."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.attachment_face_correspondence_v1 import (
    RSSvNextSocketFaceNeuralOutputV1, propose_socket_face_pairs_v1,
)


def test_product_region_context_has_no_untrained_typed_relation_paths():
    torch.manual_seed(913)
    torch.set_num_threads(1)
    assert RSSvNextConfig().typed_region_relation_policy == "PRODUCT_NO_TYPED_RELATIONS"
    cfg=replace(
        RSSvNextConfig.tiny_for_test(),
        typed_region_relation_policy="PRODUCT_NO_TYPED_RELATIONS",
    )
    model=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    )
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros(1,2,13)
    cam[:,:,3]=cam[:,:,7]=cam[:,:,11]=cam[:,:,12]=1.
    state,_,_=model.encode_observations(rgba,cam)
    assert torch.count_nonzero(state.region_relation_logits) == 0
    assert torch.allclose(state.region_relation_tokens,state.region_slot_features)
    # Product still computes region membership / same-region source evidence.
    assert state.semantic_logits.shape[-1] == cfg.region_slots+1
    out=state.region_relation_tokens.square().mean()
    out.backward()
    assert model.semantic_structure.learned_region_queries.grad is not None
    assert model.semantic_structure.pair[-1].weight.grad is None
    assert model.region_graph_to_memory[-1].weight.grad is None
    with pytest.raises(ValueError,match="PRODUCT_TYPED_QUERY_RELATIONS_DISABLED"):
        model._relation_logits(
            state,torch.tensor([[[0,1]]],dtype=torch.long),
        )


def test_face_anchor_pair_is_not_given_fake_half_probable_attachment():
    output=RSSvNextSocketFaceNeuralOutputV1(
        anchor_event_indices=(5,),
        socket_control_indices=(3,),
        held_face_logits=torch.tensor([[10.,-10.,-20.]]),
        holder_face_logits=torch.tensor([[-10.,10.,-20.]]),
        face_component_ids=torch.tensor([0,1]),
        face_region_probabilities=torch.tensor([
            [.9,.05,.05],[.05,.9,.05],
        ]),
        region_relation_logits=None,
        mesh_topology_digest="example",axis_sequence_digest="example",
    )
    p=propose_socket_face_pairs_v1(output)
    assert len(p)==1
    assert (p[0].held_face_index,p[0].holder_face_index)==(0,1)
    assert p[0].predicted_directed_relation is None
    assert p[0].state=="UNQUALIFIED_PREDICTED_PAIR_NO_TYPED_RELATION"
    assert not p[0].attachment_qualified
