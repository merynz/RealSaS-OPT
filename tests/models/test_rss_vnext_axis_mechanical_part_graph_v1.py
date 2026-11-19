from __future__ import annotations

"""AXIS can LEARN multi-branch mechanical parts without semantic invention."""
from dataclasses import replace,asdict

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6,AXISRigProgramModelV6,AXISRigProgramConfigV6,
    RigProgramEventV6 as E,RigTokenKindV6 as K,
    qualified_axis_rig_skin_sequence_loss_v6,
)
from models.axis.rig_program_decode_v6 import (
    AXISV6CodecBinding,decode_axis_v6_events,
)
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from models.rss_vnext.axis_mechanical_part_graph_v1 import (
    candidate_axis_part_pointer_choices_v1,
    decode_axis_mechanical_part_graph_v1,
)
from models.rss_vnext.hierarchical_skin_field_v1 import (
    axis_joint_group_ids_v1,axis_joint_group_ids_from_cas_events_v1,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import qualify_rig_program_v1


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(20261119)
    torch.set_num_threads(1)


def _events(*, part_ids=(0,0), disconnected=False):
    # Shared shoulder with two branches. A wing can be a tree, NOT one
    # parent-contiguous ancestor-to-end chain.
    parents=(-1,0,1,0,3)
    result=[E(K.BOS)]
    for p in parents:
        result.append(E(K.JOINT,coord_indices=(128,128,128),pointer_a=p))
    result.append(E(K.FRAMES_START))
    for i in range(len(parents)):
        result.append(E(K.FRAME,pointer_a=i,
                        frame_6d=(1.,0.,0.,0.,1.,0.)))
    result.extend([E(K.CONTROLS_START),E(K.CHAINS_START)])
    result.append(E(K.CHAIN,pointer_a=1 if disconnected else 0,
                    pointer_b=2,pointer_c=part_ids[0]))
    result.append(E(K.CHAIN,pointer_a=3 if disconnected else 0,
                    pointer_b=4,pointer_c=part_ids[1]))
    result.extend([E(K.CONSTRAINTS_START),E(K.ANCHORS_START),E(K.SKIN_START)])
    for _ in range(len(parents)*2):
        result.append(E(K.SKIN_CODE,skin_code_id=1))
    result.append(E(K.EOS))
    return tuple(result)


def _decode(events):
    cfg=AXISRigProgramConfigV6(
        scene_dim=32,hidden_dim=32,layers=1,heads=4,
        skin_vocab_size=16,max_sequence_length=128,
    )
    rig,skin=decode_axis_v6_events(
        events,config=cfg,normalization_center=(0.,0.,0.),
        normalization_scale=2.,
        mesh_lineage_hash="1"*64,feature_evidence_binding_hash="2"*64,
        carrier_evidence_hash="3"*64,carrier_topology_hash="4"*64,
        carrier_geometry_hash="5"*64,
        model_provenance="test:part",codec_binding=AXISV6CodecBinding(
            MIRA_V6_CODEC_ID,"a"*64,16,2,
        ),
    )
    return rig,skin


def test_connected_multi_branch_chains_form_one_explicit_mechanical_part():
    events=_events()
    seq=AXISSequenceV6.from_events(events,device=torch.device("cpu"))
    graph=decode_axis_mechanical_part_graph_v1(seq)
    assert graph.declared_part_count==1
    assert graph.part_by_chain==(0,0)
    assert graph.unknown_joint_count==0
    torch.testing.assert_close(
        axis_joint_group_ids_v1(seq),torch.zeros(5,dtype=torch.long),
        atol=0,rtol=0,
    )
    serialized=[{**asdict(e),"kind":int(e.kind)} for e in events]
    torch.testing.assert_close(
        axis_joint_group_ids_from_cas_events_v1(serialized),
        graph.joint_group_ids,atol=0,rtol=0,
    )
    rig,skin=_decode(events)
    qualified=qualify_rig_program_v1(rig)
    assert qualified.qualification_report["mechanical_part_group_count"]==1
    assert qualified.qualification_report["mechanical_part_semantics_qualified"] is False
    assert all(c.metadata["mechanical_part_ordinal"]==0 for c in rig.chains)
    assert len(skin.rows)==5


def test_two_different_parts_share_attachment_joint_without_stealing_it():
    seq=AXISSequenceV6.from_events(
        _events(part_ids=(0,1)),device=torch.device("cpu"),
    )
    graph=decode_axis_mechanical_part_graph_v1(seq)
    assert graph.declared_part_count==2
    assert graph.unknown_joint_count==1
    torch.testing.assert_close(
        graph.joint_group_ids,torch.tensor([-1,0,0,1,1]),rtol=0,atol=0,
    )


def test_source_without_part_group_evidence_keeps_chain_identity_not_negative_labels():
    seq=AXISSequenceV6.from_events(
        _events(part_ids=(-1,-1)),device=torch.device("cpu"),
    )
    graph=decode_axis_mechanical_part_graph_v1(seq)
    assert graph.declared_part_count==0
    assert graph.part_by_chain==(-1,-1)
    torch.testing.assert_close(
        graph.joint_group_ids,torch.tensor([-1,0,0,1,1]),rtol=0,atol=0,
    )


def test_noncausal_part_reference_or_disconnected_joint_path_rejected():
    with pytest.raises(ValueError,match="NONCAUSAL_GROUP_ADDRESS"):
        decode_axis_mechanical_part_graph_v1(AXISSequenceV6.from_events(
            _events(part_ids=(2,2)),device=torch.device("cpu"),
        ))
    with pytest.raises(ValueError,match="PART_GROUP_DISCONNECTED_CHAIN"):
        _decode(_events(disconnected=True))
    parents=(-1,0,1,0,3)
    choices=candidate_axis_part_pointer_choices_v1(
        parents=parents,existing_chains=((1,2,0),),
        start_joint=3,end_joint=4,
    )
    # The disconnected branch CANNOT join old part zero; new part or
    # explicit UNKNOWN are the only legal causal choices.
    assert choices==(-1,1)


def test_exact_source_part_graph_teacher_updates_existing_pointer_c_head():
    cfg=AXISRigProgramConfigV6(
        scene_dim=32,hidden_dim=32,layers=1,heads=4,
        skin_vocab_size=16,max_sequence_length=128,
    )
    model=AXISRigProgramModelV6(cfg).train()
    teacher=AXISSequenceV6.from_events(
        _events(),device=torch.device("cpu"),
    )
    def train(qualified:bool):
        return qualified_axis_rig_skin_sequence_loss_v6(
            model,torch.randn(1,3,32),teacher,
            tokens_per_joint=2,
            executable_constraint_operators=("IK","AIM"),
            control_intervention_qualified=False,
            structural_chains_qualified=True,
            frame_dof_qualified=False,
            mechanical_part_graph_qualified=qualified,
        )
    with pytest.raises(ValueError,match="PART_TEACHER_PROVENANCE_REQUIRED"):
        train(False)
    loss,parts=train(True)
    assert torch.isfinite(loss)
    assert "pointer_c_logits" in parts
    loss.backward()
    grad=model.pointer_c_head.weight.grad
    assert grad is not None and float(grad.abs().sum())>0


def test_compiler_rejects_forged_canonical_chain_part_metadata():
    rig,_=_decode(_events(part_ids=(0,1)))
    chain=list(rig.chains)
    chain[1]=replace(chain[1],metadata={
        **chain[1].metadata,
        "mechanical_part_ordinal":9,
    })
    from compiler.realsas_compiler_core.rig_program_ir_v1 import (
        seal_rig_program_proposal_v1,
    )
    forged=seal_rig_program_proposal_v1(replace(
        rig,chains=tuple(chain),proposal_lineage_hash="",
    ))
    with pytest.raises(Exception,match="MECHANICAL_PART_ADDRESS_INVALID"):
        qualify_rig_program_v1(forged)
