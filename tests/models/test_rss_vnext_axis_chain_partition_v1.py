from __future__ import annotations

"""Source-native chain truth must replay from AXIS's own causal program."""
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6 as E, RigTokenKindV6 as K,
)
from models.rss_vnext.axis_mechanical_chain_partition_v1 import (
    assert_qualified_chain_partition_matches_axis_v1,
    decode_axis_joint_chain_partition_v1,
)


def _seq(*, second_start=0, first_end=2):
    return AXISSequenceV6.from_events([
        E(K.BOS),
        E(K.JOINT, pointer_a=-1),  # joint 0: shared attachment root
        E(K.JOINT, pointer_a=0),   # joint 1: body+cloth chain
        E(K.JOINT, pointer_a=1),   # joint 2: cloth end
        E(K.JOINT, pointer_a=0),   # joint 3: other branch
        E(K.JOINT, pointer_a=3),   # joint 4: wing end
        E(K.FRAMES_START),
        E(K.CONTROLS_START),
        E(K.CHAINS_START),
        E(K.CHAIN, pointer_a=0, pointer_b=first_end),
        E(K.CHAIN, pointer_a=second_start, pointer_b=4),
        E(K.CONSTRAINTS_START),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
        E(K.EOS),
    ],device=torch.device("cpu"))


def test_axis_partition_is_causal_and_masks_shared_root():
    seq=_seq()
    out=decode_axis_joint_chain_partition_v1(seq)
    assert out.causal_chain_paths==((0,1,2),(0,3,4))
    assert out.parent_joint_indices==(-1,0,1,0,3)
    assert out.joint_chain_ids.tolist()==[-1,0,0,1,1]
    assert out.uncertain_joint_count==1
    assert_qualified_chain_partition_matches_axis_v1(
        seq,torch.tensor([-1,90,90,12,12],dtype=torch.long),
    )


def test_source_partition_may_rename_but_never_invent_ownership():
    seq=_seq()
    with pytest.raises(ValueError,match="UNKNOWN_OWNERSHIP"):
        assert_qualified_chain_partition_matches_axis_v1(
            seq,torch.tensor([1,90,90,12,12]),
        )
    with pytest.raises(ValueError,match="GROUPING_DRIFT"):
        assert_qualified_chain_partition_matches_axis_v1(
            seq,torch.tensor([-1,90,91,12,12]),
        )
    with pytest.raises(ValueError,match="SOURCE_PARTITION_DOMAIN_INVALID"):
        assert_qualified_chain_partition_matches_axis_v1(
            seq,torch.tensor([-1,90,90,12,12],dtype=torch.float32),
        )


def test_nonancestral_chain_and_noncausal_joint_refuse_teachers():
    with pytest.raises(ValueError,match="NONANCESTRAL_PATH"):
        decode_axis_joint_chain_partition_v1(_seq(second_start=2))
    seq=_seq()
    seq.pointer_a[0,2]=4  # joint 1 cannot parent a future joint
    with pytest.raises(ValueError,match="PARENT_NOT_CAUSAL"):
        decode_axis_joint_chain_partition_v1(seq)


def test_overlapping_chain_member_is_unknown_not_assigned_arbitrarily():
    out=decode_axis_joint_chain_partition_v1(_seq(second_start=3))
    assert out.causal_chain_paths==((0,1,2),(3,4))
    # Only chain 0 owns the root in this disjoint-path example.
    assert out.joint_chain_ids.tolist()==[0,0,0,1,1]
