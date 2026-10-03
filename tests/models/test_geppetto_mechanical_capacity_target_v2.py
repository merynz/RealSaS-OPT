from __future__ import annotations

import numpy as np
import pytest

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_capacity_target_v2 import (
    POLICY_K0,
    POLICY_K1,
    POLICY_K2,
    POLICY_K3,
    build_mechanical_capacity_target_v2,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    build_mechanical_core_target_v1,
)


def _fixture():
    # 0 assembly root
    # 1 bridge
    # 2 deform + skin
    # 3 deform low/no skin sibling
    # 4 helper child of 2
    # 5 helper leaf from assembly root
    # 6 bridge between 2 and deform child 7
    # 7 deform + skin
    parents=np.asarray([-1,0,1,1,2,0,2,6],np.int64)
    deform=np.asarray([0,0,1,1,0,0,0,1],bool)
    skin=np.zeros((4,8),np.float64)
    skin[:,2]=0.5
    skin[:,7]=0.5
    heads=np.asarray([
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,2.,0.],
        [1.,2.,0.],
        [0.,3.,0.],
        [-1.,1.,0.],
        [0.,2.5,0.],
        [0.,3.5,0.],
    ],np.float64)
    return parents,deform,skin,heads


def test_k0_is_exact_current_core_selection_and_geometry():
    parents,deform,skin,heads=_fixture()
    old=build_mechanical_core_target_v1(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads
    )
    new=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
        policy=POLICY_K0,
    )
    assert np.array_equal(new.positions_world,old.positions_world)
    assert np.array_equal(new.parent_indices,old.parent_indices)
    assert np.array_equal(new.root_mask,old.root_mask)
    assert np.array_equal(
        new.source_indices_provenance_only,
        old.source_indices_provenance_only,
    )


def test_k1_admits_all_deform_and_required_bridge_without_count_target():
    parents,deform,skin,heads=_fixture()
    t=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
        policy=POLICY_K1,
    )
    selected=set(t.source_indices_provenance_only.tolist())
    assert {2,3,7}.issubset(selected)
    assert 6 in selected  # structural bridge 2 -> 7
    assert 5 not in selected
    assert t.count==len(selected)


def test_k2_requires_independent_necessary_mask_and_can_admit_helper():
    parents,deform,skin,heads=_fixture()
    with pytest.raises(ValueError,match="necessary_control_mask"):
        build_mechanical_capacity_target_v2(
            parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
            policy=POLICY_K2,
        )
    necessary=np.zeros(len(parents),bool)
    necessary[4]=True
    t=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
        policy=POLICY_K2,necessary_control_mask=necessary,
    )
    selected=set(t.source_indices_provenance_only.tolist())
    assert 4 in selected
    assert {2,3,7}.issubset(selected)


def test_k3_is_full_legal_diagnostic_ceiling():
    parents,deform,skin,heads=_fixture()
    t=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
        policy=POLICY_K3,
    )
    assert t.count==len(parents)
    assert set(t.source_indices_provenance_only.tolist())==set(range(len(parents)))


def test_capacity_receipt_is_not_source_row_identity_hash():
    parents,deform,skin,heads=_fixture()
    a=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,bone_heads_world=heads,
        policy=POLICY_K3,
    )
    assert len(a.policy_receipt_hash)==64
    # Provenance rows remain available for audits, but the target is anonymous:
    # no source names or requested cardinality enter the constructor.
    assert not hasattr(a,"source_names")
    assert not hasattr(a,"target_joint_count")
