from __future__ import annotations

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_capacity_target_v2 import (
    POLICY_K0,
    POLICY_K1,
    POLICY_K2,
    build_mechanical_capacity_target_v2,
)


def _fixture():
    # 0 -> 1 -> 2 and an unskinned leaf 3 from 1.
    parents=np.asarray([-1,0,1,1],dtype=np.int64)
    deform=np.asarray([1,1,1,1],dtype=bool)
    heads=np.asarray([
        [0.0,0.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,2.0],
        [1.0,0.0,1.0],
    ],dtype=np.float64)
    skin=np.zeros((3,4),dtype=np.float64)
    skin[:,2]=1.0
    return parents,deform,skin,heads


def _source_set(target):
    return set(map(int,target.source_indices_provenance_only.tolist()))


def test_k2_is_intermediate_when_only_one_zero_mass_control_is_necessary():
    parents,deform,skin,heads=_fixture()

    k0=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,
        bone_heads_world=heads,policy=POLICY_K0,
    )
    k1=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,
        bone_heads_world=heads,policy=POLICY_K1,
    )
    necessary=np.asarray([0,1,0,0],dtype=bool)
    k2=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,
        bone_heads_world=heads,policy=POLICY_K2,
        necessary_control_mask=necessary,
    )

    assert _source_set(k0)=={2}
    assert _source_set(k2)=={1,2}
    assert _source_set(k1)=={0,1,2,3}
    assert _source_set(k0) < _source_set(k2) < _source_set(k1)


def test_k2_does_not_implicitly_admit_all_deform_controls():
    parents,deform,skin,heads=_fixture()
    necessary=np.zeros(4,dtype=bool)
    k0=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,
        bone_heads_world=heads,policy=POLICY_K0,
    )
    k2=build_mechanical_capacity_target_v2(
        parents=parents,deform_mask=deform,skin=skin,
        bone_heads_world=heads,policy=POLICY_K2,
        necessary_control_mask=necessary,
    )
    assert _source_set(k2)==_source_set(k0)
