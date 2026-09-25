from __future__ import annotations

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanically_meaningful_target_v2 import (
    ROLE_NAMES,
    build_mechanically_meaningful_target_v2,
)


def test_mechanically_meaningful_target_retains_function_not_helpers():
    # 0 root-motion anchor
    # 1-2 skin-supported deform chain
    # 3 zero-skin terminal continuation from 2
    # 4 detached zero-skin helper branch from root (must stay excluded)
    parents = np.asarray([-1, 0, 1, 2, 0], dtype=np.int64)
    deform = np.ones((5,), dtype=bool)
    heads = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.45],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    tails = np.asarray(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 1.8],
            [2.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    skin = np.zeros((4, 5), dtype=np.float64)
    skin[:, 1] = 0.5
    skin[:, 2] = 0.5

    target = build_mechanically_meaningful_target_v2(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads,
        bone_tails_world=tails,
    )

    assert target.count == 5
    assert int(np.count_nonzero(target.root_mask)) == 1
    assert target.role_counts == {
        "SKIN_SUPPORTED_ARTICULATION": 2,
        "STRUCTURAL_BRIDGE": 0,
        "ROOT_MOTION_ANCHOR": 1,
        "TERMINAL_EXTENSION_SOURCE": 1,
        "TERMINAL_TIP_SYNTHETIC": 1,
    }
    source = set(int(x) for x in target.source_indices_provenance_only.tolist())
    assert 4 not in source
    assert -1 in source

    # Parent-before-child is a hard training contract.
    for child, parent in enumerate(target.parent_indices.tolist()):
        assert parent < child or parent == -1

    assert set(ROLE_NAMES.values()) == {
        "SKIN_SUPPORTED_ARTICULATION",
        "STRUCTURAL_BRIDGE",
        "ROOT_MOTION_ANCHOR",
        "TERMINAL_EXTENSION_SOURCE",
        "TERMINAL_TIP_SYNTHETIC",
    }
