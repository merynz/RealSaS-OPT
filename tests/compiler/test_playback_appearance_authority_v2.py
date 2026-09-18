from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.playback_appearance_authority_v2 import (
    resolve_global_source_appearance_authority_v2,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import AppearanceProvenance


def _forwards():
    return {
        "V0": (0.0, 0.0, 1.0),
        "V1": (1.0, 0.0, 1.0),
        "V2": (1.0, 0.0, 0.0),
        "V3": (1.0, 0.0, -1.0),
        "V4": (0.0, 0.0, -1.0),
        "V5": (-1.0, 0.0, -1.0),
        "V6": (-1.0, 0.0, 0.0),
        "V7": (-1.0, 0.0, 1.0),
    }


def test_unseen_means_absent_in_all_required_views():
    views = tuple(f"V{i}" for i in range(8))
    direct = {view: np.zeros(3, dtype=np.bool_) for view in views}
    direct["V0"][0] = True
    direct["V3"][1] = True

    authority = resolve_global_source_appearance_authority_v2(direct, _forwards())

    assert authority.globally_unseen_mask.tolist() == [False, False, True]
    unseen_code = 3
    for view in views:
        codes = authority.provenance_codes_by_view[view]
        assert codes[2] == unseen_code
        assert codes[0] != unseen_code
        assert codes[1] != unseen_code


def test_target_missing_source_uses_stable_other_view_donor_not_unseen():
    views = tuple(f"V{i}" for i in range(8))
    direct = {view: np.zeros(1, dtype=np.bool_) for view in views}
    direct["V1"][0] = True

    authority = resolve_global_source_appearance_authority_v2(direct, _forwards())

    assert authority.provenance_codes_by_view["V1"][0] == 0
    assert authority.donor_view_indices_by_view["V1"][0] == 1
    assert authority.provenance_codes_by_view["V0"][0] == 1
    assert authority.donor_view_indices_by_view["V0"][0] == 1
    assert not bool(authority.globally_unseen_mask[0])


def test_motion_never_participates_in_source_donor_selection():
    views = tuple(f"V{i}" for i in range(8))
    direct = {view: np.zeros(2, dtype=np.bool_) for view in views}
    direct["V1"][:] = True
    direct["V7"][:] = True

    a = resolve_global_source_appearance_authority_v2(direct, _forwards())
    b = resolve_global_source_appearance_authority_v2(direct, _forwards())

    assert a.authority_hash == b.authority_hash
    for view in views:
        assert np.array_equal(a.provenance_codes_by_view[view], b.provenance_codes_by_view[view])
        assert np.array_equal(a.donor_view_indices_by_view[view], b.donor_view_indices_by_view[view])
