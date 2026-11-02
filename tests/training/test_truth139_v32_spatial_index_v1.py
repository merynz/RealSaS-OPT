from __future__ import annotations

import sys
import types

from tools.training.truth139_v32_spatial_index_v1 import new_rtree3


def test_truth139_v32_rtree_index_is_explicitly_three_dimensional(monkeypatch) -> None:
    observed = {}

    class Property:
        def __init__(self):
            self.dimension = 2

    class Index:
        def __init__(self, *, properties):
            observed["dimension"] = properties.dimension

    fake = types.ModuleType("rtree")
    fake.index = types.SimpleNamespace(Property=Property, Index=Index)
    monkeypatch.setitem(sys.modules, "rtree", fake)

    new_rtree3()
    assert observed["dimension"] == 3
