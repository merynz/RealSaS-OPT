"""Spatial-index helpers for Truth139 V3.2 mechanical topology courts."""
from __future__ import annotations


def new_rtree3():
    """Return an Rtree index explicitly configured for xyz AABBs.

    Rtree defaults to two dimensions.  Truth139 mechanical courts insert
    six-coordinate xyz boxes, so silently relying on that default is invalid.
    Keep the import local so repository/CI users without the optional runtime
    dependency may still import unrelated Truth139 modules.
    """
    from rtree import index

    properties = index.Property()
    properties.dimension = 3
    return index.Index(properties=properties)
