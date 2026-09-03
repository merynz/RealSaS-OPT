"""Compiler-owned substrate assembly boundaries.

This package converts admitted observation evidence into current Compiler mechanical
substrate state. Learned models stop before this boundary.
"""

from .iris_v2 import (
    attach_dtb_nd1_from_evidence,
    attach_observed_local_relations_v2,
    build_persistence_groups_v2,
    compile_surface_v2,
)

__all__ = [
    "attach_dtb_nd1_from_evidence",
    "attach_observed_local_relations_v2",
    "build_persistence_groups_v2",
    "compile_surface_v2",
]
