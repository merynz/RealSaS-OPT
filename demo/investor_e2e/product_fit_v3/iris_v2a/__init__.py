"""IRIS Reprojection-Centered V2-A.

Learned external geometry authority is camera-forward depth only. Canonical world
points are compiled analytically from exact camera rays and predicted depth.
"""

from .camera import OrthoCameraBatch, project_world, backproject_grid_depth
from .q_lattice import QLattice, build_canonical_q_lattice, visual_hull_active_mask

__all__ = [
    "OrthoCameraBatch",
    "project_world",
    "backproject_grid_depth",
    "QLattice",
    "build_canonical_q_lattice",
    "visual_hull_active_mask",
]
