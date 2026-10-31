"""Geometric-topology aware response evidence.

V1 response magnitudes are triangle-local and remain valid, but V1 pair adjacency
used raw render vertex ids. V2 preserves all V1 per-face measurements and
rebuilds pair coherence/jump over the owner-aware exact geometric topology.
"""
from __future__ import annotations
import numpy as np

from models.iris.truth.geometric_topology_v1 import face_adjacency_from_geometric_topology_v1
from models.iris.truth.presentation_response_v1 import (
    build_surface_response_arrays_v1,
    control_balanced_action_weights_v1,
)


def build_surface_response_arrays_v2(
    rest_vertices,
    faces,
    face_owner,
    isolated_position,
    actions,
    *,
    degeneracy_fraction: float = 1.0e-10,
):
    rest = np.asarray(rest_vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    owner = np.asarray(face_owner, dtype=np.int32)
    posed = np.asarray(isolated_position, dtype=np.float64)

    arrays = build_surface_response_arrays_v1(
        rest,
        tri,
        posed,
        actions,
        degeneracy_fraction=degeneracy_fraction,
    )
    weights = control_balanced_action_weights_v1(actions)

    adjacency = face_adjacency_from_geometric_topology_v1(rest, tri, owner)
    rest_centroid = rest[tri].mean(axis=1)
    posed_centroid = posed[:, tri, :].mean(axis=2)
    response = posed_centroid - rest_centroid[None, :, :]
    weighted = response * np.sqrt(weights)[:, None, None]
    flat = np.transpose(weighted, (1, 0, 2)).reshape(len(tri), -1)
    norm = np.linalg.norm(flat, axis=1)

    coherence = np.zeros(len(adjacency), dtype=np.float64)
    valid = np.zeros(len(adjacency), dtype=np.bool_)
    jump = np.zeros(len(adjacency), dtype=np.float64)
    for index, (a, b) in enumerate(adjacency):
        denom = float(norm[a] * norm[b])
        if denom > 1.0e-12:
            value = float(np.clip(np.dot(flat[a], flat[b]) / denom, -1.0, 1.0))
            coherence[index] = value
            jump[index] = 0.5 * (1.0 - value)
            valid[index] = True

    arrays = dict(arrays)
    arrays["response_pair_face"] = adjacency
    arrays["response_pair_coherence"] = coherence
    arrays["response_pair_jump"] = jump
    arrays["response_pair_valid"] = valid
    return arrays
