from __future__ import annotations

"""Exact array materialization for one frozen carrier M + carrier-native W_M.

No surface interpolation, vertex nearest-neighbour match, or semantic skin transport
is allowed here. Row identity is the exact ordered Stage19 carrier vertex id.
"""

import numpy as np

from .types import QualificationError


def carrier_skin_matrix_v1(carrier, skeleton, skin):
    if skin.carrier_evidence_hash != carrier.carrier_evidence_hash:
        raise QualificationError("CARRIER_MECHANICS_EVIDENCE_BINDING_DRIFT")
    if skin.carrier_topology_hash != carrier.topology_hash:
        raise QualificationError("CARRIER_MECHANICS_TOPOLOGY_BINDING_DRIFT")
    if skin.carrier_geometry_hash != carrier.geometry_hash:
        raise QualificationError("CARRIER_MECHANICS_GEOMETRY_BINDING_DRIFT")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("CARRIER_MECHANICS_SKELETON_BINDING_DRIFT")

    vertex_ids = tuple(map(str, carrier.ordered_vertex_ids))
    row_by_id = {str(row.carrier_vertex_id): row for row in skin.rows}
    if len(row_by_id) != len(skin.rows) or set(row_by_id) != set(vertex_ids):
        raise QualificationError("CARRIER_MECHANICS_SKIN_ROW_IDENTITY_DRIFT")
    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}
    if len(joint_index) != len(joint_ids):
        raise QualificationError("CARRIER_MECHANICS_JOINT_ID_DUPLICATE")

    weights = np.zeros((len(vertex_ids), len(joint_ids)), dtype=np.float64)
    for vi, vid in enumerate(vertex_ids):
        total = 0.0
        seen = set()
        for jid, raw_weight in row_by_id[vid].influences:
            jid = str(jid)
            if jid in seen or jid not in joint_index:
                raise QualificationError("CARRIER_MECHANICS_SKIN_JOINT_INVALID")
            seen.add(jid)
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise QualificationError("CARRIER_MECHANICS_SKIN_WEIGHT_INVALID")
            weights[vi, joint_index[jid]] = weight
            total += weight
        if abs(total - 1.0) > 1e-9:
            raise QualificationError("CARRIER_MECHANICS_SKIN_SIMPLEX_DRIFT")

    rest = np.asarray(carrier.positions, dtype=np.float64)
    faces = np.asarray(carrier.face_vertex_indices, dtype=np.int64)
    if rest.shape != (len(vertex_ids), 3):
        raise QualificationError("CARRIER_MECHANICS_POSITION_SHAPE_DRIFT")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise QualificationError("CARRIER_MECHANICS_FACE_SHAPE_DRIFT")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertex_ids):
        raise QualificationError("CARRIER_MECHANICS_FACE_INDEX_DRIFT")
    return rest, weights, faces, joint_ids


__all__ = ["carrier_skin_matrix_v1"]
