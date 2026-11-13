from __future__ import annotations

import hashlib
import json

import numpy as np

from .conditioning_v1 import TESSAConditioningV1
from .contracts_v1 import (
    TESSAMeshProposalV1,
    TESSAScalingPolicyV1,
    TESSAVertexProposalV1,
)


def _hash_payload(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assemble_tessa_proposal_v1(
    *,
    vertices_normalized: np.ndarray,
    faces: np.ndarray,
    conditioning: TESSAConditioningV1,
    component_id: str,
    model_provenance: str,
    policy: TESSAScalingPolicyV1 | None = None,
) -> TESSAMeshProposalV1:
    """Assemble decoded TESSA geometry into a typed learned proposal.

    Primary GSA anchors are deterministic routing hints only.  The Compiler must
    resolve/qualify a real SurfaceSupportBinding before a vertex can enter the
    canonical mechanical carrier.
    """
    policy = policy or TESSAScalingPolicyV1()
    policy.validate()
    v = np.asarray(vertices_normalized, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("TESSA_DECODED_VERTEX_ARRAY_INVALID")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("TESSA_DECODED_FACE_ARRAY_INVALID")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("TESSA_DECODED_FACE_INDEX_INVALID")
    if not policy.admits(vertex_count=len(v), face_count=len(f)):
        raise ValueError("TESSA_DECODED_MESH_EXCEEDS_POLICY")

    source_features = conditioning.features.detach().cpu().numpy()
    source_xyz = np.asarray(source_features[:, :3], dtype=np.float64)
    if len(source_xyz) != len(conditioning.surface_ids):
        raise ValueError("TESSA_CONDITIONING_SURFACE_ID_COUNT_DRIFT")
    allowed = np.asarray(
        [cid == component_id or cid == "UNASSIGNED" for cid in conditioning.component_ids],
        dtype=bool,
    )
    if not np.any(allowed):
        raise ValueError("TESSA_COMPONENT_HAS_NO_GSA_SUPPORT")
    allowed_indices = np.nonzero(allowed)[0]
    allowed_xyz = source_xyz[allowed_indices]

    # This nearest point is not a support binding.  It only routes the proposal
    # into the right local GSA neighborhood for deterministic Compiler support
    # resolution.  Cross-component search is forbidden here.
    diff = v[:, None, :] - allowed_xyz[None, :, :]
    nearest_local = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
    nearest = allowed_indices[nearest_local]

    center = np.asarray(conditioning.center, dtype=np.float64)
    world = v * float(conditioning.scale) + center[None, :]
    vertex_ids = tuple(f"TESSA_V1:{i:08d}" for i in range(len(v)))
    vertices = tuple(
        TESSAVertexProposalV1(
            proposal_vertex_id=vertex_ids[i],
            P=tuple(map(float, world[i].tolist())),
            primary_surface_id=str(conditioning.surface_ids[int(nearest[i])]),
            component_id=str(component_id),
            metadata={
                "anchor_semantics": "ROUTING_HINT_ONLY_NOT_SUPPORT_BINDING",
                "source_geometry_lineage_hash": conditioning.source_geometry_lineage_hash,
            },
        )
        for i in range(len(v))
    )
    faces_ids = tuple(tuple(vertex_ids[int(j)] for j in row) for row in f.tolist())
    topology_hash = _hash_payload(
        {
            "faces": faces_ids,
            "source_geometry_lineage_hash": conditioning.source_geometry_lineage_hash,
            "component_id": str(component_id),
        }
    )
    proposal = TESSAMeshProposalV1(
        vertices=vertices,
        faces=faces_ids,
        source_geometry_lineage_hash=conditioning.source_geometry_lineage_hash,
        model_provenance=str(model_provenance),
        topology_sequence_hash=topology_hash,
        metadata={
            "authority_class": "LEARNED_PROPOSAL",
            "component_id": str(component_id),
            "support_binding_qualified": False,
            "compiler_projection_required": True,
        },
    )
    proposal.validate(policy)
    return proposal
