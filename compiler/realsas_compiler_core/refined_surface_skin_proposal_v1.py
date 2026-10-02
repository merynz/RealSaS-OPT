from __future__ import annotations

"""Propose parent-constant skin transport through an exact compaction refinement.

This is a measurement hypothesis, not a learned prediction or a mechanics seal.
No nearest-neighbour matching, smoothing, anatomy, or subject identifiers enter it.
Both inverse arrays must address the same caller-verified dense source in the same
order. Every target cluster must be a subset of exactly one source cluster.
"""

import numpy as np

from .hashing import content_sha256
from .types import QualificationError, SkinInfluenceProposal, SkinProposalIR


def propose_refined_surface_skin_v1(
    *, source_surface, target_surface, skeleton, source_skin,
    dense_positions, source_inverse, target_inverse, dense_source_sha256,
):
    if not isinstance(dense_source_sha256, str) or len(dense_source_sha256) != 64:
        raise QualificationError("REFINED_SKIN_DENSE_SOURCE_ID_MISSING")
    if source_skin.surface_binding_hash != source_surface.geometry_lineage_hash:
        raise QualificationError("REFINED_SKIN_SOURCE_SURFACE_DRIFT")
    if source_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("REFINED_SKIN_SKELETON_DRIFT")
    xyz = np.asarray(dense_positions, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise QualificationError("REFINED_SKIN_DENSE_POSITIONS_INVALID")

    def verify(surface, inverse):
        raw = np.asarray(inverse)
        nodes = surface.surface_nodes
        if (raw.shape != (len(xyz),) or raw.dtype.kind not in "iu"
                or not nodes or np.any(raw < 0) or np.any(raw >= len(nodes))):
            raise QualificationError("REFINED_SKIN_INVERSE_INVALID")
        inv = raw.astype(np.int64)
        counts = np.bincount(inv, minlength=len(nodes))
        if np.any(counts == 0) or len({n.surface_id for n in nodes}) != len(nodes):
            raise QualificationError("REFINED_SKIN_CLUSTER_ACCOUNTING_INVALID")
        centers = np.zeros((len(nodes), 3), dtype=np.float64)
        np.add.at(centers, inv, xyz)
        centers /= counts[:, None]
        positions = np.asarray([n.P for n in nodes], dtype=np.float64)
        if not np.isfinite(positions).all() or not np.allclose(centers, positions, rtol=0, atol=1e-9):
            raise QualificationError("REFINED_SKIN_NODE_ORDER_OR_POSITION_DRIFT")
        return inv

    src = verify(source_surface, source_inverse)
    dst = verify(target_surface, target_inverse)
    pairs = np.unique(np.column_stack((dst, src)), axis=0)
    if len(pairs) != len(target_surface.surface_nodes):
        raise QualificationError("REFINED_SKIN_CROSS_SOURCE_CLUSTER_MERGE")
    parents = dict((int(child), int(parent)) for child, parent in pairs)
    rows = {r.surface_id: r for r in source_skin.rows}
    if len(rows) != len(source_skin.rows) or set(rows) != {n.surface_id for n in source_surface.surface_nodes}:
        raise QualificationError("REFINED_SKIN_SOURCE_ROW_ACCOUNTING_INVALID")
    joints = {j.canonical_joint_id for j in skeleton.joints}
    for row in rows.values():
        weights = dict(row.influences)
        if (len(weights) != len(row.influences) or not weights or not set(weights) <= joints
                or any(not np.isfinite(w) or w < 0 for w in weights.values())
                or abs(sum(weights.values()) - 1.0) > 1e-8):
            raise QualificationError("REFINED_SKIN_SOURCE_SIMPLEX_INVALID")
    correspondence = tuple(
        (node.surface_id, source_surface.surface_nodes[parents[i]].surface_id)
        for i, node in enumerate(target_surface.surface_nodes)
    )
    receipt = {
        "schema": "RealSaS.RefinedSurfaceSkinProposalReceipt.v1",
        "method": "EXACT_DENSE_CLUSTER_PARENT_CONSTANT_PROPOSAL",
        "source_surface_hash": source_surface.geometry_lineage_hash,
        "target_surface_hash": target_surface.geometry_lineage_hash,
        "source_skin_hash": source_skin.skin_lineage_hash,
        "source_skin_payload_hash": content_sha256(source_skin.to_dict()),
        "skeleton_hash": skeleton.skeleton_lineage_hash,
        "dense_source_sha256": dense_source_sha256,
        "source_inverse_hash": content_sha256(src.tolist()),
        "target_inverse_hash": content_sha256(dst.tolist()),
        "correspondence_hash": content_sha256(correspondence),
        "source_node_count": len(source_surface.surface_nodes),
        "target_node_count": len(target_surface.surface_nodes),
        "cross_source_merge_count": 0,
        "parent_weight_values_preserved": True,
        "new_position_prediction_claimed": False,
        "model_inference_used": False,
        "product_authority_minted": False,
        "requires_dynamic_requalification": True,
    }
    receipt["receipt_hash"] = content_sha256(receipt)
    proposal = SkinProposalIR(
        tuple(SkinInfluenceProposal(child, jid, float(weight))
              for child, parent in correspondence
              for jid, weight in rows[parent].influences),
        target_surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash,
        model_provenance="COMPILER_EXACT_COMPACTION_REFINEMENT_TRANSPORT_V1",
        metadata={"refinement_transport_receipt": receipt},
    )
    return proposal, receipt
