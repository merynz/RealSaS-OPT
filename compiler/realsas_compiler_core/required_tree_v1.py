from __future__ import annotations

"""Compiler validation/materialization for AXIS-owned hard parent trees.

This path is deliberately separate from the legacy global arborescence optimizer.
AXIS owns one required parent decision per non-root joint. The Compiler owns
admissibility: surface references, one-root/tree legality, connectivity, acyclicity,
parent-before-child serialization when declared, and canonical identity minting.
"""

from .hashing import content_sha256
from .types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualificationError,
    RiggingSurfaceIR,
    SkeletonProposalIR,
)


def _sequence_index(joint) -> int | None:
    raw = dict(joint.metadata or {}).get("sequence_index")
    if raw is None:
        return None
    value = int(raw)
    if value < 0:
        raise QualificationError("AXIS_REQUIRED_TREE_SEQUENCE_INDEX_NEGATIVE")
    return value


def qualify_required_skeleton_tree_v1(
    surface: RiggingSurfaceIR,
    proposal: SkeletonProposalIR,
    *,
    authority_bindings: dict[str, str] | None = None,
) -> QualifiedSkeletonIR:
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("AXIS_REQUIRED_TREE_SURFACE_BINDING_DRIFT")
    if not proposal.joints:
        raise QualificationError("AXIS_REQUIRED_TREE_EMPTY")

    meta = dict(proposal.metadata or {})
    if meta.get("hard_causal_parent_authority") is not True:
        raise QualificationError("AXIS_REQUIRED_TREE_AUTHORITY_NOT_DECLARED")
    if str(meta.get("compiler_role") or "") != "VALIDATE_AND_MATERIALIZE":
        raise QualificationError("AXIS_REQUIRED_TREE_COMPILER_ROLE_DRIFT")
    if meta.get("compiler_parent_reselection_allowed") is not False:
        raise QualificationError("AXIS_REQUIRED_TREE_RESELECTION_MUST_BE_FALSE")
    if str(meta.get("geometry_authority") or "") != "ORDERED_256_BIN_XYZ_ARGMAX":
        raise QualificationError("AXIS_REQUIRED_TREE_GEOMETRY_AUTHORITY_DRIFT")
    if meta.get("diffusion_runtime_authority") is not False:
        raise QualificationError("AXIS_REQUIRED_TREE_DIFFUSION_AUTHORITY_FORBIDDEN")
    if meta.get("conditional_residual_diffusion") is not False:
        raise QualificationError("AXIS_REQUIRED_TREE_CONDITIONAL_DIFFUSION_FORBIDDEN")

    joint_by = {str(j.proposal_id): j for j in proposal.joints}
    if len(joint_by) != len(proposal.joints):
        raise QualificationError("AXIS_REQUIRED_TREE_DUPLICATE_JOINT_ID")

    surface_ids = {str(n.surface_id) for n in surface.surface_nodes}
    for pid, joint in joint_by.items():
        missing = set(map(str, joint.support_surface_ids)) - surface_ids
        if missing:
            raise QualificationError(
                f"AXIS_REQUIRED_TREE_UNKNOWN_SURFACE:{pid}:{sorted(missing)}"
            )
        joint_meta = dict(joint.metadata or {})
        if joint_meta.get("hard_causal_parent_authority") is not True:
            raise QualificationError("AXIS_REQUIRED_TREE_JOINT_AUTHORITY_DRIFT")
        if joint_meta.get("discrete_xyz_authority") is not True:
            raise QualificationError("AXIS_REQUIRED_TREE_JOINT_GEOMETRY_AUTHORITY_DRIFT")
        if int(joint_meta.get("coord_bins_per_axis", -1)) != 256:
            raise QualificationError("AXIS_REQUIRED_TREE_COORD_BIN_COUNT_DRIFT")

    edges = tuple(proposal.edges)
    if len(edges) != len(joint_by) - 1:
        raise QualificationError("AXIS_REQUIRED_TREE_EDGE_COUNT_INVALID")
    if any((not e.hard_required) or e.hard_forbidden for e in edges):
        raise QualificationError("AXIS_REQUIRED_TREE_EDGE_NOT_EXACT_REQUIRED")

    parent_by_child: dict[str, str] = {}
    edge_ids: set[str] = set()
    for edge in edges:
        eid = str(edge.edge_id)
        parent = str(edge.parent_proposal_id)
        child = str(edge.child_proposal_id)
        if not eid or eid in edge_ids:
            raise QualificationError("AXIS_REQUIRED_TREE_EDGE_ID_INVALID")
        edge_ids.add(eid)
        if parent not in joint_by or child not in joint_by or parent == child:
            raise QualificationError("AXIS_REQUIRED_TREE_EDGE_ENDPOINT_INVALID")
        if child in parent_by_child:
            raise QualificationError("AXIS_REQUIRED_TREE_MULTIPLE_PARENTS")
        parent_by_child[child] = parent

    roots = tuple(sorted(set(joint_by) - set(parent_by_child)))
    if len(roots) != 1:
        raise QualificationError("AXIS_REQUIRED_TREE_EXACTLY_ONE_ROOT_REQUIRED")
    root_pid = roots[0]

    for pid in joint_by:
        seen: set[str] = set()
        cur = pid
        while cur in parent_by_child:
            if cur in seen:
                raise QualificationError("AXIS_REQUIRED_TREE_CYCLE")
            seen.add(cur)
            cur = parent_by_child[cur]
        if cur != root_pid:
            raise QualificationError("AXIS_REQUIRED_TREE_DISCONNECTED")

    seq = {pid: _sequence_index(j) for pid, j in joint_by.items()}
    known_seq = {pid: value for pid, value in seq.items() if value is not None}
    if known_seq:
        if len(known_seq) != len(joint_by):
            raise QualificationError("AXIS_REQUIRED_TREE_SEQUENCE_PARTIAL")
        values = tuple(sorted(known_seq.values()))
        if values != tuple(range(len(joint_by))):
            raise QualificationError("AXIS_REQUIRED_TREE_SEQUENCE_NOT_DENSE")
        if str(meta.get("tree_serialization") or "") != "BFS_PARENT_BEFORE_CHILD":
            raise QualificationError("AXIS_REQUIRED_TREE_SERIALIZATION_DRIFT")
        if known_seq[root_pid] != 0:
            raise QualificationError("AXIS_REQUIRED_TREE_ROOT_NOT_SEQUENCE_ZERO")
        for child, parent in parent_by_child.items():
            if known_seq[parent] >= known_seq[child]:
                raise QualificationError("AXIS_REQUIRED_TREE_PARENT_NOT_BEFORE_CHILD")

    authority = {
        str(k): str(v)
        for k, v in sorted(dict(authority_bindings or {}).items())
    }
    if any(not k or not v for k, v in authority.items()):
        raise QualificationError("AXIS_REQUIRED_TREE_AUTHORITY_BINDING_INVALID")

    semantic = {
        "schema": "RealSaS.AxisRequiredTreeSemantic.v1",
        "surface_binding_hash": surface.geometry_lineage_hash,
        "root_proposal_id": root_pid,
        "joints": [
            {
                "proposal_id": pid,
                "position": tuple(map(float, joint_by[pid].position)),
                "parent_proposal_id": parent_by_child.get(pid),
                "support_surface_ids": tuple(sorted(map(str, joint_by[pid].support_surface_ids))),
                "sequence_index": seq[pid],
            }
            for pid in sorted(joint_by)
        ],
        "authority_bindings": authority,
    }
    tree_hash = content_sha256(semantic)

    canonical = {
        pid: "J:" + content_sha256(
            {
                "required_tree_hash": tree_hash,
                "source_proposal_id": pid,
            }
        )[:20]
        for pid in joint_by
    }
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=canonical[pid],
            position=tuple(map(float, joint_by[pid].position)),
            parent_canonical_id=(
                None
                if pid == root_pid
                else canonical[parent_by_child[pid]]
            ),
            support_surface_ids=tuple(joint_by[pid].support_surface_ids),
            source_proposal_id=pid,
        )
        for pid in sorted(
            joint_by,
            key=lambda value: (
                seq[value] if seq[value] is not None else 10**9,
                value,
            ),
        )
    )
    report = {
        "solver": "AXIS_REQUIRED_TREE_VALIDATOR_V1",
        "status": "PASS_REQUIRED_TREE_VALIDATED",
        "optimality_proven": True,
        "selection_search_performed": False,
        "parent_reselection_performed": False,
        "hard_required_edge_count": len(edges),
        "joint_count": len(joints),
        "tree_semantic_hash": tree_hash,
        "tree_serialization": meta.get("tree_serialization"),
        "authority_bindings": authority,
    }
    lineage = content_sha256(
        {
            "surface": surface.geometry_lineage_hash,
            "proposal": proposal.to_dict(),
            "report": report,
            "joints": [j.to_dict() for j in joints],
        }
    )
    return QualifiedSkeletonIR(
        joints=joints,
        root_id=canonical[root_pid],
        qualification_report=report,
        skeleton_lineage_hash=lineage,
    )


__all__ = ["qualify_required_skeleton_tree_v1"]
