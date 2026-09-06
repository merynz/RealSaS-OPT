from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

from .hashing import content_sha256
from .types import QualificationError, RiggingSurfaceIR, SkeletonProposalIR


@dataclass(frozen=True)
class SkeletonAdmissionPolicyV1:
    """Explicit pre-graph admission policy.

    The current product policy is intentionally conservative:
    - learned confidence is not converted into an uncalibrated pruning threshold;
    - geometry-only duplicate fusion is forbidden because distinct controls may
      legitimately share one locus;
    - deform-joint synthesis is forbidden until a separately proven completion
      owner exists.
    """

    require_surface_support: bool = False
    minimum_confidence: float = 0.0
    geometry_duplicate_policy: str = "PRESERVE_DISTINCT_PROPOSAL_IDS"
    max_synthesized_deform_nodes: int = 0
    node_selection_policy: str = "FAIL_CLOSED_ALL_CONTRACT_VALID_PROPOSALS"
    schema_version: str = "RealSaS.SkeletonAdmissionPolicy.v1"

    def validate(self) -> None:
        if not (0.0 <= float(self.minimum_confidence) <= 1.0):
            raise ValueError("minimum_confidence must be in [0,1]")
        if self.geometry_duplicate_policy != "PRESERVE_DISTINCT_PROPOSAL_IDS":
            raise ValueError("geometry-only duplicate fusion is not product-authorized")
        if int(self.max_synthesized_deform_nodes) != 0:
            raise ValueError("deform-node completion is not product-authorized in v1")
        if self.node_selection_policy != "FAIL_CLOSED_ALL_CONTRACT_VALID_PROPOSALS":
            raise ValueError("unrecognized skeleton admission node-selection policy")


COMPAT_SKELETON_ADMISSION_POLICY_V1 = SkeletonAdmissionPolicyV1()
PRODUCT_SKELETON_ADMISSION_POLICY_V1 = SkeletonAdmissionPolicyV1(
    require_surface_support=True,
)


@dataclass(frozen=True)
class AdmittedSkeletonProposalIR:
    proposal: SkeletonProposalIR
    admission_report: dict
    admission_lineage_hash: str
    schema_version: str = "RealSaS.AdmittedSkeletonProposalIR.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_vec3(value) -> bool:
    try:
        row = tuple(float(x) for x in value)
    except Exception:
        return False
    return len(row) == 3 and all(math.isfinite(x) for x in row)


def _probability(value) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


def admit_skeleton_proposal_v1(
    surface: RiggingSurfaceIR,
    proposal: SkeletonProposalIR,
    *,
    policy: SkeletonAdmissionPolicyV1 = COMPAT_SKELETON_ADMISSION_POLICY_V1,
) -> AdmittedSkeletonProposalIR:
    """Validate and explicitly admit a proposal before canonical graph solving.

    V1 deliberately performs no hidden pruning, geometry-only fusion or synthetic
    deform-node completion. Any contract-invalid candidate makes the whole
    admission fail closed. This keeps native STOP/count responsibility visible.
    """

    policy.validate()
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("SKELETON_ADMISSION_SURFACE_LINEAGE_MISMATCH")
    if not proposal.joints:
        raise QualificationError("SKELETON_ADMISSION_EMPTY_PROPOSAL")

    surface_ids = {str(node.surface_id) for node in surface.surface_nodes}
    if not surface_ids:
        raise QualificationError("SKELETON_ADMISSION_EMPTY_SURFACE")

    joint_ids = [str(j.proposal_id) for j in proposal.joints]
    if any(not pid for pid in joint_ids):
        raise QualificationError("SKELETON_ADMISSION_EMPTY_PROPOSAL_ID")
    if len(joint_ids) != len(set(joint_ids)):
        raise QualificationError("SKELETON_ADMISSION_DUPLICATE_PROPOSAL_ID")
    joint_id_set = set(joint_ids)

    coincident = Counter()
    unsupported = []
    for joint in proposal.joints:
        pid = str(joint.proposal_id)
        if not _finite_vec3(joint.position):
            raise QualificationError(f"SKELETON_ADMISSION_BAD_POSITION:{pid}")
        if not _probability(joint.root_score):
            raise QualificationError(f"SKELETON_ADMISSION_BAD_ROOT_SCORE:{pid}")
        if not _probability(joint.confidence):
            raise QualificationError(f"SKELETON_ADMISSION_BAD_CONFIDENCE:{pid}")
        if float(joint.confidence) < float(policy.minimum_confidence):
            raise QualificationError(f"SKELETON_ADMISSION_CONFIDENCE_BELOW_POLICY:{pid}")

        support = tuple(str(sid) for sid in joint.support_surface_ids)
        if len(support) != len(set(support)):
            raise QualificationError(f"SKELETON_ADMISSION_DUPLICATE_SUPPORT_ID:{pid}")
        unknown = sorted(set(support) - surface_ids)
        if unknown:
            raise QualificationError(
                f"SKELETON_ADMISSION_UNKNOWN_SURFACE_SUPPORT:{pid}:{','.join(unknown[:8])}"
            )
        if not support:
            unsupported.append(pid)
        coincident[tuple(float(x) for x in joint.position)] += 1

    if policy.require_surface_support and unsupported:
        raise QualificationError(
            "SKELETON_ADMISSION_UNSUPPORTED_JOINTS:" + ",".join(sorted(unsupported)[:16])
        )

    edge_ids = [str(edge.edge_id) for edge in proposal.edges]
    if any(not eid for eid in edge_ids):
        raise QualificationError("SKELETON_ADMISSION_EMPTY_EDGE_ID")
    if len(edge_ids) != len(set(edge_ids)):
        raise QualificationError("SKELETON_ADMISSION_DUPLICATE_EDGE_ID")

    required_pairs = set()
    forbidden_pairs = set()
    for edge in proposal.edges:
        eid = str(edge.edge_id)
        parent = str(edge.parent_proposal_id)
        child = str(edge.child_proposal_id)
        if parent not in joint_id_set or child not in joint_id_set:
            raise QualificationError(f"SKELETON_ADMISSION_EDGE_ENDPOINT_MISSING:{eid}")
        if parent == child:
            raise QualificationError(f"SKELETON_ADMISSION_SELF_EDGE:{eid}")
        if not _probability(edge.score):
            raise QualificationError(f"SKELETON_ADMISSION_BAD_EDGE_SCORE:{eid}")
        if not _probability(edge.confidence):
            raise QualificationError(f"SKELETON_ADMISSION_BAD_EDGE_CONFIDENCE:{eid}")
        if bool(edge.hard_required) and bool(edge.hard_forbidden):
            raise QualificationError(f"SKELETON_ADMISSION_EDGE_REQUIRED_AND_FORBIDDEN:{eid}")
        pair = (parent, child)
        if edge.hard_required:
            required_pairs.add(pair)
        if edge.hard_forbidden:
            forbidden_pairs.add(pair)

    conflict = sorted(required_pairs & forbidden_pairs)
    if conflict:
        p, c = conflict[0]
        raise QualificationError(f"SKELETON_ADMISSION_PAIR_REQUIRED_AND_FORBIDDEN:{p}->{c}")

    coincident_sizes = tuple(sorted((int(n) for n in coincident.values() if n > 1), reverse=True))
    report = {
        "schema": "RealSaS.SkeletonAdmissionReport.v1",
        "status": "PASS",
        "policy": asdict(policy),
        "source_surface_hash": str(surface.geometry_lineage_hash),
        "source_proposal_hash": content_sha256(proposal),
        "input_joint_count": len(proposal.joints),
        "admitted_joint_count": len(proposal.joints),
        "rejected_joint_count": 0,
        "input_edge_count": len(proposal.edges),
        "admitted_edge_count": len(proposal.edges),
        "rejected_edge_count": 0,
        "unsupported_joint_count": len(unsupported),
        "coincident_locus_group_count": len(coincident_sizes),
        "coincident_locus_group_sizes": coincident_sizes,
        "geometry_only_fused_joint_count": 0,
        "synthesized_deform_node_count": 0,
        "native_count_preserved": True,
        "proposal_ids_are_not_canonical": True,
        "teacher_truth_used": False,
    }
    lineage = content_sha256(
        {
            "schema": "RealSaS.AdmittedSkeletonProposalIR.v1",
            "proposal": proposal,
            "admission_report": report,
        }
    )
    return AdmittedSkeletonProposalIR(
        proposal=proposal,
        admission_report=report,
        admission_lineage_hash=lineage,
    )


__all__ = [
    "SkeletonAdmissionPolicyV1",
    "AdmittedSkeletonProposalIR",
    "COMPAT_SKELETON_ADMISSION_POLICY_V1",
    "PRODUCT_SKELETON_ADMISSION_POLICY_V1",
    "admit_skeleton_proposal_v1",
]
