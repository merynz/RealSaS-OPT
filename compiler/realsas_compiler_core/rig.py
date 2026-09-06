from __future__ import annotations

from dataclasses import replace

from .canonical_graph_optimizer_authority import optimize_canonical_graph_v18_98
from .hashing import content_sha256
from .skeleton_admission_v1 import (
    COMPAT_SKELETON_ADMISSION_POLICY_V1,
    PRODUCT_SKELETON_ADMISSION_POLICY_V1,
    AdmittedSkeletonProposalIR,
    admit_skeleton_proposal_v1,
)
from .substrate.validation import (
    rigging_surface_boundary_audit_hash_v1,
    rigging_surface_topology_fingerprint_v1,
    validate_rigging_surface_ir_v1,
)
from .types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualificationError,
    RiggingSurfaceIR,
    SkeletonProposalIR,
)
from .v4_types import QualifiedSkeletonIRV2
from realsas_contracts.technical_part_graph import (
    CanonicalGraphEdgeCandidate,
    CanonicalGraphNodeCandidate,
    CanonicalGraphOptimizationRequest,
)


PRODUCT_CONDITIONING_AUTHORITY_V1 = "PRODUCT_VALIDATED_SCENE_FIRST_SIGNED_V1"
PRODUCT_TOPOLOGY_CONTRACT_V1 = "RIGGING_SURFACE_LOCAL_RELATIONS_INDEXED_V1"
PRODUCT_DEFORM_GRAPH_POLICY_V1 = "SINGLE_CONNECTED_DEFORM_TREE_V1"


def _qualify_admitted_skeleton(
    surface: RiggingSurfaceIR,
    admitted: AdmittedSkeletonProposalIR,
    *,
    run_ilp_shadow: bool,
    qualification_context: dict | None = None,
) -> QualifiedSkeletonIR:
    proposal = admitted.proposal
    joint_by = {j.proposal_id: j for j in proposal.joints}
    surface_ids = {n.surface_id for n in surface.surface_nodes}

    internal = {
        pid: "CGC:"
        + content_sha256(
            {
                "proposal_id": pid,
                "position": joint_by[pid].position,
                "surface": surface.geometry_lineage_hash,
            }
        )[:20]
        for pid in sorted(joint_by)
    }
    reverse = {v: k for k, v in internal.items()}

    nodes = []
    for pid in sorted(joint_by):
        j = joint_by[pid]
        missing = set(j.support_surface_ids) - surface_ids
        if missing:
            raise QualificationError(
                f"joint {pid} references unknown surface ids:{sorted(missing)}"
            )
        nodes.append(
            CanonicalGraphNodeCandidate(
                canonical_control_id=internal[pid],
                root_score01=float(j.root_score),
                confidence01=float(j.confidence),
                uncertainty01=max(0.0, 1.0 - float(j.confidence)),
                required=True,
                provenance={
                    "source_proposal_id": pid,
                    "support_surface_ids": tuple(j.support_surface_ids),
                    "admission_lineage_hash": admitted.admission_lineage_hash,
                },
            )
        )

    edges = []
    for e in sorted(proposal.edges, key=lambda x: x.edge_id):
        if (
            e.parent_proposal_id not in internal
            or e.child_proposal_id not in internal
        ):
            raise QualificationError(f"edge endpoint missing:{e.edge_id}")
        edges.append(
            CanonicalGraphEdgeCandidate(
                edge_key="CGE:" + content_sha256(e.to_dict())[:20],
                parent_control_id=internal[e.parent_proposal_id],
                child_control_id=internal[e.child_proposal_id],
                evidence_score01=float(e.score),
                authority_tier="MODEL_PROPOSAL_EVIDENCE",
                objective_score=float(e.score),
                uncertainty01=max(0.0, 1.0 - float(e.confidence)),
                reason=e.reason,
                hard_required=e.hard_required,
                hard_forbidden=e.hard_forbidden,
                provenance={
                    "source_edge_id": e.edge_id,
                    "admission_lineage_hash": admitted.admission_lineage_hash,
                },
            )
        )

    req = CanonicalGraphOptimizationRequest(
        request_id="RIGQ:"
        + content_sha256(
            {
                "surface": surface.geometry_lineage_hash,
                "admitted_proposal": admitted.to_dict(),
            }
        )[:24],
        attempt_id="CURRENT",
        nodes=tuple(nodes),
        edges=tuple(edges),
        root_candidate_ids=tuple(sorted(internal.values())),
        run_ilp_shadow=run_ilp_shadow,
        ilp_authority_enabled=False,
        calibration_locked=False,
        metadata={
            "proposal_ids_are_not_canonical": True,
            "surface_hash": surface.geometry_lineage_hash,
            "admission_lineage_hash": admitted.admission_lineage_hash,
            "node_set_policy": admitted.admission_report["policy"][
                "node_selection_policy"
            ],
            **dict(qualification_context or {}),
        },
    )
    res = optimize_canonical_graph_v18_98(req)
    if not res.passed:
        raise QualificationError(
            "skeleton qualification failed:"
            + ";".join(res.blockers or (res.status,))
        )
    if res.optimality_proven is not True:
        raise QualificationError(
            "skeleton qualification failed:CANONICAL_GRAPH_OPTIMALITY_NOT_PROVEN"
        )

    canonical = {
        cid: "J:"
        + content_sha256(
            {"qualified_graph": res.content_sha256, "candidate": cid}
        )[:20]
        for cid in res.selected_node_ids
    }
    q = []
    for cid in sorted(res.selected_node_ids):
        pid = reverse[cid]
        j = joint_by[pid]
        pcid = res.parent_by_child.get(cid)
        q.append(
            QualifiedJoint(
                canonical[cid],
                tuple(map(float, j.position)),
                None if pcid is None else canonical[pcid],
                tuple(j.support_surface_ids),
                pid,
            )
        )

    root = canonical[res.selected_root_control_id]
    report = {
        "solver": res.solver,
        "status": res.status,
        "objective": res.objective_value,
        "optimality_proven": res.optimality_proven,
        "blockers": res.blockers,
        "warnings": res.warnings,
        "optimizer_result_sha256": res.content_sha256,
        "proposal_to_candidate": internal,
        "candidate_to_canonical": canonical,
        "admission_lineage_hash": admitted.admission_lineage_hash,
        "admission_report": admitted.admission_report,
        **dict(qualification_context or {}),
    }
    lineage = content_sha256(
        {
            "surface": surface.geometry_lineage_hash,
            "admitted_proposal": admitted.to_dict(),
            "report": report,
            "joints": [j.to_dict() for j in q],
        }
    )
    return QualifiedSkeletonIR(tuple(q), root, report, lineage)


def qualify_skeleton(
    surface: RiggingSurfaceIR,
    proposal: SkeletonProposalIR,
    *,
    run_ilp_shadow: bool = False,
) -> QualifiedSkeletonIR:
    """Compatibility qualifier with explicit admission but no product-only gates."""
    admitted = admit_skeleton_proposal_v1(
        surface,
        proposal,
        policy=COMPAT_SKELETON_ADMISSION_POLICY_V1,
    )
    return _qualify_admitted_skeleton(
        surface,
        admitted,
        run_ilp_shadow=run_ilp_shadow,
        qualification_context={
            "rigging_route": "COMPATIBILITY_QUALIFIER_V1",
            "promotion_authority": False,
        },
    )


def _upgrade_skeleton_v2(
    legacy: QualifiedSkeletonIR,
    *,
    report_updates: dict,
) -> QualifiedSkeletonIRV2:
    roots = tuple(
        sorted(
            j.canonical_joint_id
            for j in legacy.joints
            if j.parent_canonical_id is None
        )
    )
    if not roots:
        raise QualificationError("qualified skeleton has no deform root")
    report = dict(legacy.qualification_report)
    report.update(
        {
            "schema_upgrade": "RealSaS.QualifiedSkeletonIR.v2",
            "deform_root_semantics": "EXPLICIT_SET",
            "assembly_root_semantics": "SEPARATE_NON_DEFORMING_BINDING",
            **dict(report_updates),
        }
    )
    value = QualifiedSkeletonIRV2(
        tuple(legacy.joints),
        roots,
        {},
        report,
        "",
    )
    payload = value.to_dict()
    payload.pop("skeleton_lineage_hash", None)
    return replace(value, skeleton_lineage_hash=content_sha256(payload))


def qualify_skeleton_v2(
    surface: RiggingSurfaceIR,
    proposal: SkeletonProposalIR,
    *,
    run_ilp_shadow: bool = False,
) -> QualifiedSkeletonIRV2:
    """Compatibility V2 qualifier.

    This route remains available for historical diagnostics. New scene-first
    product promotion must use compile_scene_first_rigging_v1.
    """
    legacy = qualify_skeleton(
        surface,
        proposal,
        run_ilp_shadow=run_ilp_shadow,
    )
    return _upgrade_skeleton_v2(
        legacy,
        report_updates={
            "current_optimizer_shape": "SINGLE_ARBORESCENCE_COMPATIBILITY",
            "promotion_authority": False,
            "product_route": "USE_compile_scene_first_rigging_v1",
        },
    )


def _validate_product_conditioning_certificate_v1(
    surface: RiggingSurfaceIR,
    certificate: dict,
    *,
    expected_boundary_hash: str,
    expected_topology_hash: str,
) -> None:
    cert = dict(certificate or {})
    if cert.get("schema") != "RealSaS.GeppettoProductConditioningCertificate.v1":
        raise QualificationError("RIGGING_PRODUCT_BAD_CONDITIONING_CERTIFICATE_SCHEMA")
    if cert.get("conditioning_authority") != PRODUCT_CONDITIONING_AUTHORITY_V1:
        raise QualificationError("RIGGING_PRODUCT_CONDITIONING_NOT_PRODUCT_VALIDATED")
    if cert.get("source_surface_hash") != surface.geometry_lineage_hash:
        raise QualificationError("RIGGING_PRODUCT_CONDITIONING_SURFACE_HASH_MISMATCH")
    if cert.get("surface_boundary_audit_hash") != expected_boundary_hash:
        raise QualificationError("RIGGING_PRODUCT_BOUNDARY_AUDIT_HASH_MISMATCH")
    if cert.get("local_relation_hash") != expected_topology_hash:
        raise QualificationError("RIGGING_PRODUCT_TOPOLOGY_HASH_MISMATCH")
    if cert.get("topology_contract") != PRODUCT_TOPOLOGY_CONTRACT_V1:
        raise QualificationError("RIGGING_PRODUCT_TOPOLOGY_CONTRACT_MISMATCH")
    if not str(cert.get("conditioning_hash", "")):
        raise QualificationError("RIGGING_PRODUCT_MISSING_CONDITIONING_HASH")

    got = str(cert.get("certificate_sha256", ""))
    payload = dict(cert)
    payload.pop("certificate_sha256", None)
    if not got or got != content_sha256(payload):
        raise QualificationError("RIGGING_PRODUCT_CONDITIONING_CERTIFICATE_HASH_MISMATCH")


def assert_single_deform_tree_product_v1(
    skeleton: QualifiedSkeletonIRV2,
) -> None:
    """Fail closed on anything other than one connected deform tree."""
    ids = {j.canonical_joint_id for j in skeleton.joints}
    if not ids or len(ids) != len(skeleton.joints):
        raise QualificationError("RIGGING_PRODUCT_INVALID_CANONICAL_JOINT_IDS")
    roots = tuple(skeleton.deform_root_ids)
    if len(roots) != 1 or roots[0] not in ids:
        raise QualificationError("RIGGING_PRODUCT_REQUIRES_SINGLE_DEFORM_ROOT")
    root = roots[0]

    parent = {
        j.canonical_joint_id: j.parent_canonical_id
        for j in skeleton.joints
    }
    actual_roots = {jid for jid, pid in parent.items() if pid is None}
    if actual_roots != {root}:
        raise QualificationError("RIGGING_PRODUCT_ROOT_PARENT_MISMATCH")

    for jid, pid in parent.items():
        if pid is not None and pid not in ids:
            raise QualificationError("RIGGING_PRODUCT_ILLEGAL_PARENT")
        cursor = jid
        seen = set()
        while cursor != root:
            if cursor in seen:
                raise QualificationError("RIGGING_PRODUCT_CYCLE")
            seen.add(cursor)
            next_parent = parent.get(cursor)
            if next_parent is None:
                raise QualificationError("RIGGING_PRODUCT_DISCONNECTED_FROM_ROOT")
            cursor = next_parent

    if skeleton.assembly_root_binding and not bool(
        skeleton.assembly_root_binding.get("non_deforming", False)
    ):
        raise QualificationError("RIGGING_PRODUCT_ASSEMBLY_ROOT_MUST_BE_NON_DEFORMING")


def compile_scene_first_rigging_v1(
    surface: RiggingSurfaceIR,
    proposal: SkeletonProposalIR,
    *,
    conditioning_certificate: dict,
    run_ilp_shadow: bool = False,
) -> QualifiedSkeletonIRV2:
    """Authoritative scene-first rigging route up to QualifiedSkeletonIRV2.

    Required order:
      strict RiggingSurfaceIR validation
      -> product conditioning certificate verification
      -> explicit proposal admission (no geometry fusion, no deform completion)
      -> global canonical root/parent optimization
      -> canonical ID minting
      -> single connected deform-tree product assertion.
    """

    boundary_report = validate_rigging_surface_ir_v1(
        surface,
        require_scene_first_signed_contract=True,
    )
    boundary_hash = rigging_surface_boundary_audit_hash_v1(boundary_report)
    topology_hash = rigging_surface_topology_fingerprint_v1(surface)
    _validate_product_conditioning_certificate_v1(
        surface,
        conditioning_certificate,
        expected_boundary_hash=boundary_hash,
        expected_topology_hash=topology_hash,
    )

    admitted = admit_skeleton_proposal_v1(
        surface,
        proposal,
        policy=PRODUCT_SKELETON_ADMISSION_POLICY_V1,
    )
    certificate_sha = str(conditioning_certificate["certificate_sha256"])
    legacy = _qualify_admitted_skeleton(
        surface,
        admitted,
        run_ilp_shadow=run_ilp_shadow,
        qualification_context={
            "rigging_route": "SCENE_FIRST_PRODUCT_V1",
            "promotion_authority": True,
            "surface_boundary_validation_profile": "SCENE_FIRST_SIGNED_V1",
            "surface_boundary_audit_hash": boundary_hash,
            "surface_topology_fingerprint": topology_hash,
            "conditioning_certificate_sha256": certificate_sha,
            "product_deform_graph_policy": PRODUCT_DEFORM_GRAPH_POLICY_V1,
            "geometry_only_duplicate_fusion": "FORBIDDEN",
            "deform_node_completion": "FORBIDDEN_BUDGET_0",
            "native_stop_count_authority": "GEPPETTO_PROPOSAL_CARDINALITY",
        },
    )
    value = _upgrade_skeleton_v2(
        legacy,
        report_updates={
            "current_optimizer_shape": "SINGLE_ARBORESCENCE_PRODUCT_AUTHORITY",
            "product_deform_graph_policy": PRODUCT_DEFORM_GRAPH_POLICY_V1,
            "promotion_authority": True,
            "product_route": "compile_scene_first_rigging_v1",
        },
    )
    assert_single_deform_tree_product_v1(value)
    return value


__all__ = [
    "qualify_skeleton",
    "qualify_skeleton_v2",
    "compile_scene_first_rigging_v1",
    "assert_single_deform_tree_product_v1",
    "PRODUCT_DEFORM_GRAPH_POLICY_V1",
]
