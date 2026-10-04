from __future__ import annotations

"""Promoted owner-local repair operation for one control-basis prune.

This module does not choose a control to remove and does not authorize a repair.
Selection requires controlled causal owner attribution plus a bounded repair
directive. The executor only materializes one already-authorized non-root
control deletion through the current typed pruning primitive.
"""

from dataclasses import dataclass
from typing import Any

from compiler.realsas_compiler_core.control_basis_pruning_v1 import (
    ControlPruneReceiptV1,
    prune_qualified_skeleton_control_v1,
)
from compiler.realsas_compiler_core.types import QualifiedSkeletonIR
from compiler.realsas_compiler_services.proof.repair_loop import (
    BoundedRepairOperationV1,
)

OPERATION_ID = "CONTROL_BASIS_SINGLE_PRUNE_REQUALIFICATION_V1"
OWNER_ID = "ATLAS_CONTROL_BASIS"
OPERATION_FAMILY = "prune_single_control_and_rederive_downstream"
QUALIFICATION_HASH = "4ac53ed6ee5bcd261c3f5f27fab24022f6b78bdd1f260828242fb770e22183fe"

ALLOWED_CHANGE_PATHS = (
    "mechanical.skeleton",
    "mechanical.skin",
    "motion",
    "directional_visual.direction.*.component.*.mesh_skin",
)

PROTECTED_INVARIANTS = (
    "mechanical_carrier_lineage",
    "mechanical_mesh_topology",
    "source_observation_authority",
    "same_probe_fingerprint",
    "appearance_authority",
)

QUALIFICATION_PAYLOAD = {
    "schema": "RealSaS.ControlBasisSinglePruneRepairQualification.v1",
    "operation_id": OPERATION_ID,
    "owner_id": OWNER_ID,
    "operation_family": OPERATION_FAMILY,
    "operator": "RealSaS.ControlPruneReceipt.v1",
    "hard_rules": [
        "single_non_root_control_only",
        "retained_rest_joint_positions_preserved",
        "children_reparent_to_removed_parent",
        "fresh_skeleton_lineage_required",
        "downstream_skin_motion_rederivation_required",
        "same_probe_reproof_required",
        "carrier_topology_immutable_in_this_operation",
    ],
    "qualification_evidence": [
        "tests/compiler/test_control_basis_pruning_v1.py",
        "C5.1b/C5.1d/C5.1e real Knight child-state courts",
    ],
}


@dataclass(frozen=True)
class ControlBasisPruneExecutionV1:
    source_skeleton_hash: str
    child_skeleton: QualifiedSkeletonIR
    receipt: ControlPruneReceiptV1
    changed_owner_id: str = OWNER_ID
    downstream_rederivation_required: tuple[str, ...] = (
        "MIRA_SKIN",
        "DEFORMATION_ENVELOPE",
        "MESH_SKIN_BINDING",
        "CANONICAL_PUPPET_STATE",
        "MOTION_RETARGET",
        "MOTION_COMPILE",
        "EXACT_DYNAMIC_PROOF",
    )
    schema_version: str = "RealSaS.ControlBasisPruneExecution.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_skeleton_hash": self.source_skeleton_hash,
            "child_skeleton_hash": self.child_skeleton.skeleton_lineage_hash,
            "receipt": self.receipt.to_dict(),
            "changed_owner_id": self.changed_owner_id,
            "downstream_rederivation_required": list(
                self.downstream_rederivation_required
            ),
            "product_authority_minted": False,
        }


def build_control_basis_single_prune_operation_v1(
    *,
    target_signature_ids: tuple[str, ...],
    selected_control_id: str,
    source_skeleton_hash: str,
) -> BoundedRepairOperationV1:
    signatures = tuple(sorted({str(x) for x in target_signature_ids if str(x)}))
    control_id = str(selected_control_id)
    source_hash = str(source_skeleton_hash)
    if not signatures:
        raise ValueError("CONTROL_BASIS_REPAIR_TARGET_SIGNATURES_EMPTY")
    if not control_id:
        raise ValueError("CONTROL_BASIS_REPAIR_CONTROL_ID_EMPTY")
    if not source_hash:
        raise ValueError("CONTROL_BASIS_REPAIR_SOURCE_SKELETON_HASH_EMPTY")
    return BoundedRepairOperationV1(
        operation_id=OPERATION_ID,
        owner_id=OWNER_ID,
        operation_family=OPERATION_FAMILY,
        target_signature_ids=signatures,
        qualification_hash=QUALIFICATION_HASH,
        automatic_execution_qualified=True,
        allowed_change_paths=ALLOWED_CHANGE_PATHS,
        bounded_change_spec={
            "max_removed_controls_per_child_attempt": 1,
            "root_control_removal_forbidden": True,
            "retained_rest_joint_positions_must_be_exact": True,
            "children_reparent_to_removed_parent": True,
            "fresh_skeleton_lineage_required": True,
            "carrier_topology_change_forbidden": True,
            "downstream_skin_motion_rederivation_required": True,
            "same_probe_reproof_required": True,
        },
        parameters={
            "selected_control_id": control_id,
            "source_skeleton_hash": source_hash,
        },
        protected_invariants=PROTECTED_INVARIANTS,
        metadata={
            "qualification_payload": QUALIFICATION_PAYLOAD,
            "selection_authority": "CONTROLLED_OWNER_ATTRIBUTION_ONLY",
            "executor_selects_control": False,
            "product_authority_minted": False,
        },
    )


def execute_control_basis_single_prune_v1(
    *,
    operation: BoundedRepairOperationV1,
    skeleton: QualifiedSkeletonIR,
) -> ControlBasisPruneExecutionV1:
    """Materialize one already-authorized owner-local skeleton child.

    Full child product state is intentionally not built here. The compile
    transaction must carry unchanged upstream evidence and rederive every
    downstream skeleton-bound artifact before repair credit can be evaluated.
    """
    if operation.operation_id != OPERATION_ID:
        raise ValueError("CONTROL_BASIS_REPAIR_OPERATION_ID_MISMATCH")
    if operation.owner_id != OWNER_ID:
        raise ValueError("CONTROL_BASIS_REPAIR_OWNER_MISMATCH")
    if operation.operation_family != OPERATION_FAMILY:
        raise ValueError("CONTROL_BASIS_REPAIR_FAMILY_MISMATCH")
    if operation.qualification_hash != QUALIFICATION_HASH:
        raise ValueError("CONTROL_BASIS_REPAIR_QUALIFICATION_HASH_MISMATCH")
    if tuple(operation.allowed_change_paths) != ALLOWED_CHANGE_PATHS:
        raise ValueError("CONTROL_BASIS_REPAIR_CHANGE_SCOPE_MISMATCH")

    source_hash = str(operation.parameters.get("source_skeleton_hash") or "")
    if source_hash != str(skeleton.skeleton_lineage_hash):
        raise ValueError("CONTROL_BASIS_REPAIR_SOURCE_SKELETON_DRIFT")
    control_id = str(operation.parameters.get("selected_control_id") or "")
    if not control_id:
        raise ValueError("CONTROL_BASIS_REPAIR_SELECTED_CONTROL_MISSING")

    child, receipt = prune_qualified_skeleton_control_v1(
        skeleton,
        control_id,
    )
    if receipt.source_control_count - receipt.pruned_control_count != 1:
        raise RuntimeError("CONTROL_BASIS_REPAIR_NOT_SINGLE_CONTROL_DELTA")
    if receipt.source_skeleton_hash != source_hash:
        raise RuntimeError("CONTROL_BASIS_REPAIR_RECEIPT_SOURCE_DRIFT")
    if receipt.pruned_skeleton_hash != child.skeleton_lineage_hash:
        raise RuntimeError("CONTROL_BASIS_REPAIR_RECEIPT_CHILD_DRIFT")

    return ControlBasisPruneExecutionV1(
        source_skeleton_hash=source_hash,
        child_skeleton=child,
        receipt=receipt,
    )


__all__ = [
    "OPERATION_ID",
    "OWNER_ID",
    "OPERATION_FAMILY",
    "QUALIFICATION_HASH",
    "ALLOWED_CHANGE_PATHS",
    "PROTECTED_INVARIANTS",
    "ControlBasisPruneExecutionV1",
    "build_control_basis_single_prune_operation_v1",
    "execute_control_basis_single_prune_v1",
]
