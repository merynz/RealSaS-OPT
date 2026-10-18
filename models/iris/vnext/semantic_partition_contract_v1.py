from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "RealSaS.UnifiedSemanticPartitionContract.v1"


@dataclass(frozen=True)
class SemanticPartitionContractV1:
    """Generic owner/material partition invariants for non-human morphology.

    IDs are meaningful only inside one subject. Query budgets are implementation
    capacity caps, never an ontology. Material identity, geometric connectivity
    and mechanical ownership must not be silently promoted into semantic-owner
    identity.
    """

    owner_ids_subject_local: bool = True
    material_region_ids_subject_local: bool = True
    owner_identity_permutation_invariant: bool = True
    material_region_identity_permutation_invariant: bool = True
    fixed_global_owner_class_count_forbidden: bool = True
    fixed_global_material_region_class_count_forbidden: bool = True
    query_budget_is_capacity_cap_not_ontology: bool = True
    overflow_must_abstain_or_fail: bool = True
    silent_region_merge_on_overflow_forbidden: bool = True
    material_identity_distinct_from_material_region: bool = True
    geometric_component_distinct_from_owner_region: bool = True
    mechanical_owner_joint_distinct_from_owner_region: bool = True
    owner_partition_cap_freeze_state: str = (
        "DEFER_UNTIL_QUALIFIED_OWNER_PARTITION_CENSUS"
    )

    def validate(self) -> None:
        required_true = (
            self.owner_ids_subject_local,
            self.material_region_ids_subject_local,
            self.owner_identity_permutation_invariant,
            self.material_region_identity_permutation_invariant,
            self.fixed_global_owner_class_count_forbidden,
            self.fixed_global_material_region_class_count_forbidden,
            self.query_budget_is_capacity_cap_not_ontology,
            self.overflow_must_abstain_or_fail,
            self.silent_region_merge_on_overflow_forbidden,
            self.material_identity_distinct_from_material_region,
            self.geometric_component_distinct_from_owner_region,
            self.mechanical_owner_joint_distinct_from_owner_region,
        )
        if not all(required_true):
            raise ValueError("UNIFIED_SEMANTIC_PARTITION_CONTRACT_DRIFT")
        if self.owner_partition_cap_freeze_state != (
            "DEFER_UNTIL_QUALIFIED_OWNER_PARTITION_CENSUS"
        ):
            raise ValueError("UNIFIED_OWNER_PARTITION_CAP_PREMATURELY_FROZEN")


SemanticPartitionContractV1().validate()


__all__ = ["SCHEMA_VERSION", "SemanticPartitionContractV1"]
