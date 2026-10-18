import pytest
import torch

from models.iris.vnext.contracts_v1 import (
    CanonicalViewEvidenceSetV1,
    UnifiedFieldContractV1,
    validate_canonical_view_evidence_set_v1,
)
from models.iris.vnext.contracts_v2 import (
    FExtractionSemanticContractV2,
    GeometryFieldContractV2,
    UnifiedFieldContractV2,
    validate_geometry_field_outputs_v2,
)
from models.iris.vnext.semantic_partition_contract_v1 import (
    SemanticPartitionContractV1,
)


def _views(provenance):
    return CanonicalViewEvidenceSetV1(
        images=torch.zeros(1, 8, 4, 16, 16),
        camera_features=torch.zeros(1, 8, 13),
        confidence=torch.ones(1, 8),
        provenance=tuple(provenance),
    )


def test_fit8_accepts_eight_authored_canonical_views():
    value = _views(["AUTHORED_CANONICAL"] * 8)
    validate_canonical_view_evidence_set_v1(value, require_all_authored=True)
    assert not value.generated_mask.any()


def test_product_contract_keeps_generated_views_non_source_authority():
    value = _views(["SOURCE_OBSERVED"] + ["GENERATED_CANONICAL"] * 7)
    validate_canonical_view_evidence_set_v1(value)
    assert value.observed_source_mask.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_capacity_floor_is_not_deliberately_undersized():
    v1 = UnifiedFieldContractV1()
    v2 = UnifiedFieldContractV2()
    v1.validate()
    v2.validate()
    for contract in (v1, v2):
        assert contract.scene_dim_floor >= 1024
        assert contract.scene_layers_floor >= 16
        assert contract.scene_heads_floor >= 16
        assert contract.triplane_resolution_floor >= 64
        assert contract.triplane_channels_floor >= 80
        assert contract.owner_embedding_dim >= 64
        assert contract.material_embedding_dim >= 32


def test_global_metric_sdf_cannot_be_promoted_to_corpus_invariant():
    with pytest.raises(ValueError, match="GLOBAL_METRIC_SDF_FORBIDDEN"):
        GeometryFieldContractV2(global_metric_sdf_required=True).validate()


def test_geometry_output_requires_udf_local_orientation_validity_and_uncertainty():
    outputs = {
        "unsigned_distance": torch.tensor([0.0, 1.0]),
        "oriented_level_set": torch.tensor([-0.1, 0.2]),
        "orientation_validity": torch.tensor([1.0, 0.0]),
        "geometry_log_uncertainty": torch.tensor([0.0, 0.5]),
    }
    validate_geometry_field_outputs_v2(outputs)

    bad = dict(outputs)
    bad.pop("orientation_validity")
    with pytest.raises(ValueError, match="GEOMETRY_OUTPUT_MISSING"):
        validate_geometry_field_outputs_v2(bad)


def test_extraction_sheet_identity_cannot_be_semantic_owner_material_or_component():
    for kwargs in (
        {"component_equals_local_sheet": True},
        {"material_equals_local_sheet": True},
        {"semantic_owner_equals_local_sheet": True},
    ):
        with pytest.raises(ValueError, match="SHEET_IDENTITY_COLLAPSE_FORBIDDEN"):
            FExtractionSemanticContractV2(**kwargs).validate()


def test_semantic_partition_is_subject_local_and_permutation_invariant():
    contract = SemanticPartitionContractV1()
    contract.validate()
    assert contract.owner_ids_subject_local
    assert contract.material_region_ids_subject_local
    assert contract.owner_identity_permutation_invariant
    assert contract.material_region_identity_permutation_invariant
    assert contract.fixed_global_owner_class_count_forbidden
    assert contract.fixed_global_material_region_class_count_forbidden


def test_query_budget_is_capacity_not_ontology_and_overflow_fails_closed():
    contract = SemanticPartitionContractV1()
    assert contract.query_budget_is_capacity_cap_not_ontology
    assert contract.overflow_must_abstain_or_fail
    assert contract.silent_region_merge_on_overflow_forbidden
    assert contract.owner_partition_cap_freeze_state == (
        "DEFER_UNTIL_QUALIFIED_OWNER_PARTITION_CENSUS"
    )


def test_semantic_owner_cannot_collapse_to_material_component_or_joint_owner():
    for kwargs in (
        {"material_identity_distinct_from_material_region": False},
        {"geometric_component_distinct_from_owner_region": False},
        {"mechanical_owner_joint_distinct_from_owner_region": False},
    ):
        with pytest.raises(ValueError, match="SEMANTIC_PARTITION_CONTRACT_DRIFT"):
            SemanticPartitionContractV1(**kwargs).validate()


def test_owner_partition_cap_cannot_be_frozen_before_qualified_census():
    with pytest.raises(ValueError, match="OWNER_PARTITION_CAP_PREMATURELY_FROZEN"):
        SemanticPartitionContractV1(
            owner_partition_cap_freeze_state="FIXED_64"
        ).validate()
