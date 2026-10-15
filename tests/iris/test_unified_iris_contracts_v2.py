import torch

from models.iris.vnext.contracts_v2 import (
    FExtractionSemanticContractV2,
    GeometryFieldContractV2,
    UnifiedFieldContractV2,
    validate_geometry_field_outputs_v2,
)


def test_global_metric_sdf_is_not_a_corpus_invariant():
    contract = GeometryFieldContractV2(global_metric_sdf_required=True)
    try:
        contract.validate()
    except ValueError as exc:
        assert "GLOBAL_METRIC_SDF_FORBIDDEN" in str(exc)
    else:
        raise AssertionError("expected universal SDF rejection")


def test_local_sheet_identity_cannot_collapse_to_component_material_or_owner():
    for kwargs in (
        {"component_equals_local_sheet": True},
        {"material_equals_local_sheet": True},
        {"semantic_owner_equals_local_sheet": True},
    ):
        contract = FExtractionSemanticContractV2(**kwargs)
        try:
            contract.validate()
        except ValueError as exc:
            assert "SHEET_IDENTITY_COLLAPSE_FORBIDDEN" in str(exc)
        else:
            raise AssertionError("expected local-sheet identity collapse rejection")


def test_fixed_local_patch_cardinality_is_not_required():
    contract = FExtractionSemanticContractV2(fixed_local_patch_cardinality_required=True)
    try:
        contract.validate()
    except ValueError as exc:
        assert "FIXED_LOCAL_PATCH_CARDINALITY_FORBIDDEN" in str(exc)
    else:
        raise AssertionError("expected fixed local patch cardinality rejection")


def test_geometry_outputs_require_udf_level_set_validity_and_uncertainty():
    outputs = {
        "unsigned_distance": torch.tensor([0.0, 1.0]),
        "oriented_level_set": torch.tensor([-0.1, 0.2]),
        "orientation_validity": torch.tensor([1.0, 0.0]),
        "geometry_log_uncertainty": torch.tensor([0.0, 0.5]),
    }
    validate_geometry_field_outputs_v2(outputs)

    bad = dict(outputs)
    bad.pop("orientation_validity")
    try:
        validate_geometry_field_outputs_v2(bad)
    except ValueError as exc:
        assert "GEOMETRY_OUTPUT_MISSING" in str(exc)
    else:
        raise AssertionError("expected missing validity rejection")


def test_default_v2_contract_validates():
    UnifiedFieldContractV2().validate()
