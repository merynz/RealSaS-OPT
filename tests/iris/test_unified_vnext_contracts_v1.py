import torch

from models.iris.vnext.contracts_v1 import (
    CanonicalViewEvidenceSetV1,
    UnifiedFieldContractV1,
    validate_canonical_view_evidence_set_v1,
)
from models.iris.vnext.semantic_equivalent_v1 import (
    coarse_to_fine_active_mask_v1,
    semantic_equivalent_sdf_v1,
    union_refinement_evidence_v1,
)


def _views(provenance):
    return CanonicalViewEvidenceSetV1(
        images=torch.zeros(1, 8, 4, 16, 16),
        camera_features=torch.zeros(1, 8, 13),
        confidence=torch.ones(1, 8),
        provenance=tuple(provenance),
    )


def test_fit8_phase_a_accepts_eight_authored_views():
    value = _views(["AUTHORED_CANONICAL"] * 8)
    validate_canonical_view_evidence_set_v1(value, require_all_authored=True)
    assert not value.generated_mask.any()


def test_future_one_plus_seven_contract_keeps_generated_views_non_source():
    value = _views(["SOURCE_OBSERVED"] + ["GENERATED_CANONICAL"] * 7)
    validate_canonical_view_evidence_set_v1(value)
    assert value.observed_source_mask.tolist() == [True, False, False, False, False, False, False, False]


def test_unified_field_capacity_floor_is_stdgen_class_or_larger():
    contract = UnifiedFieldContractV1()
    contract.validate()
    assert contract.scene_dim_floor >= 1024
    assert contract.scene_layers_floor >= 16
    assert contract.scene_heads_floor >= 16
    assert contract.triplane_resolution_floor >= 64
    assert contract.triplane_channels_floor >= 80
    assert contract.owner_embedding_dim >= 64
    assert contract.material_embedding_dim >= 32


def test_semantic_equivalent_sdf_keeps_original_exterior_hard_constraint():
    sdf = torch.tensor([-0.4, 0.2])
    semantic = torch.tensor([[0.8, 0.2], [0.9, 0.1]])
    out = semantic_equivalent_sdf_v1(sdf, semantic, [0])
    assert out[0] < 0
    assert out[1] >= sdf[1]
    assert out[1] > 0


def test_semantic_equivalent_sdf_discards_wrong_semantic_on_surface():
    sdf = torch.tensor([0.0, -0.1])
    semantic = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    out = semantic_equivalent_sdf_v1(sdf, semantic, [0])
    assert torch.all(out > 0)


def test_coarse_to_fine_mask_dilates_and_upsamples_inside_region():
    coarse = torch.ones(5, 5, 5)
    coarse[2, 2, 2] = -1
    active = coarse_to_fine_active_mask_v1(coarse, (10, 10, 10), dilation_kernel=3)
    assert active.dtype == torch.bool
    assert active.shape == (10, 10, 10)
    assert active.any()
    assert not active.all()


def test_generic_refinement_unions_semantic_boundary_and_uncertainty():
    geometry = torch.zeros(4, 4, 4, dtype=torch.bool)
    boundary = torch.zeros(4, 4, 4)
    uncertainty = torch.zeros(4, 4, 4)
    boundary[0, 0, 0] = 0.8
    uncertainty[3, 3, 3] = 0.9
    active = union_refinement_evidence_v1(
        geometry,
        semantic_boundary=boundary,
        uncertainty=uncertainty,
    )
    assert active[0, 0, 0]
    assert active[3, 3, 3]
