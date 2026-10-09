import numpy as np
import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_presentation_contract_v1 import source_cut_pairs, contact_residuals, relational_verdict


def test_source_cut_candidates_are_owner_scoped_not_weld_permission():
    rest = np.array([[1., 2.], [1., 2.], [1., 2.], [1.01, 2.]])
    pairs = source_cut_pairs(rest, np.array([0, 1, 2, 3]), np.array([0, 0, 1, 0]))
    assert pairs.tolist() == [[0, 1]]
    moved = rest.copy()
    moved[1, 0] += 20
    proof = contact_residuals(moved, pairs)
    assert proof["maximum_contact_residual_px"] == 20
    assert proof["contact_failure_count"] == 1
    assert relational_verdict(proof)["relational_presentation_passed"] is False


def test_missing_semantic_qualification_cannot_pass_with_perfect_replay():
    proof = {"connected_palette_relations_passed": True, "setup_identity_passed": True,
             "native_reference_byte_parity_passed": True, "area_condition_passed": True}
    verdict = relational_verdict(proof)
    assert verdict["relational_presentation_passed"] is False
    assert "qualified_dynamic_coverage_passed" in verdict["relational_presentation_blockers"]
    assert "semantic_occlusion_passed" in verdict["relational_presentation_blockers"]


def test_broken_relation_rejects_even_with_all_other_qualifications():
    keys = ("connected_palette_relations_passed", "qualified_contact_relations_passed",
            "qualified_dynamic_coverage_passed", "semantic_occlusion_passed", "setup_identity_passed",
            "frame0_relations_passed", "temporal_relations_passed")
    proof = dict.fromkeys(keys, True)
    assert relational_verdict(proof)["relational_presentation_passed"] is True
    proof["connected_palette_relations_passed"] = False
    proof.update(native_reference_byte_parity_passed=True, area_condition_passed=True)
    assert relational_verdict(proof)["relational_presentation_blockers"] == ["connected_palette_relations_passed"]


@pytest.mark.parametrize("pairs", [np.array([[0, 3]]), np.array([[-1, 0]]), np.array([[0., 1.]])])
def test_contact_probe_rejects_invalid_address(pairs):
    with pytest.raises(QualificationError):
        contact_residuals(np.zeros((2, 2)), pairs)
