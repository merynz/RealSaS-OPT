from models.canonical_model_names_v2 import (
    AXIS,
    CANONICAL_MODEL_NAMES,
    IRIS,
    MIRA,
    TESSA,
)


def test_canonical_model_stack_v2_names_and_expansions_are_unique():
    assert CANONICAL_MODEL_NAMES == ("IRIS", "TESSA", "AXIS", "MIRA")
    expansions = {IRIS.expansion, TESSA.expansion, AXIS.expansion, MIRA.expansion}
    assert len(expansions) == 4
    assert IRIS.expansion == "Image-based Relational Inference for Structure"
    assert TESSA.expansion == "Topological Evidence for Surface Structure Approximation"
    assert AXIS.expansion == "Articulation eXtraction through Inferred Structure"
    assert MIRA.expansion == "Mesh-Informed Rigging Affinity"


def test_legacy_names_are_lineage_only_not_canonical_names():
    assert TESSA.legacy_names == ()
    assert AXIS.legacy_names == ("ATLAS", "Geppetto")
    assert MIRA.legacy_names == ("Arachne",)
    legacy = set(AXIS.legacy_names + MIRA.legacy_names)
    assert legacy.isdisjoint(CANONICAL_MODEL_NAMES)


def test_names_do_not_embed_unproven_equivariance_or_invariance_claims():
    for identity in (IRIS, TESSA, AXIS, MIRA):
        expansion = identity.expansion.lower()
        assert "equivariant" not in expansion
        assert "invariant" not in expansion


def test_learned_models_do_not_claim_compiler_authority():
    for identity in (IRIS, TESSA, AXIS, MIRA):
        assert identity.canonical_authority == "LEARNED_PROPOSAL_OR_EVIDENCE_ONLY"
