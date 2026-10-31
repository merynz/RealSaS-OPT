from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "platform/internal/capability/contracts/product_capabilities_v1.json"

Artifact = tuple[str, str]

ROOT_INPUTS: frozenset[Artifact] = frozenset({("RealSaS.SourceObservation", "v1")})

# This is the executable artifact surface, not merely the final ProductRevision
# role set. Supplemental capability outputs are included because downstream
# product capabilities consume them directly.
PRODUCES: dict[str, frozenset[Artifact]] = {
    "product.input": frozenset(
        {
            ("RealSaS.ShapeInput", "v1"),
            ("RealSaS.QualifiedCameraSetIR", "v1"),
            ("RealSaS.QualifiedObservationSetIR", "v1"),
            ("RealSaS.NormalizationDomainIR", "v1"),
        }
    ),
    "product.shape": frozenset({("RealSaS.ShapeProposal", "v2")}),
    "product.mesh": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.RiggingSurfaceIR", "v1"),
        }
    ),
    "product.rig": frozenset(
        {
            ("RealSaS.CanonicalRigIR", "v2"),
            ("RealSaS.RigProgramProposalIR", "v1"),
            ("RealSaS.SkinCodeProposalIR", "v1"),
        }
    ),
    "product.skin": frozenset({("RealSaS.CanonicalSkinIR", "v2")}),
    "product.mechanics": frozenset({("RealSaS.CanonicalMechanicsIR", "v1")}),
    "product.graphics": frozenset({("RealSaS.CanonicalGraphicsIR", "v1")}),
    "product.motion": frozenset({("RealSaS.CanonicalMotionIR", "v2")}),
    "product.proof": frozenset({("RealSaS.ProductProofBundleIR", "v1")}),
}

REQUIRES: dict[str, frozenset[Artifact]] = {
    "product.input": ROOT_INPUTS,
    "product.shape": frozenset({("RealSaS.ShapeInput", "v1")}),
    "product.mesh": frozenset(
        {
            ("RealSaS.SourceObservation", "v1"),
            ("RealSaS.ShapeProposal", "v2"),
            ("RealSaS.QualifiedCameraSetIR", "v1"),
            ("RealSaS.QualifiedObservationSetIR", "v1"),
            ("RealSaS.NormalizationDomainIR", "v1"),
        }
    ),
    "product.rig": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.RiggingSurfaceIR", "v1"),
        }
    ),
    "product.skin": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.CanonicalRigIR", "v2"),
            ("RealSaS.SkinCodeProposalIR", "v1"),
        }
    ),
    "product.mechanics": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.CanonicalRigIR", "v2"),
            ("RealSaS.CanonicalSkinIR", "v2"),
        }
    ),
    "product.graphics": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.CanonicalSkinIR", "v2"),
            ("RealSaS.CanonicalMechanicsIR", "v1"),
            ("RealSaS.GraphicsEvidence", "v1"),
            ("RealSaS.CompleteAppearanceAssetIR", "v2"),
        }
    ),
    "product.motion": frozenset(
        {
            ("RealSaS.CanonicalRigIR", "v2"),
            ("RealSaS.CanonicalMechanicsIR", "v1"),
        }
    ),
    "product.proof": frozenset(
        {
            ("RealSaS.CanonicalMeshIR", "v1"),
            ("RealSaS.CanonicalRigIR", "v2"),
            ("RealSaS.CanonicalSkinIR", "v2"),
            ("RealSaS.CanonicalMechanicsIR", "v1"),
            ("RealSaS.CanonicalGraphicsIR", "v1"),
            ("RealSaS.CanonicalMotionIR", "v2"),
        }
    ),
}

# product.motion has an in-executor deterministic provider for the optional
# ControlMotionClipProposalIR. It therefore does not appear in REQUIRES.
INTERNAL_PROVIDERS: dict[str, frozenset[Artifact]] = {
    "product.motion": frozenset({("RealSaS.ControlMotionClipProposalIR", "v1")}),
}

# These are the only admitted shipping closure debts at this revision. Keeping
# them exact is useful: the court fails immediately if another capability starts
# depending on an artifact that its dependency graph cannot actually deliver.
EXPECTED_OPEN_CLOSURE: dict[str, frozenset[Artifact]] = {
    "product.graphics": frozenset(
        {
            ("RealSaS.GraphicsEvidence", "v1"),
            ("RealSaS.CompleteAppearanceAssetIR", "v2"),
        }
    )
}


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _dependencies(contract: dict, capability_id: str) -> tuple[str, ...]:
    row = contract["capabilities"][capability_id]
    return tuple(row.get("policy", {}).get("dependencies") or ())


def _accessible(contract: dict, capability_id: str) -> frozenset[Artifact]:
    out = set(ROOT_INPUTS)
    out.update(INTERNAL_PROVIDERS.get(capability_id, ()))
    for dependency in _dependencies(contract, capability_id):
        out.update(PRODUCES[dependency])
    return frozenset(out)


def _missing(contract: dict, capability_id: str) -> frozenset[Artifact]:
    return REQUIRES[capability_id] - _accessible(contract, capability_id)


def test_shipping_product_capability_set_matches_closure_court():
    contract = _contract()
    assert set(contract["capabilities"]) == set(PRODUCES) == set(REQUIRES)


def test_dependency_graph_delivers_every_required_artifact_except_declared_graphics_debt():
    contract = _contract()
    missing = {
        capability_id: _missing(contract, capability_id)
        for capability_id in REQUIRES
        if _missing(contract, capability_id)
    }
    assert missing == EXPECTED_OPEN_CLOSURE


def test_default_idle_provider_closed_the_old_motion_proposal_hole():
    contract = _contract()
    assert _missing(contract, "product.motion") == frozenset()
    assert INTERNAL_PROVIDERS["product.motion"] == frozenset(
        {("RealSaS.ControlMotionClipProposalIR", "v1")}
    )


def test_graphics_debt_is_evidence_and_appearance_only_not_geometry_or_mechanics():
    contract = _contract()
    missing = _missing(contract, "product.graphics")
    assert missing == EXPECTED_OPEN_CLOSURE["product.graphics"]
    assert ("RealSaS.CanonicalMeshIR", "v1") not in missing
    assert ("RealSaS.CanonicalSkinIR", "v2") not in missing
    assert ("RealSaS.CanonicalMechanicsIR", "v1") not in missing
