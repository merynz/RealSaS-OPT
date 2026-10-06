from __future__ import annotations

import inspect

from compiler.realsas_compiler_services.orchestrator.adapters import mesh_v2


def test_stage35_tessa_route_requires_stage19_bound_evidence_before_v2_mint() -> None:
    source = inspect.getsource(mesh_v2.qualify_canonical_mesh_stage)

    required_tokens = (
        'candidate.producer_id == "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"',
        '"18_CANONICAL_MESH_ADDRESSING_BUILD"',
        '"RealSaS.TESSACandidateBridgeEvidenceIR.v1"',
        '"19_STATIC_CANONICAL_MESH_QUALIFIED"',
        '"RealSaS.StaticCanonicalMeshQualificationIR.v1"',
        '"RealSaS.TESSAStaticCarrierBindingIR.v1"',
        "qualify_canonical_mesh_candidate_v2(",
        "**tessa_evidence",
    )
    for token in required_tokens:
        assert token in source

    mint_offset = source.index("qualify_canonical_mesh_candidate_v2(")
    assert source.index('"RealSaS.TESSACandidateBridgeEvidenceIR.v1"') < mint_offset
    assert source.index('"RealSaS.StaticCanonicalMeshQualificationIR.v1"') < mint_offset
    assert source.index('"RealSaS.TESSAStaticCarrierBindingIR.v1"') < mint_offset

    # Stage35 must not bypass the V2 dispatcher with the frozen deterministic minter.
    assert "mesh=qualify_canonical_mesh_candidate(" not in source
