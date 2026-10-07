from __future__ import annotations

import inspect

from compiler.realsas_compiler_core import qualified_mesh_v2
from compiler.realsas_compiler_core import tessa_geometry_qualification_v1
from compiler.realsas_compiler_services.orchestrator.adapters import mesh_v2


def test_stage35_tessa_route_requires_stage19_bound_evidence_before_v2_dispatch() -> None:
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

    dispatch_offset = source.index("qualify_canonical_mesh_candidate_v2(")
    assert source.index('"RealSaS.TESSACandidateBridgeEvidenceIR.v1"') < dispatch_offset
    assert source.index('"RealSaS.StaticCanonicalMeshQualificationIR.v1"') < dispatch_offset
    assert source.index('"RealSaS.TESSAStaticCarrierBindingIR.v1"') < dispatch_offset

    # Stage35 must not bypass the V2 dispatcher with the frozen deterministic minter.
    assert "mesh=qualify_canonical_mesh_candidate(" not in source


def test_tessa_v2_dispatch_is_interlocked_before_any_qualified_mesh_mint() -> None:
    dispatcher_source = inspect.getsource(
        qualified_mesh_v2.qualify_tessa_canonical_mesh_candidate_v2
    )
    validator_call = dispatcher_source.index("validate_tessa_static_carrier_binding_v1(")
    mint_start = dispatcher_source.index("id_map =")
    assert validator_call < mint_start

    # The call deliberately uses the validator default: dynamic carrier-field
    # qualification is required. Stage19 itself is the only caller that opts out.
    call_tail = dispatcher_source[validator_call:mint_start]
    assert "require_dynamic_carrier_field_binding=False" not in call_tail

    geometry_source = inspect.getsource(
        tessa_geometry_qualification_v1.validate_tessa_static_carrier_binding_v1
    )
    assert "require_dynamic_carrier_field_binding: bool = True" in geometry_source
    assert (
        tessa_geometry_qualification_v1._DYNAMIC_FIELD_BLOCKER
        == "TESSA_MIRA_CARRIER_FIELD_BINDING_NOT_PRODUCT_QUALIFIED"
    )


def test_stage19_static_binding_explicitly_does_not_claim_dynamic_product_transport() -> None:
    source = inspect.getsource(tessa_geometry_qualification_v1.bind_tessa_to_static_carrier_v1)
    required = (
        '"dynamic_carrier_field_binding_required_before_qualified_mesh": True',
        '"dynamic_carrier_field_binding_product_qualified": False',
        '"research_fit1_binding_is_product_authority": False',
        "require_dynamic_carrier_field_binding=False",
    )
    for token in required:
        assert token in source
