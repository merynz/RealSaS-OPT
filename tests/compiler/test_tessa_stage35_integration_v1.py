from __future__ import annotations

import inspect

from compiler.realsas_compiler_core import qualified_mesh_v2
from compiler.realsas_compiler_core import tessa_geometry_qualification_v1
from compiler.realsas_compiler_services.orchestrator.adapters import mesh_v2


def test_stage35_tessa_route_requires_stage19_and_exact_carrier_native_evidence() -> None:
    source = inspect.getsource(mesh_v2.qualify_canonical_mesh_stage)

    required_tokens = (
        'candidate.producer_id == "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"',
        '"18_CANONICAL_MESH_ADDRESSING_BUILD"',
        '"RealSaS.TESSACandidateBridgeEvidenceIR.v1"',
        '"19_STATIC_CANONICAL_MESH_QUALIFIED"',
        '"RealSaS.StaticCanonicalMeshQualificationIR.v1"',
        '"RealSaS.TESSAStaticCarrierBindingIR.v1"',
        '"carrier_evidence":carrier_evidence',
        '"carrier_skin":skin',
        '"post_bind_joint_frame_qualification_hash":post_bind_report_hash',
        "qualify_canonical_mesh_candidate_v2(",
        "**tessa_evidence",
    )
    for token in required_tokens:
        assert token in source

    dispatch_offset = source.index("qualify_canonical_mesh_candidate_v2(")
    assert source.index('"RealSaS.TESSACandidateBridgeEvidenceIR.v1"') < dispatch_offset
    assert source.index('"RealSaS.StaticCanonicalMeshQualificationIR.v1"') < dispatch_offset
    assert source.index('"RealSaS.TESSAStaticCarrierBindingIR.v1"') < dispatch_offset
    assert source.index('"carrier_evidence":carrier_evidence') < dispatch_offset
    assert source.index('"carrier_skin":skin') < dispatch_offset

    # Stage35 must not bypass the V2 dispatcher with the frozen deterministic minter.
    assert "mesh=qualify_canonical_mesh_candidate(" not in source


def test_tessa_dynamic_interlock_is_satisfied_only_after_carrier_native_proof() -> None:
    dispatcher_source = inspect.getsource(
        qualified_mesh_v2.qualify_tessa_canonical_mesh_candidate_v2
    )
    proof_call = dispatcher_source.index("carrier_native_dynamic_binding_hash_v1(")
    validator_call = dispatcher_source.index("validate_tessa_static_carrier_binding_v1(")
    mint_start = dispatcher_source.index("id_map =")
    assert proof_call < validator_call < mint_start

    # The Stage19 receipt itself intentionally remains static-only. The old dynamic
    # interlock is disabled only after exact M/W_M/post-bind evidence was validated.
    call_tail = dispatcher_source[validator_call:mint_start]
    assert "require_dynamic_carrier_field_binding=False" in call_tail

    public_dispatch = inspect.getsource(
        qualified_mesh_v2.qualify_canonical_mesh_candidate_v2
    )
    assert "carrier_evidence is None or carrier_skin is None" in public_dispatch
    assert "_DYNAMIC_FIELD_BLOCKER" in public_dispatch


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
