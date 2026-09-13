from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.bundle_routes import route_for
from compiler.realsas_compiler_core.compile_transaction import (
    append_compile_stage,
    artifact_binding_hash,
    compile_result_hash,
    fork_compile_transaction,
    make_artifact_binding,
    make_compile_request,
    make_stage_contract,
    seal_compile_result,
    stage_result_hash,
    start_compile_transaction,
    transaction_hash,
    validate_compile_transaction,
)
from compiler.realsas_compiler_core.types import QualificationError


def _request():
    stages = (
        make_stage_contract(
            "MOTION",
            policy_hash="policy:motion:v1",
            implementation_binding_hash="impl:motion:v1",
        ),
        make_stage_contract(
            "PROOF",
            policy_hash="policy:proof:v1",
            implementation_binding_hash="impl:proof:v1",
            dependencies=("MOTION",),
        ),
        make_stage_contract(
            "RUNTIME_EXPORT",
            policy_hash="policy:runtime:v1",
            implementation_binding_hash="impl:runtime:v1",
            dependencies=("PROOF",),
        ),
    )
    return make_compile_request(
        request_id="REQ:FIT2:E2E:0001",
        source_observation_hash="obs-hash-8view",
        source_camera_bundle_hash="camera-hash-8view",
        compiler_semantic_version="RealSaS.Compiler.v4+transaction-v1",
        policy_bundle_hash="policy-bundle-hash",
        stage_contracts=stages,
    )


def _binding(slot, stage, content_hash, *, product_hash="", proof_hash=""):
    metadata = {"source_proof_hash": proof_hash} if proof_hash else {}
    return make_artifact_binding(
        slot_id=slot,
        authority_class="TEST_AUTHORITY",
        artifact_schema=f"Test.{slot}.v1",
        artifact_id=f"{slot}:{content_hash}",
        content_hash=content_hash,
        producer_stage=stage,
        source_product_state_hash=product_hash,
        metadata=metadata,
    )


def _passed_transaction():
    tx = start_compile_transaction(_request(), transaction_id="TX:0001", attempt_id="ATTEMPT:0001")
    product = _binding("PRODUCT", "MOTION", "product-hash", product_hash="product-hash")
    tx = append_compile_stage(
        tx,
        stage_id="MOTION",
        status="PASS",
        consumed_slots=("SOURCE_OBSERVATION", "SOURCE_CAMERAS"),
        produced_bindings=(product,),
    )
    proof = _binding("PROOF_BUNDLE", "PROOF", "proof-hash", product_hash="product-hash")
    tx = append_compile_stage(
        tx,
        stage_id="PROOF",
        status="PASS",
        consumed_slots=("PRODUCT",),
        produced_bindings=(proof,),
    )
    runtime = _binding(
        "RUNTIME_PACKAGE",
        "RUNTIME_EXPORT",
        "runtime-hash",
        product_hash="product-hash",
        proof_hash="proof-hash",
    )
    tx = append_compile_stage(
        tx,
        stage_id="RUNTIME_EXPORT",
        status="PASS",
        consumed_slots=("PRODUCT", "PROOF_BUNDLE"),
        produced_bindings=(runtime,),
    )
    return tx


def test_exact_identity_happy_path_seals_one_result():
    tx = _passed_transaction()
    assert tx.status == "PASS"
    validate_compile_transaction(tx)
    result = seal_compile_result(tx)
    assert result.status == "PASS"
    assert result.product_state_hash == "product-hash"
    assert result.proof_bundle_hash == "proof-hash"
    assert result.runtime_package_hash == "runtime-hash"
    assert result.result_hash == compile_result_hash(result)
    assert result.metadata["hidden_retriangulation_forbidden"] is True
    assert result.metadata["latest_path_selection_forbidden"] is True


def test_stage_policy_or_implementation_cannot_change_after_request_freeze():
    tx = _passed_transaction()
    first = replace(tx.stage_results[0], policy_hash="post-hoc-policy")
    first = replace(first, stage_result_hash=stage_result_hash(first))
    tampered = replace(tx, stage_results=(first,) + tx.stage_results[1:], transaction_hash="")
    tampered = replace(tampered, transaction_hash=transaction_hash(tampered))
    with pytest.raises(QualificationError, match="POLICY_MUTATION_FORBIDDEN"):
        validate_compile_transaction(tampered)

    first = replace(tx.stage_results[0], implementation_binding_hash="different-impl")
    first = replace(first, stage_result_hash=stage_result_hash(first))
    tampered = replace(tx, stage_results=(first,) + tx.stage_results[1:], transaction_hash="")
    tampered = replace(tampered, transaction_hash=transaction_hash(tampered))
    with pytest.raises(QualificationError, match="IMPLEMENTATION_MUTATION_FORBIDDEN"):
        validate_compile_transaction(tampered)


def test_stage_order_and_unknown_inputs_fail_closed():
    tx = start_compile_transaction(_request(), transaction_id="TX:ORDER", attempt_id="ATTEMPT:ORDER")
    with pytest.raises(QualificationError, match="STAGE_ORDER_MISMATCH"):
        append_compile_stage(
            tx,
            stage_id="PROOF",
            status="PASS",
            consumed_slots=("SOURCE_OBSERVATION",),
            produced_bindings=(_binding("X", "PROOF", "x"),),
        )
    with pytest.raises(QualificationError, match="UNKNOWN_INPUT_SLOT"):
        append_compile_stage(
            tx,
            stage_id="MOTION",
            status="PASS",
            consumed_slots=("latest-product",),
            produced_bindings=(_binding("PRODUCT", "MOTION", "p", product_hash="p"),),
        )


def test_same_slot_cannot_be_rebound_in_place():
    tx = start_compile_transaction(_request(), transaction_id="TX:REBIND", attempt_id="ATTEMPT:REBIND")
    tx = append_compile_stage(
        tx,
        stage_id="MOTION",
        status="PASS",
        consumed_slots=("SOURCE_OBSERVATION",),
        produced_bindings=(_binding("PRODUCT", "MOTION", "p1", product_hash="p1"),),
    )
    with pytest.raises(QualificationError, match="SLOT_REBIND_FORBIDDEN"):
        append_compile_stage(
            tx,
            stage_id="PROOF",
            status="PASS",
            consumed_slots=("PRODUCT",),
            produced_bindings=(_binding("PRODUCT", "PROOF", "p2", product_hash="p2"),),
        )


def test_fail_or_abstain_is_terminal_and_requires_explicit_blocker():
    tx = start_compile_transaction(_request(), transaction_id="TX:FAIL", attempt_id="ATTEMPT:FAIL")
    with pytest.raises(QualificationError, match="REQUIRES_BLOCKER"):
        append_compile_stage(
            tx,
            stage_id="MOTION",
            status="ABSTAIN",
            consumed_slots=("SOURCE_OBSERVATION",),
        )
    tx = append_compile_stage(
        tx,
        stage_id="MOTION",
        status="ABSTAIN",
        consumed_slots=("SOURCE_OBSERVATION",),
        blockers=("insufficient_motion_authority",),
    )
    assert tx.status == "ABSTAIN"
    with pytest.raises(QualificationError, match="IS_TERMINAL"):
        append_compile_stage(
            tx,
            stage_id="PROOF",
            status="FAIL",
            consumed_slots=("SOURCE_OBSERVATION",),
            blockers=("must_not_run",),
        )


def test_child_attempt_is_new_transaction_with_exact_parent_and_request_identity():
    parent = _passed_transaction()
    child = fork_compile_transaction(
        parent,
        transaction_id="TX:CHILD",
        attempt_id="ATTEMPT:CHILD",
        carry_slots=("PRODUCT",),
    )
    assert child.parent_transaction_hash == parent.transaction_hash
    assert child.request.request_hash == parent.request.request_hash
    assert child.status == "OPEN"
    assert child.stage_results == ()
    assert {b.slot_id for b in child.artifact_bindings} == {
        "SOURCE_OBSERVATION",
        "SOURCE_CAMERAS",
        "PRODUCT",
    }


def test_latest_alias_is_not_artifact_authority():
    with pytest.raises(QualificationError, match="LATEST_ALIAS_FORBIDDEN"):
        make_artifact_binding(
            slot_id="W",
            authority_class="QUALIFIED_SKIN",
            artifact_schema="RealSaS.QualifiedSkinIR.v1",
            artifact_id="latest",
            content_hash="hash",
            producer_stage="SKIN",
        )


def test_artifact_binding_is_self_hashing_and_tamper_evident():
    value = _binding("PRODUCT", "MOTION", "p", product_hash="p")
    assert value.binding_hash == artifact_binding_hash(value)
    tampered = replace(value, content_hash="other")
    with pytest.raises(QualificationError, match="BINDING_HASH_MISMATCH"):
        from compiler.realsas_compiler_core.compile_transaction import validate_artifact_binding
        validate_artifact_binding(tampered)


def test_compile_execution_artifacts_have_one_canonical_bundle_route():
    request = _request()
    tx = start_compile_transaction(request, transaction_id="TX:ROUTE", attempt_id="ATTEMPT:ROUTE")
    result = seal_compile_result(_passed_transaction())
    assert route_for(request).filename == "compile_request_ir.json"
    assert route_for(tx).filename == "compile_transaction_ir.json"
    assert route_for(result).filename == "compile_result_ir.json"
    assert route_for(result).section == "orchestrator"
