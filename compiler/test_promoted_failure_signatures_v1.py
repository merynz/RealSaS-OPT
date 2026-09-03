from pathlib import Path

from compiler.realsas_compiler_services.proof.failure_signatures import (
    HISTORICAL_SOURCE_SHA256,
    derive_failure_signatures,
    no_owner_attribution,
)


def test_failure_signature_is_diagnostic_not_owner_attribution():
    signatures = derive_failure_signatures(
        "MESH_QUALITY",
        {"component_count": 8, "face_count": 64, "degenerate_faces": 2, "min_area": 0.0},
        status="FAIL",
    )
    assert signatures
    assert no_owner_attribution() == ()
    assert all(s["owner_attribution_status"] == "not_performed" for s in signatures)
    assert all(s["metadata"]["causal_owner_not_inferred_from_failure"] is True for s in signatures)
    assert all(s["metadata"]["repair_not_authorized_by_signature"] is True for s in signatures)
    assert all(s["metadata"]["promotion_origin_sha256"] == HISTORICAL_SOURCE_SHA256 for s in signatures)


def test_abstain_is_typed_and_unowned():
    signatures = derive_failure_signatures(
        "DEFORMATION",
        {"status": "MISSING_FIXTURE"},
        status="ABSTAIN",
    )
    assert len(signatures) == 1
    assert signatures[0]["failure_family"] == "insufficient_proof_evidence"
    assert signatures[0]["owner_attribution_status"] == "not_performed"
    assert no_owner_attribution() == ()


def test_pass_has_no_failure_evidence():
    assert derive_failure_signatures("MOTION", {}, status="PASS") == ()


def test_historical_motion_family_is_rebound_without_old_contract_stack():
    signatures = derive_failure_signatures(
        "MOTION",
        {
            "clip_count": 1,
            "effective_joint_track_count": 1,
            "max_edge_stretch_ratio": 1.8,
            "frozen_policy_thresholds": {"max_edge_stretch_ratio": 1.55},
            "failure_localization": {"view_index": 2, "mesh_id": "body"},
        },
        status="FAIL",
    )
    edge = next(s for s in signatures if s["failure_family"] == "edge_stretch_exceeded")
    assert edge["invariant"] == "bounded_edge_stretch"
    assert edge["metadata"]["historical_family_rebound"] is True
    assert edge["localization"] == {"view_index": 2, "mesh_id": "body"}


def test_current_proof_engine_cannot_infer_owner_from_failed_domain_source():
    source = Path("compiler/realsas_compiler_core/proof_engine.py").read_text(encoding="utf-8")
    assert 'owners=() if status=="PASS" else ({"owner_domain":domain},)' not in source
    assert '"owner_domain": domain' not in source
    assert "owners = no_owner_attribution()" in source
    assert "owner_attribution_requires_controlled_fault_experiment" in source
