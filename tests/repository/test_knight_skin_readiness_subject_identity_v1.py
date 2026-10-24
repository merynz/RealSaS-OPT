from __future__ import annotations

import tools.ops.inspect_knight_skin_readiness_v1 as inspector


def _empty_candidates():
    return {role: [] for role, _, _ in inspector.REQUIRED}


def test_inspector_resolves_subject_identity_without_baseline_authority(monkeypatch):
    reads = []

    def fake_get(_api_root, path):
        reads.append(path)
        if path == "/v1/subjects/by-slug/knight":
            return {
                "schema": "RealSaS.SubjectIdentity.v1",
                "subject_id": "11111111-1111-4111-8111-111111111111",
                "slug": "knight",
                "display_name": "Knight product-nearest regression witness",
            }
        raise AssertionError(path)

    monkeypatch.setattr(inspector, "_get_json", fake_get)
    monkeypatch.setattr(inspector, "_load_valid_candidates", lambda _root, _subject: (_empty_candidates(), []))
    monkeypatch.setattr(inspector, "_mira_state", lambda _root: {
        "status": "MISSING",
        "runtime_identity_ready": False,
        "research_execution_admitted": False,
    })

    report = inspector.inspect("http://example.invalid", "a" * 40)
    assert reads == ["/v1/subjects/by-slug/knight"]
    assert "/v1/ops/baselines/knight" not in reads
    assert report["status"] == "NOT_READY"
    assert report["subject"] == {
        "id": "11111111-1111-4111-8111-111111111111",
        "slug": "knight",
        "display_name": "Knight product-nearest regression witness",
        "identity_source": "PLATFORM_SUBJECT_IDENTITY_READ_V1",
    }
    assert report["platform_mutation_performed"] is False


def test_failure_receipt_preserves_reason_without_claiming_authority():
    report = inspector._failure_report("b" * 40, RuntimeError("typed-parent-missing"))
    assert report["schema"] == "RealSaS.KnightSkinReadinessInspectionFailure.v1"
    assert report["status"] == "FAILED_INSPECTION"
    assert report["source_code_sha"] == "b" * 40
    assert report["error"] == "typed-parent-missing"
    for key in (
        "scientific_pass_minted",
        "qualification_minted",
        "model_inference_executed",
        "platform_mutation_performed",
        "model_promotion_performed",
        "product_skin_activated",
        "product_revision_minted",
        "genericity_claimed",
    ):
        assert report[key] is False
