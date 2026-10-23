from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_capability_versions import capability_specs, capability_versions
from tools.platform_release_snapshot import snapshot


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def test_product_capability_versions_are_complete_and_semantic():
    specs = capability_specs()
    versions = capability_versions()
    assert set(versions) == {"product.input", "product.shape"}
    assert set(specs) == set(versions)
    assert specs["product.input"]["parameters"]["responsibility_ids"] == [
        "input.model_observation_lowering"
    ]
    assert specs["product.shape"]["parameters"]["responsibility_ids"] == [
        "shape.shared_scene_evidence",
        "shape.topology_proposal",
    ]
    for capability_id, version in versions.items():
        assert set(version) == {
            "implementation_sha256",
            "policy_sha256",
            "parameters_sha256",
        }, capability_id
        assert all(_is_sha256(value) for value in version.values()), capability_id
        assert version["policy_sha256"] == mainline.content_sha256(
            specs[capability_id]["policy"]
        )
        assert version["parameters_sha256"] == mainline.content_sha256(
            specs[capability_id]["parameters"]
        )
    assert (
        versions["product.input"]["implementation_sha256"]
        != versions["product.shape"]["implementation_sha256"]
    )


def test_release_snapshot_v2_carries_non_stage_capability_versions():
    plan = mainline.load_json(mainline.PLAN_PATH)
    request = snapshot(
        plan,
        {"subject_id": "not-release-authority"},
        name="capability-release-v2",
        purpose="RESEARCH",
        created_by="ci",
    )
    assert request["manifest"]["contract_version"] == "RealSaS.EngineReleaseManifest.v2"
    assert request["capability_versions"] == capability_versions()
    assert "subject_id" not in str(request)
