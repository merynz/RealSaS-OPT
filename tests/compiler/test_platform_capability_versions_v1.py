from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_capability_versions import (
    capability_implementations,
    capability_specs,
    product_model_requirements,
)
from tools.platform_release_snapshot import snapshot


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def test_product_capability_implementations_are_complete_and_semantic():
    specs = capability_specs()
    implementations = capability_implementations()
    assert set(implementations) == {
        "product.input",
        "product.shape",
        "product.mesh",
        "product.rig",
    }
    assert set(specs) == set(implementations)
    assert specs["product.input"]["parameters"]["responsibility_ids"] == [
        "input.model_observation_lowering"
    ]
    assert specs["product.shape"]["parameters"]["responsibility_ids"] == [
        "shape.shared_scene_evidence",
        "shape.topology_proposal",
    ]
    assert specs["product.mesh"]["parameters"]["responsibility_ids"] == [
        "mesh.canonical_identity_mint",
        "mesh.source_fidelity_qualification",
        "mesh.static_topology_qualification",
    ]
    assert specs["product.rig"]["parameters"]["responsibility_ids"] == [
        "rig.canonical_identity_mint",
        "rig.structure_proposal",
        "rig.structure_qualification",
    ]
    assert "rig.frame_observability_qualification" not in str(specs["product.rig"])
    assert specs["product.shape"]["parameters"]["required_models"] == [
        {
            "family": "IRIS",
            "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3",
        },
        {
            "family": "TESSA",
            "architecture_id": "RealSaS.TESSA.Structuralizer.v3",
        },
    ]
    assert specs["product.mesh"]["parameters"]["required_models"] == []
    assert specs["product.rig"]["parameters"]["required_models"] == [
        {
            "family": "AXIS",
            "architecture_id": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
        }
    ]
    assert product_model_requirements() == [
        {
            "family": "AXIS",
            "architecture_id": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
        },
        {
            "family": "IRIS",
            "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3",
        },
        {
            "family": "TESSA",
            "architecture_id": "RealSaS.TESSA.Structuralizer.v3",
        },
    ]
    for capability_id, implementation in implementations.items():
        assert set(implementation) == {"implementation_sha256"}, capability_id
        assert _is_sha256(implementation["implementation_sha256"]), capability_id
    assert len(
        {
            implementations["product.input"]["implementation_sha256"],
            implementations["product.shape"]["implementation_sha256"],
            implementations["product.mesh"]["implementation_sha256"],
            implementations["product.rig"]["implementation_sha256"],
        }
    ) == 4


def test_release_snapshot_v2_carries_implementation_and_model_requirements():
    plan = mainline.load_json(mainline.PLAN_PATH)
    request = snapshot(
        plan,
        {"subject_id": "not-release-authority"},
        name="capability-release-v2",
        purpose="RESEARCH",
        created_by="ci",
    )
    assert request["manifest"]["contract_version"] == "RealSaS.EngineReleaseManifest.v2"
    assert request["capability_implementations"] == capability_implementations()
    assert "capability_versions" not in request
    for row in request["capability_implementations"].values():
        assert set(row) == {"implementation_sha256"}
        assert "policy_sha256" not in row
        assert "parameters_sha256" not in row
    assert request["manifest"]["model_requirements"] == [
        {
            "family": "AXIS",
            "architecture_id": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
            "channel": "RESEARCH",
        },
        {
            "family": "IRIS",
            "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3",
            "channel": "RESEARCH",
        },
        {
            "family": "TESSA",
            "architecture_id": "RealSaS.TESSA.Structuralizer.v3",
            "channel": "RESEARCH",
        },
    ]
    assert "subject_id" not in str(request)
