import hashlib
import json

import pytest

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import (
    SUBJECT_AGNOSTIC_RELEASE_CONTRACT,
    graph_node_sha256,
    released_execution_plan,
    subject_agnostic_release_parameters,
    verify_execution_version,
    verify_source_inputs,
)
from tools.platform_release_snapshot import snapshot


def _go_graph(plan):
    fields = ("ordinal", "id", "title", "group", "depends_on", "adapter", "manifest_keys")
    policy = ("fail_closed", "cacheable", "output_hash_required", "product_pass_authority")
    return {
        "schema": "RealSaS.StageGraph.v1",
        "stage_count": len(plan["stages"]),
        "stages": [
            {
                **{key: row[key] for key in fields},
                "policy": {key: bool(row["policy"].get(key, False)) for key in policy},
            }
            for row in plan["stages"]
        ],
    }


def test_release_snapshot_does_not_embed_subject_values():
    plan = mainline.load_json(mainline.PLAN_PATH)
    first = {"subject_id": "alpha", "source_files": [{"role": "alpha-only"}]}
    second = {"subject_id": "beta", "source_files": [{"role": "beta-only"}]}
    a = snapshot(plan, first, name="generic-release", purpose="RESEARCH", created_by="ci")
    b = snapshot(plan, second, name="generic-release", purpose="RESEARCH", created_by="ci")
    assert a == b
    encoded = json.dumps(a, sort_keys=True)
    assert "alpha-only" not in encoded and "beta-only" not in encoded
    for stage, binding in zip(plan["stages"], a["manifest"]["stages"]):
        parameters = binding["semantic_parameters"]
        assert parameters["contract"] == SUBJECT_AGNOSTIC_RELEASE_CONTRACT
        assert parameters["manifest_keys"] == list(stage.get("manifest_keys") or [])
        assert "manifest" not in parameters


def test_same_release_verifies_two_arbitrary_subject_manifests(monkeypatch):
    base = mainline.load_json(mainline.PLAN_PATH)
    plan = released_execution_plan(base, _go_graph(base), mode="RESEARCH")
    stage = plan["stages"][1]
    monkeypatch.setattr(mainline, "_adapter_impl_hash", lambda _: "a" * 64)
    request = {
        "stage_id": stage["id"],
        "graph_node_sha256": graph_node_sha256(stage),
        "implementation_sha256": "a" * 64,
        "policy_sha256": mainline.content_sha256(stage["policy"]),
        "semantic_parameters": subject_agnostic_release_parameters(stage),
    }
    manifest_a = {"source_license": {"source_pack": "pack-a", "license_name": "A", "license_ref": "urn:a"}}
    manifest_b = {"source_license": {"source_pack": "pack-b", "license_name": "B", "license_ref": "urn:b"}}
    verify_execution_version(plan, request, manifest_a)
    verify_execution_version(plan, request, manifest_b)


def test_subject_input_cas_still_rejects_cross_subject_section_swap():
    section_a = {"source_pack": "pack-a", "license_name": "A", "license_ref": "urn:a"}
    section_b = {"source_pack": "pack-b", "license_name": "B", "license_ref": "urn:b"}
    raw_b = json.dumps(section_b, sort_keys=True).encode()
    ref_b = {
        "role": "subject:manifest:source_license",
        "content_sha256": hashlib.sha256(raw_b).hexdigest(),
        "size_bytes": len(raw_b),
    }
    with pytest.raises(RuntimeError, match="MANIFEST_SECTION_DRIFT"):
        verify_source_inputs(
            manifest={"source_license": section_a},
            inputs=[ref_b],
            read_object=lambda _: raw_b,
        )


def test_stage_release_parameters_are_generic_and_subject_authority_is_forbidden():
    plan = mainline.load_json(mainline.PLAN_PATH)
    stage = plan["stages"][9]
    parameters = subject_agnostic_release_parameters(
        stage,
        {"external_evidence_roles": ["receipt", "checkpoint", "result"]},
    )
    assert parameters["external_evidence_roles"] == ["receipt", "checkpoint", "result"]
    with pytest.raises(RuntimeError, match="SUBJECT_AUTHORITY_OVERRIDE_FORBIDDEN"):
        subject_agnostic_release_parameters(stage, {"manifest": {"subject": "forbidden"}})


def test_snapshot_accepts_release_scoped_external_stage_contract():
    plan = mainline.load_json(mainline.PLAN_PATH)
    stage_id = plan["stages"][9]["id"]
    parameters = {
        "schema": "RealSaS.StageReleaseParameters.v1",
        "stages": {
            stage_id: {
                "external_evidence_roles": ["receipt", "checkpoint", "result"],
            }
        },
    }
    request = snapshot(
        plan,
        {"subject_id": "not-release-authority"},
        name="external-generic-release",
        purpose="RESEARCH",
        created_by="ci",
        stage_parameters=parameters,
    )
    binding = next(row for row in request["manifest"]["stages"] if row["stage_id"] == stage_id)
    assert binding["semantic_parameters"]["external_evidence_roles"] == ["receipt", "checkpoint", "result"]
