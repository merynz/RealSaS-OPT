from copy import deepcopy

import pytest

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import (
    graph_node_sha256, released_execution_plan, verify_execution_version,
)
from tools.platform_release_snapshot import snapshot


def go_graph(plan):
    fields = ("ordinal", "id", "title", "group", "depends_on", "adapter", "manifest_keys")
    policy = ("fail_closed", "cacheable", "output_hash_required", "product_pass_authority")
    return {"schema": "RealSaS.StageGraph.v1", "stage_count": len(plan["stages"]),
            "stages": [{**{key: deepcopy(row[key]) for key in fields},
                        "policy": {key: bool(row["policy"].get(key, False)) for key in policy}}
                       for row in plan["stages"]]}


def test_real_go_projection_preserves_full_policy_and_exact_plan_hash():
    base = mainline.load_json(mainline.PLAN_PATH)
    plan = released_execution_plan(base, go_graph(base))
    assert plan == base
    assert mainline.validate_plan(plan) == mainline.validate_plan(base)
    for stage, original in zip(plan["stages"], base["stages"]):
        assert stage["policy"] == original["policy"]
        assert graph_node_sha256(stage) == graph_node_sha256(original)


@pytest.mark.parametrize("graph", [None, {}])
def test_missing_release_graph_is_not_silently_assigned_current_main(graph):
    with pytest.raises(RuntimeError, match="GRAPH_REQUIRED"):
        released_execution_plan(mainline.load_json(mainline.PLAN_PATH), graph)


def test_rewired_research_graph_is_actual_engine_plan_and_changes_plan_identity():
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    graph["stages"][1]["depends_on"] = []
    plan = released_execution_plan(base, graph, mode="RESEARCH")
    assert plan["stages"][1]["depends_on"] == []
    assert base["stages"][1]["depends_on"] != []
    assert mainline.content_sha256(plan) != mainline.content_sha256(base)
    assert graph_node_sha256(plan["stages"][0]) == graph_node_sha256(base["stages"][0])
    with pytest.raises(RuntimeError, match="PRODUCT_RELEASE_PLAN_DRIFT"):
        released_execution_plan(base, graph, mode="PRODUCT")


@pytest.mark.parametrize("bad_dep", ["99_UNKNOWN", "46_PRODUCT_CLOSURE_SEAL"])
def test_unknown_or_cyclic_dependency_fails_before_engine_execution(bad_dep):
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    graph["stages"][0]["depends_on"] = [bad_dep]
    with pytest.raises(RuntimeError, match="FUTURE_OR_UNKNOWN_DEPENDENCY"):
        released_execution_plan(base, graph, mode="RESEARCH")


def test_duplicate_dependency_fails_closed():
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    graph["stages"][1]["depends_on"] *= 2
    with pytest.raises(RuntimeError, match="DUPLICATE_DEPENDENCY"):
        released_execution_plan(base, graph, mode="RESEARCH")


def test_small_research_network_can_omit_product_closure():
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    graph["stages"] = graph["stages"][:2]
    graph["stage_count"] = 2
    plan = released_execution_plan(base, graph, mode="RESEARCH")
    ledger = mainline.build_fresh_run_ledger(plan, run_id="RESEARCH_NETWORK",
                                           subject_id="SUBJECT_FREE_NETWORK",
                                           manifest_ref="manifest.json",
                                           execution_class="IMPLEMENTATION_AUDIT",
                                           require_product_pass_authority=False)
    mainline.validate_ledger(plan, ledger, require_product_pass_authority=False)
    assert ledger["total_count"] == 2
    with pytest.raises(RuntimeError, match="PRODUCT_PASS_AUTHORITY_DRIFT"):
        mainline.validate_plan(plan)


def test_added_research_node_is_reconstructed_without_invalidating_independent_node():
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    node = deepcopy(graph["stages"][0])
    node.update(id="47_RESEARCH_PROBE", ordinal=47, depends_on=[node["id"]])
    graph["stages"].append(node)
    graph["stage_count"] = 47
    plan = released_execution_plan(base, graph, mode="RESEARCH")
    assert plan["stages"][-1]["id"] == "47_RESEARCH_PROBE"
    assert graph_node_sha256(plan["stages"][0]) == graph_node_sha256(base["stages"][0])


def test_reconstructed_full_policy_matches_release_execution_binding(monkeypatch):
    base = mainline.load_json(mainline.PLAN_PATH)
    plan = released_execution_plan(base, go_graph(base), mode="RESEARCH")
    stage = plan["stages"][-1]
    monkeypatch.setattr(mainline, "_adapter_impl_hash", lambda _: "a" * 64)
    request = {"stage_id": stage["id"], "graph_node_sha256": graph_node_sha256(stage),
               "implementation_sha256": "a" * 64,
               "policy_sha256": mainline.content_sha256(base["stages"][-1]["policy"]),
               "semantic_parameters": {"manifest": mainline._manifest_subset({}, stage)}}
    verify_execution_version(plan, request, {})
    request["policy_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="POLICY_DRIFT"):
        verify_execution_version(plan, request, {})


def test_snapshot_preflight_matches_engine_for_small_research_graph():
    base = mainline.load_json(mainline.PLAN_PATH)
    base["stages"] = base["stages"][:2]
    base["stage_count"] = 2
    request = snapshot(base, {}, name="research-network", purpose="RESEARCH", created_by="ci")
    reconstructed = released_execution_plan(mainline.load_json(mainline.PLAN_PATH),
                                            request["manifest"]["dag"], mode="RESEARCH")
    assert mainline.content_sha256(reconstructed) == request["manifest"]["pipeline_plan_sha256"]


def test_real_engine_executes_rewired_network_not_canonical_dependencies(tmp_path, monkeypatch):
    pytest.importorskip("temporalio")
    import asyncio
    from temporalio.testing import ActivityEnvironment
    from compiler.realsas_compiler_services.platform_worker import worker
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path / "authority"))
    monkeypatch.setenv("REALSAS_ARTIFACT_ROOT", str(tmp_path / "cas"))
    run_id = "SUBJECT_FREE_REWIRED_ENGINE"
    manifest = {"run_id": run_id, "subject_id": "SUBJECT_FREE_NETWORK",
                "execution_class": "IMPLEMENTATION_AUDIT",
                "source_license": {"source_pack": "test", "license_name": "test", "license_ref": "urn:test"}}
    mainline.atomic_json(mainline.run_manifest_path(run_id), manifest)
    base = mainline.load_json(mainline.PLAN_PATH)
    graph = go_graph(base)
    graph["stages"] = graph["stages"][:2]
    graph["stage_count"] = 2
    graph["stages"][1]["depends_on"] = []
    plan = released_execution_plan(base, graph, mode="RESEARCH")
    stage = plan["stages"][1]
    request = {"stage_id": stage["id"], "allowed_execute_stage_ids": [stage["id"]],
               "compiler_run_id": run_id, "released_graph": graph, "execution_mode": "RESEARCH",
               "pipeline_plan_sha256": mainline.content_sha256(plan),
               "graph_node_sha256": graph_node_sha256(stage),
               "implementation_sha256": mainline._adapter_impl_hash(stage["adapter"]),
               "policy_sha256": mainline.content_sha256(stage["policy"]),
               "semantic_parameters": {"manifest": mainline._manifest_subset(manifest, stage)}}
    import json
    source = worker._put_cas_bytes(json.dumps(manifest["source_license"]).encode())
    request["source_inputs"] = [{**source, "role": "subject:manifest:source_license"}]
    environment = ActivityEnvironment()
    heartbeats = []
    environment.on_heartbeat = lambda *details: heartbeats.append(details)
    # Exercise the actual async activity -> worker thread boundary. Replacing
    # heartbeat with a no-op hid the live SDK's event-loop error.
    result = asyncio.run(environment.run(worker.execute_compile_stage, request))
    assert heartbeats
    assert result["status"] == "PASS"
    ledger = mainline.load_json(mainline.run_ledger_path(run_id))
    assert ledger["stages"][0]["status"] == "PENDING"
    assert ledger["stages"][1]["status"] == "PASS"
    assert result["outputs"][0]["payload_schema"] == "RealSaS.SourceLicenseSeal.v1"
    assert worker._read_cas_object(result["outputs"][0])
    wrong = dict(request, pipeline_plan_sha256=mainline.content_sha256(base))
    with pytest.raises(RuntimeError, match="PLATFORM_PLAN_SHA_DRIFT"):
        worker._execute_stage_core(wrong)


def test_async_engine_heartbeat_stays_on_event_loop_and_propagates_failure():
    pytest.importorskip("temporalio")
    import asyncio
    import threading
    from temporalio.testing import ActivityEnvironment
    from compiler.realsas_compiler_services.platform_worker import worker

    owner_thread = threading.get_ident()
    environment = ActivityEnvironment()
    heartbeat_threads = []
    environment.on_heartbeat = lambda *_: heartbeat_threads.append(threading.get_ident())

    def synchronous_work(_):
        assert threading.get_ident() != owner_thread
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        raise ValueError("EXACT_ENGINE_FAILURE")

    with pytest.raises(ValueError, match="EXACT_ENGINE_FAILURE"):
        asyncio.run(environment.run(worker._with_heartbeat, synchronous_work, {"stage_id": "TEST"}))
    assert heartbeat_threads and all(thread == owner_thread for thread in heartbeat_threads)


def test_imported_source_must_match_manifest_section_and_actual_bytes(tmp_path):
    import hashlib
    import json
    from compiler.realsas_compiler_services.platform_worker.stage_inputs import verify_source_inputs
    data = b"sealed input"
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / "proposal.npz"
    path.write_bytes(data)
    manifest = {"model": {"proposal_path": str(path), "proposal_sha256": digest}}
    section = json.dumps(manifest["model"]).encode()
    ref = {"role": "subject:manifest:model", "content_sha256": hashlib.sha256(section).hexdigest(), "size_bytes": len(section)}
    verify_source_inputs(manifest=manifest, inputs=[ref], read_object=lambda _: section)
    wrong = {"model": {**manifest["model"], "proposal_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="MANIFEST_SECTION_DRIFT"):
        verify_source_inputs(manifest=wrong, inputs=[ref], read_object=lambda _: section)
    path.write_bytes(b"stale source")
    with pytest.raises(RuntimeError, match="SOURCE_INPUT_FILE_DRIFT"):
        verify_source_inputs(manifest=manifest, inputs=[ref], read_object=lambda _: section)
    raw = {"role": "subject:proposal", "content_sha256": digest, "size_bytes": len(data)}
    with pytest.raises(RuntimeError, match="SOURCE_INPUT_NOT_REFERENCED"):
        verify_source_inputs(manifest={}, inputs=[raw], read_object=lambda _: data)
