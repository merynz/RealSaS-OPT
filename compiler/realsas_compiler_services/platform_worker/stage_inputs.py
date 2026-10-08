"""Materialize Platform-bound immutable stage results for Engine adapters.

The compiler ledger is an execution view. Registry CAS bytes remain immutable;
only paths and run-local fingerprints in that view are reconstructed.
"""
from pathlib import Path
import json
import os
import tempfile

from compiler.realsas_compiler_services.orchestrator import mainline


def released_execution_plan(base_plan, released_graph, *, mode="PRODUCT"):
    """Reconstruct execution metadata without losing the full Compiler policy.

    The four scheduling policy fields belong to the released graph. Additional
    scientific policy/metadata belongs to the shipped Compiler plan. The hash of
    this complete reconstruction must match the release's pipeline_plan_sha256.
    """
    if not released_graph:
        raise RuntimeError("ENGINE_RELEASE_GRAPH_REQUIRED")
    if mode not in {"RESEARCH", "PRODUCT"}:
        raise RuntimeError("ENGINE_EXECUTION_MODE_REQUIRED")
    graph = dict(released_graph)
    if graph.get("schema") != "RealSaS.StageGraph.v1":
        raise RuntimeError("ENGINE_RELEASE_GRAPH_SCHEMA_DRIFT")
    stages = graph.get("stages")
    if not isinstance(stages, list) or int(graph.get("stage_count", -1)) != len(stages):
        raise RuntimeError("ENGINE_RELEASE_GRAPH_CARDINALITY_DRIFT")
    base_stages = mainline._stage_map(base_plan)
    reconstructed = []
    seen = set()
    for ordinal, node in enumerate(stages, 1):
        if node.get("ordinal") != ordinal or node.get("id") in seen:
            raise RuntimeError("ENGINE_RELEASE_GRAPH_ORDER_OR_ID_DRIFT")
        if any(dep not in seen for dep in node.get("depends_on", ())):
            raise RuntimeError("ENGINE_RELEASE_GRAPH_FUTURE_OR_UNKNOWN_DEPENDENCY")
        deps = node.get("depends_on", ())
        if len(deps) != len(set(deps)):
            raise RuntimeError("ENGINE_RELEASE_GRAPH_DUPLICATE_DEPENDENCY")
        stage = dict(base_stages.get(node.get("id"), {}))
        policy = dict(stage.get("policy", {}))
        policy.update(node.get("policy", {}))
        stage.update(node)
        stage["manifest_keys"] = list(node.get("manifest_keys") or [])
        stage["policy"] = policy
        reconstructed.append(stage)
        seen.add(node["id"])
    plan = dict(base_plan)
    plan["stage_count"] = len(stages)
    plan["stages"] = reconstructed
    mainline.validate_plan(plan, require_product_pass_authority=mode == "PRODUCT")
    if mode == "PRODUCT" and mainline.content_sha256(plan) != mainline.content_sha256(base_plan):
        raise RuntimeError("ENGINE_PRODUCT_RELEASE_PLAN_DRIFT")
    return plan


def graph_node_sha256(stage):
    return mainline.content_sha256({
        "id": stage["id"], "depends_on": list(stage.get("depends_on") or []),
        "adapter": stage["adapter"], "manifest_keys": list(stage.get("manifest_keys") or []),
        "policy": {key: bool(stage.get("policy", {}).get(key, False)) for key in
                   ("fail_closed", "cacheable", "output_hash_required", "product_pass_authority")},
    })


def hydrate_stage_inputs(*, plan, ledger, manifest, inputs, read_object, mode):
    stages = mainline._stage_map(plan)
    rows = mainline._ledger_map(ledger)
    supplied = {}
    for item in inputs:
        stage_id = str(item["stage_id"])
        if stage_id in supplied or stage_id not in stages:
            raise RuntimeError("ENGINE_STAGE_INPUT_IDENTITY_INVALID:" + stage_id)
        supplied[stage_id] = item["artifact"]
    for stage_id in mainline.topological_stage_ids(plan):
        if stage_id not in supplied:
            continue
        ref = supplied[stage_id]
        if ref["artifact_type"] != "RealSaS.StageResultManifest" or ref["schema_version"] != "v1":
            raise RuntimeError("ENGINE_STAGE_MANIFEST_TYPE_DRIFT:" + stage_id)
        cached = json.loads(read_object(ref))
        if cached.get("schema") != "RealSaS.StageResultManifest.v1" or cached.get("stage_id") != stage_id or cached.get("expected_semantic_sha256") != ref["semantic_sha256"]:
            raise RuntimeError("ENGINE_STAGE_MANIFEST_IDENTITY_DRIFT:" + stage_id)
        status = cached.get("compiler_status")
        if status != "PASS" and not (mode == "RESEARCH" and status == "PASS_DEMO_ONLY"):
            raise RuntimeError("ENGINE_STAGE_CACHE_QUALIFICATION_DRIFT:" + stage_id)
        stage = stages[stage_id]
        if cached.get("graph_node_sha256") != graph_node_sha256(stage):
            raise RuntimeError("ENGINE_STAGE_CACHE_GRAPH_DRIFT:" + stage_id)
        parameters = cached.get("semantic_parameters") or {}
        if "manifest" not in parameters or parameters["manifest"] != mainline._manifest_subset(manifest,stage):
            raise RuntimeError("ENGINE_STAGE_CACHE_MANIFEST_DRIFT:" + stage_id)
        impl = mainline._adapter_impl_hash(stage["adapter"])
        if cached.get("implementation_sha256") != impl:
            raise RuntimeError("ENGINE_STAGE_CACHE_IMPLEMENTATION_DRIFT:" + stage_id)
        if any(not mainline.dependency_status_admissible(ledger, rows[dep]["status"]) for dep in stage["depends_on"]):
            raise RuntimeError("ENGINE_STAGE_CACHE_DEPENDENCY_MISSING:" + stage_id)
        root = (mainline.authority_root() / "runs" / str(ledger["run_id"]) / "artifacts" / stage_id).resolve()
        outputs = []
        destinations = set()
        for output in cached.get("outputs", ()):
            relative = Path(str(output.get("relative_path") or ""))
            payload_schema = str(output.get("payload_schema") or "")
            if not payload_schema or relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
                raise RuntimeError("ENGINE_CACHE_PORTABLE_OUTPUT_REQUIRED:" + stage_id)
            path = (root / relative).resolve()
            if root not in path.parents or path in destinations:
                raise RuntimeError("ENGINE_CACHE_OUTPUT_PATH_DRIFT:" + stage_id)
            destinations.add(path)
            data = read_object(output)
            # No payload rewriting: mechanical/art/weight identities stay exact.
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    handle.write(data)
                    temporary = handle.name
                os.replace(temporary, path)
            outputs.append({"path": str(path), "sha256": output["content_sha256"],
                            "bytes": output["size_bytes"], "schema": payload_schema,
                            "authority_class": output["authority_class"]})
        if not outputs:
            raise RuntimeError("ENGINE_CACHE_OUTPUTS_REQUIRED:" + stage_id)
        fingerprint, policy = mainline._fingerprint(plan, ledger, manifest, stage, impl)
        if cached.get("policy_sha256") != policy:
            raise RuntimeError("ENGINE_STAGE_CACHE_POLICY_DRIFT:" + stage_id)
        rows[stage_id].update(status=status, outputs=outputs, input_fingerprint=fingerprint,
                              implementation_hash=impl, policy_hash=policy,
                              diagnostics_hash=cached.get("diagnostics_hash") or "", blockers=[],
                              platform_artifact_id=ref["id"], wall_seconds=0.0)
    mainline._refresh(plan, ledger)


def verify_execution_version(plan, request, manifest):
    stage = mainline._stage_map(plan)[str(request["stage_id"])]
    if graph_node_sha256(stage) != request.get("graph_node_sha256"):
        raise RuntimeError("ENGINE_RELEASE_GRAPH_DRIFT")
    if mainline._adapter_impl_hash(stage["adapter"]) != request.get("implementation_sha256"):
        raise RuntimeError("ENGINE_RELEASE_IMPLEMENTATION_DRIFT")
    if mainline.content_sha256(stage["policy"]) != request.get("policy_sha256"):
        raise RuntimeError("ENGINE_RELEASE_POLICY_DRIFT")
    parameters = request.get("semantic_parameters") or {}
    if "manifest" not in parameters:
        raise RuntimeError("ENGINE_RELEASE_MANIFEST_BINDING_REQUIRED")
    if parameters["manifest"] != mainline._manifest_subset(manifest, stage):
        raise RuntimeError("ENGINE_RELEASE_MANIFEST_DRIFT")
