"""Materialize Platform-bound immutable stage results for Engine adapters.

The compiler ledger is an execution view. Registry CAS bytes remain immutable;
only paths and run-local fingerprints in that view are reconstructed.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

from compiler.realsas_compiler_services.orchestrator import mainline


SUBJECT_AGNOSTIC_RELEASE_CONTRACT = "RealSaS.SubjectAgnosticStageRelease.v1"


def subject_agnostic_release_parameters(stage, overrides=None):
    """Build release-owned parameters without embedding subject values.

    Manifest values are owned by SubjectInput/CAS. Release parameters may add
    stage-level execution contracts (for example external evidence role names),
    but they may not replace the declared manifest read-set contract.
    """
    extra = dict(overrides or {})
    forbidden = {"contract", "manifest", "manifest_keys"}.intersection(extra)
    if forbidden:
        raise RuntimeError("RELEASE_SUBJECT_AUTHORITY_OVERRIDE_FORBIDDEN:" + ",".join(sorted(forbidden)))
    out = {
        "contract": SUBJECT_AGNOSTIC_RELEASE_CONTRACT,
        "manifest_keys": list(stage.get("manifest_keys") or []),
    }
    out.update(extra)
    for key in ("external_evidence_roles", "optional_external_evidence_roles"):
        roles = out.get(key)
        if roles is None:
            continue
        if not isinstance(roles, (list, tuple)) or not roles:
            raise RuntimeError("RELEASE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
        normalized = [str(role) for role in roles]
        if (len(normalized) != len(set(normalized))
                or any(not role or role != role.strip() for role in normalized)):
            raise RuntimeError("RELEASE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
        out[key] = normalized
    required = set(out.get("external_evidence_roles") or ())
    optional = set(out.get("optional_external_evidence_roles") or ())
    if required.intersection(optional):
        raise RuntimeError("RELEASE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
    return out


def _verify_release_parameters(stage, parameters, manifest):
    """Accept legacy subject-bound releases, prefer subject-agnostic releases."""
    parameters = dict(parameters or {})
    if "manifest" in parameters:
        # Backward compatibility for already-sealed Knight/dev releases only.
        if parameters["manifest"] != mainline._manifest_subset(manifest, stage):
            raise RuntimeError("ENGINE_RELEASE_MANIFEST_DRIFT")
        return
    if parameters.get("contract") != SUBJECT_AGNOSTIC_RELEASE_CONTRACT:
        raise RuntimeError("ENGINE_RELEASE_SUBJECT_AGNOSTIC_CONTRACT_REQUIRED")
    if list(parameters.get("manifest_keys") or []) != list(stage.get("manifest_keys") or []):
        raise RuntimeError("ENGINE_RELEASE_MANIFEST_READ_SET_DRIFT")
    normalized = {}
    for key in ("external_evidence_roles", "optional_external_evidence_roles"):
        roles = parameters.get(key)
        if roles is None:
            normalized[key] = []
            continue
        if not isinstance(roles, list) or not roles:
            raise RuntimeError("ENGINE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
        if (len(roles) != len(set(roles))
                or any(not isinstance(role, str) or not role or role != role.strip() for role in roles)):
            raise RuntimeError("ENGINE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
        normalized[key] = roles
    if set(normalized["external_evidence_roles"]).intersection(normalized["optional_external_evidence_roles"]):
        raise RuntimeError("ENGINE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")


def released_execution_plan(base_plan, released_graph, *, mode="PRODUCT"):
    """Reconstruct execution metadata without losing the full Compiler policy."""
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


def _manifest_file_refs(value):
    if isinstance(value, list):
        for item in value:
            yield from _manifest_file_refs(item)
    elif isinstance(value, dict):
        for key, path in value.items():
            if key == "path":
                digest = value.get("expected_sha256") or value.get("sha256")
            elif key.endswith("_path"):
                digest = value.get(key[:-5] + "_sha256")
            else:
                digest = None
            if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
                yield Path(path).expanduser().resolve(), digest
        for item in value.values():
            yield from _manifest_file_refs(item)


def _materialize_external_input(manifest, role, ref, data):
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        raise RuntimeError("ENGINE_EXTERNAL_INPUT_RUN_ID_REQUIRED")
    clean_role = role.removeprefix("external:")
    if not clean_role or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in clean_role):
        raise RuntimeError("ENGINE_EXTERNAL_INPUT_ROLE_INVALID:" + role)
    root = (mainline.authority_root() / "runs" / run_id / "external_inputs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{clean_role}.{ref['content_sha256']}").resolve()
    if root not in path.parents:
        raise RuntimeError("ENGINE_EXTERNAL_INPUT_PATH_ESCAPE")
    if path.exists():
        current = path.read_bytes()
        if current != data:
            raise RuntimeError("ENGINE_EXTERNAL_INPUT_EXISTING_BYTES_DRIFT:" + role)
    else:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            handle.write(data)
            temporary = handle.name
        os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": str(ref["content_sha256"]),
        "artifact_id": str(ref["id"]),
        "semantic_sha256": str(ref["semantic_sha256"]),
    }


def verify_source_inputs(*, manifest, inputs, read_object):
    """Verify subject bytes and materialize external stage evidence.

    `manifest` is the Engine's in-memory copy. External refs are injected under
    `_platform_external_inputs`; the persisted run-manifest bytes are untouched.
    """
    seen = set()
    external = {}
    for ref in inputs:
        role = str(ref.get("role") or "")
        if role in seen or not (role.startswith("subject:") or role.startswith("external:")):
            raise RuntimeError("ENGINE_SOURCE_INPUT_ROLE_INVALID")
        seen.add(role)
        data = read_object(ref)
        if hashlib.sha256(data).hexdigest() != ref["content_sha256"] or len(data) != ref["size_bytes"]:
            raise RuntimeError("ENGINE_SOURCE_INPUT_CAS_DRIFT:" + role)
        if role.startswith("external:"):
            clean = role.removeprefix("external:")
            external[clean] = _materialize_external_input(manifest, role, ref, data)
            continue
        if role.startswith("subject:manifest:"):
            key = role[len("subject:manifest:"):]
            if key not in manifest or json.loads(data) != manifest[key]:
                raise RuntimeError("ENGINE_SOURCE_MANIFEST_SECTION_DRIFT:" + role)
            refs = list(_manifest_file_refs(manifest[key]))
        else:
            refs = [(path, digest) for path, digest in _manifest_file_refs(manifest)
                    if digest == ref["content_sha256"]]
            if not refs:
                raise RuntimeError("ENGINE_SOURCE_INPUT_NOT_REFERENCED:" + role)
        for path, digest in refs:
            if not path.is_file() or mainline.sha256_file(path) != digest:
                raise RuntimeError("ENGINE_SOURCE_INPUT_FILE_DRIFT:" + role)
    manifest["_platform_external_inputs"] = external
    return external


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
        _verify_release_parameters(stage, cached.get("semantic_parameters") or {}, manifest)
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
    _verify_release_parameters(stage, request.get("semantic_parameters") or {}, manifest)
