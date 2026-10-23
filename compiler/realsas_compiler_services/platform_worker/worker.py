from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.model_inputs import materialize_model_releases
from compiler.realsas_compiler_services.platform_worker.shape_executor_v1 import execute_shape_capability_v1
from compiler.realsas_compiler_services.platform_worker.stage_inputs import hydrate_stage_inputs, verify_execution_version, released_execution_plan, verify_source_inputs

ENGINE_TASK_QUEUE = "realsas-engine-v1"
EXECUTE_STAGE_ACTIVITY = "engine.execute_compile_stage.v1"
EXECUTE_CAPABILITY_ACTIVITY = "engine.execute_capability.v1"
RENDER_TAIL_ACTIVITY = "engine.render_runtime_tail.v1"


def _artifact_root() -> Path:
    raw = os.environ.get("REALSAS_ARTIFACT_ROOT", "").strip()
    if not raw:
        raise RuntimeError("REALSAS_ARTIFACT_ROOT_REQUIRED")
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cas_key(digest: str) -> str:
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError("CAS_SHA256_INVALID")
    return f"cas/sha256/{digest[:2]}/{digest[2:4]}/{digest}"


def _cas_path(storage_key: str) -> Path:
    root = _artifact_root()
    target = (root / storage_key).resolve()
    if target == root or root not in target.parents:
        raise RuntimeError("CAS_STORAGE_KEY_ESCAPES_ROOT")
    return target


def _put_cas_bytes(data: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    key = _cas_key(digest)
    target = _cas_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = target.read_bytes()
        if hashlib.sha256(current).hexdigest() != digest or len(current) != len(data):
            raise RuntimeError("CAS_EXISTING_OBJECT_DRIFT")
    else:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return {
        "storage_key": key,
        "content_sha256": digest,
        "size_bytes": len(data),
    }


def _put_cas_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    expected = mainline.sha256_file(path)
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise RuntimeError("ENGINE_OUTPUT_HASH_DRIFT")
    return _put_cas_bytes(data)


def _read_cas_object(ref: dict[str, Any]) -> bytes:
    path = _cas_path(str(ref["storage_key"]))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != str(ref["content_sha256"]):
        raise RuntimeError("CAS_OBJECT_CONTENT_DRIFT")
    if len(data) != int(ref["size_bytes"]):
        raise RuntimeError("CAS_OBJECT_SIZE_DRIFT")
    return data


def _ledger_row(ledger: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for row in ledger.get("stages") or ():
        if str(row.get("id")) == stage_id:
            return row
    raise RuntimeError(f"ENGINE_STAGE_NOT_IN_LEDGER:{stage_id}")


def _stage_result_outputs(row: dict[str, Any], *, run_id: str, stage_id: str) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for index, output in enumerate(row.get("outputs") or ()):
        path = Path(str(output["path"])).resolve()
        if not path.is_file():
            raise RuntimeError(f"ENGINE_OUTPUT_MISSING:{path}")
        sealed = _put_cas_file(path)
        if sealed["content_sha256"] != str(output["sha256"]):
            raise RuntimeError(f"ENGINE_OUTPUT_LEDGER_HASH_DRIFT:{path}")
        output_rows.append(
            {
                "relative_path": path.relative_to(mainline.authority_root() / "runs" / run_id / "artifacts" / stage_id).as_posix(),
                "payload_schema": str(output["schema"]),
                "role": f"output:{index:04d}",
                "artifact_type": "RealSaS.EngineOutput",
                "schema_version": "v1",
                "storage_key": sealed["storage_key"],
                "content_sha256": sealed["content_sha256"],
                "size_bytes": sealed["size_bytes"],
                "authority_class": str(output.get("authority_class") or "ENGINE_OUTPUT"),
            }
        )
    return output_rows


def _typed_stage_capability_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project stage payload schemas into typed capability artifacts.

    Direct compile/replay keeps RealSaS.EngineOutput/v1. Only the capability DAG
    receives the typed projection, so downstream product roles can select exact
    IR contracts without file paths, ordinals, or hidden run-ledger lookups.
    """
    typed: list[dict[str, Any]] = []
    for raw in outputs:
        row = dict(raw)
        schema = str(row.get("payload_schema") or "")
        head, separator, version = schema.rpartition(".")
        if (
            separator
            and head.startswith("RealSaS.")
            and version.startswith("v")
            and version[1:].isdigit()
        ):
            row["artifact_type"] = head
            row["schema_version"] = version
        typed.append(row)
    return typed


def _external_wait_result(stage_id: str, required: tuple[str, ...], available: set[str]) -> dict[str, Any]:
    missing = [role for role in required if role not in available]
    diagnostics = {
        "schema": "RealSaS.ExternalEvidenceWait.v1",
        "stage_id": stage_id,
        "required_roles": list(required),
        "available_roles": sorted(available),
        "missing_roles": missing,
        "qualification_minted": False,
    }
    return {
        "stage_id": stage_id,
        "status": "WAITING_EXTERNAL_EVIDENCE",
        "executed_stage_ids": [],
        "outputs": [],
        "diagnostics_hash": hashlib.sha256(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "failure": None,
    }


def _execute_stage_core(request: dict[str, Any]) -> dict[str, Any]:
    stage_id = str(request["stage_id"])
    allowed = tuple(map(str, request.get("allowed_execute_stage_ids") or ()))
    if stage_id not in allowed:
        raise RuntimeError(f"ENGINE_STAGE_OUTSIDE_PLATFORM_SCOPE:{stage_id}")

    run_id = str(request["compiler_run_id"])
    canonical_plan = mainline.load_json(mainline.PLAN_PATH)
    mainline.validate_plan(canonical_plan)
    mode = str(request.get("execution_mode") or "")
    plan = released_execution_plan(canonical_plan, request.get("released_graph"), mode=mode)
    require_product = mode == "PRODUCT"
    plan_sha = mainline.validate_plan(plan, require_product_pass_authority=require_product)
    if plan_sha != str(request["pipeline_plan_sha256"]):
        raise RuntimeError("ENGINE_PLATFORM_PLAN_SHA_DRIFT")

    ledger_path = mainline.run_ledger_path(run_id)
    manifest_path = mainline.run_manifest_path(run_id)
    manifest = mainline.load_json(manifest_path)
    verify_execution_version(plan, request, manifest)
    external = verify_source_inputs(manifest=manifest, inputs=request.get("source_inputs") or (), read_object=_read_cas_object)
    materialize_model_releases(
        manifest=manifest,
        models=request.get("model_releases") or (),
        read_object=_read_cas_object,
    )
    semantic_parameters = dict(request.get("semantic_parameters") or {})
    required_external = tuple(map(str, semantic_parameters.get("external_evidence_roles") or ()))
    optional_external = tuple(map(str, semantic_parameters.get("optional_external_evidence_roles") or ()))
    for roles in (required_external, optional_external):
        if len(roles) != len(set(roles)) or any(not role or role != role.strip() for role in roles):
            raise RuntimeError("ENGINE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
    if set(required_external).intersection(optional_external):
        raise RuntimeError("ENGINE_EXTERNAL_EVIDENCE_ROLE_CONTRACT_INVALID")
    if required_external and any(role not in external for role in required_external):
        return _external_wait_result(stage_id, required_external, set(external))

    if not ledger_path.exists():
        ledger = mainline.build_fresh_run_ledger(
            plan, run_id=run_id, subject_id=str(manifest["subject_id"]),
            manifest_ref=str(manifest_path),
            execution_class=str(manifest.get("execution_class") or "WITNESS"),
            require_product_pass_authority=require_product,
        )
        mainline.atomic_json(ledger_path, ledger)
    ledger = mainline.load_json(ledger_path)
    mainline.validate_ledger(plan, ledger, require_product_pass_authority=require_product)
    mainline._validate_run_manifest_identity(
        manifest,
        run_id=run_id,
        subject_id=str(ledger.get("subject_id") or ""),
    )

    if mode == "PRODUCT" and str(ledger.get("execution_class")) == "DEMO_WITNESS":
        raise RuntimeError("ENGINE_PRODUCT_DEMO_LEDGER_FORBIDDEN")
    hydrate_stage_inputs(plan=plan, ledger=ledger, manifest=manifest,
                         inputs=request.get("input_stages") or (), read_object=_read_cas_object, mode=mode)
    mainline.atomic_json(ledger_path, ledger)
    mainline.validate_ledger(plan, ledger, require_product_pass_authority=require_product)
    stage = mainline._stage_map(plan)[stage_id]
    by_id = mainline._ledger_map(ledger)
    unpassed = [
        str(dep)
        for dep in stage.get("depends_on", ())
        if not mainline.dependency_status_admissible(
            ledger, str(by_id[str(dep)].get("status") or "")
        )
    ]
    if unpassed:
        return {
            "stage_id": stage_id,
            "status": "FAIL",
            "executed_stage_ids": [],
            "outputs": [],
            "diagnostics_hash": "",
            "failure": {
                "code": "ENGINE_RUN_DEPENDENCY_NOT_PASS",
                "class": "ENGINE_LEDGER",
                "reported_owner_stage_id": stage_id,
                "diagnostics": {"unpassed_dependencies": unpassed},
            },
        }

    mainline._run_stage(
        plan=plan,
        ledger=ledger,
        manifest=manifest,
        manifest_path=manifest_path,
        run_id=run_id,
        stage_id=stage_id,
        ledger_path=ledger_path,
    )
    ledger = mainline.load_json(ledger_path)
    row = _ledger_row(ledger, stage_id)
    status = str(row.get("status") or "FAIL")
    if status in mainline.PASS_STATUSES:
        outputs = _stage_result_outputs(row, run_id=run_id, stage_id=stage_id)
        return {
            "stage_id": stage_id,
            "status": status,
            "executed_stage_ids": [stage_id],
            "outputs": outputs,
            "diagnostics_hash": str(row.get("diagnostics_hash") or ""),
            "failure": None,
        }
    return {
        "stage_id": stage_id,
        "status": status,
        "executed_stage_ids": [stage_id],
        "outputs": [],
        "diagnostics_hash": str(row.get("diagnostics_hash") or ""),
        "failure": {
            "code": str((row.get("blockers") or ["COMPILER_STAGE_FAILED"])[0]),
            "class": "COMPILER_STAGE",
            "reported_owner_stage_id": stage_id,
            "diagnostics": {
                "blockers": list(row.get("blockers") or ()),
                "wall_seconds": float(row.get("wall_seconds") or 0.0),
            },
        },
    }


@activity.defn(name=EXECUTE_STAGE_ACTIVITY)
async def execute_compile_stage(request: dict[str, Any]) -> dict[str, Any]:
    return await _with_heartbeat(_execute_stage_core, request)


async def _with_heartbeat(function, request):
    task = asyncio.create_task(asyncio.to_thread(function, request))
    while not task.done():
        activity.heartbeat({"stage_id": request.get("stage_id"), "phase": "running"})
        await asyncio.wait({task}, timeout=10.0)
    return await task


def _resolve_stage_capability_binding(request: dict[str, Any]) -> tuple[str, str]:
    metadata = dict(request.get("capability_metadata") or {})
    params = dict(request.get("goal_parameters") or {})
    stage_id = str(metadata.get("stage_id") or "")
    raw_bridge = params.get("compiler_bridge")
    bridge = dict(raw_bridge) if isinstance(raw_bridge, dict) else {}
    legacy_run_id = str(params.get("compiler_run_id") or "")
    bridge_run_id = str(bridge.get("compiler_run_id") or "")
    if bridge:
        if str(bridge.get("schema") or "") != "RealSaS.CapabilityCompilerBridge.v1":
            raise RuntimeError("STAGE_CAPABILITY_BRIDGE_SCHEMA_DRIFT")
        if legacy_run_id and legacy_run_id != bridge_run_id:
            raise RuntimeError("STAGE_CAPABILITY_RUN_BINDING_DRIFT")
        if str(bridge.get("pipeline_plan_sha256") or "") != str(request.get("pipeline_plan_sha256") or ""):
            raise RuntimeError("STAGE_CAPABILITY_PLAN_BINDING_DRIFT")
    run_id = bridge_run_id or legacy_run_id
    if not stage_id or not run_id:
        raise RuntimeError("STAGE_CAPABILITY_RUN_BINDING_REQUIRED")
    return stage_id, run_id


@activity.defn(name=EXECUTE_CAPABILITY_ACTIVITY)
async def execute_capability(request: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if (
        str(request.get("kind") or "") == "MODULE"
        and metadata.get("executor_contract") == "RealSaS.ProductInputInference.v1"
    ):
        from compiler.realsas_compiler_services.platform_worker.input_executor_v1 import (
            execute_input_capability_v1,
        )

        def run_input(value):
            return execute_input_capability_v1(
                value, read_object=_read_cas_object, put_bytes=_put_cas_bytes
            )
        return await _with_heartbeat(run_input, request)

    if (
        str(request.get("kind") or "") == "MODULE"
        and metadata.get("executor_contract") == "RealSaS.ProductShapeInference.v1"
    ):
        def run_shape_v1(value):
            return execute_shape_capability_v1(
                value, read_object=_read_cas_object, put_bytes=_put_cas_bytes
            )
        return await _with_heartbeat(run_shape_v1, request)

    if (
        str(request.get("kind") or "") == "MODULE"
        and metadata.get("executor_contract") == "RealSaS.ProductShapeInference.v2"
    ):
        from compiler.realsas_compiler_services.platform_worker.shape_executor_v2 import (
            execute_shape_capability_v2,
        )

        def run_shape_v2(value):
            return execute_shape_capability_v2(
                value, read_object=_read_cas_object, put_bytes=_put_cas_bytes
            )
        return await _with_heartbeat(run_shape_v2, request)

    if (
        str(request.get("kind") or "") == "MODULE"
        and metadata.get("executor_contract") == "RealSaS.ProductMeshCompile.v1"
    ):
        from compiler.realsas_compiler_services.platform_worker.mesh_executor_v1 import (
            execute_mesh_capability_v1,
        )

        def run_mesh(value):
            return execute_mesh_capability_v1(
                value, read_object=_read_cas_object, put_bytes=_put_cas_bytes
            )
        return await _with_heartbeat(run_mesh, request)

    if (
        str(request.get("kind") or "") == "MODULE"
        and metadata.get("executor_contract") == "RealSaS.ProductRigInference.v1"
    ):
        from compiler.realsas_compiler_services.platform_worker.rig_executor_v1 import (
            execute_rig_capability_v1,
        )

        def run_rig(value):
            return execute_rig_capability_v1(
                value, read_object=_read_cas_object, put_bytes=_put_cas_bytes
            )
        return await _with_heartbeat(run_rig, request)

    if str(request.get("kind") or "") != "STAGE":
        return {
            "capability_id": str(request.get("capability_id") or ""),
            "status": "FAIL",
            "outputs": [],
            "failure": {
                "code": "CAPABILITY_EXECUTOR_NOT_IMPLEMENTED",
                "class": "CAPABILITY",
                "reported_owner_module_id": str(request.get("owner_module_id") or ""),
                "diagnostics": {"kind": request.get("kind")},
            },
        }
    try:
        stage_id, run_id = _resolve_stage_capability_binding(request)
    except RuntimeError as exc:
        return {
            "capability_id": str(request.get("capability_id") or ""),
            "status": "FAIL",
            "outputs": [],
            "failure": {
                "code": str(exc),
                "class": "CAPABILITY",
                "reported_owner_module_id": str(request.get("owner_module_id") or ""),
                "diagnostics": {"stage_id": str(metadata.get("stage_id") or "")},
            },
        }
    def root_ref(ref):
        role = str(ref.get("role", ""))
        if role.startswith("subject:") or role.startswith("external:"):
            return dict(ref)
        return {**ref, "role": "subject:" + role}
    stage_request = {
        "released_graph": request.get("released_graph"),
        "graph_node_sha256": request.get("stage_graph_node_sha256"),
        "stage_id": stage_id,
        "allowed_execute_stage_ids": [stage_id],
        "compiler_run_id": run_id,
        "pipeline_plan_sha256": request.get("pipeline_plan_sha256"),
        "execution_mode": "RESEARCH",
        "implementation_sha256": request["implementation_sha256"],
        "policy_sha256": request["policy_sha256"],
        "semantic_parameters": request.get("stage_semantic_parameters") or {},
        "input_stages": [{"stage_id": ref["role"][6:], "artifact": ref} for ref in request.get("input_artifacts", ()) if str(ref.get("role", "")).startswith("stage:")],
        "source_inputs": [root_ref(ref) for ref in request.get("input_artifacts", ()) if not str(ref.get("role", "")).startswith("stage:")],
        "model_releases": request.get("model_releases") or [],
    }
    stage_result = await _with_heartbeat(_execute_stage_core, stage_request)
    failure = stage_result.get("failure")
    if failure:
        failure = {
            "code": failure.get("code"),
            "class": failure.get("class"),
            "reported_owner_module_id": str(request.get("owner_module_id") or ""),
            "diagnostics": failure.get("diagnostics") or {},
        }
    return {
        "capability_id": str(request["capability_id"]),
        "status": "PASS" if stage_result["status"] in mainline.PASS_STATUSES else "FAIL",
        "outputs": _typed_stage_capability_outputs(stage_result.get("outputs") or []),
        "diagnostics": {
            "compiler_run_id": run_id,
            "stage_id": stage_id,
            "stage_status": stage_result["status"],
            "stage_diagnostics_hash": str(stage_result.get("diagnostics_hash") or ""),
            "typed_capability_artifact_projection": True,
        },
        "failure": failure,
    }


def _select_motion_clip(motion: dict[str, Any], requested: str) -> dict[str, Any]:
    clips = list(motion.get("clips") or ())
    for clip in clips:
        aliases = {str(clip.get("clip_id") or ""), str(clip.get("intent") or ""), str(clip.get("display_name") or "")}
        if requested in aliases:
            return dict(clip)
    raise RuntimeError(f"RENDER_CLIP_NOT_FOUND:{requested}")


def _render_tail_sync(request: dict[str, Any]) -> dict[str, Any]:
    runtime_ref = dict(request["runtime_package"])
    motion_ref = dict(request["motion"])
    view_spec = dict(request.get("view_spec") or {})
    settings = dict(request.get("render_settings") or {})

    player = Path(os.environ.get("REALSAS_RUNTIME_V2_PLAYER", "")).expanduser().resolve()
    if not player.is_file():
        raise RuntimeError("REALSAS_RUNTIME_V2_PLAYER_REQUIRED")

    runtime_bytes = _read_cas_object(runtime_ref)
    motion_bytes = _read_cas_object(motion_ref)
    motion = json.loads(motion_bytes.decode("utf-8"))

    clip_name = str(settings.get("clip") or "run")
    clip = _select_motion_clip(motion, clip_name)
    clip_id = str(clip.get("clip_id") or clip_name)
    frames_meta = list(clip.get("frames") or ())
    frame_count = int(clip.get("frame_count") or len(frames_meta))
    if frame_count <= 0:
        raise RuntimeError("RENDER_CLIP_HAS_NO_FRAMES")

    view_id = str(view_spec.get("view") or "V0")
    resolution = int(settings.get("resolution") or 512)
    fps = float(settings.get("fps") or 24.0)
    if resolution <= 0 or fps <= 0:
        raise RuntimeError("RENDER_SETTINGS_INVALID")

    with tempfile.TemporaryDirectory(prefix="realsas-render-tail-") as tmp_raw:
        tmp = Path(tmp_raw)
        package = tmp / "product_runtime_v2.rss"
        package.write_bytes(runtime_bytes)
        images: list[Image.Image] = []
        for frame_index in range(frame_count):
            rgba = tmp / f"{frame_index:04d}.rgba"
            provenance = tmp / f"{frame_index:04d}.prov"
            owner = tmp / f"{frame_index:04d}.owner"
            completed = subprocess.run(
                [str(player), str(package), "--clip", clip_id, "--view", view_id,
                 "--frame", str(frame_index), "--out-rgba", str(rgba),
                 "--out-provenance", str(provenance), "--out-owner", str(owner)],
                check=False, capture_output=True, text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError("RUNTIME_RENDER_FAILED:" + completed.stderr.strip() + ":" + completed.stdout.strip())
            raw = rgba.read_bytes()
            expected_bytes = resolution * resolution * 4
            if len(raw) != expected_bytes:
                raise RuntimeError(f"RUNTIME_RENDER_RGBA_SIZE_DRIFT:{len(raw)}!={expected_bytes}")
            images.append(Image.frombytes("RGBA", (resolution, resolution), raw).copy())

        gif_path = tmp / f"{clip_id}__{view_id}.gif"
        duration_ms = max(1, round(1000.0 / fps))
        images[0].save(gif_path, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0, disposal=2)
        sealed = _put_cas_file(gif_path)

    return {
        "status": "PASS",
        "output_path": sealed["storage_key"],
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "diagnostics": {
            "runtime_only": True, "upstream_compile_stage_count": 0,
            "clip_id": clip_id, "view_id": view_id, "frame_count": frame_count,
            "fps": fps, "resolution": resolution,
            "native_player_sha256": mainline.sha256_file(player),
        },
    }


@activity.defn(name=RENDER_TAIL_ACTIVITY)
async def render_runtime_tail(request: dict[str, Any]) -> dict[str, Any]:
    return await _with_heartbeat(_render_tail_sync, request)


async def main() -> None:
    address = os.environ.get("REALSAS_TEMPORAL_ADDRESS", "127.0.0.1:7233")
    namespace = os.environ.get("REALSAS_TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=ENGINE_TASK_QUEUE,
        activities=[execute_compile_stage, execute_capability, render_runtime_tail],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
