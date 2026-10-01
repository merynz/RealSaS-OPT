from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from compiler.realsas_compiler_services.orchestrator.mainline import (
    authority_root,
    run_ledger_path,
    run_manifest_path,
)

ENGINE_TASK_QUEUE = "realsas-engine-v1"
EXECUTE_CAPABILITY_ACTIVITY = "engine.execute_capability.v1"
RUNTIME_RENDER_CAPABILITY = "runtime.render_compiler_run"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolved(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        p = repo_root / p
    return p.resolve()


def _stage(ledger: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for row in ledger.get("stages") or ():
        if str(row.get("id")) == stage_id:
            return dict(row)
    raise RuntimeError(f"PLATFORM_ENGINE_STAGE_MISSING:{stage_id}")


def _stage_output(
    ledger: dict[str, Any],
    stage_id: str,
    *,
    authority_class: str | None = None,
    json_schema: str | None = None,
) -> Path:
    row = _stage(ledger, stage_id)
    if not str(row.get("status", "")).startswith("PASS"):
        raise RuntimeError(f"PLATFORM_ENGINE_STAGE_NOT_PASS:{stage_id}:{row.get('status')}")
    for output in row.get("outputs") or ():
        if authority_class and str(output.get("authority_class")) != authority_class:
            continue
        path = _resolved(str(output.get("path") or ""))
        if not path.is_file():
            continue
        expected = str(output.get("sha256") or "")
        if len(expected) != 64 or _sha256(path) != expected:
            raise RuntimeError(f"PLATFORM_ENGINE_STAGE_OUTPUT_DRIFT:{stage_id}:{path}")
        if json_schema:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("schema_version") or payload.get("schema") or "") != json_schema:
                continue
        return path
    label = authority_class or json_schema or "ANY"
    raise RuntimeError(f"PLATFORM_ENGINE_STAGE_OUTPUT_NOT_FOUND:{stage_id}:{label}")


def _projection_payload(ledger: dict[str, Any]) -> dict[str, Any]:
    row = _stage(ledger, "42_RUNTIME_PROJECTION_AND_CAA_BINDING")
    if not str(row.get("status", "")).startswith("PASS"):
        raise RuntimeError("PLATFORM_ENGINE_STAGE42_NOT_PASS")
    accepted = {
        "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1",
        "RealSaS.RuntimeProjectionIR.v2",
    }
    for output in row.get("outputs") or ():
        path = _resolved(str(output.get("path") or ""))
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        expected = str(output.get("sha256") or "")
        if len(expected) != 64 or _sha256(path) != expected:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema_version") or payload.get("schema") or "")
        if schema in accepted:
            return payload
    raise RuntimeError("PLATFORM_ENGINE_STAGE42_PROJECTION_NOT_FOUND")


def _select(rows: list[dict[str, Any]], key: str, requested: str, fallback: str) -> dict[str, Any]:
    requested = requested.strip()
    if requested:
        exact = [row for row in rows if str(row.get(key)) == requested]
        if len(exact) == 1:
            return exact[0]
        folded = [
            row for row in rows
            if requested.casefold() in str(row.get(key, "")).casefold()
        ]
        if len(folded) == 1:
            return folded[0]
        raise RuntimeError(f"PLATFORM_ENGINE_SELECTOR_AMBIGUOUS_OR_MISSING:{key}:{requested}")
    exact_fallback = [row for row in rows if str(row.get(key)).casefold() == fallback.casefold()]
    if len(exact_fallback) == 1:
        return exact_fallback[0]
    folded = [row for row in rows if fallback.casefold() in str(row.get(key, "")).casefold()]
    if len(folded) == 1:
        return folded[0]
    if not rows:
        raise RuntimeError(f"PLATFORM_ENGINE_SELECTOR_EMPTY:{key}")
    return rows[0]


def _cas_put(data: bytes) -> tuple[str, str, int]:
    root_raw = os.environ.get("REALSAS_ARTIFACT_ROOT", "").strip()
    if not root_raw:
        raise RuntimeError("REALSAS_ARTIFACT_ROOT_REQUIRED")
    digest = hashlib.sha256(data).hexdigest()
    key = f"cas/sha256/{digest[:2]}/{digest[2:4]}/{digest}"
    root = Path(root_raw).expanduser().resolve()
    target = root / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_bytes()
        if hashlib.sha256(existing).hexdigest() != digest or len(existing) != len(data):
            raise RuntimeError("PLATFORM_ENGINE_CAS_COLLISION_OR_CORRUPTION")
    else:
        temp = target.with_name(target.name + ".tmp-" + str(os.getpid()))
        temp.write_bytes(data)
        os.replace(temp, target)
    return key, digest, len(data)


def _render_compiler_run(request: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("operation") != "RENDER_COMPILER_RUN":
        raise RuntimeError("PLATFORM_ENGINE_RUNTIME_OPERATION_DRIFT")
    if metadata.get("fit_train_calibration") is not False:
        raise RuntimeError("PLATFORM_ENGINE_RENDER_FIT_TRAIN_GUARD_DRIFT")

    params = dict(request.get("goal_parameters") or {})
    run_id = str(params.get("compiler_run_id") or "").strip()
    if not run_id:
        raise RuntimeError("PLATFORM_ENGINE_RENDER_COMPILER_RUN_ID_REQUIRED")

    ledger_path = run_ledger_path(run_id)
    manifest_path = run_manifest_path(run_id)
    if not ledger_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"PLATFORM_ENGINE_COMPILER_RUN_NOT_FOUND:{run_id}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(ledger.get("run_id")) != run_id:
        raise RuntimeError("PLATFORM_ENGINE_COMPILER_RUN_LEDGER_IDENTITY_DRIFT")
    manifest_run_id = str(manifest.get("run_id") or "")
    if manifest_run_id and manifest_run_id != run_id:
        raise RuntimeError("PLATFORM_ENGINE_COMPILER_RUN_MANIFEST_IDENTITY_DRIFT")

    package = _stage_output(
        ledger,
        "43_RSS_MATERIALIZE_COMPACT",
        authority_class="RUNTIME_V2_RSS_PACKAGE",
    )
    projection = _projection_payload(ledger)
    clips = [dict(row) for row in projection.get("clips") or ()]
    views = [dict(row) for row in projection.get("views") or ()]
    clip = _select(clips, "clip_id", str(params.get("clip_id") or ""), "run")
    view = _select(views, "view_id", str(params.get("view_id") or ""), "V0")

    runtime_cfg = dict(manifest.get("runtime") or {})
    player_ref = dict(runtime_cfg.get("native_player") or {})
    player = _resolved(str(player_ref.get("path") or ""))
    player_sha = str(player_ref.get("sha256") or "")
    if not player.is_file() or len(player_sha) != 64 or _sha256(player) != player_sha:
        raise RuntimeError("PLATFORM_ENGINE_NATIVE_PLAYER_IDENTITY_DRIFT")

    frame_count = int(clip.get("frame_count") or 0)
    if frame_count <= 0:
        raise RuntimeError("PLATFORM_ENGINE_RENDER_FRAME_COUNT_INVALID")
    resolution = int(dict(view.get("camera") or {}).get("resolution") or 0)
    if resolution <= 0:
        resolution = int(view.get("source_width") or 0)
    if resolution <= 0:
        raise RuntimeError("PLATFORM_ENGINE_RENDER_RESOLUTION_INVALID")

    requested_fps = float(params.get("fps") or 0.0)
    duration_seconds = float(clip.get("duration_seconds") or 0.0)
    fps = requested_fps if requested_fps > 0 else (
        max(1.0, (frame_count - 1) / duration_seconds)
        if duration_seconds > 0 and frame_count > 1
        else 24.0
    )
    gif_duration_ms = max(1, round(1000.0 / fps))

    temp_root = Path(tempfile.mkdtemp(prefix="realsas-platform-render-"))
    try:
        frames: list[Image.Image] = []
        renderer_tokens: set[str] = set()
        for frame_index in range(frame_count):
            stem = temp_root / f"{frame_index:04d}"
            rgba_path = stem.with_suffix(".rgba")
            prov_path = stem.with_suffix(".prov")
            owner_path = stem.with_suffix(".owner")
            completed = subprocess.run(
                [
                    str(player),
                    str(package),
                    "--clip", str(clip["clip_id"]),
                    "--view", str(view["view_id"]),
                    "--frame", str(frame_index),
                    "--out-rgba", str(rgba_path),
                    "--out-provenance", str(prov_path),
                    "--out-owner", str(owner_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "PLATFORM_ENGINE_NATIVE_RENDER_FAILED:"
                    + completed.stderr.strip()[:1200]
                )
            token = next(
                (
                    value
                    for value in (
                        "renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D",
                        "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH",
                    )
                    if value in completed.stdout
                ),
                "",
            )
            if not token:
                raise RuntimeError("PLATFORM_ENGINE_NATIVE_RENDERER_CONTRACT_DRIFT")
            renderer_tokens.add(token)
            raw = rgba_path.read_bytes()
            if len(raw) != resolution * resolution * 4:
                raise RuntimeError("PLATFORM_ENGINE_NATIVE_RGBA_SIZE_DRIFT")
            frame = Image.frombytes("RGBA", (resolution, resolution), raw).copy()
            frames.append(frame)

        gif_path = temp_root / "render.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        gif_bytes = gif_path.read_bytes()
        gif_key, gif_sha, gif_size = _cas_put(gif_bytes)

        report = {
            "schema": "RealSaS.DeveloperRenderReport.v1",
            "execution_id": request.get("execution_id"),
            "compiler_run_id": run_id,
            "source_ledger_sha256": _sha256(ledger_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "package_sha256": _sha256(package),
            "native_player_sha256": player_sha,
            "clip_id": str(clip["clip_id"]),
            "view_id": str(view["view_id"]),
            "frame_count": frame_count,
            "resolution": resolution,
            "fps": fps,
            "renderer_tokens": sorted(renderer_tokens),
            "upstream_stage_executions": 0,
            "fit_train_calibration_executed": False,
            "product_promotion_executed": False,
            "render_content_sha256": gif_sha,
        }
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        report_key, report_sha, report_size = _cas_put(report_bytes)
        return {
            "capability_id": RUNTIME_RENDER_CAPABILITY,
            "status": "PASS",
            "outputs": [
                {
                    "role": "render",
                    "artifact_type": "RealSaS.DeveloperRender",
                    "schema_version": "v1",
                    "storage_key": gif_key,
                    "content_sha256": gif_sha,
                    "size_bytes": gif_size,
                    "authority_class": "DEVELOPER_RENDER_NON_PRODUCT",
                },
                {
                    "role": "report",
                    "artifact_type": "RealSaS.DeveloperRenderReport",
                    "schema_version": "v1",
                    "storage_key": report_key,
                    "content_sha256": report_sha,
                    "size_bytes": report_size,
                    "authority_class": "DEVELOPER_RENDER_EVIDENCE",
                },
            ],
            "diagnostics": report,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


@activity.defn(name=EXECUTE_CAPABILITY_ACTIVITY)
async def execute_capability(request: dict[str, Any]) -> dict[str, Any]:
    requested = str(request.get("requested_executor_activity") or "")
    if requested != EXECUTE_CAPABILITY_ACTIVITY:
        return {
            "capability_id": str(request.get("capability_id") or ""),
            "status": "FAIL",
            "outputs": [],
            "failure": {
                "code": "CAPABILITY_EXECUTOR_ACTIVITY_DRIFT",
                "class": "CONTRACT",
                "reported_owner_module_id": str(request.get("owner_module_id") or ""),
                "diagnostics": {"requested_executor_activity": requested},
            },
        }
    capability_id = str(request.get("capability_id") or "")
    try:
        if capability_id == RUNTIME_RENDER_CAPABILITY:
            return await asyncio.to_thread(_render_compiler_run, request)
        raise RuntimeError(f"PLATFORM_ENGINE_CAPABILITY_UNSUPPORTED:{capability_id}")
    except Exception as exc:
        return {
            "capability_id": capability_id,
            "status": "FAIL",
            "outputs": [],
            "failure": {
                "code": type(exc).__name__.upper(),
                "class": "ENGINE_CAPABILITY",
                "reported_owner_module_id": str(request.get("owner_module_id") or ""),
                "diagnostics": {"message": str(exc)},
            },
        }


async def _main() -> None:
    address = os.environ.get("REALSAS_TEMPORAL_ADDRESS", "127.0.0.1:7233")
    namespace = os.environ.get("REALSAS_TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=ENGINE_TASK_QUEUE,
        activities=[execute_capability],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(_main())
