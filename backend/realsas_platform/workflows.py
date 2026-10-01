from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


@dataclass(frozen=True)
class CompileSubjectCommand:
    command_id: str
    subject_id: str
    attempt_id: str
    engine_release_id: str
    subject_semantic_sha256: str
    target_stage_id: str


@dataclass(frozen=True)
class RenderCommand:
    command_id: str
    subject_id: str
    product_revision_id: str
    render_request_id: str
    render_request_semantic_sha256: str


_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(minutes=1),
    backoff_coefficient=2.0,
    maximum_attempts=5,
)

_STAGE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)


def compile_activity_sequence(plan: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Pure contract used by workflow/tests: one durable activity per stage."""
    rows = []
    for stage in tuple(plan.get("stages") or ()):
        stage_id = str(stage["stage_id"])
        action = str(stage["action"])
        if action == "REUSE":
            rows.append((stage_id, "bind_reused_stage"))
        elif action == "EXECUTE":
            rows.append((stage_id, "execute_compile_stage"))
        else:
            raise ValueError(f"UNKNOWN_RESOLUTION_ACTION:{stage_id}:{action}")
    return tuple(rows)


@workflow.defn
class CompileSubjectWorkflow:
    @workflow.run
    async def run(self, command: CompileSubjectCommand) -> dict[str, Any]:
        plan = await workflow.execute_activity(
            "resolve_compile_plan",
            command,
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_ACTIVITY_RETRY,
        )

        completed: list[dict[str, Any]] = []
        allowed_execute_stage_ids = [
            str(row["stage_id"])
            for row in plan["stages"]
            if str(row["action"]) == "EXECUTE"
        ]
        for stage_id, activity_name in compile_activity_sequence(dict(plan)):
            request = {
                "command_id": command.command_id,
                "subject_id": command.subject_id,
                "attempt_id": command.attempt_id,
                "engine_release_id": command.engine_release_id,
                "stage_id": stage_id,
                "resolution": next(
                    row for row in plan["stages"] if str(row["stage_id"]) == stage_id
                ),
                "allowed_execute_stage_ids": allowed_execute_stage_ids,
            }
            if activity_name == "bind_reused_stage":
                result = await workflow.execute_activity(
                    activity_name,
                    request,
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=_ACTIVITY_RETRY,
                )
            else:
                result = await workflow.execute_activity(
                    activity_name,
                    request,
                    start_to_close_timeout=timedelta(hours=4),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=_STAGE_RETRY,
                )
            completed.append(dict(result))

        return dict(
            await workflow.execute_activity(
                "finalize_compile_attempt",
                {
                    "command": command,
                    "plan": plan,
                    "completed_stages": completed,
                },
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_ACTIVITY_RETRY,
            )
        )


@workflow.defn
class RenderWorkflow:
    @workflow.run
    async def run(self, command: RenderCommand) -> dict[str, Any]:
        resolution = await workflow.execute_activity(
            "resolve_render_request",
            command,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_ACTIVITY_RETRY,
        )
        if bool(resolution.get("cache_hit")):
            return dict(resolution)
        forbidden = tuple(resolution.get("forbidden_stage_ids") or ())
        if forbidden:
            raise RuntimeError("PRODUCT_RENDER_FORBIDDEN_STAGE:" + ",".join(forbidden))
        result = await workflow.execute_activity(
            "execute_render_tail",
            resolution,
            start_to_close_timeout=timedelta(hours=2),
            heartbeat_timeout=timedelta(minutes=1),
            retry_policy=_ACTIVITY_RETRY,
        )
        return dict(result)
