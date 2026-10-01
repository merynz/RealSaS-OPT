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
    desired_revision_spec_sha256: str


@dataclass(frozen=True)
class RenderCommand:
    command_id: str
    subject_id: str
    product_revision_id: str
    render_request_semantic_sha256: str


_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(minutes=1),
    backoff_coefficient=2.0,
    maximum_attempts=5,
)


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
        result = await workflow.execute_activity(
            "execute_compile_plan",
            plan,
            start_to_close_timeout=timedelta(hours=12),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=_ACTIVITY_RETRY,
        )
        return dict(result)


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
