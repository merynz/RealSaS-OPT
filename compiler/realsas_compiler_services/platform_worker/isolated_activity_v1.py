"""Run each Engine activity in its own supervised process and process group.

Scientific adapters use native libraries with process-global state. They must not
share a Python address space merely because Temporal activities are concurrent.
Request/result envelopes and logs remain diagnostic evidence; the original stage
result and all existing version, input and independent-court gates are unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import uuid

SCHEMA = "RealSaS.IsolatedEngineActivity.v1"
OPERATIONS = {"compile_stage", "capability", "render_tail", "probe"}


def write(path, value):
    staged = path.with_suffix('.partial')
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    staged.replace(path)


class IsolatedActivityError(RuntimeError):
    pass


async def terminate_group(process, *, grace_seconds=5):
    if process.returncode is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        pass
    # The leader may exit before an uncooperative Blender/native descendant.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def activity_capacity():
    value = int(os.environ.get('REALSAS_ENGINE_MAX_CONCURRENT_ACTIVITIES', '1'))
    if not 1 <= value <= 64:
        raise ValueError('ENGINE_ACTIVITY_CAPACITY_OUT_OF_RANGE')
    return value


async def run_isolated(operation, request, *, heartbeat, interval_seconds=5):
    if operation not in OPERATIONS:
        raise IsolatedActivityError('ENGINE_ACTIVITY_OPERATION_UNSUPPORTED')
    root = Path(os.environ['REALSAS_AUTHORITY_ROOT']).resolve() / 'engine_activities'
    serialized = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    request_sha = hashlib.sha256(serialized).hexdigest()
    scope = root / (request_sha + '-' + uuid.uuid4().hex)
    scope.mkdir(parents=True, exist_ok=False)
    request_path, result_path = scope / 'REQUEST.json', scope / 'RESULT.json'
    request_path.write_bytes(serialized)
    env = {**os.environ, 'REALSAS_ENGINE_PROCESS_CHILD': '1'}
    process = None
    supervision = {'schema': SCHEMA, 'operation': operation, 'request_sha256': request_sha,
                   'training_authorized': False}
    with (scope / 'PROCESS.log').open('wb') as log:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', __name__, '--operation', operation,
                '--request', str(request_path), '--result', str(result_path),
                stdout=log, stderr=asyncio.subprocess.STDOUT,
                env=env, start_new_session=True)
            supervision.update(pid=process.pid, state='RUNNING')
            write(scope / 'SUPERVISION.json', supervision)
            while process.returncode is None:
                heartbeat({'phase': 'isolated_process', 'pid': process.pid,
                           'stage_id': request.get('stage_id'), 'request_sha256': request_sha})
                try:
                    await asyncio.wait_for(process.wait(), timeout=interval_seconds)
                except asyncio.TimeoutError:
                    pass
            if process.returncode != 0 or not result_path.is_file():
                raise IsolatedActivityError('ENGINE_ACTIVITY_PROCESS_FAILED:' + str(process.returncode))
            envelope = json.loads(result_path.read_text())
            if envelope.get('schema') != SCHEMA or envelope.get('request_sha256') != request_sha:
                raise IsolatedActivityError('ENGINE_ACTIVITY_RESULT_BINDING_DRIFT')
            if envelope.get('ok') is not True:
                raise IsolatedActivityError(str(envelope.get('error_type')) + ':' + str(envelope.get('message')))
            supervision.update(state='COMPLETED', exit_code=process.returncode)
            return envelope['result']
        except BaseException as exc:
            if process is not None:
                await terminate_group(process)
            supervision.update(state='CANCELLED' if isinstance(exc, asyncio.CancelledError) else 'FAILED',
                               error_type=type(exc).__name__, reason=str(exc))
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise IsolatedActivityError(str(exc) + ':receipt=' + str(scope / 'SUPERVISION.json')) from exc
        finally:
            write(scope / 'SUPERVISION.json', supervision)


def execute(operation, request):
    if operation == 'probe':
        return {'pid': os.getpid(), 'process_group': os.getpgrp(), 'parent_pid': os.getppid()}
    from compiler.realsas_compiler_services.platform_worker import worker
    if operation == 'compile_stage':
        return worker._execute_stage_core(request)
    if operation == 'render_tail':
        return worker._render_tail_sync(request)
    return asyncio.run(worker.execute_capability(request))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', choices=sorted(OPERATIONS), required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    raw = args.request.read_bytes()
    result = {'schema': SCHEMA, 'request_sha256': hashlib.sha256(raw).hexdigest(), 'pid': os.getpid()}
    try:
        if os.environ.get('REALSAS_ENGINE_PROCESS_CHILD') != '1':
            raise RuntimeError('ENGINE_ACTIVITY_CHILD_CONTEXT_REQUIRED')
        result.update(ok=True, result=execute(args.operation, json.loads(raw)))
    except Exception as exc:
        result.update(ok=False, error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    write(args.result, result)


if __name__ == '__main__':
    main()
