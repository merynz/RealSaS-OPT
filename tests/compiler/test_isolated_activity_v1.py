import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from compiler.realsas_compiler_services.platform_worker.isolated_activity_v1 import (
    run_isolated, terminate_group, IsolatedActivityError, activity_capacity,
)


def test_concurrent_activities_have_distinct_processes_and_bound_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv('REALSAS_AUTHORITY_ROOT', str(tmp_path))
    beats = []
    async def run():
        return await asyncio.gather(*(run_isolated('probe', {'ordinal': i}, heartbeat=beats.append) for i in range(2)))
    results = asyncio.run(run())
    assert len({r['pid'] for r in results}) == 2
    assert all(r['pid'] == r['process_group'] and r['pid'] != os.getpid() for r in results)
    assert len(beats) >= 2
    for path in tmp_path.glob('engine_activities/*/RESULT.json'):
        receipt = json.loads(path.read_text())
        supervision = json.loads(path.with_name('SUPERVISION.json').read_text())
        assert receipt['ok'] is True
        assert receipt['request_sha256'] == supervision['request_sha256']
        assert supervision['state'] == 'COMPLETED'


def test_child_stage_preflight_failure_is_retained_not_returned_as_pass(tmp_path, monkeypatch):
    pytest.importorskip('temporalio')
    monkeypatch.setenv('REALSAS_AUTHORITY_ROOT', str(tmp_path))
    with pytest.raises(IsolatedActivityError, match='ENGINE_STAGE_OUTSIDE_PLATFORM_SCOPE'):
        asyncio.run(run_isolated('compile_stage', {'stage_id': 'OUTSIDE', 'allowed_execute_stage_ids': []}, heartbeat=lambda _: None))
    result = json.loads(next(tmp_path.glob('engine_activities/*/RESULT.json')).read_text())
    assert result['ok'] is False
    assert result['error_type'] == 'RuntimeError'
    assert json.loads(next(tmp_path.glob('engine_activities/*/SUPERVISION.json')).read_text())['state'] == 'FAILED'


@pytest.mark.parametrize('leader_exits', [False, True])
def test_cancellation_terminates_native_descendants(tmp_path, leader_exits):
    async def run():
        child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        leader_code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); print(p.pid,flush=True); " + ("sys.exit(0)" if leader_exits else "time.sleep(60)")
        process = await asyncio.create_subprocess_exec(sys.executable, '-c', leader_code, child_code,
            start_new_session=True, stdout=asyncio.subprocess.PIPE)
        descendant = int(await asyncio.wait_for(process.stdout.readline(), timeout=5))
        if leader_exits:
            await process.wait()
        await terminate_group(process, grace_seconds=.1)
        return process, descendant
    process, descendant = asyncio.run(run())
    assert process.returncode is not None
    path = Path(f'/proc/{descendant}/stat')
    assert not path.exists() or path.read_text().split()[2] == 'Z'


def test_zero_capacity_cannot_enable_unbounded_native_concurrency(monkeypatch):
    monkeypatch.setenv('REALSAS_ENGINE_MAX_CONCURRENT_ACTIVITIES', '0')
    with pytest.raises(ValueError, match='CAPACITY_OUT_OF_RANGE'):
        activity_capacity()
