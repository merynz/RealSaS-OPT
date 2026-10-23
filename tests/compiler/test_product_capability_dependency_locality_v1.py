from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_generic_platform_worker_import_does_not_load_torch():
    # This regression measures only scientific dependency locality. Some broad
    # compiler banks intentionally do not install the operational Temporal SDK,
    # so provide the smallest import-time facade worker.py needs and then fail
    # closed on any attempted Torch import. We are not testing Temporal behavior
    # here; dedicated Platform worker banks own that contract.
    script = r'''
import builtins
import sys
import types

activity = types.ModuleType("temporalio.activity")
def defn(*args, **kwargs):
    return lambda fn: fn
activity.defn = defn
activity.heartbeat = lambda *args, **kwargs: None

client = types.ModuleType("temporalio.client")
client.Client = type("Client", (), {})
worker_module = types.ModuleType("temporalio.worker")
worker_module.Worker = type("Worker", (), {})
temporalio = types.ModuleType("temporalio")
temporalio.activity = activity

sys.modules["temporalio"] = temporalio
sys.modules["temporalio.activity"] = activity
sys.modules["temporalio.client"] = client
sys.modules["temporalio.worker"] = worker_module

original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("TORCH_IMPORT_FORBIDDEN_AT_GENERIC_WORKER_IMPORT")
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
from compiler.realsas_compiler_services.platform_worker import worker
assert worker.EXECUTE_CAPABILITY_ACTIVITY == "engine.execute_capability.v1"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
