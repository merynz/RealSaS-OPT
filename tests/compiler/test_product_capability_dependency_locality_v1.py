from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_generic_platform_worker_import_does_not_load_torch():
    script = r'''
import builtins
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
