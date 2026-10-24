from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_generic_platform_worker_import_does_not_load_torch_and_projects_typed_stage_artifacts():
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
from compiler.realsas_compiler_services.platform_worker import rig_executor_v1
from compiler.realsas_compiler_services.platform_worker import skin_executor_v1
assert worker.EXECUTE_CAPABILITY_ACTIVITY == "engine.execute_capability.v1"
assert rig_executor_v1.RIG_EXECUTOR_CONTRACT_V1 == "RealSaS.ProductRigInference.v1"
assert skin_executor_v1.SKIN_EXECUTOR_CONTRACT_V1 == "RealSaS.ProductSkinInference.v1"
assert not any(name == "models.axis" or name.startswith("models.axis.") for name in sys.modules)
assert not any(name == "models.mira" or name.startswith("models.mira.") for name in sys.modules)
assert "torch" not in sys.modules
rows = worker._typed_stage_capability_outputs([
    {"artifact_type":"RealSaS.EngineOutput", "schema_version":"v1", "payload_schema":"RealSaS.NormalizationDomainIR.v1"},
    {"artifact_type":"RealSaS.EngineOutput", "schema_version":"v1", "payload_schema":"UNVERSIONED_DIAGNOSTIC"},
])
assert rows[0]["artifact_type"] == "RealSaS.NormalizationDomainIR"
assert rows[0]["schema_version"] == "v1"
assert rows[1]["artifact_type"] == "RealSaS.EngineOutput"
assert rows[1]["schema_version"] == "v1"

request = {
    "capability_metadata": {"stage_id": "17_MECHANICAL_PARTITION_QUALIFIED"},
    "pipeline_plan_sha256": "a" * 64,
    "goal_parameters": {
        "compiler_bridge": {
            "schema": "RealSaS.CapabilityCompilerBridge.v1",
            "compiler_run_id": "run-canonical",
            "pipeline_plan_sha256": "a" * 64,
        }
    },
}
assert worker._resolve_stage_capability_binding(request) == (
    "17_MECHANICAL_PARTITION_QUALIFIED", "run-canonical"
)

legacy = {
    "capability_metadata": {"stage_id": "05_CAMERA_CANONICALIZATION"},
    "pipeline_plan_sha256": "b" * 64,
    "goal_parameters": {"compiler_run_id": "legacy-run"},
}
assert worker._resolve_stage_capability_binding(legacy) == (
    "05_CAMERA_CANONICALIZATION", "legacy-run"
)

mismatch = {
    **request,
    "goal_parameters": {
        **request["goal_parameters"],
        "compiler_run_id": "other-run",
    },
}
try:
    worker._resolve_stage_capability_binding(mismatch)
except RuntimeError as exc:
    assert str(exc) == "STAGE_CAPABILITY_RUN_BINDING_DRIFT"
else:
    raise AssertionError("bridge/top-level run identity drift was accepted")

plan_drift = {
    **request,
    "pipeline_plan_sha256": "c" * 64,
}
try:
    worker._resolve_stage_capability_binding(plan_drift)
except RuntimeError as exc:
    assert str(exc) == "STAGE_CAPABILITY_PLAN_BINDING_DRIFT"
else:
    raise AssertionError("compiler bridge plan drift was accepted")
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
