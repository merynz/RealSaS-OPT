#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

CORE_PATH = Path(__file__).with_name("oracle_z_representation_probe_v2.py")


def _load_core():
    spec = importlib.util.spec_from_file_location("oracle_z_representation_probe_v2_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("ORACLE_Z_V2_CORE_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # module body is executing. Register the dynamically loaded module before
    # exec_module; this is wrapper plumbing only and does not alter the sealed
    # scientific core or any court constant.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _github_context() -> dict[str, str]:
    keys = (
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_SHA",
        "GITHUB_REF",
        "RUNNER_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
    )
    return {key: os.environ.get(key, "").strip() for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gh = _github_context()
    required = ("GITHUB_REPOSITORY", "GITHUB_WORKFLOW", "GITHUB_RUN_ID", "GITHUB_SHA")
    missing = [key for key in required if not gh[key]]
    if missing:
        raise RuntimeError("GITHUB_SCIENTIFIC_CONTEXT_REQUIRED:" + ",".join(missing))

    # Compatibility provenance for the sealed core. These values identify this
    # GitHub run; they do not imply Platform registry objects exist.
    os.environ["REALSAS_EXPERIMENT_ID"] = f"github-fast:{gh['GITHUB_REPOSITORY']}:{gh['GITHUB_WORKFLOW']}:{gh['GITHUB_RUN_ID']}"
    os.environ["REALSAS_ATTEMPT_ID"] = f"github-run-attempt:{gh.get('GITHUB_RUN_ATTEMPT') or '1'}"
    os.environ["REALSAS_ENGINE_RELEASE_ID"] = "github-fast-probe-no-engine-release"
    os.environ["REALSAS_CANONICAL_CODE_SHA"] = gh["GITHUB_SHA"]

    core = _load_core()
    result = core.run_scientific()
    legacy = result.pop("platform_context", None)
    result["mode"] = "GITHUB_SCIENTIFIC"
    result["scientific_authority"] = True
    result["execution_authority"] = "SEALED_GITHUB_WORKFLOW_REPORT"
    result["execution_context"] = gh
    result["execution_amendment"] = "ORACLE_Z_V2_FAST_EXECUTION_AMENDMENT_V1.md"
    if legacy is not None:
        result["legacy_core_provenance_compatibility"] = legacy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": result["mode"],
        "status": result["status"],
        "execution_authority": result["execution_authority"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
