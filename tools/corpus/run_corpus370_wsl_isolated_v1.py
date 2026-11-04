#!/usr/bin/env python3
"""Run a source-locked Corpus370 command in its own bounded systemd user scope.

The GitHub Runner.Listener and platform engine remain OUTSIDE this scope.
The cgroup OOM killer can terminate the CORPUS worker without taking the
runner down. MemoryHigh throttles/reclaims; MemoryMax is a hard limit. The
worker's exact source truth/checkpoints are not modified by this wrapper.

Fail CLOSED if cgroups or telemetry are unavailable; never silently fall
back to unrestricted execution. No deletions, sysctl, service mutations or
model/training authority grants are performed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from tools.corpus.check_corpus370_runner_capacity_v1 import assess
from tools.corpus.diagnose_corpus370_wsl_runner_v1 import diagnose


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def build_command(*, unit: str, high_mib: int, max_mib: int,
                  swap_mib: int, cpu_quota: int, child: list[str]) -> list[str]:
    require(bool(re.fullmatch(r"[a-z][a-z0-9-]{3,60}", unit)),
            "WSL_SCOPE_UNSAFE_UNIT_NAME")
    require(1024 <= high_mib < max_mib <= 5120
            and 0 <= swap_mib <= 2048
            and 50 <= cpu_quota <= 200,
            "WSL_SCOPE_RESOURCE_LIMITS_INVALID")
    require(bool(child) and all(bool(x) for x in child),
            "WSL_SCOPE_EMPTY_CHILD_COMMAND")
    return [
        "systemd-run", "--user", "--scope",
        "--unit=" + unit,
        "--property=MemoryHigh=" + str(high_mib) + "M",
        "--property=MemoryMax=" + str(max_mib) + "M",
        "--property=MemorySwapMax=" + str(swap_mib) + "M",
        "--property=CPUQuota=" + str(cpu_quota) + "%",
        "--", *child,
    ]


def execute(*, unit: str, high_mib: int, max_mib: int, swap_mib: int,
            cpu_quota: int, child: list[str], minimum_ram_gib: float,
            minimum_disk_gib: float, report_path: Path | None) -> int:
    command = build_command(unit=unit, high_mib=high_mib, max_mib=max_mib,
                            swap_mib=swap_mib, cpu_quota=cpu_quota, child=child)
    require(shutil.which("systemd-run") is not None,
            "WSL_SCOPE_SYSTEMD_RUN_UNAVAILABLE")
    evidence = diagnose(include_journal=False)
    gate = assess(evidence, minimum_ram_gib=minimum_ram_gib,
                  minimum_disk_gib=minimum_disk_gib)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "scope_command_except_child": command[:-len(child)],
            "gate": gate,
            "host_diagnostics": evidence,
            "source_truth_or_training_authorized": False,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "event":"CORPUS370_WSL_ISOLATION_PREFLIGHT",
        "unit":unit,"gate_status":gate["status"],
        "available_ram_gib":gate["available_ram_gib"],
        "available_root_disk_gib":gate["available_root_disk_gib"],
        "blockers":gate["blockers"],
        "memory_high_mib":high_mib,"memory_max_mib":max_mib,
        "memory_swap_max_mib":swap_mib,"cpu_quota_percent":cpu_quota,
    }, sort_keys=True),flush=True)
    require(not gate["blockers"], "WSL_SCOPE_RESOURCE_PREFLIGHT_BLOCKED")
    # Run the source-locked worker directly; no shell and no fallback.
    # systemd-run --scope waits for process exit and propagates failure.
    result = subprocess.run(command, check=False)
    print(json.dumps({
        "event":"CORPUS370_WSL_ISOLATED_WORKER_EXIT",
        "unit":unit,"exit_code":result.returncode,
        "no_training_authority":True,
    },sort_keys=True),flush=True)
    return int(result.returncode)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--unit",required=True)
    p.add_argument("--high-mib",type=int,default=3584)
    p.add_argument("--max-mib",type=int,default=4608)
    p.add_argument("--swap-mib",type=int,default=1024)
    p.add_argument("--cpu-quota",type=int,default=180)
    p.add_argument("--minimum-ram-gib",type=float,default=5.0)
    p.add_argument("--minimum-disk-gib",type=float,default=12.0)
    p.add_argument("--report-path",type=Path)
    p.add_argument("child",nargs=argparse.REMAINDER)
    a=p.parse_args()
    child=a.child[1:] if a.child and a.child[0]=="--" else a.child
    require(a.minimum_ram_gib>=4.0 and a.minimum_disk_gib>=8.0,
            "WSL_SCOPE_GATE_MINIMUM_CANNOT_BE_WEAKENED")
    status=execute(unit=a.unit,high_mib=a.high_mib,max_mib=a.max_mib,
                   swap_mib=a.swap_mib,cpu_quota=a.cpu_quota,
                   child=child,minimum_ram_gib=a.minimum_ram_gib,
                   minimum_disk_gib=a.minimum_disk_gib,report_path=a.report_path)
    raise SystemExit(status)


if __name__=="__main__":
    main()
