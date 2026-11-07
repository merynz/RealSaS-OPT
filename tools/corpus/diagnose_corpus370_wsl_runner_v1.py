#!/usr/bin/env python3
"""Read-only WSL/self-hosted runner post-mortem.

This diagnostic NEVER deletes source datasets, court packets, native teachers,
checkpoints, WSL caches, logs or package files, and never restarts the runner.
Kernel logs can disappear after WSL restarts: inaccessible evidence is UNKNOWN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

SCHEMA = "RealSaS.Corpus370WSLRunnerDiagnostic.v1"
EVIDENCE = re.compile(
    r"out of memory|oom[-_ ]killer|killed process|memory cgroup out of memory|"
    r"cannot allocate memory|segfault|general protection fault|"
    r"systemd.*killed|kernel panic|cgroup.*oom",
    flags=re.I,
)
ROOTS = (
    "/home/monster/realsas_authority",
    "/home/monster/actions-runner/_work",
    "/home/monster/.cache",
    "/tmp",
)

def read_text(path: str, limit: int = 100_000) -> str | None:
    try:
        return Path(path).read_bytes()[:limit].decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None

def run(args: list[str], timeout: int = 12) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"return_code": p.returncode,
                "stdout": p.stdout[-30_000:], "stderr": p.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"return_code": None, "error": type(exc).__name__}

def kernel_matches(response: dict) -> dict:
    if response.get("return_code") != 0:
        return {"accessible": False, "lines": [], "count": None}
    hits = [line[-500:] for line in response.get("stdout", "").splitlines()
            if EVIDENCE.search(line)]
    return {"accessible": True, "lines": hits[-60:], "count": len(hits)}

def parse_memory(raw: str | None) -> dict:
    result = {}
    for line in (raw or "").splitlines():
        match = re.fullmatch(r"([\w()]+):\s+(\d+) kB", line.strip())
        if match:
            result[match[1]] = int(match[2])*1024
    return {key: result.get(key) for key in
            ("MemTotal","MemAvailable","SwapTotal","SwapFree","Committed_AS")}

def evaluate(memory: dict, disk: dict, cgroup_events: str | None) -> list[str]:
    flags = []
    mem = memory.get("MemAvailable")
    swap = memory.get("SwapFree")
    total = memory.get("MemTotal")
    if mem is None:
        flags.append("RAM_AVAILABLE_UNKNOWN")
    elif total and mem < min(2*1024**3, total//4):
        flags.append("RAM_CURRENTLY_LOW")
    if swap is None:
        flags.append("SWAP_UNKNOWN")
    elif swap == 0:
        flags.append("SWAP_CURRENTLY_EXHAUSTED_OR_UNCONFIGURED")
    root = disk.get("/")
    if root is not None and root["free_bytes"] < 8*1024**3:
        flags.append("ROOT_DISK_CURRENTLY_LOW")
    if cgroup_events is None:
        flags.append("CGROUP_OOM_COUNTER_UNAVAILABLE")
    else:
        match = re.search(r"^oom_kill\s+(\d+)$", cgroup_events, re.M)
        if match and int(match[1]) > 0:
            flags.append("CGROUP_HAS_HISTORICAL_OOM_KILL_COUNTER")
    return flags

def diagnose(*, include_journal: bool = True) -> dict:
    memory = parse_memory(read_text("/proc/meminfo"))
    mounts = {}
    for target in ("/", "/home/monster/realsas_authority", "/tmp"):
        try:
            stat = shutil.disk_usage(target)
            mounts[target] = {"total_bytes":stat.total,
                              "used_bytes":stat.used,"free_bytes":stat.free}
        except OSError:
            mounts[target] = None
    dmesg = kernel_matches(run(["dmesg", "--time-format", "iso"]))
    journal = kernel_matches(run(["journalctl", "-k", "--since", "today",
                                  "--no-pager", "-o", "short-iso"],timeout=20)) if include_journal else {
                                      "accessible": False, "lines":[], "count":None}
    cgroup_events = read_text("/sys/fs/cgroup/memory.events")
    disk = {k:v for k,v in mounts.items() if v is not None}
    return {
        "schema": SCHEMA,
        "status": "POSTMORTEM_CAPTURED__CAUSE_NEEDS_EVIDENCE",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": os.environ.get("RUNNER_NAME"),
        "wsl_detected": "microsoft" in (read_text("/proc/version") or "").lower(),
        "memory_bytes": memory,
        "disk_bytes": mounts,
        "cgroup_events": cgroup_events,
        "cgroup_memory_max": read_text("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_current": read_text("/sys/fs/cgroup/memory.current"),
        "dmesg": dmesg, "journal": journal,
        "uptime": read_text("/proc/uptime"),
        "loadavg": read_text("/proc/loadavg"),
        "diagnostic_flags": evaluate(memory,disk,cgroup_events),
        "target_paths_present": {p:Path(p).exists() for p in ROOTS},
        "read_only": True,
        "training_authorized": False,
        "notes": ("No OOM line after WSL restart DOES NOT rule out previous OOM. "
                  "Windows Reliability Monitor / Event Viewer may retain host crash evidence."),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--skip-journal",action="store_true")
    a=p.parse_args()
    r=diagnose(include_journal=not a.skip_journal)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":r["status"],"wsl_detected":r["wsl_detected"],
                      "memory_bytes":r["memory_bytes"],
                      "disk_bytes":r["disk_bytes"],
                      "diagnostic_flags":r["diagnostic_flags"],
                      "kernel_oom":r["dmesg"]["lines"][-8:],
                      "journal_oom":r["journal"]["lines"][-8:]},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
