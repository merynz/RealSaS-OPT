#!/usr/bin/env python3
"""Conservative, read-only resource gate before Corpus370 heavy Blender work.

Use before starting a WSL runner to avoid repeating a memory/disk exhaustion
incident. This does NOT establish historical OOM root cause.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus.diagnose_corpus370_wsl_runner_v1 import diagnose

GIB = 1024 ** 3

def assess(report: dict, *, minimum_ram_gib: float = 4.0,
           minimum_disk_gib: float = 12.0) -> dict:
    mem = report.get("memory_bytes") or {}
    root = (report.get("disk_bytes") or {}).get("/") or {}
    available = mem.get("MemAvailable")
    free_disk = root.get("free_bytes")
    blockers = []
    warnings = []
    if available is None:
        blockers.append("RAM_AVAILABILITY_UNKNOWN")
    elif available < minimum_ram_gib * GIB:
        blockers.append("AVAILABLE_RAM_BELOW_CONSERVATIVE_BLENDER_GATE")
    if free_disk is None:
        blockers.append("DISK_AVAILABILITY_UNKNOWN")
    elif free_disk < minimum_disk_gib * GIB:
        blockers.append("DISK_FREE_BELOW_CONSERVATIVE_CORPUS_GATE")
    if (mem.get("SwapTotal") or 0) == 0:
        warnings.append("NO_CONFIGURED_WSL_SWAP")
    elif (mem.get("SwapFree") or 0) < GIB/2:
        warnings.append("SWAP_NEARLY_EXHAUSTED")
    if (report.get("dmesg") or {}).get("accessible") is False:
        warnings.append("PREVIOUS_KERNEL_OOM_NOT_VERIFIABLE_FROM_DMESG")
    if (report.get("dmesg") or {}).get("lines"):
        warnings.append("KERNEL_HAS_OOM_OR_CRASH_INDICATORS_REVIEW_BEFORE_WORK")
    return {
        "schema": "RealSaS.Corpus370RunnerResourceGate.v1",
        "status": ("BLOCKED_INSUFFICIENT_OR_UNKNOWN_RESOURCES" if blockers
                   else "PASS_CONSERVATIVE_PREFLIGHT_NOT_BENCHMARKED"),
        "available_ram_gib": round(available/GIB,2) if available is not None else None,
        "available_root_disk_gib": round(free_disk/GIB,2) if free_disk is not None else None,
        "minimum_ram_gib":minimum_ram_gib,
        "minimum_disk_gib":minimum_disk_gib,
        "blockers":blockers,
        "warnings":warnings,
        "truth_or_training_authority":False,
    }

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path)
    p.add_argument("--minimum-ram-gib",type=float,default=4.0)
    p.add_argument("--minimum-disk-gib",type=float,default=12.0)
    args=p.parse_args()
    if not (args.minimum_ram_gib >= 2.0 and args.minimum_disk_gib >= 5.0):
        p.error("Conservative minimums cannot be lowered below RAM=2 GiB, disk=5 GiB.")
    system=diagnose()
    result=assess(system,minimum_ram_gib=args.minimum_ram_gib,
                  minimum_disk_gib=args.minimum_disk_gib)
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({"resources":result,"diagnostics":system},
                                          sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if not result["blockers"] else 2)

if __name__=="__main__":
    main()
