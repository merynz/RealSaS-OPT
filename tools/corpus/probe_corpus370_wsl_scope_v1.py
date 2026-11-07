#!/usr/bin/env python3
"""Inside-the-scope proof that the kernel is enforcing Corpus370 memory limits."""
from __future__ import annotations

import json
from pathlib import Path

HIGH=3584*1024**2
MAX=4608*1024**2
SWAP=1024*1024**2

def inspect(*, proc_cgroup:Path=Path("/proc/self/cgroup"),
            fs_root:Path=Path("/sys/fs/cgroup"))->dict:
    rows=[x.split("::",1)[1] for x in proc_cgroup.read_text().splitlines()
          if x.startswith("0::")]
    if len(rows)!=1:
        raise RuntimeError("CORPUS370_SCOPE_CGROUP_V2_REQUIRED")
    # cgroup v2 path is an absolute POSIX path, interpreted RELATIVE to root.
    rel=rows[0].lstrip("/")
    if ".." in Path(rel).parts:
        raise RuntimeError("CORPUS370_SCOPE_CGROUP_ESCAPE")
    actual=(fs_root/rel).resolve()
    if not actual.is_relative_to(fs_root.resolve()):
        raise RuntimeError("CORPUS370_SCOPE_CGROUP_ESCAPE")
    def value(field:str)->int:
        raw=(actual/field).read_text().strip()
        if raw=="max":
            raise RuntimeError("CORPUS370_SCOPE_UNLIMITED_"+field.upper().replace(".","_"))
        return int(raw)
    maximum=value("memory.max")
    high=value("memory.high")
    swap=value("memory.swap.max")
    if maximum>MAX or high>HIGH or swap>SWAP:
        raise RuntimeError("CORPUS370_SCOPE_LIMIT_DRIFT")
    if maximum<1024**3:
        raise RuntimeError("CORPUS370_SCOPE_UNEXPECTED_TIGHT_MEMORY_LIMIT")
    return {"status":"PASS_KERNEL_CGROUP_MEMORY_LIMITS_EFFECTIVE",
            "memory_high_bytes":high,"memory_max_bytes":maximum,
            "swap_max_bytes":swap,"memory_peak_bytes":(
                (actual/"memory.peak").read_text().strip()
                if (actual/"memory.peak").is_file() else None),
            "training_authorized":False}

def main():
    result=inspect()
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
