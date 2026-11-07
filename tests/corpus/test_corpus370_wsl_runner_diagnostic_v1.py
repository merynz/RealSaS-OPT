from __future__ import annotations

from tools.corpus.diagnose_corpus370_wsl_runner_v1 import (
    evaluate, parse_memory, kernel_matches,
)

def test_oom_indicator_is_captured_without_claiming_timeout():
    evidence = kernel_matches({"return_code":0,"stdout":
        "2026-11-04 kernel: Out of memory: Killed process 1234 (blender) total-vm...\n"})
    assert evidence["accessible"]
    assert evidence["count"] == 1
    assert "Killed process" in evidence["lines"][0]

def test_unavailable_kernel_is_unknown_not_oom_absent():
    r = kernel_matches({"return_code":1,"stdout":"permission denied"})
    assert not r["accessible"]
    assert r["count"] is None

def test_linux_meminfo_is_bytes():
    r = parse_memory("MemTotal:       8192000 kB\nMemAvailable:  1024000 kB\nSwapTotal:      2097152 kB\nSwapFree:            0 kB\n")
    assert r["MemTotal"] == 8192000*1024
    assert r["MemAvailable"] == 1024000*1024
    assert r["SwapFree"] == 0

def test_insufficient_ram_swap_and_disk_reported_distinctly():
    flags = evaluate(
        {"MemTotal":8*1024**3,"MemAvailable":100*1024**2,"SwapFree":0},
        {"/":{"free_bytes":2*1024**3}},
        "low 3\noom 4\noom_kill 2\n"
    )
    assert "RAM_CURRENTLY_LOW" in flags
    assert "SWAP_CURRENTLY_EXHAUSTED_OR_UNCONFIGURED" in flags
    assert "ROOT_DISK_CURRENTLY_LOW" in flags
    assert "CGROUP_HAS_HISTORICAL_OOM_KILL_COUNTER" in flags

def test_no_previous_kernel_evidence_does_not_prove_no_oom():
    flags = evaluate({"MemTotal":8*1024**3,"MemAvailable":5*1024**3,
                      "SwapFree":3*1024**3},
                      {"/":{"free_bytes":20*1024**3}},None)
    assert "CGROUP_OOM_COUNTER_UNAVAILABLE" in flags
