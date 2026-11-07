from __future__ import annotations

from tools.corpus.check_corpus370_runner_capacity_v1 import GIB, assess

def report(mem_gib=6,disk_gib=30,swap_total=2,swap_free=2):
    return {
        "memory_bytes":{"MemAvailable":mem_gib*GIB,
                        "SwapTotal":swap_total*GIB,
                        "SwapFree":swap_free*GIB},
        "disk_bytes":{"/":{"free_bytes":disk_gib*GIB}},
        "dmesg":{"accessible":True,"lines":[]},
    }

def test_healthy_host_passes_without_authorizing_truth():
    o=assess(report())
    assert o["status"]=="PASS_CONSERVATIVE_PREFLIGHT_NOT_BENCHMARKED"
    assert o["truth_or_training_authority"] is False

def test_ram_pressure_fails_closed():
    o=assess(report(mem_gib=1))
    assert "AVAILABLE_RAM_BELOW_CONSERVATIVE_BLENDER_GATE" in o["blockers"]

def test_disk_pressure_fails_closed():
    o=assess(report(disk_gib=3))
    assert "DISK_FREE_BELOW_CONSERVATIVE_CORPUS_GATE" in o["blockers"]

def test_unknown_telemetry_never_implies_safety():
    o=assess({"memory_bytes":{},"disk_bytes":{},"dmesg":{}})
    assert "RAM_AVAILABILITY_UNKNOWN" in o["blockers"]
    assert "DISK_AVAILABILITY_UNKNOWN" in o["blockers"]

def test_swap_exhaustion_is_explicit_warning():
    o=assess(report(swap_free=0))
    assert "SWAP_NEARLY_EXHAUSTED" in o["warnings"]

def test_missing_swap_is_explicit_warning():
    o=assess(report(swap_total=0,swap_free=0))
    assert "NO_CONFIGURED_WSL_SWAP" in o["warnings"]

def test_historical_oom_warning_does_not_claim_current_resource_failure():
    r=report()
    r["dmesg"]["lines"]=["Out of memory: Killed process blender"]
    o=assess(r)
    assert "KERNEL_HAS_OOM_OR_CRASH_INDICATORS_REVIEW_BEFORE_WORK" in o["warnings"]
    assert o["status"]=="PASS_CONSERVATIVE_PREFLIGHT_NOT_BENCHMARKED"
