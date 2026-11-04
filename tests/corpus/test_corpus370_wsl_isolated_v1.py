from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.corpus import run_corpus370_wsl_isolated_v1 as isolate


def safe_host():
    gib = 1024**3
    return {
        "memory_bytes": {
            "MemAvailable": int(6.9*gib),
            "MemTotal": int(7.7*gib),
            "SwapTotal": 2*gib,
            "SwapFree": 2*gib,
        },
        "disk_bytes": {"/": {"free_bytes": 676*gib}},
        "dmesg": {"accessible": True, "lines": []},
        "journal": {"accessible": False, "lines": []},
    }


def test_cgroup_limits_and_child_argv_are_exact():
    cmd=isolate.build_command(unit="realsas-c370-native-123456",
        high_mib=3584,max_mib=4608,swap_mib=1024,cpu_quota=180,
        child=["/bin/python3","-u","worker.py","--max-frames","16"])
    assert cmd[:3]==["systemd-run","--user","--scope"]
    assert "--property=MemoryHigh=3584M" in cmd
    assert "--property=MemoryMax=4608M" in cmd
    assert "--property=MemorySwapMax=1024M" in cmd
    assert "--property=CPUQuota=180%" in cmd
    assert cmd[-5:]==["/bin/python3","-u","worker.py","--max-frames","16"]
    assert "bash" not in cmd
    assert "-c" not in cmd


@pytest.mark.parametrize("bad",[
    {"high_mib":4608},
    {"max_mib":6000},
    {"high_mib":900},
    {"swap_mib":3000},
    {"cpu_quota":250},
    {"unit":"../../escape"},
    {"child":[]},
])
def test_unbounded_or_unsafe_scope_configuration_is_rejected(bad):
    kw=dict(unit="realsas-c370-ray-1234",
            high_mib=3584,max_mib=4608,swap_mib=1024,cpu_quota=180,
            child=["/usr/bin/true"])
    kw.update(bad)
    with pytest.raises(ValueError,match="WSL_SCOPE_"):
        isolate.build_command(**kw)


def test_worker_executes_only_inside_bounded_scope(tmp_path,monkeypatch):
    sent=[]
    monkeypatch.setattr(isolate.shutil,"which",lambda name:"/usr/bin/systemd-run")
    monkeypatch.setattr(isolate,"diagnose",lambda **kwargs:safe_host())
    monkeypatch.setattr(isolate.subprocess,"run",
                        lambda args,**kwargs:(sent.append(args) or SimpleNamespace(returncode=0)))
    receipt=tmp_path/"resource_receipt.json"
    code=isolate.execute(unit="realsas-c370-native-1234",high_mib=3584,
        max_mib=4608,swap_mib=1024,cpu_quota=180,
        child=["/usr/bin/true"],minimum_ram_gib=5.,
        minimum_disk_gib=12.,report_path=receipt)
    assert code==0
    assert len(sent)==1
    assert sent[0][0]=="systemd-run"
    assert "--property=MemoryMax=4608M" in sent[0]
    assert json.loads(receipt.read_text())["source_truth_or_training_authorized"] is False


def test_low_available_ram_fails_before_worker(tmp_path,monkeypatch):
    host=safe_host()
    host["memory_bytes"]["MemAvailable"]=1*1024**3
    monkeypatch.setattr(isolate.shutil,"which",lambda name:"/usr/bin/systemd-run")
    monkeypatch.setattr(isolate,"diagnose",lambda **kwargs:host)
    monkeypatch.setattr(isolate.subprocess,"run",
        lambda *args,**kwargs: pytest.fail("Unbounded worker must never run"))
    with pytest.raises(ValueError,match="WSL_SCOPE_RESOURCE_PREFLIGHT_BLOCKED"):
        isolate.execute(unit="realsas-c370-native-1234",high_mib=3584,
          max_mib=4608,swap_mib=1024,cpu_quota=180,
          child=["/usr/bin/true"],minimum_ram_gib=5.,
          minimum_disk_gib=12.,report_path=tmp_path/"resources.json")


def test_systemd_unavailable_never_falls_back_to_unlimited_worker(monkeypatch):
    monkeypatch.setattr(isolate.shutil,"which",lambda name:None)
    monkeypatch.setattr(isolate.subprocess,"run",
                        lambda *a,**k: pytest.fail("Unsafe fallback launched"))
    with pytest.raises(ValueError,match="WSL_SCOPE_SYSTEMD_RUN_UNAVAILABLE"):
        isolate.execute(unit="realsas-c370-native-1234",high_mib=3584,
          max_mib=4608,swap_mib=1024,cpu_quota=180,
          child=["/usr/bin/true"],minimum_ram_gib=5.,
          minimum_disk_gib=12.,report_path=None)


def test_scope_worker_failure_is_propagated(monkeypatch):
    monkeypatch.setattr(isolate.shutil,"which",lambda name:"/usr/bin/systemd-run")
    monkeypatch.setattr(isolate,"diagnose",lambda **kwargs:safe_host())
    monkeypatch.setattr(isolate.subprocess,"run",
                       lambda *a,**k:SimpleNamespace(returncode=137))
    assert isolate.execute(unit="realsas-c370-native-1234",high_mib=3584,
        max_mib=4608,swap_mib=1024,cpu_quota=180,child=["/usr/bin/false"],
        minimum_ram_gib=5.,minimum_disk_gib=12.,report_path=None)==137


@pytest.mark.parametrize("flow",[
    "corpus370-v5-native-temporal-all-v4-real.yml",
    "corpus370-v5-source-locus-geometric-ray-all352-real-v1.yml",
    "corpus370-v5-source-palette-intervention-pilot-real-v1.yml",
])
def test_heavy_workflows_manual_only_and_cgroup_isolated(flow):
    from pathlib import Path
    repo=Path(__file__).resolve().parents[2]
    src=(repo/".github"/"workflows"/flow).read_text()
    top=src.split("\non:",1)[1].split("\npermissions:",1)[0]
    assert "workflow_dispatch:" in top
    assert "\n  push:" not in top
    assert "run_corpus370_wsl_isolated_v1.py" in src
    assert "--max-mib 4608" in src
    assert "--minimum-ram-gib 5" in src
