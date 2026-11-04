from __future__ import annotations

import pytest

from tools.corpus.probe_corpus370_wsl_scope_v1 import HIGH,MAX,SWAP,inspect


def fixture(tmp_path, *, high=HIGH, maximum=MAX, swap=SWAP):
    proc=tmp_path/"proc_cgroup"
    proc.write_text("0::/user.slice/c370.scope\n")
    fs=tmp_path/"sys_cgroup"
    scope=fs/"user.slice"/"c370.scope"
    scope.mkdir(parents=True)
    (scope/"memory.high").write_text(str(high))
    (scope/"memory.max").write_text(str(maximum))
    (scope/"memory.swap.max").write_text(str(swap))
    return proc,fs


def test_cgroup_effective_kernel_limits_are_measured(tmp_path):
    proc,fs=fixture(tmp_path)
    r=inspect(proc_cgroup=proc,fs_root=fs)
    assert r["status"]=="PASS_KERNEL_CGROUP_MEMORY_LIMITS_EFFECTIVE"
    assert r["memory_max_bytes"]==MAX
    assert r["training_authorized"] is False


@pytest.mark.parametrize("field",["memory.max","memory.high","memory.swap.max"])
def test_unbounded_scope_is_a_hard_failure(tmp_path,field):
    proc,fs=fixture(tmp_path)
    (fs/"user.slice"/"c370.scope"/field).write_text("max")
    with pytest.raises(RuntimeError,match="CORPUS370_SCOPE_UNLIMITED"):
        inspect(proc_cgroup=proc,fs_root=fs)


def test_scope_drift_fails_even_if_action_reports_success(tmp_path):
    proc,fs=fixture(tmp_path,maximum=MAX+1024**2)
    with pytest.raises(RuntimeError,match="CORPUS370_SCOPE_LIMIT_DRIFT"):
        inspect(proc_cgroup=proc,fs_root=fs)


def test_cgroup_v1_not_admitted(tmp_path):
    proc,fs=fixture(tmp_path)
    proc.write_text("1:memory:/old.path\n")
    with pytest.raises(RuntimeError,match="CGROUP_V2_REQUIRED"):
        inspect(proc_cgroup=proc,fs_root=fs)
