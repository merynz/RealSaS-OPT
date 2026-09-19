from __future__ import annotations

from pathlib import Path

import compiler.realsas_compiler_services.orchestrator.adapters.runtime_native_v1 as native


def test_batch_transport_executes_one_multi_node_call(monkeypatch,tmp_path):
    calls=[]
    sentinel=object()
    monkeypatch.setattr(native,"_cache",lambda ctx,archive,player: sentinel)

    def fake_render_many(cache,requests):
        calls.append((cache,tuple(requests)))
        return [
            {"render_cache_hit":False,"native_batch_seconds":0.25},
            {"render_cache_hit":False,"native_batch_seconds":0.25},
            {"render_cache_hit":True},
        ]

    monkeypatch.setattr(native,"render_many_v4",fake_render_many)
    rows,perf=native._batch_render(
        {"authority_root":tmp_path},
        archive=tmp_path/"x.rss",
        player=tmp_path/"player",
        requests=(
            {"clip":"run","view":"V0","time_seconds":0.0,"out_png":tmp_path/"0.png"},
            {"clip":"run","view":"V1","time_seconds":0.0,"out_png":tmp_path/"1.png"},
            {"clip":"run","view":"V2","time_seconds":0.0,"out_png":tmp_path/"2.png"},
        ),
    )

    assert len(calls)==1
    assert calls[0][0] is sentinel
    assert len(calls[0][1])==3
    assert len(rows)==3
    assert perf["cache_hit_count"]==1
    assert perf["cache_miss_count"]==2
    assert perf["native_batch_process_count"]==1
    assert perf["native_batch_seconds"]==0.25
    assert perf["scalar_subprocess_loop_used"] is False


def test_all_warm_hits_require_no_native_batch_process(monkeypatch,tmp_path):
    monkeypatch.setattr(native,"_cache",lambda ctx,archive,player: object())
    monkeypatch.setattr(
        native,
        "render_many_v4",
        lambda cache,requests:[
            {"render_cache_hit":True},
            {"render_cache_hit":True},
        ],
    )
    _rows,perf=native._batch_render(
        {"authority_root":tmp_path},
        archive=Path(tmp_path/"x.rss"),
        player=Path(tmp_path/"player"),
        requests=({"x":1},{"x":2}),
    )
    assert perf["cache_hit_count"]==2
    assert perf["cache_miss_count"]==0
    assert perf["native_batch_process_count"]==0
    assert perf["native_batch_seconds"]==0.0
    assert perf["scalar_subprocess_loop_used"] is False
