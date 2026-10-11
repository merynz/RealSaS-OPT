import json
import os
import subprocess
from pathlib import Path

from tools.architecture_contracts import (
    ADAPTERS,
    ContractScanner,
    compare_maps,
    payload_contracts,
)
from tools.serve_architecture import LiveMap, checkout_token


def source(root, module, text):
    path = root / (module.replace(".", "/") + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_scanner_follows_selected_override_and_called_helper_only(tmp_path):
    source(
        tmp_path,
        ADAPTERS + "old",
        "def adapter(ctx):\n return stage_output_payload(ctx, 'wrong', 'Old')\n",
    )
    source(
        tmp_path,
        ADAPTERS + "new",
        """
def unrelated(ctx):
    return stage_output_payload(ctx, 'wrong', 'Unrelated')
def helper(ctx, stage, schema='Mesh'):
    helper(ctx, stage, schema)
    return stage_output_payload(ctx, stage, schema)
def adapter(ctx):
    def uncalled():
        return stage_output_payload(ctx, 'wrong', 'Nested')
    helper(ctx, 'mesh')
    stage_output_payload(ctx, ctx['dynamic'], 'Unknown')
""",
    )
    source(
        tmp_path,
        ADAPTERS + "dispatch",
        "from .old import *\nfrom .new import adapter\n",
    )
    value = ContractScanner(tmp_path).inspect(ADAPTERS + "dispatch:adapter", {"mesh"})
    assert [(r["producer"], r["schema"]) for r in value["reads"]] == [("mesh", "Mesh")]
    assert value["reads"][0]["location"]["function"] == "helper"
    assert len(value["unresolved_reads"]) == 1


def test_schema_read_distinguishes_ancestor_from_unbound(tmp_path):
    source(
        tmp_path,
        ADAPTERS + "sample",
        "def adapter(ctx):\n stage_output_payload(ctx, 'a', 'Mesh')\n stage_output_payload(ctx, 'missing', 'X')\n",
    )
    plan = {
        "stages": [
            {"id": sid, "depends_on": deps, "adapter": ADAPTERS + "sample:adapter"}
            for sid, deps in [("a", []), ("b", ["a"]), ("c", ["b"])]
        ]
    }
    reads = payload_contracts(plan, tmp_path)["c"]["reads"]
    assert reads[0]["binding"] == "ANCESTOR"
    assert reads[1]["binding"] == "UNBOUND"


def init_repo(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source(root, "tools.sample", "x=1\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )


def test_live_cache_invalidates_same_size_same_mtime_and_new_sources(tmp_path):
    init_repo(tmp_path)
    generated = []

    def generate():
        generated.append(1)
        return {"nodes": [], "edges": [], "source": {}}

    live = LiveMap(tmp_path, generate)
    assert live.read()["status"] == "CURRENT"
    live.read()
    assert len(generated) == 1
    path = tmp_path / "tools/sample.py"
    stat = path.stat()
    path.write_text("x=2\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    live.read()
    assert len(generated) == 2
    token = checkout_token(tmp_path)
    source(tmp_path, "tools.new", "x=3")
    assert checkout_token(tmp_path) != token


def test_live_generation_failure_keeps_last_good_and_retries(tmp_path):
    init_repo(tmp_path)
    good = {"nodes": [], "edges": [], "source": {}}
    live = LiveMap(tmp_path, lambda: good)
    live.read()
    source(tmp_path, "tools.new", "bad")

    def fail():
        raise RuntimeError("invalid graph")

    live.generate = fail
    result = live.read()
    assert result["status"] == "STALE" and result["map"] == good
    assert result["error"] == "invalid graph"
    live.generate = lambda: good
    assert live.read()["status"] == "CURRENT"


def test_map_diff_reports_removed_stage_and_unclassified_edit():
    before = {
        "nodes": [{"id": "seal"}],
        "edges": [],
        "source": {"fingerprinted_files": [{"path": "AGENTS.md", "sha256": "old"}]},
    }
    after = {
        "nodes": [],
        "edges": [],
        "source": {"fingerprinted_files": [{"path": "AGENTS.md", "sha256": "new"}]},
    }
    result = compare_maps(before, after)
    assert result["removed_stages"] == ["seal"]
    assert (
        result["changed_files"][0]["classification"]
        == "UNCLASSIFIED_OR_GLOBAL__INSPECT"
    )
    assert "fingerprinted_files" not in result["after_source"]


def test_http_view_is_loopback_and_only_serves_fixed_routes(tmp_path):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from tools.serve_architecture import handler_for
    import pytest

    init_repo(tmp_path)
    page = tmp_path / "tools/architecture_map.html"
    page.write_text("<script>__ARCHITECTURE_DATA__</script>")
    good = {"nodes": [], "edges": [], "source": {}}
    view = LiveMap(tmp_path, lambda: good)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(view))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urlopen(base + "/").read() == b"<script>null</script>"
        response = urlopen(base + "/api/map")
        assert response.headers["Cache-Control"] == "no-store"
        assert json.load(response)["status"] == "CURRENT"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../../AGENTS.md")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
