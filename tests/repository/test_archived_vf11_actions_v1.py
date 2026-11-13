"""Historical GitHub Actions source preserved while main uses bounded RSS CI."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "historical/ci_workflows/vf11_20260921/RELOCATION_MANIFEST_V1.json"
WF = ROOT / ".github/workflows"

def _events_and_push_branches(source: str):
    lines = source.splitlines()
    assert "on:" in lines
    start = lines.index("on:") + 1
    end = next((i for i in range(start, len(lines))
                if lines[i] and not lines[i][0].isspace() and not lines[i].startswith("#")), len(lines))
    block = lines[start:end]
    events = {m.group(1) for line in block
              if (m := re.fullmatch(r"  ([a-z_]+):(?:.*)", line))}
    assert "push" in events and events <= {"push", "workflow_dispatch"}, events
    p = next(i for i, line in enumerate(block) if line.startswith("  push:"))
    end = next((i for i in range(p+1,len(block)) if re.match(r"^  [a-z_]+:",block[i])),len(block))
    rows = block[p+1:end]
    i = next((i for i, line in enumerate(rows) if line.startswith("    branches:")), None)
    assert i is not None
    value = rows[i].split(":",1)[1].strip()
    if value.startswith("["):
        assert value.endswith("]")
        branches = [a.strip().strip("\"'") for a in value[1:-1].split(",") if a.strip()]
    else:
        branches = []
        for line in rows[i+1:]:
            if line.startswith("      - "):
                branches.append(line.split("- ",1)[1].strip().strip("\"'"))
            elif line.strip():
                break
    return events, branches

def test_archived_vf11_workflows_preserve_exact_git_blobs_and_trigger_scope():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "RealSaS.HistoricalWorkflowRelocation.v1"
    assert manifest["archived_count"] == len(manifest["paths"]) == 207
    old = set()
    for row in manifest["paths"]:
        src, dst = row["old_path"], row["new_path"]
        assert src.startswith(".github/workflows/") and src not in old
        old.add(src)
        assert dst == src.replace(".github/workflows/", "historical/ci_workflows/vf11_20260921/", 1)
        assert not (ROOT / src).exists(), src
        data = (ROOT / dst).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + bytes([0]) + data).hexdigest()
        assert actual == row["git_blob_sha1"], dst
        events, branches = _events_and_push_branches(data.decode("utf-8"))
        assert branches == ["research/vf11-certified-adaptive-20260921"], (dst, branches)

def test_active_ci_roles_and_budget_survive_vf11_archival():
    p = json.loads((ROOT / "canonical/CI_EXECUTION_AUTHORITY_V1.json").read_text())["pr_policy"]
    for file in (*p["fast_on_synchronize"], *p["full_on_open_reopen_ready_or_explicit_dispatch"]):
        assert (WF / file).is_file(), file
    remaining = list(WF.glob("*.yml")) + list(WF.glob("*.yaml"))
    assert len(remaining) == 222
    budget = json.loads((ROOT / "canonical/PLATFORM_CI_OPERATIONS_AUTHORITY_V1.json").read_text())["workflow_budget"]
    assert 222 <= budget["max_workflow_files"] <= 230
