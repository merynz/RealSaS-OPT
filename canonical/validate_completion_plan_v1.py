#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "canonical" / "CANONICAL_REALSaS_COMPLETION_MATRIX_20260902.json"
ALLOWED = {"KEEP","KEEP_ADAPT","KEEP_QUALIFY","PORT_REBIND","PORT_QUALIFY","PORT_ADAPT","BUILD","BUILD_ADAPT","BUILD_FIX","BUILD_SELECTIVE_PORT","BUILD_PROMOTE_IF_NEEDED","BUILD_AFTER_SOURCE_GATE","FIX","FIX_BLOCKING","FIX_QUALIFY","KEEP_BUILD_BRIDGE","DEFER_CONDITIONAL"}


def main() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["optimizer_authorized"] is False
    assert data["generalization_authorized"] is False
    assert "ZERO_ARCHITECTURE_UNKNOWN" in data["status"]
    rows = {}
    for row in data["rows"]:
        rid = row["id"]
        assert rid not in rows, rid
        assert row["d"] in ALLOWED, (rid, row["d"])
        assert str(row["target"]).strip(), (rid, "target")
        assert str(row["gate"]).strip(), (rid, "gate")
        assert "UNKNOWN" not in row["d"].upper(), rid
        rows[rid] = row
    for row in data["rows"]:
        for dep in row["deps"]:
            assert dep in rows, (row["id"], dep)

    # Dependency graph must be a DAG.
    visiting, done = set(), set()
    def visit(rid: str) -> None:
        if rid in done:
            return
        assert rid not in visiting, f"dependency cycle at {rid}"
        visiting.add(rid)
        for dep in rows[rid]["deps"]:
            visit(dep)
        visiting.remove(rid)
        done.add(rid)
    for rid in rows:
        visit(rid)

    # Known audit blockers cannot silently disappear.
    assert rows["GEP-03"]["d"] == "FIX_BLOCKING"
    assert ">=328" in rows["GEP-03"]["gate"]
    assert rows["AR-05"]["d"] == "FIX_BLOCKING"
    assert rows["SURF-03"]["d"] == "PORT_QUALIFY"
    assert rows["FIT-01"]["deps"] == ["INTEG-02"]
    assert rows["SURF-02"]["deps"] == ["AUTH-01"]
    print(f"PASS completion matrix rows={len(rows)} dependencies=DAG optimizer=blocked")


if __name__ == "__main__":
    main()
