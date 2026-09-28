from __future__ import annotations

import json, subprocess
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())

def expected_descendants(source):
    out={source};changed=True
    while changed:
        changed=False
        for stage in PLAN["stages"]:
            sid=stage["id"]
            if sid in out: continue
            if any(dep in out for dep in stage.get("depends_on",[])):
                out.add(sid);changed=True
    return tuple(
        s["id"] for s in sorted(PLAN["stages"],key=lambda x:x["ordinal"])
        if s["id"] in out
    )

ledger=mainline.build_fresh_run_ledger(
    PLAN,
    run_id="AUDIT_INCREMENTAL",
    subject_id="SUBJECT_FREE_INCREMENTAL_AUDIT",
    manifest_ref="/tmp/audit/run_manifest.json",
    execution_class="IMPLEMENTATION_AUDIT",
)
for row in ledger["stages"]:
    row["status"]="PASS"
    row["outputs"]=[{
      "path":"/tmp/fake",
      "sha256":"0"*64,
      "bytes":0,
      "schema":"Audit.Fake.v1",
      "authority_class":"AUDIT_ONLY",
    }]
mainline._refresh(PLAN,ledger)

cases={}
for source in (
  "14_GSA_BUILD",
  "18_CANONICAL_MESH_ADDRESSING_BUILD",
  "32_SKIN_QUALIFIED",
  "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
  "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
):
    clone=json.loads(json.dumps(ledger))
    actual=mainline._invalidate_dependents(PLAN,clone,source,"AUDIT")
    expected=expected_descendants(source)
    cases[source]={
      "expected":expected,
      "actual":actual,
      "matches_exactly":tuple(actual)==tuple(expected),
      "upstream_preserved_count":sum(
        1 for row in clone["stages"]
        if row["id"] not in expected and row["status"]=="PASS"
      ),
      "invalidated_count":len(actual),
    }

payload={
  "schema":"RealSaS.IncrementalInvalidationAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "cases":cases,
  "all_declared_descendant_invalidations_exact":all(c["matches_exactly"] for c in cases.values()),
  "claim_boundary":"Validates the declared DAG invalidation algorithm only. Hidden reads/back-edges/fingerprint gaps must be eliminated before this is sufficient for semantic incremental correctness.",
}
out=ROOT/"canonical"/"V2_INCREMENTAL_INVALIDATION_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["all_declared_descendant_invalidations_exact"]:
    raise SystemExit(2)
