from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.orchestrator import mainline

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())


def fake_ledger(plan_hash: str):
    rows=[]
    for stage in PLAN["stages"]:
        rows.append({
            "id":stage["id"],
            "outputs":[{
                "sha256":content_sha256({"stage":stage["id"]}),
                "bytes":1,
                "schema":"Audit.Output.v1",
                "authority_class":"AUDIT_ONLY",
            }],
        })
    return {
        "run_id":"AUDIT_RUN_A",
        "pipeline_plan_sha256":plan_hash,
        "stages":rows,
    }


def fake_manifest():
    # Manifest subset only needs declared keys to exist; None is a legal audit
    # placeholder for fingerprint sensitivity testing.
    keys={k for stage in PLAN["stages"] for k in stage.get("manifest_keys",[])}
    return {k:None for k in keys}


plan_hash_a=content_sha256(PLAN)
plan_b=copy.deepcopy(PLAN)
# Semantically irrelevant to Stage01 itself: add a downstream dependency edge
# already implied transitively, only to demonstrate global plan coupling.
stage46=next(s for s in plan_b["stages"] if s["id"]=="46_PRODUCT_CLOSURE_SEAL")
deps=list(stage46["depends_on"])
if "42_RUNTIME_PROJECTION_AND_CAA_BINDING" not in deps:
    deps.append("42_RUNTIME_PROJECTION_AND_CAA_BINDING")
else:
    stage46["policy"]=dict(stage46.get("policy") or {})
    stage46["policy"]["audit_downstream_only_marker"]="NON_PRODUCT_AUDIT"
stage46["depends_on"]=deps
plan_hash_b=content_sha256(plan_b)

manifest=fake_manifest()
stage01=next(s for s in PLAN["stages"] if s["id"]=="01_SOURCE_BYTES_SEALED")
impl01=mainline._adapter_impl_hash(stage01["adapter"])
ledger_a=fake_ledger(plan_hash_a)
ledger_b=fake_ledger(plan_hash_b)
fp_a,_=mainline._fingerprint(PLAN,ledger_a,manifest,stage01,impl01)
fp_b,_=mainline._fingerprint(PLAN,ledger_b,manifest,stage01,impl01)

ledger_run_b=copy.deepcopy(ledger_a)
ledger_run_b["run_id"]="AUDIT_RUN_B"
fp_run_b,_=mainline._fingerprint(PLAN,ledger_run_b,manifest,stage01,impl01)

module_name=stage01["adapter"].partition(":")[0]
closure=mainline._local_import_closure(module_name)

payload={
  "schema":"RealSaS.V2IncrementalFingerprintPortabilityAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "stage01":{
    "adapter":stage01["adapter"],
    "implementation_import_closure_module_count":len(closure),
    "implementation_import_closure_modules":[name for name,_ in closure],
    "fingerprint_original":fp_a,
    "fingerprint_after_downstream_only_plan_change":fp_b,
    "fingerprint_after_run_id_change_only":fp_run_b,
  },
  "findings":[
    {
      "id":"GLOBAL_PIPELINE_PLAN_HASH_FORCES_UNRELATED_STAGE_FINGERPRINT_DRIFT",
      "severity":"P0",
      "confirmed":fp_a!=fp_b,
      "class":"INCREMENTAL_REUSE_GRANULARITY_GAP",
      "consequence":"A plan edit that does not change Stage01 semantics still changes Stage01 input identity; current resume verification can therefore invalidate from the front of the pipeline after architecture/DAG edits.",
      "design_before_code":"In transactional versioned DAG attempts, bind each stage to its stage-local semantic contract/dependency schema and transaction lineage. Keep a global plan seal for governance, but do not make unrelated plan edits poison every reusable artifact."
    },
    {
      "id":"RUN_ID_IS_PART_OF_STAGE_INPUT_FINGERPRINT",
      "severity":"P1",
      "confirmed":fp_a!=fp_run_b,
      "class":"CROSS_ATTEMPT_REUSE_GAP",
      "consequence":"Identical semantic inputs in a different run/attempt cannot reuse the same stage fingerprint without an explicit carry-forward/import mechanism.",
      "design_before_code":"Define transaction-child carry-forward by exact artifact hash and semantic contract, distinct from witness/run identity. Provenance must retain both original producer and child transaction binding."
    },
    {
      "id":"ADAPTER_IMPLEMENTATION_HASH_USES_FULL_LOCAL_IMPORT_CLOSURE",
      "severity":"P2",
      "confirmed":len(closure)>1,
      "class":"CONSERVATIVE_IMPLEMENTATION_INVALIDATION",
      "consequence":"A change anywhere in a shared local import closure can invalidate a stage even when its actually executed semantic slice is unchanged.",
      "design_before_code":"Evaluate function-level/static call-closure hashing or explicit semantic implementation versions. Preserve conservative correctness; optimize only if broad invalidation materially harms the <=60s contract."
    }
  ],
  "claim_boundary":"This audit measures fingerprint portability, not declared descendant invalidation. The latter is separately proven exact by V2_INCREMENTAL_INVALIDATION_AUDIT_V1.",
}
out=ROOT/"canonical"/"V2_INCREMENTAL_FINGERPRINT_PORTABILITY_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not all(row["confirmed"] for row in payload["findings"]):
    raise SystemExit(2)
