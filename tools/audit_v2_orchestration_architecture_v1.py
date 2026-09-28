from __future__ import annotations

import ast, json, subprocess
from collections import defaultdict, deque
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json"
MAINLINE=ROOT/"compiler/realsas_compiler_services/orchestrator/mainline.py"
TX=ROOT/"compiler/realsas_compiler_core/compile_transaction.py"
REPAIR=ROOT/"compiler/realsas_compiler_services/proof/repair_loop.py"
V2ARCH=ROOT/"compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py"

plan=json.loads(PLAN.read_text())
stages=plan["stages"]
by={s["id"]:s for s in stages}
ordinal={s["id"]:s["ordinal"] for s in stages}

# Declared DAG check
indeg={s["id"]:0 for s in stages}
children=defaultdict(list)
for s in stages:
    for d in s.get("depends_on",[]):
        indeg[s["id"]]+=1
        children[d].append(s["id"])
q=deque([sid for sid,d in indeg.items() if d==0])
order=[]
while q:
    x=q.popleft(); order.append(x)
    for y in children[x]:
        indeg[y]-=1
        if indeg[y]==0:q.append(y)
declared_is_dag=len(order)==len(stages)

mainline=MAINLINE.read_text()
tx=TX.read_text()
repair=REPAIR.read_text()
v2arch=V2ARCH.read_text()

evidence={
  "declared_46_stage_plan_is_dag":declared_is_dag,
  "mainline_descendant_invalidation_present":"def _invalidate_dependents" in mainline,
  "mainline_pass_fingerprint_revalidation_present":"def _verify_existing_passes" in mainline and "_fingerprint(" in mainline,
  "transaction_immutable_attempt_identity_present":"attempt_id: str" in tx and "parent_transaction_hash" in tx,
  "transaction_fork_present":"def fork_compile_transaction" in tx,
  "transaction_slot_rebind_forbidden":"COMPILE_ARTIFACT_SLOT_REBIND_FORBIDDEN" in tx,
  "repair_parent_child_state_hash_present":"parent_product_state_hash" in repair and "child_product_state_hash" in repair,
  "repair_same_probe_reproof_present":"reproof" in repair.lower(),
  "stage18_reads_stage35_repair_artifacts":(
      '"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"' in v2arch
      and "repair_candidate_path" in v2arch
  ),
}

# Compute declared descendants for representative invalidation sources.
def descendants(source):
    out={source}; changed=True
    while changed:
        changed=False
        for s in stages:
            if s["id"] in out: continue
            if any(d in out for d in s.get("depends_on",[])):
                out.add(s["id"]); changed=True
    return [s["id"] for s in sorted(stages,key=lambda z:z["ordinal"]) if s["id"] in out]

representative={
  sid:descendants(sid)
  for sid in (
    "14_GSA_BUILD",
    "18_CANONICAL_MESH_ADDRESSING_BUILD",
    "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
    "42_RUNTIME_V4_PROJECTION_BUILD",
  )
  if sid in by
}

# Pure DAG is exact only if no downstream->upstream reads exist.
pure_dag_exact=declared_is_dag and not evidence["stage18_reads_stage35_repair_artifacts"]

recommendation={
 "architecture_id":"TRANSACTIONAL_VERSIONED_DAG_ATTEMPTS_V1",
 "shape":{
   "outer":"COMPILE_TRANSACTION_STATE_MACHINE",
   "inner":"IMMUTABLE_ACYCLIC_STAGE_DAG_PER_ATTEMPT",
   "repair":"FAIL/ABSTAIN -> typed RepairDirective -> fork child attempt",
   "carry_forward":"unchanged exact artifact bindings may be carried by hash into child attempt",
   "mutation":"in-place upstream mutation and downstream filesystem back-edges forbidden",
 },
 "why_not_single_static_dag":[
   "Stage35 qualification can legitimately request a Stage18-owned topology change, which is feedback.",
   "Encoding that feedback as Stage18 reading Stage35 files creates a hidden cycle and invalidates DAG fingerprints.",
 ],
 "why_not_general_mutable_state_machine":[
   "Would weaken exact dependency identities, parallelism, cacheability, and local invalidation.",
   "Current mainline DAG already has useful descendant invalidation and per-stage fingerprints.",
 ],
 "why_hybrid_is_preferred":[
   "Preserves DAG parallelism and minimal recompute inside each immutable attempt.",
   "Represents repair feedback explicitly as parent->child transaction lineage rather than a cycle.",
   "Matches existing CompileTransactionIR fork semantics and bounded repair parent/child hashes.",
 ],
 "incremental_expectation":[
   "Within an attempt: changed stage identity invalidates only declared descendants.",
   "Across repair attempts: carry only artifacts proven unaffected by the directive owner/scope; re-run repaired owner and its true descendants.",
   "Full Stage01-46 rerun is required only when a changed authority is an ancestor of all later stages or when input/policy/compiler semantic version changes globally.",
 ],
}

payload={
 "schema":"RealSaS.OrchestrationArchitectureAudit.v1",
 "status":"AUDIT_DECISION__NO_IMPLEMENTATION_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "declared_graph":{
   "stage_count":len(stages),
   "is_dag":declared_is_dag,
 },
 "evidence":evidence,
 "representative_declared_invalidation_sets":representative,
 "pure_static_dag_exact_for_current_system":pure_dag_exact,
 "recommendation":recommendation,
 "implementation_authorized":False,
 "claim_boundary":"Architecture recommendation only. Hidden read-set/fingerprint gaps and all semantic audit findings must be closed before relying on minimal incremental rerun.",
}
out=ROOT/"canonical"/"V2_ORCHESTRATION_ARCHITECTURE_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
