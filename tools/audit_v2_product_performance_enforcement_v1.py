from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"canonical/PRODUCT_COMPILE_PERFORMANCE_CONTRACT_V1_20260927.json"
PLAN=ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json"
MAINLINE=ROOT/"compiler/realsas_compiler_services/orchestrator/mainline.py"
CLOSURE=ROOT/"compiler/realsas_compiler_services/orchestrator/adapters/closure_v2.py"

contract=json.loads(CONTRACT.read_text())
plan=json.loads(PLAN.read_text())
mainline=MAINLINE.read_text()
closure=CLOSURE.read_text()
hard=float(contract["latency_budget_seconds"]["hard_total_compile_max"])

contract_name=CONTRACT.name
runtime_refs=[]
for path in (PLAN,MAINLINE,CLOSURE):
    src=path.read_text()
    if contract_name in src or "hard_total_compile_max" in src:
        runtime_refs.append(str(path.relative_to(ROOT)))

wall_telemetry=(
    "wall_seconds" in mainline
    and "perf_counter()" in mainline
)
stage46=next(s for s in plan["stages"] if s["id"]=="46_PRODUCT_CLOSURE_SEAL")
stage46_policy=json.dumps(stage46.get("policy") or {},sort_keys=True)
stage46_has_perf=(
    "performance" in stage46_policy.lower()
    or "latency" in stage46_policy.lower()
    or "wall" in stage46_policy.lower()
    or "60" in stage46_policy
)
closure_has_perf=(
    "hard_total_compile_max" in closure
    or "compile_latency" in closure
    or "total_compile" in closure
)

payload={
  "schema":"RealSaS.V2ProductPerformanceEnforcementAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "contract":{
    "path":str(CONTRACT.relative_to(ROOT)),
    "hard_total_compile_max_seconds":hard,
    "quality_threshold_relaxation_for_speed_forbidden":bool(
      contract["principles"]["quality_threshold_relaxation_for_speed_forbidden"]
    ),
  },
  "execution":{
    "stage_wall_seconds_telemetry_present":wall_telemetry,
    "current_plan_or_mainline_or_stage46_contract_refs":runtime_refs,
    "stage46_policy_has_performance_gate":stage46_has_perf,
    "closure_adapter_has_total_compile_gate":closure_has_perf,
  },
  "finding":{
    "id":"PRODUCT_COMPILE_60S_CONTRACT_NOT_EXECUTABLE_PRODUCT_GATE",
    "severity":"P1",
    "confirmed":bool(
      wall_telemetry
      and not runtime_refs
      and not stage46_has_perf
      and not closure_has_perf
    ),
    "class":"PERFORMANCE_AUTHORITY_GAP",
    "consequence":"The orchestrator measures per-stage wall time but current 46-stage product closure does not fail or abstain when the end-to-end product compile exceeds the sealed <=60s requirement.",
    "design_before_code":"Freeze timing semantics first: cold versus warm/cache compile, parallel physical-pass wall clock, external training excluded from product inference, hardware class, and percentile requirement. Then make the sealed total-latency budget an explicit product qualification without relaxing any quality gate."
  }
}
out=ROOT/"canonical"/"V2_PRODUCT_PERFORMANCE_ENFORCEMENT_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["finding"]["confirmed"]:
    raise SystemExit(2)
