from __future__ import annotations

import json, subprocess
from collections import defaultdict, deque
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())
FP=json.loads((ROOT/"canonical/V2_FINGERPRINT_COMPLETENESS_AUDIT_V1_20260928.json").read_text())

stages=[str(s["id"]) for s in PLAN["stages"]]
by={str(s["id"]):s for s in PLAN["stages"]}

# dependency edge orientation: upstream -> consumer
declared=set()
for s in PLAN["stages"]:
    sid=str(s["id"])
    for dep in s.get("depends_on",[]):
        declared.add((str(dep),sid))

actual=set(declared)
hidden=[]
for f in FP.get("findings",[]):
    cls=str(f.get("class") or "")
    stage=str(f.get("stage") or "")
    dep=str(f.get("dependency") or "")
    if not stage or not dep:
        continue
    if cls in {"FINGERPRINT_DEPENDENCY_GAP","DAG_BACK_EDGE","FILESYSTEM_SIDE_CHANNEL"}:
        actual.add((dep,stage))
        if (dep,stage) not in declared:
            hidden.append((dep,stage,cls))

# Add the known direct filesystem repair side-channel if static scanner classification
# used referenced_stage instead of dependency.
for f in FP.get("findings",[]):
    if str(f.get("class") or "") in {"FILESYSTEM_SIDE_CHANNEL_BACK_EDGE"}:
        stage=str(f.get("stage") or "")
        dep=str(f.get("referenced_stage") or "")
        if stage and dep:
            actual.add((dep,stage))
            if (dep,stage) not in declared:
                hidden.append((dep,stage,str(f.get("class"))))

def topo(edges):
    children=defaultdict(list)
    indeg={x:0 for x in stages}
    for a,b in edges:
        if a not in indeg or b not in indeg: continue
        children[a].append(b);indeg[b]+=1
    q=deque(sorted([x for x,d in indeg.items() if d==0],key=lambda x:int(by[x]["ordinal"])))
    order=[]
    while q:
        x=q.popleft();order.append(x)
        for y in children[x]:
            indeg[y]-=1
            if indeg[y]==0:q.append(y)
    cyclic=[x for x,d in indeg.items() if d>0]
    return len(order)==len(stages),order,cyclic

def descendants(source,edges):
    ch=defaultdict(set)
    for a,b in edges: ch[a].add(b)
    seen={source};stack=[source]
    while stack:
        x=stack.pop()
        for y in ch[x]:
            if y not in seen:
                seen.add(y);stack.append(y)
    return sorted(seen,key=lambda x:int(by[x]["ordinal"]))

declared_is_dag,declared_order,declared_cycle=topo(declared)
actual_is_dag,actual_order,actual_cycle=topo(actual)

# Explicitly prove the Stage18<->Stage35 cycle if both directions/path exist.
def path_exists(src,dst,edges):
    ch=defaultdict(set)
    for a,b in edges:ch[a].add(b)
    seen=set();stack=[src]
    while stack:
        x=stack.pop()
        if x==dst:return True
        if x in seen:continue
        seen.add(x);stack.extend(ch[x]-seen)
    return False

stage18="18_CANONICAL_MESH_ADDRESSING_BUILD"
stage35="35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
cycle_witness={
    "declared_path_18_to_35":path_exists(stage18,stage35,declared),
    "actual_hidden_edge_35_to_18":(stage35,stage18) in actual,
}
cycle_witness["cycle_confirmed"]=all(cycle_witness.values())

probes={}
for sid in [
    "14_GSA_BUILD",
    "18_CANONICAL_MESH_ADDRESSING_BUILD",
    "23_COMPLETE_APPEARANCE_ASSET_BAKED",
    "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
    "42_RUNTIME_V2_PROJECTION_BUILT",
]:
    probes[sid]={
      "declared_invalidation_set":descendants(sid,declared),
      "declared_invalidation_count":len(descendants(sid,declared)),
      "actual_augmented_reachability_set":descendants(sid,actual),
      "actual_augmented_reachability_count":len(descendants(sid,actual)),
    }

payload={
  "schema":"RealSaS.V2DAGTruthfulnessAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "declared_graph":{
    "edge_count":len(declared),
    "is_dag":declared_is_dag,
    "topological_order":declared_order,
  },
  "augmented_actual_read_graph":{
    "edge_count":len(actual),
    "hidden_edge_count":len(set((a,b) for a,b,_ in hidden)),
    "hidden_edges":[{"upstream":a,"consumer":b,"class":c} for a,b,c in sorted(set(hidden))],
    "is_dag":actual_is_dag,
    "cyclic_nodes":actual_cycle,
  },
  "stage18_stage35_cycle_witness":cycle_witness,
  "invalidation_probes":probes,
  "verdict":{
    "declared_dag_valid":declared_is_dag,
    "runtime_dataflow_truthfully_represented_by_declared_dag":actual_is_dag and not hidden,
    "incremental_resume_safe_before_dependency_cleanup":False if hidden or not actual_is_dag else True,
    "recommended_architecture":"EPOCHAL_DAG_WITH_EXPLICIT_REPAIR_DIRECTIVE_EDGES",
    "reason":"Keep each compile epoch acyclic; a failed Stage35 emits an immutable RepairDirective consumed by Stage18 of the next epoch. Do not let Stage18 read downstream artifacts from the same epoch.",
  },
  "claim_boundary":"Static+plan audit of declared dependencies and scanner-confirmed direct/side-channel reads. It proves graph truthfulness issues, not the optimal repair operator.",
}
out=ROOT/"canonical"/"V2_DAG_TRUTHFULNESS_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,sort_keys=True))
if not declared_is_dag:
    raise SystemExit("DECLARED_PLAN_NOT_DAG")
if not cycle_witness["cycle_confirmed"]:
    raise SystemExit("EXPECTED_STAGE18_STAGE35_CYCLE_NOT_CONFIRMED")
