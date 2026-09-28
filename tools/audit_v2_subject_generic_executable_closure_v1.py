from __future__ import annotations

import ast, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())

SUBJECT_TOKENS=(
    "KNIGHT","SUBJECT2","QUATERNIUS_KNIGHT","DEMO_RUN_V1","DEMO_IDLE_V1",
    "DEMO_SLASH_V1","MAGE_FIT","PROMOTED_MAGE"
)
ALLOWED_GOVERNANCE_PATHS={
    "compiler/realsas_compiler_services/orchestrator/mainline.py",
}
ALLOWED_MAINLINE_LITERALS={
    ".github/workflows/subject2_knight_observation_preflight.yml",
}

def module_path(module:str):
    p=ROOT/Path(*module.split(".")).with_suffix(".py")
    return p if p.is_file() else None

def adapter_files():
    out=set()
    for stage in PLAN["stages"]:
        module,_,_=str(stage["adapter"]).partition(":")
        p=module_path(module)
        if p: out.add(p)
    return out

files=set(adapter_files())
for base in (
    ROOT/"compiler/realsas_compiler_core",
    ROOT/"models/iris",
    ROOT/"models/geppetto",
    ROOT/"models/arachne",
):
    if base.exists():
        files.update(base.rglob("*.py"))

rows=[]
violations=[]
for path in sorted(files):
    rel=str(path.relative_to(ROOT))
    src=path.read_text(encoding="utf-8",errors="replace")
    hits=[]
    for lineno,line in enumerate(src.splitlines(),1):
        upper=line.upper()
        toks=[t for t in SUBJECT_TOKENS if t in upper]
        if not toks:
            continue
        allowed=False
        reason=""
        if rel in ALLOWED_GOVERNANCE_PATHS:
            stripped=line.strip().strip('"').strip("'")
            if any(lit in line for lit in ALLOWED_MAINLINE_LITERALS):
                allowed=True
                reason="IMPLEMENTATION_CLOSURE_GOVERNANCE_REFERENCE_ONLY"
        hit={"line":lineno,"tokens":toks,"text":line.strip()[:500],"allowed":allowed,"reason":reason}
        hits.append(hit)
        if not allowed:
            violations.append({"path":rel,**hit})
    if hits:
        rows.append({"path":rel,"hits":hits})

payload={
  "schema":"RealSaS.SubjectGenericExecutableClosureAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "scanned_file_count":len(files),
  "subject_tokens":SUBJECT_TOKENS,
  "files_with_hits":rows,
  "violations":violations,
  "violation_count":len(violations),
  "finding":{
    "id":"EXECUTABLE_CLOSURE_SUBJECT_GENERICITY",
    "passed":len(violations)==0,
    "claim_boundary":"Scans current 46-stage adapters, compiler core, and promoted IRIS/Geppetto/Arachne Python implementation. Witness manifests/tools/tests/workflows are outside algorithmic closure except the explicitly classified mainline governance reference.",
  },
}
out=ROOT/"canonical"/"V2_SUBJECT_GENERIC_EXECUTABLE_CLOSURE_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,sort_keys=True))
if violations:
    raise SystemExit(2)
