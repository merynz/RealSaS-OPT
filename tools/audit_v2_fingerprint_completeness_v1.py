from __future__ import annotations

import ast, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())
STAGE_RE=re.compile(r"^\d{2}_[A-Z0-9_]+$")


def module_path(name:str):
    p=ROOT/Path(*name.split(".")).with_suffix(".py")
    return p if p.is_file() else None


def fmap(tree):
    return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def closure(fm, entry):
    seen=set(); stack=[entry]
    while stack:
        x=stack.pop()
        if x in seen or x not in fm: continue
        seen.add(x)
        for n in ast.walk(fm[x]):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in fm:
                stack.append(n.func.id)
    return seen


def srcseg(src,n):
    try:return ast.get_source_segment(src,n) or ""
    except:return ""


def string_const(n):
    return n.value if isinstance(n,ast.Constant) and isinstance(n.value,str) else None


def attribute_chain(n):
    parts=[]
    while isinstance(n,ast.Subscript):
        key=string_const(n.slice)
        if key is None:return None
        parts.append(key); n=n.value
    if isinstance(n,ast.Name) and n.id=="ctx":
        return list(reversed(parts))
    return None


def scan_stage(stage):
    module,_,entry=stage["adapter"].partition(":")
    p=module_path(module)
    out={
      "stage_id":stage["id"],"adapter":stage["adapter"],
      "declared_dependencies":stage.get("depends_on",[]),
      "declared_manifest_keys":stage.get("manifest_keys",[]),
      "direct_stage_reads":[],"manifest_reads":[],"undeclared_manifest_reads":[],
      "canonical_repo_reads":[],"run_artifact_stage_refs":[],"env_reads":[],
    }
    if p is None:return out
    src=p.read_text(); tree=ast.parse(src); fm=fmap(tree); cl=closure(fm,entry)
    for fnname in sorted(cl):
        fn=fm[fnname]
        for n in ast.walk(fn):
            if isinstance(n,ast.Call):
                callee=""
                if isinstance(n.func,ast.Name):callee=n.func.id
                elif isinstance(n.func,ast.Attribute):callee=n.func.attr
                if callee in {"stage_output_payload","_stage_output_payload"}:
                    vals=[string_const(a) for a in n.args]
                    sid=next((x for x in vals if x and STAGE_RE.fullmatch(x)),None)
                    if sid:
                        out["direct_stage_reads"].append({"stage_id":sid,"function":fnname,"line":n.lineno})
                if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="os" and n.func.attr in {"getenv"}:
                    out["env_reads"].append({"function":fnname,"line":n.lineno,"source":srcseg(src,n)[:300]})
            if isinstance(n,ast.Subscript):
                ch=attribute_chain(n)
                if ch and len(ch)>=2 and ch[0]=="run_manifest":
                    key=ch[1]
                    out["manifest_reads"].append({"key":key,"function":fnname,"line":n.lineno})
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="get":
                ch=attribute_chain(n.func.value)
                if ch and ch==["run_manifest"] and n.args:
                    key=string_const(n.args[0])
                    if key:
                        out["manifest_reads"].append({"key":key,"function":fnname,"line":n.lineno})

        fs=srcseg(src,fn)
        # Literal canonical policy/contract reads from repo_root are outside run fingerprint
        if "repo_root" in fs and "canonical" in fs:
            literals=re.findall(r'["\']([^"\']+\.(?:json|md|yaml|yml))["\']',fs)
            for lit in literals:
                out["canonical_repo_reads"].append({"path_literal":lit,"function":fnname})
        if "run_root" in fs and "artifacts" in fs:
            for sid in re.findall(r'["\'](\d{2}_[A-Z0-9_]+)["\']',fs):
                out["run_artifact_stage_refs"].append({"stage_id":sid,"function":fnname})

    # dedupe
    for k in ("direct_stage_reads","manifest_reads","canonical_repo_reads","run_artifact_stage_refs","env_reads"):
        uniq=[]; seen=set()
        for row in out[k]:
            key=json.dumps(row,sort_keys=True)
            if key not in seen: seen.add(key); uniq.append(row)
        out[k]=uniq
    declared=set(out["declared_manifest_keys"])
    out["undeclared_manifest_reads"]=[r for r in out["manifest_reads"] if r["key"] not in declared]
    return out


def ancestors(sid,by):
    seen=set(); stack=list(by[sid].get("depends_on",[]))
    while stack:
        x=stack.pop()
        if x in seen:continue
        seen.add(x);stack.extend(by[x].get("depends_on",[]))
    return seen


rows=[scan_stage(s) for s in PLAN["stages"]]
by={s["id"]:s for s in PLAN["stages"]}
ordinal={s["id"]:s["ordinal"] for s in PLAN["stages"]}
findings=[]
for row in rows:
    sid=row["stage_id"]; declared=set(row["declared_dependencies"]); anc=ancestors(sid,by)
    for rr in row["direct_stage_reads"]:
        dep=rr["stage_id"]
        if dep==sid:continue
        if dep not in declared:
            findings.append({
              "id":f"FP_UNDECLARED_DIRECT_STAGE_READ__{sid}__{dep}",
              "severity":"HIGH" if dep in anc else "CRITICAL",
              "class":"FINGERPRINT_DEPENDENCY_GAP",
              "stage":sid,"dependency":dep,"transitive_ancestor":dep in anc,
              "evidence":rr,
            })
        if dep in ordinal and ordinal[dep]>ordinal[sid]:
            findings.append({
              "id":f"FP_BACK_EDGE__{sid}__{dep}","severity":"CRITICAL",
              "class":"DAG_BACK_EDGE","stage":sid,"dependency":dep,"evidence":rr,
            })
    for rr in row["run_artifact_stage_refs"]:
        dep=rr["stage_id"]
        if dep!=sid and dep in ordinal and ordinal[dep]>ordinal[sid]:
            findings.append({
              "id":f"FP_FILESYSTEM_BACK_EDGE__{sid}__{dep}","severity":"CRITICAL",
              "class":"FILESYSTEM_SIDE_CHANNEL","stage":sid,"dependency":dep,"evidence":rr,
            })
    for rr in row["undeclared_manifest_reads"]:
        findings.append({
          "id":f"FP_UNDECLARED_MANIFEST_KEY__{sid}__{rr['key']}",
          "severity":"CRITICAL","class":"MANIFEST_FINGERPRINT_GAP",
          "stage":sid,"manifest_key":rr["key"],"evidence":rr,
        })
    for rr in row["canonical_repo_reads"]:
        findings.append({
          "id":f"FP_REPO_POLICY_READ_REVIEW__{sid}__{rr['path_literal']}",
          "severity":"REVIEW","class":"REPO_FILE_FINGERPRINT_REVIEW",
          "stage":sid,"evidence":rr,
        })
    for rr in row["env_reads"]:
        findings.append({
          "id":f"FP_ENV_READ_REVIEW__{sid}__{rr['line']}",
          "severity":"REVIEW","class":"ENVIRONMENT_FINGERPRINT_REVIEW",
          "stage":sid,"evidence":rr,
        })

# stale architecture/readiness facts
arch=(ROOT/"canonical/SYSTEM_ARCHITECTURE_V2.md").read_text()
if "MESH_WEIGHT_TYPED_SEAM_OPEN" in arch or "exact schema unsealed" in arch:
    findings.append({
      "id":"DOC_SYSTEM_ARCH_V2_MWB_SEAM_STATUS_STALE","severity":"HIGH",
      "class":"CANONICAL_DOCUMENTATION_DRIFT",
      "evidence":"SYSTEM_ARCHITECTURE_V2 still says mesh-weight typed seam open/unsealed while MESH_WEIGHT_BINDING_CONTRACT_V1 and Stage36 typed mesh-skin transfer exist.",
    })

readiness=json.loads((ROOT/"canonical/V2_IMPLEMENTATION_READINESS.json").read_text())
if readiness.get("status")!="READY_FOR_WITNESS_EXECUTION":
    findings.append({
      "id":"GOV_READINESS_REVOKED","severity":"CRITICAL","class":"GOVERNANCE_STATE",
      "evidence":{"status":readiness.get("status"),"seal":readiness.get("readiness_seal")},
    })

report={
 "schema":"RealSaS.V2FingerprintCompletenessAudit.v1",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "stage_rows":rows,
 "findings":findings,
 "counts":{
   "total":len(findings),
   "critical":sum(x["severity"]=="CRITICAL" for x in findings),
   "high":sum(x["severity"]=="HIGH" for x in findings),
   "review":sum(x["severity"]=="REVIEW" for x in findings),
 },
 "claim_boundary":"Static read-set audit; REVIEW findings require semantic confirmation before promotion to defects.",
}
out=ROOT/"canonical"/"V2_FINGERPRINT_COMPLETENESS_AUDIT_V1_20260928.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("FP_AUDIT",json.dumps(report["counts"],sort_keys=True))
for f in findings:
    if f["severity"] in {"CRITICAL","HIGH"}:
        print("FP_FINDING",json.dumps(f,sort_keys=True))
