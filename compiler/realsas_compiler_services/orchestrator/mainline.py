from __future__ import annotations
import argparse, ast, hashlib, importlib, importlib.util, json, os, re
from time import perf_counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[3]
PLAN_PATH=ROOT/"canonical"/"MAINLINE_EXECUTION_PLAN_V1.json"
LEDGER_PATH=ROOT/"canonical"/"ACTIVE_RUN_V1.json"
PASS_STATUSES={"PASS","CACHE_HIT"}
FAIL_STATUSES={"FAIL","ABSTAIN","BLOCKED"}
_STAGE_RE=re.compile(r"^\d{2}_[A-Z0-9_]+$")

def _canon(v:Any)->Any:
    if isinstance(v,dict): return {str(k):_canon(v[k]) for k in sorted(v)}
    if isinstance(v,(list,tuple)): return [_canon(x) for x in v]
    return v

def canonical_bytes(v:Any)->bytes:
    return json.dumps(_canon(v),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")

def content_sha256(v:Any)->str: return hashlib.sha256(canonical_bytes(v)).hexdigest()

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def load_json(path:Path)->dict: return json.loads(path.read_text(encoding="utf-8"))

def atomic_json(path:Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tmp.replace(path)

def validate_plan(plan:dict)->str:
    if plan.get("schema")!="RealSaS.MainlineExecutionPlan.v1": raise RuntimeError("MAINLINE_PLAN_SCHEMA_DRIFT")
    stages=list(plan.get("stages") or ())
    if int(plan.get("stage_count",-1))!=40 or len(stages)!=40: raise RuntimeError("MAINLINE_PLAN_REQUIRES_40_STAGES")
    if plan.get("canonical_branch")!="main": raise RuntimeError("MAINLINE_PLAN_BRANCH_DRIFT")
    if plan.get("subject_specific_code_forbidden") is not True: raise RuntimeError("MAINLINE_PLAN_GENERICITY_DRIFT")
    seen=set(); ids=[]
    for ordinal,stage in enumerate(stages,1):
        sid=str(stage.get("id","")); ids.append(sid)
        if int(stage.get("ordinal",-1))!=ordinal or not _STAGE_RE.fullmatch(sid): raise RuntimeError(f"MAINLINE_PLAN_STAGE_ORDER_DRIFT:{sid}")
        if any(dep not in seen for dep in tuple(stage.get("depends_on") or ())): raise RuntimeError(f"MAINLINE_PLAN_DEPENDENCY_NOT_EARLIER:{sid}")
        seen.add(sid)
        if not str(stage.get("adapter","")).strip(): raise RuntimeError(f"MAINLINE_PLAN_ADAPTER_MISSING:{sid}")
        keys=stage.get("manifest_keys")
        if not isinstance(keys,list) or not all(isinstance(x,str) and x for x in keys): raise RuntimeError(f"MAINLINE_PLAN_MANIFEST_SCOPE_MISSING:{sid}")
        policy=dict(stage.get("policy") or {})
        if policy.get("fail_closed") is not True or policy.get("output_hash_required") is not True: raise RuntimeError(f"MAINLINE_PLAN_FAIL_CLOSED_POLICY_DRIFT:{sid}")
    if len(ids)!=len(set(ids)): raise RuntimeError("MAINLINE_PLAN_DUPLICATE_STAGE")
    return content_sha256(plan)

def validate_ledger(plan:dict,ledger:dict)->None:
    plan_hash=validate_plan(plan)
    if ledger.get("schema")!="RealSaS.ActiveRunLedger.v1": raise RuntimeError("ACTIVE_RUN_LEDGER_SCHEMA_DRIFT")
    if ledger.get("canonical_branch")!="main": raise RuntimeError("ACTIVE_RUN_LEDGER_BRANCH_DRIFT")
    if ledger.get("pipeline_plan_sha256")!=plan_hash: raise RuntimeError(f"ACTIVE_RUN_LEDGER_PLAN_HASH_DRIFT:{ledger.get('pipeline_plan_sha256')}!={plan_hash}")
    rows=list(ledger.get("stages") or ())
    if len(rows)!=40: raise RuntimeError("ACTIVE_RUN_LEDGER_STAGE_COUNT_DRIFT")
    if [x.get("id") for x in rows]!=[x["id"] for x in plan["stages"]]: raise RuntimeError("ACTIVE_RUN_LEDGER_STAGE_ID_DRIFT")
    complete=sum(x.get("status") in PASS_STATUSES for x in rows)
    by_id={row["id"]:row for row in rows}
    for stage,row in zip(plan["stages"],rows):
        if row.get("status") not in PASS_STATUSES:
            continue
        for dep in stage.get("depends_on",()):
            if by_id[dep].get("status") not in PASS_STATUSES:
                raise RuntimeError(f"ACTIVE_RUN_LEDGER_PASS_WITH_UNPASSED_DEPENDENCY:{stage['id']}:{dep}")
    if int(ledger.get("completed_count",-1))!=complete: raise RuntimeError("ACTIVE_RUN_LEDGER_PROGRESS_DRIFT")
    expected=next((x["id"] for x in rows if x.get("status") not in PASS_STATUSES),None)
    if ledger.get("next_stage")!=expected: raise RuntimeError("ACTIVE_RUN_LEDGER_NEXT_STAGE_DRIFT")

def authority_root()->Path:
    raw=os.environ.get("REALSAS_AUTHORITY_ROOT","").strip()
    return Path(raw).expanduser().resolve() if raw else (Path.home()/"realsas_authority").resolve()

def run_manifest_path(run_id:str)->Path: return authority_root()/"runs"/run_id/"run_manifest.json"

def _local_module_path(module_name:str)->Path|None:
    rel=Path(*str(module_name).split("."))
    file_path=(ROOT/rel).with_suffix(".py")
    if file_path.is_file():
        return file_path.resolve()
    init_path=ROOT/rel/"__init__.py"
    if init_path.is_file():
        return init_path.resolve()
    return None


def _local_import_closure(module_name:str)->tuple[tuple[str,str],...]:
    seen:set[str]=set()
    rows:list[tuple[str,str]]=[]

    def visit(name:str)->None:
        if name in seen:
            return
        path=_local_module_path(name)
        if path is None:
            return
        seen.add(name)
        rows.append((name,sha256_file(path)))
        try:
            tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        except Exception as exc:
            raise RuntimeError(f"MAINLINE_IMPLEMENTATION_AST_INVALID:{name}:{exc}") from exc
        package=name.rpartition(".")[0]
        for node in ast.walk(tree):
            candidates:list[str]=[]
            if isinstance(node,ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    relative="."*int(node.level)+(node.module or "")
                    try:
                        base=importlib.util.resolve_name(relative,package or name)
                    except Exception:
                        base=""
                else:
                    base=str(node.module or "")
                if base:
                    candidates.append(base)
                    for alias in node.names:
                        child=f"{base}.{alias.name}"
                        if _local_module_path(child) is not None:
                            candidates.append(child)
            for candidate in candidates:
                visit(candidate)

    visit(module_name)
    return tuple(sorted(rows))


def _adapter_impl_hash(adapter:str)->str:
    if adapter=="UNBOUND": return hashlib.sha256(b"UNBOUND").hexdigest()
    module_name,sep,fn_name=adapter.partition(":")
    if not sep or not module_name or not fn_name: raise RuntimeError(f"MAINLINE_ADAPTER_ID_INVALID:{adapter}")
    module=importlib.import_module(module_name)
    if not callable(getattr(module,fn_name,None)):
        raise RuntimeError(f"MAINLINE_ADAPTER_CALLABLE_MISSING:{adapter}")
    closure=_local_import_closure(module_name)
    if not closure:
        raise RuntimeError(f"MAINLINE_IMPLEMENTATION_CLOSURE_EMPTY:{adapter}")
    return content_sha256({
        "schema":"RealSaS.AdapterImplementationClosure.v1",
        "adapter":adapter,
        "local_python_import_closure":[{"module":name,"sha256":digest} for name,digest in closure],
    })

def _manifest_subset(manifest:dict,stage:dict)->dict:
    return {key:manifest.get(key) for key in stage["manifest_keys"]}

def _fingerprint(plan:dict,ledger:dict,manifest:dict,stage:dict,impl_hash:str)->tuple[str,str]:
    by_id={row["id"]:row for row in ledger["stages"]}; deps=[]
    for dep in stage.get("depends_on",()):
        deps.extend(str(out["sha256"]) for out in by_id[dep].get("outputs",()) if str(out.get("sha256","")))
    policy_hash=content_sha256(stage["policy"])
    return content_sha256({
        "schema":"RealSaS.StageInputFingerprint.v1",
        "run_id":ledger["run_id"],
        "stage_id":stage["id"],
        "manifest_subset":_manifest_subset(manifest,stage),
        "dependency_output_sha256":deps,
        "policy_hash":policy_hash,
        "implementation_hash":impl_hash,
        "pipeline_plan_sha256":ledger["pipeline_plan_sha256"],
    }),policy_hash

def _dependency_blockers(stage:dict,ledger:dict)->list[str]:
    by_id={row["id"]:row for row in ledger["stages"]}
    blockers=[]
    for dep in stage.get("depends_on",()):
        row=by_id.get(dep)
        if row is None:
            blockers.append(f"DEPENDENCY_UNKNOWN:{dep}")
            continue
        if row.get("status") not in PASS_STATUSES:
            blockers.append(f"DEPENDENCY_NOT_PASS:{dep}:{row.get('status','')}")
            continue
        if not _outputs_verify(row):
            blockers.append(f"DEPENDENCY_OUTPUT_IDENTITY_INVALID:{dep}")
    return blockers


def _outputs_verify(row:dict)->bool:
    outputs=list(row.get("outputs") or ())
    if not outputs: return False
    for out in outputs:
        path=Path(str(out.get("path",""))).expanduser()
        digest=str(out.get("sha256",""))
        if not path.is_file() or len(digest)!=64 or sha256_file(path)!=digest: return False
    return True

def _refresh(ledger:dict)->None:
    rows=ledger["stages"]
    ledger["completed_count"]=sum(x["status"] in PASS_STATUSES for x in rows)
    ledger["total_count"]=len(rows)
    ledger["next_stage"]=next((x["id"] for x in rows if x["status"] not in PASS_STATUSES),None)
    if ledger["completed_count"]==len(rows): ledger["status"]="PASS__ALL_40_STAGES"
    elif any(x["status"] in FAIL_STATUSES for x in rows): ledger["status"]="BLOCKED_AT_"+str(ledger["next_stage"] or "UNKNOWN")
    else: ledger["status"]="ACTIVE"

def _invalidate_dependents(plan:dict,ledger:dict,stage_id:str,reason:str)->tuple[str,...]:
    invalid={str(stage_id)}
    changed=True
    while changed:
        changed=False
        for stage in plan["stages"]:
            sid=stage["id"]
            if sid in invalid:
                continue
            if any(dep in invalid for dep in stage.get("depends_on",())):
                invalid.add(sid); changed=True
    invalidated=[]
    for row in ledger["stages"]:
        if row["id"] not in invalid:
            continue
        row.update(status="PENDING",input_fingerprint="",implementation_hash="",policy_hash="",outputs=[],diagnostics_hash="",blockers=[],wall_seconds=0.0,performance={})
        invalidated.append(row["id"])
    ledger.setdefault("history",[]).append({
        "event":"DEPENDENCY_SUBGRAPH_INVALIDATED",
        "source_stage":str(stage_id),
        "reason":reason,
        "invalidated_stages":invalidated,
    })
    _refresh(ledger)
    return tuple(invalidated)

def _seal_outputs(outputs:list[dict])->list[dict]:
    sealed=[]
    for out in outputs:
        path=Path(str(out["path"])).expanduser().resolve()
        if not path.is_file(): raise RuntimeError(f"STAGE_OUTPUT_MISSING:{path}")
        digest=sha256_file(path); expected=str(out.get("sha256","") or "")
        if expected and expected!=digest: raise RuntimeError(f"STAGE_OUTPUT_SHA_MISMATCH:{path}")
        sealed.append({"path":str(path),"sha256":digest,"bytes":path.stat().st_size,"authority_class":str(out.get("authority_class","SEALED_STAGE_OUTPUT")),"schema":str(out.get("schema","UNSPECIFIED"))})
    if not sealed: raise RuntimeError("STAGE_PASS_REQUIRES_OUTPUT")
    return sealed

def execute(run_id:str,*,from_stage:str="",to_stage:str="",resume:bool=True)->int:
    plan=load_json(PLAN_PATH); ledger=load_json(LEDGER_PATH); validate_ledger(plan,ledger)
    if ledger["run_id"]!=run_id: raise RuntimeError(f"ACTIVE_RUN_ID_MISMATCH:{ledger['run_id']}!={run_id}")
    manifest_path=run_manifest_path(run_id)
    if not manifest_path.is_file(): raise RuntimeError(f"RUN_MANIFEST_MISSING:{manifest_path}")
    manifest=load_json(manifest_path)
    ids=[x["id"] for x in plan["stages"]]
    start=ids.index(from_stage) if from_stage else 0; end=ids.index(to_stage) if to_stage else len(ids)-1
    if start>end: raise RuntimeError("MAINLINE_STAGE_RANGE_INVALID")
    for prior_index in range(start):
        prior_stage=plan["stages"][prior_index]; prior_row=ledger["stages"][prior_index]
        if prior_row.get("status") not in PASS_STATUSES:
            continue
        prior_impl=_adapter_impl_hash(prior_stage["adapter"])
        prior_fingerprint,prior_policy_hash=_fingerprint(plan,ledger,manifest,prior_stage,prior_impl)
        if (
            prior_row.get("input_fingerprint")!=prior_fingerprint
            or prior_row.get("implementation_hash")!=prior_impl
            or prior_row.get("policy_hash")!=prior_policy_hash
            or not _outputs_verify(prior_row)
        ):
            _invalidate_dependents(plan,ledger,prior_stage["id"],"STALE_UPSTREAM_BEFORE_REQUESTED_START")
            atomic_json(LEDGER_PATH,ledger)
            return 2
    for index in range(start,end+1):
        stage=plan["stages"][index]; row=ledger["stages"][index]
        impl_hash=_adapter_impl_hash(stage["adapter"]); fingerprint,policy_hash=_fingerprint(plan,ledger,manifest,stage,impl_hash)
        if resume and row["status"] in PASS_STATUSES:
            if row.get("input_fingerprint")==fingerprint and _outputs_verify(row): continue
            _invalidate_dependents(plan,ledger,stage["id"],"STALE_PASS_IDENTITY"); atomic_json(LEDGER_PATH,ledger); row=ledger["stages"][index]
        if not resume and row["status"] in PASS_STATUSES:
            _invalidate_dependents(plan,ledger,stage["id"],"FORCED_RERUN")
            atomic_json(LEDGER_PATH,ledger)
            row=ledger["stages"][index]
            impl_hash=_adapter_impl_hash(stage["adapter"]); fingerprint,policy_hash=_fingerprint(plan,ledger,manifest,stage,impl_hash)
        dependency_blockers=_dependency_blockers(stage,ledger)
        if dependency_blockers:
            row.update(status="BLOCKED",attempts=int(row.get("attempts",0))+1,input_fingerprint=fingerprint,implementation_hash=impl_hash,policy_hash=policy_hash,outputs=[],diagnostics_hash=content_sha256({"dependency_blockers":dependency_blockers}),blockers=dependency_blockers)
            _refresh(ledger); atomic_json(LEDGER_PATH,ledger); return 2
        if stage["adapter"]=="UNBOUND":
            row.update(status="BLOCKED",attempts=int(row.get("attempts",0))+1,input_fingerprint=fingerprint,implementation_hash=impl_hash,policy_hash=policy_hash,outputs=[],diagnostics_hash="",blockers=["STAGE_ADAPTER_UNBOUND"])
            _refresh(ledger); atomic_json(LEDGER_PATH,ledger); return 2
        module_name,_,fn_name=stage["adapter"].partition(":"); fn=getattr(importlib.import_module(module_name),fn_name)
        row.update(status="RUNNING",attempts=int(row.get("attempts",0))+1,input_fingerprint=fingerprint,implementation_hash=impl_hash,policy_hash=policy_hash,outputs=[],diagnostics_hash="",blockers=[])
        _refresh(ledger); atomic_json(LEDGER_PATH,ledger)
        ctx={"repo_root":ROOT,"authority_root":authority_root(),"run_root":authority_root()/"runs"/run_id,"run_id":run_id,"run_manifest_path":manifest_path,"run_manifest":manifest,"stage":stage,"ledger":ledger}
        started=perf_counter()
        try:
            result=dict(fn(ctx) or {}); elapsed=perf_counter()-started; status=str(result.get("status","FAIL")).upper()
            if status!="PASS":
                row.update(
                    status=status if status in FAIL_STATUSES else "FAIL",
                    diagnostics_hash=content_sha256(result.get("diagnostics",{})),
                    blockers=list(result.get("blockers") or ["STAGE_ADAPTER_REPORTED_FAILURE"]),
                    wall_seconds=float(elapsed),
                    performance=dict(result.get("performance") or {}),
                )
                _refresh(ledger); atomic_json(LEDGER_PATH,ledger); return 2
            row.update(
                status="PASS",
                outputs=_seal_outputs(list(result.get("outputs") or ())),
                diagnostics_hash=content_sha256(result.get("diagnostics",{})),
                blockers=[],
                wall_seconds=float(elapsed),
                performance=dict(result.get("performance") or {}),
            )
        except Exception as exc:
            elapsed=perf_counter()-started
            row.update(status="FAIL",outputs=[],diagnostics_hash=content_sha256({"exception_type":type(exc).__name__,"message":str(exc)}),blockers=[f"EXCEPTION:{type(exc).__name__}:{exc}"],wall_seconds=float(elapsed),performance={})
            _refresh(ledger); atomic_json(LEDGER_PATH,ledger); raise
        _refresh(ledger); atomic_json(LEDGER_PATH,ledger)
    validate_ledger(plan,ledger); return 0

def status_text(plan:dict,ledger:dict)->str:
    validate_ledger(plan,ledger)
    lines=[f"run={ledger['run_id']} status={ledger['status']} progress={ledger['completed_count']}/{ledger['total_count']}",f"next={ledger.get('next_stage') or 'NONE'}"]
    for stage,row in zip(plan["stages"],ledger["stages"]):
        mark="x" if row["status"] in PASS_STATUSES else ("!" if row["status"] in FAIL_STATUSES else ("~" if row["status"]=="RUNNING" else " "))
        lines.append(f"[{mark}] {stage['ordinal']:02d}/40 {stage['id']} — {row['status']}")
    return "\n".join(lines)

def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("validate-plan"); sub.add_parser("status"); run=sub.add_parser("execute")
    run.add_argument("--run-id",required=True); run.add_argument("--from-stage",default=""); run.add_argument("--to-stage",default=""); run.add_argument("--no-resume",action="store_true")
    args=parser.parse_args(argv); plan=load_json(PLAN_PATH); ledger=load_json(LEDGER_PATH)
    if args.command=="validate-plan":
        validate_ledger(plan,ledger); print(f"MAINLINE_PLAN_PASS stages=40 plan_sha256={ledger['pipeline_plan_sha256']}"); return 0
    if args.command=="status": print(status_text(plan,ledger)); return 0
    return execute(args.run_id,from_stage=args.from_stage,to_stage=args.to_stage,resume=not args.no_resume)

if __name__=="__main__": raise SystemExit(main())
