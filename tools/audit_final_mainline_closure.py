#!/usr/bin/env python3
from __future__ import annotations
import ast, fnmatch, json, re, subprocess, sys
from collections import deque
from pathlib import Path

LOCAL_PREFIXES=("compiler.","models.","runtime.","tools.")
def git(*args):
    p=subprocess.run(["git",*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode: raise RuntimeError(p.stderr)
    return p.stdout
def exists(ref,path):
    return subprocess.run(["git","cat-file","-e",f"{ref}:{path}"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def show(ref,path): return git("show",f"{ref}:{path}")
def sha(ref,path): return git("rev-parse",f"{ref}:{path}").strip() if exists(ref,path) else None
def paths(ref): return [x for x in git("ls-tree","-r","--name-only",ref).splitlines() if x]
def module_path(ref,module):
    rel=module.replace(".","/")
    for p in (rel+".py",rel+"/__init__.py"):
        if exists(ref,p): return p
def path_module(path):
    if path.endswith("/__init__.py"): return path[:-12].replace("/",".")
    if path.endswith(".py"): return path[:-3].replace("/",".")
    return ""
def resolve_from(cur,level,target):
    if level<=0:return target or ""
    parts=cur.split(".")[:-1]
    up=max(level-1,0)
    if up: parts=parts[:-up] if up<=len(parts) else []
    if target:parts.extend(target.split("."))
    return ".".join(parts)
def imports(ref,path):
    src=show(ref,path);cur=path_module(path);tree=ast.parse(src,filename=path);out=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            for a in n.names:
                if a.name.startswith(LOCAL_PREFIXES) and module_path(ref,a.name):out.add(a.name)
        elif isinstance(n,ast.ImportFrom):
            base=resolve_from(cur,int(n.level or 0),n.module)
            if base.startswith(LOCAL_PREFIXES) and module_path(ref,base):out.add(base)
            for a in n.names:
                if a.name=="*":continue
                child=f"{base}.{a.name}" if base else a.name
                if child.startswith(LOCAL_PREFIXES) and module_path(ref,child):out.add(child)
    return out
def literal(src,name):
    t=ast.parse(src)
    for n in t.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            ts=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(x,ast.Name) and x.id==name for x in ts):
                return ast.literal_eval(n.value)
    raise KeyError(name)

research=sys.argv[1]; baseline=sys.argv[2]; out=sys.argv[3]
plan_path="canonical/MAINLINE_EXECUTION_PLAN_V2.json"
mainline_path="compiler/realsas_compiler_services/orchestrator/mainline.py"
plan=json.loads(show(research,plan_path)); src=show(research,mainline_path)
assert int(plan["stage_count"])==46 and len(plan["stages"])==46
assert plan["canonical_branch"]=="main"
assert plan["subject_specific_code_forbidden"] is True
allp=set(paths(research))
selected={plan_path,mainline_path}
for p in literal(src,"IMPLEMENTATION_CLOSURE_STATIC_PATHS"):
    if p in allp:selected.add(p)
for pat in literal(src,"IMPLEMENTATION_CLOSURE_DYNAMIC_GLOBS"):
    selected.update(p for p in allp if fnmatch.fnmatch(p,pat))
for root in literal(src,"IMPLEMENTATION_CLOSURE_TEST_ROOTS"):
    pref=root.rstrip("/")+"/"
    selected.update(p for p in allp if p.startswith(pref) and p.endswith(".py"))
seeds=set()
for s in sorted(plan["stages"],key=lambda x:int(x["ordinal"])):
    mod=s["adapter"].split(":",1)[0]; p=module_path(research,mod)
    if not p:raise RuntimeError("missing adapter "+mod)
    selected.add(p);seeds.add(p)
q=deque(sorted(seeds));parsed=set();mods=set()
while q:
    p=q.popleft()
    if p in parsed:continue
    parsed.add(p)
    for m in imports(research,p):
        mods.add(m);cp=module_path(research,m)
        if cp:selected.add(cp)
        if cp and cp not in parsed:q.append(cp)
forbidden=set(literal(src,"CURRENT_V2_FORBIDDEN_IMPORT_MODULES"))
bad=sorted(mods&forbidden)
if bad:raise RuntimeError("forbidden donor imports:"+",".join(bad))
subject=[p for p in selected if p.startswith(("compiler/","models/","runtime/")) and re.search(r"(knight|mage|fit1|subject[_-]?2)",p,re.I)]
if subject:raise RuntimeError("subject product code:"+",".join(subject))
changed=[]
for p in sorted(selected):
    rs,bs=sha(research,p),sha(baseline,p)
    if rs!=bs:changed.append({"path":p,"baseline_sha":bs,"research_sha":rs,"new":bs is None})
report={"schema":"RealSaS.FinalMainlineClosureDiff.v1","research":research,"baseline":baseline,
"selected_count":len(selected),"changed_count":len(changed),"adapter_python_count":len(parsed),
"changed_files":changed,
"stage_adapters":[{"ordinal":s["ordinal"],"id":s["id"],"adapter":s["adapter"],"depends_on":s.get("depends_on",[])} for s in sorted(plan["stages"],key=lambda x:int(x["ordinal"]))]}
Path(out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("SELECTED",len(selected));print("CHANGED",len(changed));print("ADAPTER_PY",len(parsed))
for row in changed: print(("NEW " if row["new"] else "CHG ")+row["path"])
