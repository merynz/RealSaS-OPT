from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from compiler.realsas_compiler_core.mesh._historical_v05 import (
    HISTORICAL_CDT_SOURCE_SHA256,
    historical_cdt_source_bytes,
    load_historical_cdt_module_v05,
)

ROOT=Path(__file__).resolve().parents[1]

raw=historical_cdt_source_bytes()
src=raw.decode("utf-8")
tree=ast.parse(src)
functions={}
classes={}
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
        seg=ast.get_source_segment(src,node) or ""
        functions[node.name]={
            "lineno":node.lineno,
            "end_lineno":getattr(node,"end_lineno",None),
            "args":[a.arg for a in node.args.args],
            "kwonlyargs":[a.arg for a in node.args.kwonlyargs],
            "defaults_count":len(node.args.defaults),
            "source":seg,
        }
    elif isinstance(node,ast.ClassDef):
        classes[node.name]={
            "lineno":node.lineno,
            "methods":[x.name for x in node.body if isinstance(x,ast.FunctionDef)],
        }

module=load_historical_cdt_module_v05()
fn=module.triangulate_production_cdt
sig=str(inspect.signature(fn))
target=functions.get("triangulate_production_cdt")
if target is None:
    raise SystemExit("HISTORICAL_CDT_TARGET_FUNCTION_MISSING")

tokens={
    t:(t in target["source"])
    for t in (
        "constraints","constraint_edges","boundary","polygon","holes","segments",
        "constraint_split_count","boundary_vertex_count","Steiner","steiner",
        "triangles","vertices",
    )
}
calls=sorted({
    x.func.id if isinstance(x.func,ast.Name) else x.func.attr
    for x in ast.walk(ast.parse(target["source"]))
    if isinstance(x,ast.Call) and isinstance(x.func,(ast.Name,ast.Attribute))
})

payload={
    "schema":"RealSaS.HistoricalCDTV05CapabilityAudit.v1",
    "source_sha256":HISTORICAL_CDT_SOURCE_SHA256,
    "source_bytes":len(raw),
    "runtime_signature":sig,
    "target_function":{
        "lineno":target["lineno"],
        "end_lineno":target["end_lineno"],
        "args":target["args"],
        "kwonlyargs":target["kwonlyargs"],
        "token_presence":tokens,
        "calls":calls,
        "source_excerpt":target["source"],
    },
    "top_level_function_names":sorted(functions),
    "classes":classes,
}
out=ROOT/"canonical"/"HISTORICAL_CDT_V05_CAPABILITY_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print("HISTORICAL_CDT_V05_CAPABILITY_AUDIT_PASS",json.dumps({
    "signature":sig,"tokens":tokens,"calls":calls,
},sort_keys=True))
