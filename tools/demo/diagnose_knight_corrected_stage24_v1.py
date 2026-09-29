from __future__ import annotations
import argparse, json
from pathlib import Path
from tools.demo.render_knight_motion_preview_v1 import _ctx
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import qualify_complete_appearance_stage

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    ctx=_ctx(a.authority_root.resolve(),a.run_id)
    ctx["stage"]={"id":"24_COMPLETE_APPEARANCE_QUALIFIED"}
    result=qualify_complete_appearance_stage(ctx)
    print("STAGE24_DIAGNOSTICS="+json.dumps({
        "status":result.get("status"),
        "blockers":result.get("blockers") or [],
        "diagnostics":result.get("diagnostics") or {},
    },sort_keys=True))

if __name__=="__main__":
    main()
