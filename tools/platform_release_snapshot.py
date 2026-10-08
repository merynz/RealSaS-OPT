#!/usr/bin/env python3
"""Export an EngineRelease request without loading model runtimes or fitting.

Pins source implementation, policy and each stage's actual manifest read set.
The Go API owns sealing; this command only prepares a reviewable request.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import released_execution_plan
from tools.realsas_architecture import lightweight_implementation_closure


def snapshot(plan, manifest, *, name, purpose, created_by):
    require_product = purpose == "PRODUCT"
    closure = lightweight_implementation_closure(plan, require_product_pass_authority=require_product)
    implementations = {row["stage_id"]: row["implementation_hash"]
                       for row in closure["adapter_implementation_closures"]}
    request = {"created_by": created_by, "manifest": {
        "pipeline_plan_sha256": mainline.validate_plan(plan, require_product_pass_authority=require_product),
        "dag": {"schema": "RealSaS.StageGraph.v1", "stage_count": len(plan["stages"]),
                "stages": [{**{key: stage[key] for key in
                            ("ordinal", "id", "title", "group", "depends_on", "adapter", "manifest_keys")},
                            "policy": {key: bool(stage["policy"].get(key, False)) for key in
                                       ("fail_closed", "cacheable", "output_hash_required", "product_pass_authority")}}
                           for stage in plan["stages"]]},
        "contract_version": "RealSaS.EngineReleaseManifest.v1",
        "name": name, "purpose": purpose,
        "stages": [{"ordinal": stage["ordinal"], "stage_id": stage["id"],
                    "implementation_sha256": implementations[stage["id"]],
                    "policy_sha256": mainline.content_sha256(stage["policy"]),
                    "semantic_parameters": {"manifest": mainline._manifest_subset(manifest, stage)}}
                   for stage in plan["stages"]],
    }}
    # Do not seal a release which this shipped Engine cannot reconstruct.
    reconstructed = released_execution_plan(mainline.load_json(mainline.PLAN_PATH),
                                            request["manifest"]["dag"], mode=purpose)
    if mainline.content_sha256(reconstructed) != request["manifest"]["pipeline_plan_sha256"]:
        raise RuntimeError("RELEASE_EXECUTION_METADATA_NOT_SHIPPED_ON_MAIN")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=mainline.PLAN_PATH,
                        help="Explicit versioned research DAG; defaults to canonical main plan")
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", choices=("RESEARCH", "PRODUCT"), required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = snapshot(mainline.load_json(args.plan),
                     mainline.load_json(args.run_manifest), name=args.name,
                     purpose=args.purpose, created_by=args.created_by)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(f"EngineRelease request: {args.out}; {len(value['manifest']['stages'])} stages; no execution")


if __name__ == "__main__":
    main()
