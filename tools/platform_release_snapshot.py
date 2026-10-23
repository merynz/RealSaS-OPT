#!/usr/bin/env python3
"""Export a subject-agnostic EngineRelease request without executing models.

EngineRelease owns executable code/policy/DAG and stage-level execution
contracts. Subject-specific manifest values belong to SubjectInput/CAS and are
bound at compile time, never duplicated into the release identity.

V2 requests also carry version identities for first-class product capabilities
which live outside the legacy Compiler stage DAG. Platform derives stage
capability versions from the sealed stage bindings and seals both views under
the same immutable EngineRelease ID. Learned ModelRelease requirements are
also derived from the same product-role/responsibility contracts; they are not
maintained as a parallel hand-authored list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import (
    released_execution_plan,
    subject_agnostic_release_parameters,
)
from tools.platform_capability_versions import (
    capability_versions,
    product_model_requirements,
)
from tools.realsas_architecture import lightweight_implementation_closure


def _stage_parameter_overrides(payload):
    if payload is None:
        return {}
    if payload.get("schema") != "RealSaS.StageReleaseParameters.v1":
        raise RuntimeError("RELEASE_STAGE_PARAMETERS_SCHEMA")
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("RELEASE_STAGE_PARAMETERS_REQUIRED")
    return stages


def snapshot(plan, manifest=None, *, name, purpose, created_by, stage_parameters=None):
    del manifest  # Legacy call compatibility; subject values are intentionally not release authority.
    require_product = purpose == "PRODUCT"
    closure = lightweight_implementation_closure(plan, require_product_pass_authority=require_product)
    implementations = {row["stage_id"]: row["implementation_hash"]
                       for row in closure["adapter_implementation_closures"]}
    overrides = _stage_parameter_overrides(stage_parameters)
    known = {stage["id"] for stage in plan["stages"]}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise RuntimeError("RELEASE_STAGE_PARAMETERS_UNKNOWN_STAGE:" + ",".join(unknown))
    model_channel = "PRODUCTION" if require_product else "RESEARCH"
    model_requirements = [
        {**row, "channel": model_channel}
        for row in product_model_requirements()
    ]
    request = {
        "created_by": created_by,
        "capability_versions": capability_versions(),
        "manifest": {
            "pipeline_plan_sha256": mainline.validate_plan(
                plan, require_product_pass_authority=require_product
            ),
            "dag": {
                "schema": "RealSaS.StageGraph.v1",
                "stage_count": len(plan["stages"]),
                "stages": [
                    {
                        **{
                            key: stage[key]
                            for key in (
                                "ordinal",
                                "id",
                                "title",
                                "group",
                                "depends_on",
                                "adapter",
                                "manifest_keys",
                            )
                        },
                        "policy": {
                            key: bool(stage["policy"].get(key, False))
                            for key in (
                                "fail_closed",
                                "cacheable",
                                "output_hash_required",
                                "product_pass_authority",
                            )
                        },
                    }
                    for stage in plan["stages"]
                ],
            },
            "contract_version": "RealSaS.EngineReleaseManifest.v2",
            "name": name,
            "purpose": purpose,
            "model_requirements": model_requirements,
            "stages": [
                {
                    "ordinal": stage["ordinal"],
                    "stage_id": stage["id"],
                    "implementation_sha256": implementations[stage["id"]],
                    "policy_sha256": mainline.content_sha256(stage["policy"]),
                    "semantic_parameters": subject_agnostic_release_parameters(
                        stage, overrides.get(stage["id"])
                    ),
                }
                for stage in plan["stages"]
            ],
        },
    }
    # Do not seal a release which this shipped Engine cannot reconstruct.
    reconstructed = released_execution_plan(
        mainline.load_json(mainline.PLAN_PATH), request["manifest"]["dag"], mode=purpose
    )
    if mainline.content_sha256(reconstructed) != request["manifest"]["pipeline_plan_sha256"]:
        raise RuntimeError("RELEASE_EXECUTION_METADATA_NOT_SHIPPED_ON_MAIN")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Legacy compatibility only; subject values are not release authority",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=mainline.PLAN_PATH,
        help="Explicit versioned research DAG; defaults to canonical main plan",
    )
    parser.add_argument(
        "--stage-parameters",
        type=Path,
        help="Optional RealSaS.StageReleaseParameters.v1 subject-agnostic stage contracts",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", choices=("RESEARCH", "PRODUCT"), required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stage_parameters = (
        None if args.stage_parameters is None else mainline.load_json(args.stage_parameters)
    )
    manifest = None if args.run_manifest is None else mainline.load_json(args.run_manifest)
    value = snapshot(
        mainline.load_json(args.plan),
        manifest,
        name=args.name,
        purpose=args.purpose,
        created_by=args.created_by,
        stage_parameters=stage_parameters,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"EngineRelease request: {args.out}; {len(value['manifest']['stages'])} stages; "
        f"{len(value['capability_versions'])} product capabilities; "
        f"{len(value['manifest']['model_requirements'])} learned model requirements; no execution"
    )


if __name__ == "__main__":
    main()
