#!/usr/bin/env python3
"""Export a subject-agnostic EngineRelease request without executing models.

EngineRelease owns executable code/policy/DAG and stage-level execution
contracts. Subject-specific manifest values belong to SubjectInput/CAS and are
bound at compile time, never duplicated into the release identity.

V2 requests also carry version identities for first-class product capabilities
which live outside the legacy Compiler stage DAG. Platform derives stage
capability versions from the sealed stage bindings and seals both views under
the same immutable EngineRelease ID. Product capability admission is projected
from the exact released DAG using the shared product semantic contract; learned
ModelRelease requirements are then limited to those admitted roles.
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
    capability_specs,
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


def _admitted_product_capability_ids(plan) -> tuple[str, ...]:
    specs = capability_specs()
    stage_capabilities = {
        "compiler.stage/" + str(stage["id"])
        for stage in plan.get("stages", [])
    }
    memo: dict[str, bool] = {}
    visiting: set[str] = set()

    def unavailable(capability_id: str) -> bool:
        if capability_id in memo:
            return memo[capability_id]
        if capability_id in visiting:
            raise RuntimeError("PRODUCT_CAPABILITY_DEPENDENCY_CYCLE:" + capability_id)
        if capability_id not in specs:
            raise RuntimeError("PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:" + capability_id)
        visiting.add(capability_id)
        try:
            policy = specs[capability_id]["policy"]
            dependencies = policy.get("dependencies") or []
            if not isinstance(dependencies, list):
                raise RuntimeError("PRODUCT_CAPABILITY_DEPENDENCIES_TYPE:" + capability_id)
            for dependency in dependencies:
                dependency = str(dependency)
                if dependency in stage_capabilities:
                    continue
                if dependency.startswith("compiler.stage/"):
                    memo[capability_id] = True
                    return True
                if dependency.startswith("product."):
                    if dependency not in specs:
                        raise RuntimeError(
                            "PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:"
                            + capability_id
                            + ":"
                            + dependency
                        )
                    if unavailable(dependency):
                        memo[capability_id] = True
                        return True
                    continue
                raise RuntimeError(
                    "PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:"
                    + capability_id
                    + ":"
                    + dependency
                )
            memo[capability_id] = False
            return False
        finally:
            visiting.discard(capability_id)

    return tuple(
        capability_id
        for capability_id in sorted(specs)
        if not unavailable(capability_id)
    )


def _release_capability_versions(admitted_ids: tuple[str, ...]):
    all_versions = capability_versions()
    admitted = set(admitted_ids)
    unknown = sorted(admitted - set(all_versions))
    if unknown:
        raise RuntimeError("RELEASE_PRODUCT_CAPABILITY_VERSION_UNKNOWN:" + ",".join(unknown))
    return {
        capability_id: all_versions[capability_id]
        for capability_id in sorted(admitted)
    }


def _release_model_requirements(admitted_ids: tuple[str, ...]):
    # product_model_requirements() validates the complete source catalog against
    # the responsibility ledger. Release projection then selects only families
    # actually required by admitted product roles.
    global_rows = {
        str(row["family"]): dict(row)
        for row in product_model_requirements()
    }
    specs = capability_specs()
    active_families: set[str] = set()
    for capability_id in admitted_ids:
        required_models = specs[capability_id]["parameters"].get("required_models") or []
        if not isinstance(required_models, list):
            raise RuntimeError("PRODUCT_MODEL_REQUIREMENTS_TYPE:" + capability_id)
        for row in required_models:
            family = str(row.get("family") or "").strip()
            if not family or family not in global_rows:
                raise RuntimeError(
                    "RELEASE_PRODUCT_MODEL_REQUIREMENT_UNKNOWN:"
                    + capability_id
                    + ":"
                    + family
                )
            active_families.add(family)
    return [global_rows[family] for family in sorted(active_families)]


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
    admitted_capabilities = _admitted_product_capability_ids(plan)
    model_channel = "PRODUCTION" if require_product else "RESEARCH"
    model_requirements = [
        {**row, "channel": model_channel}
        for row in _release_model_requirements(admitted_capabilities)
    ]
    request = {
        "created_by": created_by,
        "capability_versions": _release_capability_versions(admitted_capabilities),
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
