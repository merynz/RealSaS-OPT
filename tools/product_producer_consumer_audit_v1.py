from __future__ import annotations

"""Read-only, source-grounded product producer/consumer wiring court.

This is an engineering audit, NEVER a ProductRevision or scientific authority.
A PASS here says registered declarations line up; it does not prove inference,
source qualification, portable export, UI editing, or native playback.
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]

# These are interface relationships, not another authority for version numbers:
# exact artifact types and versions always come from the live Platform descriptors.
EDGES: dict[str, dict[str, tuple[str, str]]] = {
    "product.input": {},
    "product.shape": {"product.input": ("input_artifact_type", "input_schema_version")},
    "product.mesh": {"product.shape": ("input_shape_artifact_type", "input_shape_schema_version")},
    "product.rig": {"product.mesh": ("input_mesh_artifact_type", "input_mesh_schema_version")},
    "product.skin": {
        "product.mesh": ("input_mesh_artifact_type", "input_mesh_schema_version"),
        "product.rig": ("input_rig_artifact_type", "input_rig_schema_version"),
    },
    "product.mechanics": {
        "product.mesh": ("input_mesh_artifact_type", "input_mesh_schema_version"),
        "product.rig": ("input_rig_artifact_type", "input_rig_schema_version"),
        "product.skin": ("input_skin_artifact_type", "input_skin_schema_version"),
    },
    "product.graphics": {
        "product.mesh": ("input_mesh_artifact_type", "input_mesh_schema_version"),
        "product.skin": ("input_skin_artifact_type", "input_skin_schema_version"),
        "product.mechanics": ("input_mechanics_artifact_type", "input_mechanics_schema_version"),
    },
    "product.motion": {
        "product.rig": ("input_rig_artifact_type", "input_rig_schema_version"),
        "product.mechanics": ("input_mechanics_artifact_type", "input_mechanics_schema_version"),
    },
}

V2_MODEL_MODULES = (
    "models/axis/rig_program_model_v6.py",
    "models/axis/rig_program_decode_v6.py",
    "models/axis/runtime_checkpoint_v2.py",
    "models/axis/scene_conditioning_v6.py",
    "models/mira/skin_fsq_codec_v6.py",
    "models/mira/conditioning_v6.py",
    "models/mira/runtime_checkpoint_v2.py",
)


def read_snapshot(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    return {
        "capabilities": json.loads((
            root / "platform/internal/capability/contracts/product_capabilities_v1.json"
        ).read_text(encoding="utf-8"))["capabilities"],
        "go_catalog": (root / "platform/internal/capability/release_compose.go").read_text(encoding="utf-8"),
        "worker": (root / "compiler/realsas_compiler_services/platform_worker/worker.py").read_text(encoding="utf-8"),
        "mesh_executor": (root / "compiler/realsas_compiler_services/platform_worker/mesh_executor_v1.py").read_text(encoding="utf-8"),
        "go_revision": (root / "platform/internal/control/product_render.go").read_text(encoding="utf-8"),
        "editor": (root / "product/living_compile/server.py").read_text(encoding="utf-8"),
        "authoring": (root / "product/living_compile/authoring.py").read_text(encoding="utf-8"),
        "source_files": {
            path: (root / path).is_file()
            for path in (
                *V2_MODEL_MODULES,
                "compiler/realsas_compiler_core/canonical_motion_ir_v2.py",
                "compiler/realsas_compiler_core/control_motion_runtime_v2.py",
                "compiler/realsas_compiler_services/platform_worker/mechanics_executor_v1.py",
                "compiler/realsas_compiler_services/platform_worker/motion_executor_v2.py",
                "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp",
            )
        },
    }


def _worker_dispatch_contracts(source: str) -> set[str]:
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_capability"
    ]
    if len(functions) != 1:
        raise ValueError("WORKER_EXECUTE_CAPABILITY_ENTRYPOINT_MISSING")
    contracts: set[str] = set()
    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq) or len(node.comparators) != 1:
            continue
        left, right = node.left, node.comparators[0]
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Attribute)
            and left.func.attr == "get"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Constant)
            and left.args[0].value == "executor_contract"
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
        ):
            contracts.add(right.value)
    return contracts


def _go_registered_roles(source: str) -> set[str]:
    start = source.find("func ProductProviders()")
    end = source.find("func productProviderCatalog(", start)
    if start < 0 or end < 0:
        raise ValueError("GO_PRODUCT_PROVIDER_CATALOG_MISSING")
    return {
        "product." + match.lower()
        for match in re.findall(r"\bProduct([A-Za-z]+)Provider\{\}", source[start:end])
    }


def _sealed_output_types(source: str) -> set[tuple[str, str]]:
    """Find literal _seal(artifact_type=..., schema_version=...) call sites."""
    tree = ast.parse(source)
    out: set[tuple[str, str]] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "_seal":
            continue
        words = {
            keyword.arg: keyword.value.value
            for keyword in call.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        }
        if "artifact_type" in words and "schema_version" in words:
            out.add((words["artifact_type"], words["schema_version"]))
    return out


def _dict_boolean(source: str, key: str) -> bool | None:
    """Only read literal Python boolean values, never infer from comments."""
    values = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for k, value in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                    values.append(value.value)
    return values[0] if len(values) == 1 else None


def evaluate(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    caps = snapshot["capabilities"]
    catalog = _go_registered_roles(snapshot["go_catalog"])
    worker = _worker_dispatch_contracts(snapshot["worker"])
    mesh_outputs = _sealed_output_types(snapshot["mesh_executor"])
    issues: list[str] = []
    edges: list[dict[str, str]] = []
    registered: list[dict[str, str]] = []

    if set(caps) != catalog:
        issues.append("REGISTERED_GO_JSON_CATALOG_DRIFT")
    for capability_id, row in sorted(caps.items()):
        meta = row.get("metadata") or {}
        policy = row.get("policy") or {}
        contract = meta.get("executor_contract")
        registered.append({
            "capability_id": capability_id,
            "output": str(meta.get("output_artifact_type") or "") + "/" + str(meta.get("output_schema_version") or ""),
            "executor_contract": str(contract or ""),
        })
        if contract not in worker:
            issues.append("REGISTERED_EXECUTOR_DISPATCH_MISSING:" + capability_id)
        if not row.get("entry_module"):
            issues.append("REGISTERED_ENTRY_MODULE_MISSING:" + capability_id)
        recipe = EDGES.get(capability_id)
        if recipe is None:
            issues.append("REGISTERED_EDGE_RECIPE_MISSING:" + capability_id)
            continue
        actual_deps = {
            dep for dep in policy.get("dependencies", ())
            if isinstance(dep, str) and dep.startswith("product.")
        }
        if actual_deps != set(recipe):
            issues.append("REGISTERED_DEPENDENCY_SET_DRIFT:" + capability_id)
        for parent_id, (input_type_key, input_version_key) in recipe.items():
            producer = caps.get(parent_id)
            if producer is None:
                issues.append("REGISTERED_PARENT_MISSING:" + capability_id + ":" + parent_id)
                continue
            producer_meta = producer["metadata"]
            consumer_type = str(meta.get(input_type_key) or "")
            consumer_version = str(meta.get(input_version_key) or "")
            parent_type = str(producer_meta.get("output_artifact_type") or "")
            parent_version = str(producer_meta.get("output_schema_version") or "")
            status = (
                "MATCH"
                if consumer_type and consumer_version
                and (consumer_type, consumer_version) == (parent_type, parent_version)
                else "MISMATCH"
            )
            edges.append({
                "producer": parent_id,
                "consumer": capability_id,
                "produces": parent_type + "/" + parent_version,
                "consumes": consumer_type + "/" + consumer_version,
                "status": status,
            })
            if status != "MATCH":
                issues.append("REGISTERED_ARTIFACT_SCHEMA_MISMATCH:" + parent_id + "->" + capability_id)

    # product.mesh emits more than its primary canonical output; this is actual
    # _seal call-site evidence, not an assertion based on documentation.
    for extra in (("RealSaS.RiggingSurfaceIR", "v1"),
                  ("RealSaS.TESSAMaterialSupportFieldIR", "v1")):
        if extra not in mesh_outputs:
            issues.append("MESH_SECONDARY_OUTPUT_MISSING:" + extra[0])

    files = snapshot["source_files"]
    missing_models = [path for path in V2_MODEL_MODULES if not files.get(path, False)]
    mechanics_registered = "product.mechanics" in caps
    motion_registered = "product.motion" in caps
    graphics_registered = "product.graphics" in caps
    motion_source_present = all(files.get(p, False) for p in (
        "compiler/realsas_compiler_core/canonical_motion_ir_v2.py",
        "compiler/realsas_compiler_core/control_motion_runtime_v2.py",
    ))
    native_source_present = bool(files.get("runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp", False))

    # Stage40 qualified motion frames and Stage43 .rss remain the sealed product
    # route. Merely importing CanonicalMotion.v2 is NOT a consuming render bridge.
    go_revision = snapshot["go_revision"]
    legacy_revision = all(s in go_revision for s in (
        '"40_MOTION_COMPILE_RUN"', '"43_RSS_MATERIALIZE_COMPACT"',
        '"46_PRODUCT_CLOSURE_SEAL"', '"RealSaS.QualifiedMotionArtifact"',
    ))
    old_frame_consumer = (
        'motion.get("clips")' in snapshot["worker"]
        and 'clip.get("frames")' in snapshot["worker"]
    )
    v2_render_bridge = (
        '"RealSaS.CanonicalMotionIR"' in go_revision
        and "canonical_motion_from_dict_v2" in snapshot["worker"]
        and not old_frame_consumer
    )
    editor_compile = _dict_boolean(snapshot["editor"], "compile_available")
    edits_requal = _dict_boolean(snapshot["authoring"], "requires_compiler_requalification")

    blockers: list[str] = []
    if not mechanics_registered:
        blockers.append("MECHANICS_NOT_REGISTERED_AS_PRODUCT_CAPABILITY")
    if not motion_registered:
        blockers.append("MOTION_V2_NOT_REGISTERED_AS_PRODUCT_CAPABILITY")
    if not graphics_registered:
        blockers.append("GRAPHICS_NOT_REGISTERED_AS_PRODUCT_CAPABILITY")
    if "product.proof" not in caps:
        blockers.append("PROOF_NOT_REGISTERED_AS_PRODUCT_CAPABILITY")
    if missing_models:
        blockers.append("V2_RIG_SKIN_MODEL_RUNTIME_SOURCE_DEPENDENCIES_MISSING")
    if not v2_render_bridge:
        blockers.append("CANONICAL_MOTION_V2_NOT_CONSUMED_BY_PRODUCT_RENDER")
    if editor_compile is not True:
        blockers.append("EDITOR_COMPILE_NOT_CONNECTED")
    if edits_requal is not False:
        blockers.append("EDIT_LAYER_NOT_REQUALIFIED_AS_PRODUCT")
    if not native_source_present:
        blockers.append("NATIVE_RUNTIME_SOURCE_MISSING")

    return {
        "schema": "RealSaS.ProductProducerConsumerWiringAudit.v1",
        "authority": "READ_ONLY_SOURCE_WIRING_DIAGNOSTIC__NO_PRODUCT_OR_SCIENTIFIC_PASS",
        "registered_contract_status": "PASS" if not issues else "FAIL",
        "registered_capabilities": registered,
        "registered_edges": edges,
        "registered_issues": sorted(set(issues)),
        "v2_candidate": {
            "missing_model_runtime_sources": missing_models,
            "motion_source_present": motion_source_present,
            "mechanics_registered": mechanics_registered,
            "motion_registered": motion_registered,
            "graphics_registered": graphics_registered,
        },
        "runtime_consumption": {
            "legacy_stage40_43_46_product_route_present": legacy_revision,
            "frame_baked_render_consumer_present": old_frame_consumer,
            "canonical_motion_v2_render_bridge_verified": v2_render_bridge,
            "native_player_source_present": native_source_present,
        },
        "editor": {
            "compile_available_literal": editor_compile,
            "edits_require_compiler_requalification": edits_requal,
        },
        "product_completion_blockers": blockers,
        "product_e2e_qualified": False,
        "product_e2e_proof": "NOT_EXECUTED_OR_MEASURED_BY_THIS_AUDIT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--strict-registered", action="store_true",
                        help="Fail on registered producer/consumer incompatibilities")
    args = parser.parse_args()
    report = evaluate(read_snapshot(args.root))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict_registered and report["registered_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
