#!/usr/bin/env python3
"""Dependency-free static audit of the current IRIS/TESSA V3 apparatus.

Only Python stdlib source/AST inspection is used. The audit intentionally does
not import torch or any RealSaS model module, instantiate a model, run inference,
train, promote a ModelRelease, or mutate product state.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "RealSaS.IRISTESSAV3ApparatusStaticContractAudit.v1"
AUDIT_STATUS = "STATIC_FACTS_SEALED__READINESS_NOT_DECIDED"

SOURCE_PATHS = {
    "input": "models/iris/vnext/inference_input_v1.py",
    "iris_config": "models/iris/vnext/contracts_v1.py",
    "scene": "models/iris/vnext/scene_core_v1.py",
    "iris_model": "models/iris/vnext/model_v3.py",
    "relations": "models/iris/vnext/heads_v3.py",
    "presentation": "models/iris/vnext/presentation_evidence_v1.py",
    "bridge": "models/iris/vnext/structure_v4.py",
    "tessa_v2": "models/tessa/v2/model_v2.py",
    "tessa_objective": "models/tessa/v2/objective_v2.py",
    "tessa_config": "models/tessa/v3/config_v3.py",
    "tessa_model": "models/tessa/v3/model_v3.py",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(repo_root: Path, rel: str) -> str:
    path = repo_root / rel
    if not path.is_file():
        raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_SOURCE_MISSING:" + rel)
    return path.read_text(encoding="utf-8")


def _tree(text: str, label: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_PARSE_FAILED:" + label) from exc


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_DRIFT:" + label)


def _literal(node: ast.AST, label: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_NONLITERAL:" + label) from exc


def _module_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return _literal(node.value, name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is None:
                break
            return _literal(node.value, name)
    raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_ASSIGNMENT_MISSING:" + name)


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_CLASS_MISSING:" + class_name)


def _class_default(tree: ast.Module, class_name: str, field_name: str) -> Any:
    cls = _class_node(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field_name:
            if node.value is None:
                break
            return _literal(node.value, f"{class_name}.{field_name}")
    raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_FIELD_MISSING:" + class_name + "." + field_name)


def _class_assignment(tree: ast.Module, class_name: str, name: str) -> Any:
    cls = _class_node(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return _literal(node.value, f"{class_name}.{name}")
    raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_CLASS_ASSIGNMENT_MISSING:" + class_name + "." + name)


def _function_node(tree: ast.Module, function_name: str, *, class_name: str | None = None) -> ast.FunctionDef:
    body: list[ast.stmt]
    if class_name is None:
        body = tree.body
    else:
        body = _class_node(tree, class_name).body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node  # type: ignore[return-value]
    label = f"{class_name + '.' if class_name else ''}{function_name}"
    raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_FUNCTION_MISSING:" + label)


def _node_source(text: str, node: ast.AST) -> str:
    lines = text.splitlines()
    start = int(getattr(node, "lineno")) - 1
    end = int(getattr(node, "end_lineno"))
    return "\n".join(lines[start:end])


def audit(repo_root: Path, expected_parent_sha: str) -> dict[str, Any]:
    texts = {key: _read(repo_root, rel) for key, rel in SOURCE_PATHS.items()}
    trees = {key: _tree(text, key) for key, text in texts.items()}

    view_count = int(_module_assignment(trees["input"], "VIEW_COUNT_V1"))
    patch_grid = int(_module_assignment(trees["input"], "PATCH_GRID_V1"))
    camera_dim = int(_module_assignment(trees["input"], "CAMERA_FEATURE_DIM_V1"))
    _require("PATCH_TOKEN_COUNT_V1 = PATCH_GRID_V1 * PATCH_GRID_V1" in texts["input"], "PATCH_TOKEN_DERIVATION")
    patch_tokens = patch_grid * patch_grid

    iris_fields = {
        name: int(_class_default(trees["iris_config"], "IRISVNextConfigV1", name))
        for name in (
            "scene_dim",
            "plane_size",
            "plane_channels",
            "local_feature_dim",
            "extraction_training_resolution",
            "surface_witness_count",
        )
    }
    tessa_fields = {
        name: int(_class_default(trees["tessa_config"], "TESSAStructuralizerConfigV3", name))
        for name in (
            "d_model",
            "n_heads",
            "decoder_layers",
            "scene_memory_tokens",
            "surface_memory_tokens",
            "presentation_memory_tokens",
            "conditioning_token_budget",
            "local_attention_window",
            "query_chunk_size",
            "max_faces",
            "max_vertices",
        )
    }
    static_outputs = tuple(_class_assignment(trees["relations"], "GraphicsRelationHeadV3", "STATIC_OUTPUTS"))
    view_outputs = tuple(_class_assignment(trees["relations"], "GraphicsRelationHeadV3", "VIEW_OUTPUTS"))

    rgba_fn = _node_source(texts["input"], _function_node(trees["input"], "rgba_to_white_composited_rgb_v1"))
    frontend_fn = _node_source(texts["input"], _function_node(trees["input"], "build_scene_inputs_from_rgba_v1"))
    scene_init = _node_source(texts["scene"], _function_node(trees["scene"], "__init__", class_name="SharedSceneTriplaneVNextV1"))
    scene_forward = _node_source(texts["scene"], _function_node(trees["scene"], "forward", class_name="SharedSceneTriplaneVNextV1"))
    model_init = _node_source(texts["iris_model"], _function_node(trees["iris_model"], "__init__", class_name="IRISVNextEvidenceModelV3"))
    static_fn = _node_source(texts["relations"], _function_node(trees["relations"], "static_relations", class_name="GraphicsRelationHeadV3"))
    view_fn = _node_source(texts["relations"], _function_node(trees["relations"], "view_relations", class_name="GraphicsRelationHeadV3"))
    typed_forward = _node_source(texts["tessa_v2"], _function_node(trees["tessa_v2"], "forward", class_name="_TypedMemoryStreamV2"))
    tri_forward = _node_source(texts["tessa_model"], _function_node(trees["tessa_model"], "forward", class_name="TESSATriStreamMemoryV3"))
    window_forward = _node_source(texts["tessa_v2"], _function_node(trees["tessa_v2"], "forward", class_name="WindowedCausalSelfAttentionV2"))

    _require(view_count == 8, "VIEW_COUNT")
    _require(patch_grid == 37, "PATCH_GRID")
    _require(camera_dim == 13, "CAMERA_FEATURE_DIM")
    _require("rgba[:, :, :3] * alpha + (1.0 - alpha)" in rgba_fn, "RGBA_WHITE_COMPOSITE")
    _require("extract_dinov2_s_patch_map_v2(dino_model, rgb)" in frontend_fn, "DINO_RGB_FRONTEND")
    _require("nn.TransformerDecoder" in scene_init, "SCENE_QUERY_DECODER")
    _require("nn.ConvTranspose2d" in scene_init and "stride=2" in scene_init, "TRIPLANE_UPSAMPLE_X2")
    _require("self.decoder(q, memory)" in scene_forward, "SCENE_QUERIES_ATTEND_MEMORY")
    _require("self.mechanical_demand" in model_init, "MECHANICAL_HEAD")
    _require("self.appearance" in model_init, "APPEARANCE_HEAD")
    _require("self.relations" in model_init, "RELATION_HEAD")
    _require("RayStackVisibilityHeadV3" not in model_init, "RAY_HEAD_NOT_INSTANTIATED")
    _require("point_a" not in static_fn and "point_b" not in static_fn and "camera_features" not in static_fn, "STATIC_NO_EXPLICIT_GEOMETRY")
    _require("relation_camera_geometry_v3" in view_fn and "camera_features" in view_fn, "VIEW_EXPLICIT_GEOMETRY")
    _require("self.cross(" in typed_forward, "TYPED_STREAM_CROSS_ATTENTION")
    _require("self.refinement(memory)" in typed_forward, "TYPED_STREAM_REFINEMENT")
    _require("self.scene(" in tri_forward and "self.surface(" in tri_forward and "self.presentation(" in tri_forward, "TESSA_THREE_STREAMS")
    _require("torch.cat((scene, surface, presentation), dim=1)" in tri_forward, "TESSA_LATE_CONCAT")
    _require("local_attention_window" in window_forward, "TESSA_LOCAL_WINDOW")

    total_conditioning = (
        tessa_fields["scene_memory_tokens"]
        + tessa_fields["surface_memory_tokens"]
        + tessa_fields["presentation_memory_tokens"]
    )
    _require(total_conditioning == tessa_fields["conditioning_token_budget"], "TESSA_MATCHED_TOKEN_BUDGET")

    plane_output_resolution = iris_fields["plane_size"] * 2
    source_hashes = {
        rel: _sha256(repo_root / rel)
        for rel in SOURCE_PATHS.values()
    }

    facts = {
        "observation": {
            "view_count": view_count,
            "native_rgba_resolution": [1024, 1024],
            "foundation_frontend": "WHITE_COMPOSITED_RGB_TO_FROZEN_DINOV2_S",
            "alpha_independent_foundation_channel": False,
            "dino_patch_grid": [patch_grid, patch_grid],
            "dino_tokens_per_view": patch_tokens,
            "camera_feature_dim": camera_dim,
        },
        "iris_scene": {
            "scene_dim": iris_fields["scene_dim"],
            "plane_query_resolution": [iris_fields["plane_size"], iris_fields["plane_size"]],
            "plane_channels": iris_fields["plane_channels"],
            "triplane_output_resolution": [plane_output_resolution, plane_output_resolution],
            "local_h_dim": iris_fields["local_feature_dim"],
            "extraction_training_resolution": iris_fields["extraction_training_resolution"],
            "surface_witness_count": iris_fields["surface_witness_count"],
            "source_token_cross_view_self_attention_before_scene_queries": False,
            "multi_view_interaction_route": "LEARNED_SCENE_QUERIES_ATTEND_FLATTENED_ALL_VIEW_MEMORY",
        },
        "iris_relations": {
            "static_outputs": list(static_outputs),
            "view_outputs": list(view_outputs),
            "static_pair_inputs": ["H_a", "H_b"],
            "static_explicit_point_geometry": False,
            "static_explicit_camera_geometry": False,
            "view_explicit_pair_geometry": True,
            "view_explicit_camera_features": True,
            "ray_stack_visibility_head_instantiated_by_main_v3": False,
        },
        "tessa": {
            "d_model": tessa_fields["d_model"],
            "heads": tessa_fields["n_heads"],
            "decoder_layers": tessa_fields["decoder_layers"],
            "conditioning_tokens": {
                "scene": tessa_fields["scene_memory_tokens"],
                "surface_mechanics": tessa_fields["surface_memory_tokens"],
                "presentation": tessa_fields["presentation_memory_tokens"],
                "total": total_conditioning,
            },
            "stream_compression": "INDEPENDENT_TYPED_PERCEIVER_STYLE_BEFORE_CONCAT",
            "cross_stream_interaction_before_independent_compression": False,
            "local_attention_window": tessa_fields["local_attention_window"],
            "query_chunk_size": tessa_fields["query_chunk_size"],
            "max_faces": tessa_fields["max_faces"],
            "max_vertices": tessa_fields["max_vertices"],
        },
    }

    open_risks = [
        {
            "id": "A1_RGBA_COLLISION",
            "class": "IDENTIFIABILITY",
            "fact": "foundation scene input sees white-composited RGB rather than independent RGBA",
            "required_probe": "construct distinct RGBA observations with identical composited RGB and graphics-relevant alpha/source-support differences",
        },
        {
            "id": "A3_SPATIAL_ACCESS",
            "class": "REPRESENTATION_RESOLUTION",
            "fact": f"foundation grid={patch_grid}x{patch_grid}; triplane={plane_output_resolution}x{plane_output_resolution}; extraction_train={iris_fields['extraction_training_resolution']}; surface_witness_count={iris_fields['surface_witness_count']}",
            "required_probe": "controlled thin/separated/interleaving structure accessibility at H(x), stratified below and above source-token scale",
        },
        {
            "id": "B1_IRIS_FUSION_PLACEMENT",
            "class": "ATTENTION_PLACEMENT",
            "fact": "view tokens first interact through shared scene-query attention",
            "required_probe": "cross-view correspondence/reconciliation accessibility under controlled ambiguity",
        },
        {
            "id": "B2_TESSA_FUSION_PLACEMENT",
            "class": "ATTENTION_PLACEMENT",
            "fact": "scene/surface/presentation are independently compressed before concatenation",
            "required_probe": "typed endpoint addressability survives independent compression; otherwise preregister matched early/mid/late fusion ablation",
        },
        {
            "id": "C1_STATIC_PAIR_GEOMETRY",
            "class": "RELATION_EXPRESSIVITY",
            "fact": "static relation path receives H pair but not explicit relative point/camera geometry",
            "required_probe": "causal relation task where identical local appearance requires relative spatial reasoning",
        },
        {
            "id": "C3_RAY_VISIBILITY_WIRING",
            "class": "IMPLEMENTATION_WIRING",
            "fact": "complete-ray head exists separately but is not instantiated by main IRIS V3",
            "required_probe": "trace training and product producer path before any visibility learning claim",
        },
        {
            "id": "G2_SEQUENCE_WINDOW",
            "class": "SEQUENCE_MEMORY",
            "fact": f"local causal window={tessa_fields['local_attention_window']}, max_faces={tessa_fields['max_faces']}",
            "required_probe": "measure real teacher topology token-length distribution and long-range dependency strata before architectural interpretation",
        },
    ]

    return {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "expected_parent_main_sha": expected_parent_sha,
        "source_hashes": source_hashes,
        "facts": facts,
        "open_risks": open_risks,
        "training_executed": False,
        "model_inference_executed": False,
        "scientific_hypothesis_pass_claimed": False,
        "causal_truth_training_authorized": False,
        "fit8_training_authorized": False,
        "product_authority_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--expected-parent-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo_root.resolve(), args.expected_parent_sha.strip().lower())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
