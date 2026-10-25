#!/usr/bin/env python3
"""Static, zero-training audit of the current IRIS/TESSA V3 research apparatus.

This executable seals source-level facts needed before causal Truth Foundry or
FIT8 learning experiments are allowed.  It does not instantiate the large
models, run inference, train, promote a ModelRelease, or mutate product state.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3
from models.iris.vnext.inference_input_v1 import (
    CAMERA_FEATURE_DIM_V1,
    PATCH_GRID_V1,
    PATCH_TOKEN_COUNT_V1,
    VIEW_COUNT_V1,
    build_scene_inputs_from_rgba_v1,
    rgba_to_white_composited_rgb_v1,
)
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.scene_core_v1 import (
    IRISVNextSceneCoreV1,
    SharedSceneTriplaneVNextV1,
)
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v2.model_v2 import _TypedMemoryStreamV2, WindowedCausalSelfAttentionV2
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSATriStreamMemoryV3


AUDIT_SCHEMA = "RealSaS.IRISTESSAV3ApparatusStaticContractAudit.v1"
AUDIT_STATUS = "STATIC_FACTS_SEALED__READINESS_NOT_DECIDED"

_SOURCE_PATHS = (
    "models/iris/vnext/inference_input_v1.py",
    "models/iris/vnext/contracts_v1.py",
    "models/iris/vnext/scene_core_v1.py",
    "models/iris/vnext/model_v3.py",
    "models/iris/vnext/heads_v3.py",
    "models/iris/vnext/presentation_evidence_v1.py",
    "models/iris/vnext/structure_v4.py",
    "models/tessa/v2/model_v2.py",
    "models/tessa/v2/objective_v2.py",
    "models/tessa/v3/config_v3.py",
    "models/tessa/v3/model_v3.py",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(obj: Any) -> str:
    return inspect.getsource(obj)


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_DRIFT:" + label)


def audit(repo_root: Path, expected_parent_sha: str) -> dict[str, Any]:
    cfg = IRISVNextConfigV1()
    tcfg = TESSAStructuralizerConfigV3()
    cfg.validate()
    tcfg.validate()

    rgba_source = _source(rgba_to_white_composited_rgb_v1)
    frontend_source = _source(build_scene_inputs_from_rgba_v1)
    scene_source = _source(SharedSceneTriplaneVNextV1)
    scene_core_source = _source(IRISVNextSceneCoreV1)
    model_source = _source(IRISVNextEvidenceModelV3)
    relation_source = _source(GraphicsRelationHeadV3)
    static_relation_source = _source(GraphicsRelationHeadV3.static_relations)
    view_relation_source = _source(GraphicsRelationHeadV3.view_relations)
    bridge_source = _source(IRISVNextStructuralizationBridgeV4)
    typed_stream_source = _source(_TypedMemoryStreamV2)
    tri_stream_source = _source(TESSATriStreamMemoryV3)
    window_source = _source(WindowedCausalSelfAttentionV2)

    # These are source-contract invariants, not scientific conclusions.
    _require(VIEW_COUNT_V1 == 8, "VIEW_COUNT")
    _require(PATCH_GRID_V1 == 37, "PATCH_GRID")
    _require(PATCH_TOKEN_COUNT_V1 == PATCH_GRID_V1 * PATCH_GRID_V1, "PATCH_TOKEN_COUNT")
    _require(CAMERA_FEATURE_DIM_V1 == 13, "CAMERA_FEATURE_DIM")
    _require("rgb = rgba[:, :, :3] * alpha + (1.0 - alpha)" in rgba_source, "RGBA_WHITE_COMPOSITE")
    _require("extract_dinov2_s_patch_map_v2(dino_model, rgb)" in frontend_source, "DINO_RGB_FRONTEND")
    _require("nn.TransformerDecoder" in scene_source, "SCENE_QUERY_DECODER")
    _require("memory).reshape(b, 3 * p * p" not in scene_source, "SOURCE_TOKEN_SELF_ATTENTION_UNEXPECTED")
    _require("self.upsample = nn.ConvTranspose2d" in scene_source, "TRIPLANE_UPSAMPLE")
    _require("stride=2" in scene_source, "TRIPLANE_UPSAMPLE_STRIDE")
    _require("FROZEN_DINOV2_S_IMAGE_TOKENS_PLUS_CAMERAS" in scene_core_source, "SCENE_INPUT_SEMANTICS")

    _require("self.mechanical_demand" in model_source, "MECHANICAL_HEAD")
    _require("self.appearance" in model_source, "APPEARANCE_HEAD")
    _require("self.relations" in model_source, "RELATION_HEAD")
    _require("RayStackVisibilityHeadV3" not in model_source, "RAY_HEAD_NOT_INSTANTIATED")

    _require("feature_a" in static_relation_source and "feature_b" in static_relation_source, "STATIC_H_PAIR")
    _require("point_a" not in static_relation_source and "point_b" not in static_relation_source, "STATIC_NO_POINT_GEOMETRY")
    _require("camera_features" not in static_relation_source, "STATIC_NO_CAMERA")
    _require("relation_camera_geometry_v3" in view_relation_source, "VIEW_EXPLICIT_PAIR_GEOMETRY")
    _require("camera_features" in view_relation_source, "VIEW_CAMERA_ACCESS")

    _require("presentation" in bridge_source, "BRIDGE_PRESENTATION_STREAM")
    _require("nn.MultiheadAttention" in typed_stream_source, "TYPED_STREAM_CROSS_ATTENTION")
    _require("TransformerEncoder" in typed_stream_source, "TYPED_STREAM_REFINEMENT")
    _require("self.scene(" in tri_stream_source, "TESSA_SCENE_STREAM")
    _require("self.surface(" in tri_stream_source, "TESSA_SURFACE_STREAM")
    _require("self.presentation(" in tri_stream_source, "TESSA_PRESENTATION_STREAM")
    _require("torch.cat((scene, surface, presentation), dim=1)" in tri_stream_source, "TESSA_LATE_CONCAT")
    _require("local_attention_window" in window_source, "TESSA_LOCAL_WINDOW")

    source_hashes = {}
    for rel in _SOURCE_PATHS:
        path = repo_root / rel
        if not path.is_file():
            raise RuntimeError("IRIS_TESSA_V3_STATIC_AUDIT_SOURCE_MISSING:" + rel)
        source_hashes[rel] = _sha256(path)

    plane_output_resolution = int(cfg.plane_size) * 2
    conditioning_tokens = {
        "scene": int(tcfg.scene_memory_tokens),
        "surface_mechanics": int(tcfg.surface_memory_tokens),
        "presentation": int(tcfg.presentation_memory_tokens),
        "total": int(tcfg.total_conditioning_tokens),
    }

    facts = {
        "observation": {
            "view_count": VIEW_COUNT_V1,
            "native_rgba_resolution": [1024, 1024],
            "foundation_frontend": "WHITE_COMPOSITED_RGB_TO_FROZEN_DINOV2_S",
            "alpha_independent_foundation_channel": False,
            "dino_patch_grid": [PATCH_GRID_V1, PATCH_GRID_V1],
            "dino_tokens_per_view": PATCH_TOKEN_COUNT_V1,
            "camera_feature_dim": CAMERA_FEATURE_DIM_V1,
        },
        "iris_scene": {
            "scene_dim": int(cfg.scene_dim),
            "plane_query_resolution": [int(cfg.plane_size), int(cfg.plane_size)],
            "plane_channels": int(cfg.plane_channels),
            "triplane_output_resolution": [plane_output_resolution, plane_output_resolution],
            "local_h_dim": int(cfg.local_feature_dim),
            "source_token_cross_view_self_attention_before_scene_queries": False,
            "multi_view_interaction_route": "LEARNED_SCENE_QUERIES_ATTEND_FLATTENED_ALL_VIEW_MEMORY",
        },
        "iris_relations": {
            "static_outputs": list(GraphicsRelationHeadV3.STATIC_OUTPUTS),
            "view_outputs": list(GraphicsRelationHeadV3.VIEW_OUTPUTS),
            "static_pair_inputs": ["H_a", "H_b"],
            "static_explicit_point_geometry": False,
            "static_explicit_camera_geometry": False,
            "view_explicit_pair_geometry": True,
            "view_explicit_camera_features": True,
            "ray_stack_visibility_head_instantiated_by_main_v3": False,
        },
        "tessa": {
            "d_model": int(tcfg.d_model),
            "heads": int(tcfg.n_heads),
            "decoder_layers": int(tcfg.decoder_layers),
            "conditioning_tokens": conditioning_tokens,
            "stream_compression": "INDEPENDENT_TYPED_PERCEIVER_STYLE_BEFORE_CONCAT",
            "cross_stream_interaction_before_independent_compression": False,
            "local_attention_window": int(tcfg.local_attention_window),
            "query_chunk_size": int(tcfg.query_chunk_size),
            "max_faces": int(tcfg.max_faces),
            "max_vertices": int(tcfg.max_vertices),
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
            "fact": f"foundation grid={PATCH_GRID_V1}x{PATCH_GRID_V1}; triplane={plane_output_resolution}x{plane_output_resolution}",
            "required_probe": "controlled thin/separated/interleaving structure accessibility at H(x)",
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
            "fact": f"local causal window={tcfg.local_attention_window}, max_faces={tcfg.max_faces}",
            "required_probe": "measure real teacher topology token-length distribution and long-range dependency strata",
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
