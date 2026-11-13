from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import io

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import _codec_config


def _png_from_mask(mask):
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[:, :, :3] = (180, 35, 15)
    rgba[:, :, 3] = np.asarray(mask, dtype=np.uint8) * 255
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    return out.getvalue()


def _triangle_source_and_scene():
    cfg = TESSAStructuralizerConfigV3(
        coordinate_bins=128,
        d_model=32, n_heads=4, decoder_layers=2,
        scene_memory_tokens=16, surface_memory_tokens=32,
        presentation_memory_tokens=16, conditioning_token_budget=64,
        memory_source_width=16, memory_refinement_layers=0,
        local_attention_window=128, query_chunk_size=16,
        gradient_checkpointing=False,
    )
    q = (30, 25, 64, 98, 25, 64, 64, 104, 64)
    tokens = (
        cfg.BOS, cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN,
        cfg.FACE_BREAK, *q, cfg.CHART_END, cfg.COMPONENT_END, cfg.EOS,
    )
    decoded = decode_tessa_asset_sequence_v1(tokens, cfg=_codec_config(cfg))
    base = decoded.vertices_normalized * 2.
    face = decoded.faces
    originals = []
    for i in range(2):
        points = base[:, :2].copy()
        points[:, 0] *= (1. if i == 0 else -1.)
        points[:, 1] *= -1.
        low = points.min(axis=0)
        high = points.max(axis=0)
        scale = 37. / max(high - low)
        projected = (points - .5 * (high + low)) * scale + 32.
        triangles = [tuple(tuple(float(x) for x in projected[int(k)]) for k in tri)
                     for tri in face]
        mask = np.frombuffer(rasterize_triangles_half_integer_top_left(
            triangles, width=64, height=64,
        ), dtype=np.uint8).reshape(64, 64).copy()
        originals.append(_png_from_mask(mask))
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=tuple(originals),
        source_rights_claim_sha256="a" * 64,
        view_claims=("FRONT", "BACK"),
    )
    scene = {
        "tessa_topology_token_ids": np.asarray([tokens], dtype=np.int64),
        "tessa_candidate_base_vertices": np.asarray([base], dtype=np.float32),
        "tessa_candidate_rest_vertices": np.asarray([base], dtype=np.float32),
        "tessa_candidate_faces": face.astype(np.int64),
        "tessa_candidate_vertex_components": np.asarray(decoded.vertex_component_indices, dtype=np.int64),
        "tessa_candidate_face_components": np.asarray(decoded.face_component_indices, dtype=np.int64),
    }
    receipt = {
        "tessa_decoder_config": asdict(cfg),
        "source_binding_sha256": source.source_binding_sha256,
        "checkpoint_sha256": "b" * 64,
    }
    return source, scene, receipt


def _run(source, scene, receipt):
    return admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest="c" * 64,
    )


def test_true_original_alpha_views_mint_reopenable_static_canonical_mesh():
    source, scene, receipt = _triangle_source_and_scene()
    mesh, static, court = _run(source, scene, receipt)
    assert mesh is not None
    assert static is not None
    assert court["source_static_silhouette_qualified"] is True
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1
    assert canonical_mesh_from_dict_v1(mesh.to_dict()) == mesh
    assert not court["original_camera_calibrated"]
    assert not court["camera_hypotheses_are_observed"]
    assert not court["source_rgb_surface_ownership_qualified"]
    assert not court["product_revision_minted"]
    assert static.qualification_report["source_fidelity_scope"] == (
        "TWO_ORIGINAL_ALPHA_MASKS_WITH_FITTED_CAMERA_HYPOTHESES"
    )


def test_raw_mesh_real_source_shape_mismatch_rejects_even_consistent_tessa():
    source, scene, receipt = _triangle_source_and_scene()
    opaque = np.zeros((64, 64), dtype=np.uint8)
    opaque[14:50, 15:47] = 1
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=(source.original_files[0], _png_from_mask(opaque)),
        source_rights_claim_sha256=source.source_rights_claim_sha256,
        view_claims=source.view_claims,
    )
    mesh, static, court = _run(source, scene, receipt)
    assert mesh is None and static is None
    assert court["view_results"][0]["accepted"]
    assert not court["view_results"][1]["accepted"]


def test_raw_mesh_rejects_opaque_scene_as_unknown_character_silhouette():
    source, scene, receipt = _triangle_source_and_scene()
    rgba = np.ones((64, 64, 4), dtype=np.uint8) * 255
    opaque = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(opaque, format="PNG")
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=(opaque.getvalue(), source.original_files[1]),
        source_rights_claim_sha256=source.source_rights_claim_sha256,
        view_claims=("FRONT", "BACK"),
    )
    mesh, static, court = _run(source, scene, receipt)
    assert mesh is None and static is None
    assert court["view_results"][0]["blocker"] == "SOURCE_ALPHA_IS_EMPTY_OR_FULL_FRAME"


def test_raw_mesh_requires_original_front_back_claim_for_static_hypothesis():
    source, scene, receipt = _triangle_source_and_scene()
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=source.original_files,
        source_rights_claim_sha256=source.source_rights_claim_sha256,
        view_claims=("UNKNOWN", "UNKNOWN"),
    )
    mesh, static, court = _run(source, scene, receipt)
    assert mesh is None and static is None
    assert all(not row["accepted"] for row in court["view_results"])


def test_raw_mesh_cannot_reinterpret_forged_tessa_triangle_as_valid_token_decode():
    source, scene, receipt = _triangle_source_and_scene()
    scene["tessa_candidate_faces"] = np.array([[0, 2, 1]], dtype=np.int64)
    with pytest.raises(RuntimeError, match="TOKEN_SURFACE_MISMATCH"):
        _run(source, scene, receipt)
    source, scene, receipt = _triangle_source_and_scene()
    receipt.pop("tessa_decoder_config")
    with pytest.raises(RuntimeError, match="CONFIG_NOT_PINNED"):
        _run(source, scene, receipt)
