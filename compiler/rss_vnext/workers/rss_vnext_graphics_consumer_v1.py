from __future__ import annotations

"""One-Shape-inference RSS.vNext Graphics candidate rebind.

The native Graphics worker may consume the same immutable source/evidence CAS
that created the canonical puppet. It must not require a manually submitted
GraphicsEvidence artifact, reload a neural checkpoint or pretend that region
affinities are source-observed cut/occlusion/appearance truths.
"""

from hashlib import sha256
import json
from typing import Any, Callable

from compiler.realsas_compiler_core.graphics_v1 import (
    compile_neural_region_graphics_on_canonical_puppet_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_cas_parent_v1 import (
    select_rss_vnext_parent_v1 as _one,
    read_rss_vnext_parent_cas_v1 as _cas,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_product_fanout_v1 import (
    bind_rss_vnext_fanout_refs_v1,
    read_rss_vnext_neural_evidence_v1,
    validate_rss_vnext_fanout_index_v1,
)


def _raw_two_image_graphics_v1(
    request: dict[str, Any], *, canonical_mesh, canonical_rig,
    canonical_skin, read_object: Callable,
):
    """Consume real raw Shape H(x) candidate channels on admitted TESSA faces.

    The original two image alphas qualify only a static mesh silhouette.
    Learned 3D slot/face relations cannot turn into source color, view
    occlusion, material ownership, drawable truth or attachment contact.
    """
    import numpy as np
    from dataclasses import replace
    from compiler.realsas_compiler_core.hashing import content_sha256
    from compiler.realsas_compiler_core.graphics_v1 import (
        _rss_typed_face_graphics_candidates_v1,
        build_graphics_evidence_v1,
        compile_graphics_on_canonical_puppet_v1,
        compiled_graphics_hash_v1,
    )
    from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
        decode_uncalibrated_two_image_source_v1,
    )
    from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
        RAW_SOURCE, RAW_SCENE, RAW_RECEIPT, RAW_VIEWS, RAW_CONFIG, RAW_DECODER,
        SCENE_REQUIRED, MESH_KEYS, AXIS_MIRA_KEYS, RAW_GRAPHICS_KEYS,
        _archive, _validate_candidate,
    )
    from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
        admit_raw_two_view_mesh_v1,
    )
    rows = request.get("input_artifacts") or ()
    originals = [
        row for row in rows if type(row) is dict
        and row.get("artifact_type") == RAW_SOURCE
        and row.get("schema_version") == "v1"
    ]
    if len(originals) not in (1, 2) or any(
        x.get(k) != originals[0].get(k)
        for x in originals[1:]
        for k in ("content_sha256", "size_bytes", "storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_ORIGINAL_SOURCE_BINDING")
    source_ref = originals[0]
    source = decode_uncalibrated_two_image_source_v1(_cas(
        source_ref, read_object, limit=70 * 1024 * 1024,
    ))
    required = {
        RAW_SCENE: "v1", RAW_RECEIPT: "v1", RAW_VIEWS: "v1",
        RAW_CONFIG: "v1", RAW_DECODER: "v1",
    }
    parents = {}
    for kind, version in required.items():
        parents[kind] = _one(request, kind, version)
    receipt_data = _cas(parents[RAW_RECEIPT], read_object, limit=2 * 1024 * 1024)
    try:
        receipt = json.loads(receipt_data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_RECEIPT_NOT_JSON") from exc
    if (
        type(receipt) is not dict
        or receipt.get("source_content_sha256") != source_ref["content_sha256"]
        or receipt.get("source_binding_sha256") != source.source_binding_sha256
        or receipt.get("scene_content_sha256") != parents[RAW_SCENE]["content_sha256"]
        or receipt.get("generated_views_content_sha256") != parents[RAW_VIEWS]["content_sha256"]
        or receipt.get("unified_model_config_content_sha256") != parents[RAW_CONFIG]["content_sha256"]
        or receipt.get("embedded_mira_decoder_content_sha256") != parents[RAW_DECODER]["content_sha256"]
        or receipt.get("original_camera_calibrated") is not False
        or receipt.get("source_ray_evidence_qualified") is not False
        or receipt.get("product_revision_minted") is not False
    ):
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_SOURCE_OR_CHECKPOINT_PARENT_DRIFT")
    scene_ref = parents[RAW_SCENE]
    scene = _archive(
        _cas(scene_ref, read_object),
        required=SCENE_REQUIRED | {"region_relation_valid_mask"},
        allowed=SCENE_REQUIRED | MESH_KEYS | AXIS_MIRA_KEYS | RAW_GRAPHICS_KEYS,
    )
    measured = _validate_candidate(scene, receipt)
    if measured["mesh_state"] != "UNQUALIFIED_MESH_CANDIDATE":
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_TOPOLOGY_ABSTAIN")
    replay_mesh, _, source_measurement = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest=scene_ref["content_sha256"],
    )
    if (
        replay_mesh is None or replay_mesh != canonical_mesh
        or not source_measurement["source_static_silhouette_qualified"]
        or canonical_rig.mesh_lineage_hash != canonical_mesh.mesh_lineage_hash
        or canonical_skin.mesh_lineage_hash != canonical_mesh.mesh_lineage_hash
        or canonical_skin.deform_structure_hash != canonical_rig.deform_structure_hash
    ):
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_CANONICAL_PARENT_NOT_REPLAYED")

    n = len(canonical_mesh.vertices)
    face_rows = scene["tessa_candidate_faces"]
    region_logits = scene["region_relation_logits"]
    region_valid = scene["region_relation_valid_mask"]
    membership = scene["tessa_candidate_semantic_logits"]
    if (
        region_logits.ndim != 4 or region_logits.shape[0] != 1
        or region_logits.shape[1] < 2
        or region_logits.shape[1] != region_logits.shape[2]
        or region_logits.shape[3] != 3
        or region_valid.shape != region_logits.shape[:3]
        or region_valid.dtype.kind not in "iu"
        or not np.isin(region_valid, [0, 1]).all()
        or membership.ndim != 3 or membership.shape != (
            1, n, region_logits.shape[1] + 1
        )
        or len(face_rows) != len(canonical_mesh.faces)
        or len(canonical_mesh.faces) > 32768
    ):
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_TYPED_REGION_ABI_INVALID")
    original_faces = tuple(
        tuple(canonical_mesh.vertices[int(i)].canonical_mesh_vertex_id for i in row)
        for row in face_rows
    )
    if original_faces != canonical_mesh.faces:
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_TESSA_FACE_WINDING_OR_ORDER_DRIFT")

    # Re-evaluate anonymous face occupancy, with the UNKNOWN channel intact.
    logits = membership[0].astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    face_probs = {}
    for idx, triangle in enumerate(face_rows):
        face_mean = probs[triangle].mean(axis=0)
        face_probs[idx] = (
            tuple(float(v) for v in face_mean[:-1]),
            float(face_mean[-1]),
        )
    edge_faces = {}
    for i, face in enumerate(canonical_mesh.faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((a, b))), []).append(i)
    pairs = set()
    for indices in edge_faces.values():
        for i, a in enumerate(indices):
            for b in indices[i + 1:]:
                pairs.add(tuple(sorted((a, b))))
    if len(pairs) > 4096:
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_FACE_RELATION_BUDGET")
    source_evidence_hash = content_sha256({
        "schema": "RealSaS.RSSvNext.RawGraphicsEvidenceBinding.v1",
        "original_source_sha256": source_ref["content_sha256"],
        "scene_sha256": scene_ref["content_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "canonical_mesh_lineage_hash": canonical_mesh.mesh_lineage_hash,
    })
    rows, coverage = _rss_typed_face_graphics_candidates_v1(
        scene={
            "region_relation_logits": region_logits[0].astype(float).tolist(),
            "region_relation_valid_mask": region_valid[0].astype(bool).tolist(),
        },
        face_probabilities=face_probs,
        adjacent_pairs=tuple(sorted(pairs)),
        evidence_hash=source_evidence_hash,
    )
    evidence = build_graphics_evidence_v1(
        mesh_binding_hash=canonical_mesh.mesh_lineage_hash,
        static_relations=rows,
        view_relations=(),
        metadata={
            "source_neural_evidence_hash": source_evidence_hash,
            "real_source_static_alpha_replayed": True,
            "source_RGB_appearance_not_qualified": True,
            "candidate_region_unknown_retained": True,
            "typed_candidate_pair_coverage": coverage,
            "source_2d_drawable_truth": False,
            "predicted_source_occlusion_not_observed": True,
            "original_source_camera_observed": False,
        },
    )
    compiled = compile_graphics_on_canonical_puppet_v1(
        mesh=canonical_mesh, rig=canonical_rig, skin=canonical_skin,
        source_graphics_binding_hash=source_evidence_hash,
        evidence=evidence,
    )
    from compiler.realsas_compiler_core.graphics_v1 import GraphicsViewStateV1
    if compiled.views:
        raise RuntimeError("RSS_VNEXT_RAW_GRAPHICS_UNOBSERVED_VIEW_LABEL_DRIFT")
    # The C++ Graphics transport requires a complete ordered view domain.
    # The two user image claims and the six *predicted* source-normalized
    # directions are not measured camera rays or source occlusion. Every
    # canonical fragment stays eligible for geometry-depth evaluation:
    # visible_fragment_ids is a candidate execution mask, not a learned
    # visibility label or measured per-view alpha ownership.
    candidates = tuple(sorted(f.fragment_id for f in compiled.fragments))
    virtual_views = tuple(
        GraphicsViewStateV1(
            view_index=i,
            visible_fragment_ids=candidates,
            order_edges=(),
            metadata={
                "camera_authority": "VIRTUAL_PRODUCT_DIRECTION_NOT_SOURCE_CAMERA",
                "visibility_known_from_original_source": False,
                "visibility_unknown_fragment_ids": candidates,
                "unobserved_visibility_raster_policy": "DRAW_GEOMETRY_AND_DEFER_OCCLUSION_TO_DEPTH",
                "source_order_edges_observed": False,
                "pose_conditioned_source_visibility_qualified": False,
                "view_was_generated_or_hypothesized": True,
            },
        ) for i in range(8)
    )
    bound = replace(compiled, views=virtual_views, metadata={
        **compiled.metadata,
        "origin": "RAW_TWO_IMAGE_UNIFIED_H_TYPED_GRAPHICS_CANDIDATES",
        "source_static_silhouette_passed": True,
        "qualified_drawable_source_truth": False,
        "qualified_source_occlusion": False,
        "qualified_source_attachments": False,
        "independent_editability_proven": False,
        "source_pixels_win_where_observed": False,
        "source_rgb_surface_owner_qualified": False,
        "camera_hypotheses_not_observations": True,
        "eight_virtual_runtime_views_not_source_observations": True,
        "source_view_visibility_unqualified": True,
        "unobserved_draw_order_deferred_to_geometry_depth": True,
        "typed_candidate_pair_coverage": coverage,
    }, graph_hash="")
    compiled = replace(bound, graph_hash=compiled_graphics_hash_v1(bound))
    receipt = {
        "schema": "RealSaS.RSSvNext.GraphicsNativeCandidateReceipt.v1",
        "unified_checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_binding_sha256": source.source_binding_sha256,
        "source_evidence_binding_court_hash": content_sha256(source_measurement),
        "neural_evidence_hash": source_evidence_hash,
        "canonical_mesh_lineage_hash": canonical_mesh.mesh_lineage_hash,
        "canonical_rig_lineage_hash": canonical_rig.rig_lineage_hash,
        "canonical_skin_lineage_hash": canonical_skin.skin_lineage_hash,
        "candidate_graph_hash": compiled.graph_hash,
        "graphics_source_observation_qualified": False,
        "source_hidden_surface_qualified": False,
        "pose_conditioned_visibility_qualified": False,
        "virtual_runtime_view_count": 8,
        "virtual_runtime_views_are_original_observations": False,
        "attachment_socket_authority": False,
        "product_revision_minted": False,
    }
    receipt["receipt_hash"] = content_sha256(receipt)
    return compiled, receipt


def compile_rss_vnext_shape_native_graphics_v1(
    request: dict[str, Any], *,
    canonical_mesh,
    canonical_rig,
    canonical_skin,
    read_object: Callable[[dict[str, Any]], bytes],
):
    """Compile provisional candidate layers from exact original source CAS.

    Source geometry / source edit / dynamic z-order are separate admission
    domains and remain explicitly UNQUALIFIED here.
    """
    if any(
        type(row) is dict
        and row.get("artifact_type") == "RealSaS.RSSvNextUncalibratedTwoImageSource"
        for row in request.get("input_artifacts") or ()
    ):
        if request.get("model_releases"):
            raise RuntimeError("RSS_VNEXT_GRAPHICS_SECOND_MODEL_RELEASE_FORBIDDEN")
        return _raw_two_image_graphics_v1(
            request, canonical_mesh=canonical_mesh,
            canonical_rig=canonical_rig, canonical_skin=canonical_skin,
            read_object=read_object,
        )
    if request.get("model_releases"):
        # A downstream consumer never runs another checkpoint. Inherited
        # envelope model refs should be absent under the unified catalog.
        raise RuntimeError("RSS_VNEXT_GRAPHICS_SECOND_MODEL_RELEASE_FORBIDDEN")
    ref = _one(request, "RealSaS.RSSvNextTwoViewSource", "v1")
    source = validate_rss_vnext_source_ref_v1(ref, _cas(ref, read_object))
    index_ref = _one(
        request, "RealSaS.RSSvNextProductEvidenceFanout", "v1",
    )
    raw = _cas(index_ref, read_object, limit=8 * 1024 * 1024)
    try:
        claimed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RSS_VNEXT_GRAPHICS_INDEX_JSON_INVALID") from exc
    if type(claimed) is not dict:
        raise RuntimeError("RSS_VNEXT_GRAPHICS_INDEX_OBJECT_REQUIRED")
    index = validate_rss_vnext_fanout_index_v1(
        raw,
        expected_source_content_sha256=ref["content_sha256"],
        expected_source_binding_sha256=source.source_binding_sha256,
        expected_checkpoint_sha256=claimed.get("unified_checkpoint_sha256"),
        expected_model_release_behavior_sha256=claimed.get(
            "unified_model_release_behavior_sha256"
        ),
    )
    children = bind_rss_vnext_fanout_refs_v1(
        index,
        input_artifacts=list(request.get("input_artifacts") or ()),
    )
    evidence = read_rss_vnext_neural_evidence_v1(
        index, child_refs=children, read_object=read_object,
    )
    binding = validate_rss_vnext_source_evidence_binding_v1(
        source=source,
        evidence=evidence,
        expected_source_binding_sha256=index["source_binding_sha256"],
    )
    if (
        evidence.checkpoint_sha256 != index["unified_checkpoint_sha256"]
        or evidence.evidence_hash not in canonical_mesh.proposal_evidence_refs
        or canonical_rig.mesh_lineage_hash != canonical_mesh.mesh_lineage_hash
        or canonical_skin.mesh_lineage_hash != canonical_mesh.mesh_lineage_hash
    ):
        raise RuntimeError("RSS_VNEXT_GRAPHICS_SOURCE_OR_CANONICAL_PARENT_DRIFT")

    # TESSA exact order is part of the canonical mesh, not a flexible
    # nearest-neighbor reconstruction. Core Graphics checks every vertex,
    # component and oriented face against the original inference evidence.
    mapping = tuple(
        row.source_candidate_vertex_id for row in canonical_mesh.vertices
    )
    if any(
        name != f"RSSV:{i:08d}" for i, name in enumerate(mapping)
    ):
        raise RuntimeError("RSS_VNEXT_GRAPHICS_TESSA_VERTEX_ORDER_DRIFT")
    compiled = compile_neural_region_graphics_on_canonical_puppet_v1(
        evidence=evidence,
        expected_checkpoint_sha256=index["unified_checkpoint_sha256"],
        candidate_vertex_ids_by_index=mapping,
        mesh=canonical_mesh,
        rig=canonical_rig,
        skin=canonical_skin,
    )
    meta = compiled.metadata
    if any(meta.get(name) is not False for name in (
        "qualified_drawable_source_truth",
        "qualified_source_occlusion",
        "qualified_source_attachments",
        "independent_editability_proven",
        "source_pixels_win_where_observed",
    )):
        raise RuntimeError("RSS_VNEXT_GRAPHICS_PREMATURE_PRESENTATION_AUTHORITY")
    receipt = {
        "schema": "RealSaS.RSSvNext.GraphicsNativeCandidateReceipt.v1",
        "unified_checkpoint_sha256": index["unified_checkpoint_sha256"],
        "source_binding_sha256": source.source_binding_sha256,
        "source_evidence_binding_court_hash": binding.court_hash,
        "neural_evidence_hash": evidence.evidence_hash,
        "canonical_mesh_lineage_hash": canonical_mesh.mesh_lineage_hash,
        "canonical_rig_lineage_hash": canonical_rig.rig_lineage_hash,
        "canonical_skin_lineage_hash": canonical_skin.skin_lineage_hash,
        "candidate_graph_hash": compiled.graph_hash,
        "graphics_source_observation_qualified": False,
        "source_hidden_surface_qualified": False,
        "pose_conditioned_visibility_qualified": False,
        "attachment_socket_authority": False,
        "product_revision_minted": False,
    }
    from compiler.realsas_compiler_core.hashing import content_sha256
    receipt["receipt_hash"] = content_sha256(receipt)
    return compiled, receipt


__all__ = ["compile_rss_vnext_shape_native_graphics_v1"]
