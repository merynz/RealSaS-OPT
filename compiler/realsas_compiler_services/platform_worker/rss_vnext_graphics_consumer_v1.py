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
from compiler.realsas_compiler_services.platform_worker.rss_vnext_rig_executor_v3 import (
    _cas, _one,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_product_fanout_v1 import (
    bind_rss_vnext_fanout_refs_v1,
    read_rss_vnext_neural_evidence_v1,
    validate_rss_vnext_fanout_index_v1,
)


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
