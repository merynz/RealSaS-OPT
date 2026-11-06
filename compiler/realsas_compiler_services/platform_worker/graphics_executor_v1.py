from __future__ import annotations

"""CAS-native product.graphics executor.

Graphics is the complete visual/presentation domain. CAA is its source-preserving
appearance sublayer; role-free relation evidence supplies fragmentation,
visibility/occlusion and draw-order evidence. Compiler validates/binds these
inputs but does not synthesize missing perception.
"""

import json
from typing import Any, Callable

GRAPHICS_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductGraphicsCompile.v1"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"GRAPHICS_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0]


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("GRAPHICS_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("GRAPHICS_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _graphics_evidence_from_dict(payload: dict[str, Any]):
    from compiler.realsas_compiler_core.graphics_v1 import (
        GraphicsEvidenceV1,
        GraphicsStaticRelationEvidenceV1,
        GraphicsViewRelationEvidenceV1,
        graphics_evidence_hash_v1,
    )

    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.GraphicsEvidence.v1":
        raise RuntimeError("GRAPHICS_EVIDENCE_SCHEMA_DRIFT")
    static = tuple(
        GraphicsStaticRelationEvidenceV1(
            face_a=int(row["face_a"]),
            face_b=int(row["face_b"]),
            same_drawable_affinity=float(row["same_drawable_affinity"]),
            cut_probability=float(row["cut_probability"]),
            attachment_affinity=float(row["attachment_affinity"]),
            alpha_boundary_probability=float(row["alpha_boundary_probability"]),
            appearance_boundary_probability=float(row["appearance_boundary_probability"]),
            confidence=float(row["confidence"]),
            evidence_ref=str(row.get("evidence_ref") or ""),
            schema_version=str(
                row.get("schema_version") or "RealSaS.GraphicsStaticRelationEvidence.v1"
            ),
        )
        for row in payload.get("static_relations") or ()
    )
    view = tuple(
        GraphicsViewRelationEvidenceV1(
            view_index=int(row["view_index"]),
            face_a=int(row["face_a"]),
            face_b=int(row["face_b"]),
            front_probability=float(row["front_probability"]),
            occlusion_probability=float(row["occlusion_probability"]),
            visibility_probability_a=float(row["visibility_probability_a"]),
            visibility_probability_b=float(row["visibility_probability_b"]),
            confidence=float(row["confidence"]),
            evidence_ref=str(row.get("evidence_ref") or ""),
            schema_version=str(
                row.get("schema_version") or "RealSaS.GraphicsViewRelationEvidence.v1"
            ),
        )
        for row in payload.get("view_relations") or ()
    )
    value = GraphicsEvidenceV1(
        mesh_binding_hash=str(payload["mesh_binding_hash"]),
        static_relations=static,
        view_relations=view,
        evidence_hash=str(payload["evidence_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.evidence_hash != graphics_evidence_hash_v1(value):
        raise RuntimeError("GRAPHICS_EVIDENCE_HASH_DRIFT")
    return value


def _indexed_faces(mesh) -> tuple[tuple[int, int, int], ...]:
    index_by_id = {
        str(vertex.canonical_mesh_vertex_id): index
        for index, vertex in enumerate(mesh.vertices)
    }
    if len(index_by_id) != len(mesh.vertices):
        raise RuntimeError("GRAPHICS_MESH_VERTEX_ID_DUPLICATE")
    try:
        return tuple(
            tuple(index_by_id[str(vertex_id)] for vertex_id in face)
            for face in mesh.faces
        )
    except KeyError as exc:
        raise RuntimeError("GRAPHICS_MESH_FACE_VERTEX_UNKNOWN") from exc


def execute_graphics_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != GRAPHICS_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("GRAPHICS_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("GRAPHICS_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("GRAPHICS_DIRECT_MODEL_RELEASE_FORBIDDEN")

    from compiler.realsas_compiler_core.appearance_authority_v2 import (
        complete_appearance_asset_from_dict,
    )
    from compiler.realsas_compiler_core.canonical_graphics_ir_v1 import (
        mint_canonical_graphics_v1,
    )
    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
        canonical_mechanics_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        canonical_mesh_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
        canonical_skin_from_dict_v2,
    )
    from compiler.realsas_compiler_core.graphics_v1 import (
        GraphicsCompilePolicyV1,
        compile_graphics_v1,
    )

    mesh = canonical_mesh_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMeshIR", "v1"))
    )
    skin = canonical_skin_from_dict_v2(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalSkinIR", "v2"))
    )
    mechanics = canonical_mechanics_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMechanicsIR", "v1"))
    )
    evidence = _graphics_evidence_from_dict(
        _json_payload(read_object, _one(request, "RealSaS.GraphicsEvidence", "v1"))
    )
    appearance = complete_appearance_asset_from_dict(
        _json_payload(read_object, _one(request, "RealSaS.CompleteAppearanceAssetIR", "v2"))
    )

    policy_payload = dict(metadata.get("presentation_compile_policy") or {})
    policy = GraphicsCompilePolicyV1(
        same_drawable_min=float(policy_payload.get("same_drawable_min", 0.65)),
        cut_max=float(policy_payload.get("cut_max", 0.35)),
        boundary_max=float(policy_payload.get("boundary_max", 0.50)),
        relation_confidence_min=float(
            policy_payload.get("relation_confidence_min", 0.50)
        ),
        front_probability_min=float(policy_payload.get("front_probability_min", 0.65)),
        occlusion_probability_min=float(
            policy_payload.get("occlusion_probability_min", 0.50)
        ),
        visibility_probability_min=float(
            policy_payload.get("visibility_probability_min", 0.50)
        ),
        attachment_affinity_min=float(
            policy_payload.get("attachment_affinity_min", 0.70)
        ),
        mechanical_contact_min=float(
            policy_payload.get("mechanical_contact_min", 0.75)
        ),
        support_band_alpha_boundary_min=float(
            policy_payload.get("support_band_alpha_boundary_min", 0.65)
        ),
        maximum_support_band_pixels=int(
            policy_payload.get("maximum_support_band_pixels", 2)
        ),
    )

    presentation = compile_graphics_v1(
        canonical_mesh_binding_hash=mesh.mesh_lineage_hash,
        canonical_mesh_skin_binding_hash=skin.skin_lineage_hash,
        source_graphics_binding_hash=appearance.asset_hash,
        faces=_indexed_faces(mesh),
        evidence=evidence,
        mechanical_attachment_evidence=None,
        policy=policy,
    )
    graphics = mint_canonical_graphics_v1(
        mesh,
        skin,
        mechanics,
        presentation=presentation,
        appearance=appearance,
    )
    sealed = put_bytes(_json_bytes(graphics))
    output = {
        "role": "canonical_graphics",
        "artifact_type": "RealSaS.CanonicalGraphicsIR",
        "schema_version": "v1",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "COMPILER_CANONICAL_GRAPHICS",
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "schema": "RealSaS.ProductGraphicsDiagnostics.v1",
            "graphics_lineage_hash": graphics.graphics_lineage_hash,
            "presentation_graph_hash": graphics.presentation_graph_hash,
            "appearance_asset_hash": graphics.appearance_asset_hash,
            "fragment_count": len(graphics.presentation.fragments),
            "view_count": len(graphics.presentation.views),
            "appearance_is_caa_sublayer": True,
            "graphics_owns_all_visual_presentation": True,
            "compiler_perception_synthesis_performed": False,
            "product_revision_pass_claimed": False,
        },
        "failure": None,
    }


__all__ = ["GRAPHICS_EXECUTOR_CONTRACT_V1", "execute_graphics_capability_v1"]
