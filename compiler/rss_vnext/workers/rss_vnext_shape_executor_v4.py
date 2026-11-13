from __future__ import annotations

"""One product.shape role: real raw two-image construction or calibrated V3.

The raw branch actually runs the pinned canonical diffusion -> six-view
RGB/normal diffusion -> same IRIS/TESSA/AXIS/MIRA scene. Outputs are strictly
CANDIDATES, since the raw user's original cameras and pixel->3D depth are
unknown. No synthetic source ray or pseudo-calibrated .npz is exported.

The legacy calibrated V3 branch remains executable under the V4 active role.
Downstream mesh/graphics/proof cannot silently consume raw candidates as the
calibrated V1 evidence ABI; that requires separate qualification work.
"""

from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    materialize_model_releases,
)
from .rss_vnext_input_executor_v3 import RAW_ARTIFACT_TYPE
from .rss_vnext_shape_executor_v3 import (
    execute_rss_vnext_shape_capability_v3, MODEL_FAMILY,
    MODEL_ARCHITECTURE, CHECKPOINT_ARTIFACT_TYPE,
    _production_vision_factory,
)

EXECUTOR_CONTRACT = "RealSaS.ProductShapeInference.v4"
RAW_SCENE_ARTIFACT = "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
RAW_GENERATED_VIEWS_ARTIFACT = "RealSaS.RSSvNextGeneratedConditioningViews"
RAW_RECEIPT_ARTIFACT = "RealSaS.RSSvNextUncalibratedConstructionReceipt"
_MAX_RAW_NEURAL_BYTES = 256 * 1024 * 1024


def _cas_ref(
    payload: bytes, *, kind: str, role: str,
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    if not 1 <= len(payload) <= _MAX_RAW_NEURAL_BYTES:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_CAS_OUTPUT_BUDGET")
    receipt = put_bytes(payload)
    if (
        type(receipt) is not dict
        or receipt.get("content_sha256") != sha256(payload).hexdigest()
        or receipt.get("size_bytes") != len(payload)
        or not receipt.get("storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_CAS_OUTPUT_DRIFT")
    return {
        "role": role, "artifact_type": kind, "schema_version": "v1",
        "storage_key": receipt["storage_key"],
        "content_sha256": receipt["content_sha256"],
        "size_bytes": receipt["size_bytes"],
        "authority_class": "LEARNED_RAW_SOURCE_SCENE_CANDIDATE_NO_ORIGINAL_CAMERA",
    }


def _tensor_archive(**arrays) -> bytes:
    extracted = {}
    for key, tensor in arrays.items():
        if tensor is None:
            raise RuntimeError("RSS_VNEXT_SHAPE_V4_REQUIRED_TENSOR_ABSENT:" + key)
        raw = tensor.detach().cpu().contiguous().numpy()
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise RuntimeError("RSS_VNEXT_SHAPE_V4_NONFINITE_OR_TYPED_SCENE:" + key)
        extracted[key] = raw
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **extracted)
    return buffer.getvalue()



def _tessa_candidate_tensors_v1(proposal: dict[str, Any]) -> tuple[dict, dict]:
    """Persist *all* decoded TESSA topology/surface proposals in the H(x) CAS.

    Model inference has already happened. Dropping the topology token stream
    here would force an unpinned second decoding in Mesh or silently erase a
    valid candidate. Neither tokens nor predicted vertices imply canonical
    source, ray, drawable, skin, or product authority.
    """
    import torch
    from models.rss_vnext.animation_ready_mesh import RSSvNextStructuralMeshCandidate

    if type(proposal) is not dict:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_TESSA_PROPOSAL_REQUIRED")
    tokens = proposal.get("structural_token_proposal")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1
        or tokens.dtype not in (torch.int32, torch.int64)
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_TESSA_TOKEN_DOMAIN")
    mesh = proposal.get("mesh_candidate")
    status = proposal.get("mesh_status")
    if status not in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE") or (
        (status == "ABSTAIN") != (mesh is None)
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_TESSA_STATUS_OR_CANDIDATE_DRIFT")
    tensors = {"tessa_topology_token_ids": tokens}
    receipt = {
        "tessa_token_count": int(tokens.shape[1]),
        "tessa_mesh_candidate_present": mesh is not None,
        "tessa_mesh_candidate_vertex_count": 0,
        "tessa_mesh_candidate_face_count": 0,
        "tessa_mesh_candidate_role_schema_id": None,
        "tessa_mesh_candidate_authority": "UNQUALIFIED_LEARNED_PROPOSAL_ONLY",
    }
    if mesh is None:
        return tensors, receipt
    if (
        type(mesh) is not RSSvNextStructuralMeshCandidate
        or mesh.canonical_mesh_authority
        or mesh.source_ray_evidence is not None
        or mesh.source_authority != "LEARNED_PREDICTED_TOPOLOGY"
        or mesh.signed_solid_authority != "UNQUALIFIED"
        or tuple(int(v) for v in tokens[0].tolist()) != mesh.topology_token_ids
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_TESSA_PREMATURE_SOURCE_AUTHORITY")
    mesh.validate()
    tensors.update({
        "tessa_candidate_base_vertices": mesh.base_vertices_iris,
        "tessa_candidate_rest_vertices": mesh.rest_vertices_iris,
        "tessa_candidate_faces": mesh.faces,
        "tessa_candidate_vertex_components": mesh.vertex_component_ids,
        "tessa_candidate_face_components": mesh.face_component_ids,
        "tessa_candidate_local_features": mesh.local_features,
        "tessa_candidate_normals": mesh.normals,
        "tessa_candidate_semantic_logits": mesh.semantic_logits,
        "tessa_candidate_presentation_role_logits": mesh.presentation_role_logits,
        "tessa_candidate_affordance_logits": mesh.affordance_logits,
        "tessa_candidate_graphics_logits": mesh.graphics_logits,
        "tessa_candidate_static_quality_logits": mesh.static_quality_logits,
        "tessa_candidate_unsigned_distance": mesh.geometry["unsigned_distance"],
        "tessa_candidate_sign_logits": mesh.geometry["sign_logit"],
    })
    receipt.update({
        "tessa_mesh_candidate_vertex_count": int(mesh.rest_vertices_iris.shape[1]),
        "tessa_mesh_candidate_face_count": int(mesh.faces.shape[0]),
        "tessa_mesh_candidate_role_schema_id": mesh.presentation_role_schema_id,
    })
    return tensors, receipt


def _raw_source_ref(request: dict[str, Any]) -> dict[str, Any] | None:
    refs = request.get("input_artifacts") or ()
    if not isinstance(refs, (list, tuple)) or any(type(ref) is not dict for ref in refs):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_ARTIFACT_REFS_INVALID")
    raw = [
        ref for ref in refs
        if ref.get("artifact_type") == RAW_ARTIFACT_TYPE
        and ref.get("schema_version") == "v1"
    ]
    calibrated = [
        ref for ref in refs
        if ref.get("artifact_type") == "RealSaS.RSSvNextTwoViewSource"
        and ref.get("schema_version") == "v1"
    ]
    if raw and calibrated:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_TWO_DIFFERENT_SOURCE_AUTHORITIES")
    if not raw:
        return None
    if len(raw) not in (1, 2) or any(
        row.get(key) != raw[0].get(key)
        for row in raw[1:]
        for key in ("content_sha256", "size_bytes", "storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_RAW_GO_DUPLICATE_OR_BYTES_DRIFT")
    return raw[0]


def execute_rss_vnext_shape_capability_v4(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
    test_vision_encoder_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    metadata = request.get("capability_metadata") or {}
    if (
        type(request) is not dict or type(metadata) is not dict
        or request.get("capability_id") != "product.shape"
        or request.get("kind") != "MODULE"
        or metadata.get("executor_contract") != EXECUTOR_CONTRACT
        or not str(request.get("execution_id") or "")
        or (request.get("goal_parameters") or {}).get("product_mode") is False
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_ROLE_OR_MODE_CONTRACT")
    source = _raw_source_ref(request)
    if source is None:
        # The same role accepts legacy calibrated V1 source as an explicitly
        # measured input, without changing its already-tested evidence ABI.
        legacy = {
            **request,
            "capability_metadata": {
                **metadata, "executor_contract": "RealSaS.ProductShapeInference.v3",
            },
        }
        return execute_rss_vnext_shape_capability_v3(
            legacy, read_object=read_object, put_bytes=put_bytes,
            test_vision_encoder_factory=test_vision_encoder_factory,
        )
    from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
        decode_uncalibrated_two_image_source_v1,
    )
    raw = read_object(source)
    if (
        type(raw) is not bytes
        or source.get("content_sha256") != sha256(raw).hexdigest()
        or type(source.get("size_bytes")) is not int
        or source.get("size_bytes") != len(raw)
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_RAW_CAS_DRIFT")
    source_object = decode_uncalibrated_two_image_source_v1(raw)
    models = list(request.get("model_releases") or ())
    if (
        len(models) != 1 or models[0].get("family") != MODEL_FAMILY
        or models[0].get("architecture_id") != MODEL_ARCHITECTURE
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_SINGLE_MODEL_RELEASE_REQUIRED")
    released = materialize_model_releases(
        manifest={"run_id": "rss-vnext-" + str(request["execution_id"])},
        models=models, read_object=read_object,
    )
    if len(released) != 1:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_MODEL_RELEASE_DRIFT")
    model_release = released[0]
    checkpoint = dict(model_release.get("checkpoint") or {})
    if (
        checkpoint.get("artifact_type") != CHECKPOINT_ARTIFACT_TYPE
        or checkpoint.get("schema_version") != "v2"
        or not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_file()
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_THREE_STAGE_CHECKPOINT_REQUIRED")
    vision = (
        _production_vision_factory() if test_vision_encoder_factory is None
        else test_vision_encoder_factory()
    )
    from models.rss_vnext.inference_checkpoint_v1 import load_inference_checkpoint_v1
    from models.rss_vnext.realsas_raw_two_image_inference_v1 import (
        run_uncalibrated_two_image_neural_scene_v1,
    )
    model = load_inference_checkpoint_v1(
        checkpoint["path"],
        expected_sha256=checkpoint["content_sha256"],
        vision_encoder=vision,
        product_mode=test_vision_encoder_factory is None,
    )
    bundle = getattr(model, "_pinned_neural_bundle", None)
    if bundle is None or bundle.checkpoint_sha256 != checkpoint["content_sha256"]:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_SINGLE_NEURAL_BUNDLE_REQUIRED")
    construction = run_uncalibrated_two_image_neural_scene_v1(
        raw_cas_bytes=raw,
        expected_raw_cas_sha256=source["content_sha256"],
        neural_bundle=bundle,
        max_topology_tokens=metadata.get("maximum_new_tokens"),
    )
    if (
        construction.source.source_binding_sha256
        != source_object.source_binding_sha256
        or construction.original_camera_features is not None
        or construction.construction.product_revision_minted
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_PREMATURE_SOURCE_AUTHORITY")
    proposal = construction.construction.proposal
    state = proposal["state"]
    if state.camera_features is not None:
        raise RuntimeError("RSS_VNEXT_SHAPE_V4_SOURCE_CAMERA_WAS_INVENTED")
    tessa_tensors, tessa_receipt = _tessa_candidate_tensors_v1(proposal)
    scene_bytes = _tensor_archive(
        **tessa_tensors,
        scene_triplanes=state.scene_triplanes,
        query_xyz=state.query_xyz,
        local_features=state.local_features,
        semantic_logits=state.semantic_logits,
        graphics_logits=state.graphics_logits,
        presentation_role_logits=state.presentation_role_logits,
        region_relation_logits=state.region_relation_logits,
    )
    generated_bytes = _tensor_archive(
        canonical_rgba=construction.construction.canonical_rgba,
        generated_rgba=construction.construction.generated_rgba,
        generated_normals=construction.construction.generated_normals,
        generated_virtual_cameras=construction.construction.generated_cameras,
    )
    scene_ref = _cas_ref(
        scene_bytes, kind=RAW_SCENE_ARTIFACT,
        role="rss_uncalibrated_neural_scene", put_bytes=put_bytes,
    )
    views_ref = _cas_ref(
        generated_bytes, kind=RAW_GENERATED_VIEWS_ARTIFACT,
        role="rss_predicted_conditioning_views", put_bytes=put_bytes,
    )
    receipt = {
        "schema": "RealSaS.RSSvNext.UncalibratedConstructionReceipt.v1",
        "source_content_sha256": source["content_sha256"],
        "source_binding_sha256": source_object.source_binding_sha256,
        "source_rights_claim_sha256": source_object.source_rights_claim_sha256,
        "source_rights_verified": False,
        "model_release_behavior_identity_sha256": model_release["behavior_identity_sha256"],
        "checkpoint_sha256": checkpoint["content_sha256"],
        "two_stage_checkpoint_binding_sha256": (
            construction.construction.construction_checkpoint_binding_sha256
        ),
        "derived_conditioning_sha256": construction.derived_conditioning_sha256,
        "scene_content_sha256": scene_ref["content_sha256"],
        "generated_views_content_sha256": views_ref["content_sha256"],
        "original_camera_calibrated": False,
        "original_pixel_count": 2,
        "generated_view_count": 6,
        "source_ray_evidence_qualified": False,
        "generated_views_as_original_source": False,
        "canonical_mesh_qualified": False,
        "product_revision_minted": False,
        "mesh_status": proposal["mesh_status"],
        **tessa_receipt,
    }
    receipt_ref = _cas_ref(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
         + "\n").encode(),
        kind=RAW_RECEIPT_ARTIFACT,
        role="rss_uncalibrated_construction_receipt",
        put_bytes=put_bytes,
    )
    return {
        "capability_id": "product.shape",
        "status": "PASS",
        "failure": None,
        "outputs": [scene_ref, views_ref, receipt_ref],
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductShapeDiagnostics.v4",
            "mode": "TWO_UNCALIBRATED_ORIGINAL_IMAGES_TO_GENERATED_H",
            "source_binding_sha256": source_object.source_binding_sha256,
            "unified_checkpoint_sha256": checkpoint["content_sha256"],
            "original_camera_calibrated": False,
            "source_visibility_qualified": False,
            "native_parity_qualified": False,
            "source_contact_qualified": False,
            "mechanics_qualified": False,
            "canonical_mesh_qualified": False,
            "product_revision_minted": False,
            "mesh_court_pending_unprojected_source": True,
        },
    }


__all__ = [
    "EXECUTOR_CONTRACT",
    "RAW_SCENE_ARTIFACT", "RAW_GENERATED_VIEWS_ARTIFACT",
    "RAW_RECEIPT_ARTIFACT", "execute_rss_vnext_shape_capability_v4",
]
