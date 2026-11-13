from __future__ import annotations

"""Single Mesh role: calibrated V2 replay or exact raw two-image CAS examination.

No original camera is invented. This court reads the TESSA candidate which
Shape already generated; it does not load a second model or re-run inference.
Topology integrity is measurable, but original-source appearance, projection
and signed 3D truth remain UNKNOWN. Product Mesh must fail closed until a
separate source/appearance admission court can qualify a CanonicalMeshIR.
"""

from hashlib import sha256
import io
import json
from typing import Any, Callable
from zipfile import ZipFile

import numpy as np

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    decode_uncalibrated_two_image_source_v1,
)
from .rss_vnext_mesh_executor_v2 import (
    execute_rss_vnext_mesh_capability_v2,
    PARENT_BINDING_SCHEMA,
)

EXECUTOR_CONTRACT = "RealSaS.ProductMeshCompile.v3"
RAW_SOURCE = "RealSaS.RSSvNextUncalibratedTwoImageSource"
RAW_SCENE = "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
RAW_VIEWS = "RealSaS.RSSvNextGeneratedConditioningViews"
RAW_RECEIPT = "RealSaS.RSSvNextUncalibratedConstructionReceipt"
RAW_TYPES = {RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT}
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

SCENE_REQUIRED = {
    "scene_triplanes", "query_xyz", "local_features",
    "semantic_logits", "graphics_logits", "presentation_role_logits",
    "region_relation_logits", "tessa_topology_token_ids",
}
MESH_KEYS = {
    "tessa_candidate_base_vertices", "tessa_candidate_rest_vertices",
    "tessa_candidate_faces", "tessa_candidate_vertex_components",
    "tessa_candidate_face_components", "tessa_candidate_local_features",
    "tessa_candidate_normals", "tessa_candidate_semantic_logits",
    "tessa_candidate_presentation_role_logits",
    "tessa_candidate_affordance_logits", "tessa_candidate_graphics_logits",
    "tessa_candidate_static_quality_logits",
    "tessa_candidate_unsigned_distance", "tessa_candidate_sign_logits",
}
VIEWS_REQUIRED = {
    "canonical_rgba", "generated_rgba",
    "generated_normals", "generated_virtual_cameras",
}


def _digest(value: Any) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _cas(ref: dict, reader: Callable[[dict], bytes]) -> bytes:
    blob = reader(ref)
    if (
        type(blob) is not bytes or len(blob) > MAX_ARCHIVE_BYTES
        or type(ref.get("size_bytes")) is not int
        or ref["size_bytes"] != len(blob)
        or ref.get("content_sha256") != sha256(blob).hexdigest()
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_PARENT_CAS_BYTES_DRIFT")
    return blob


def _archive(blob: bytes, required: set[str], allowed: set[str]) -> dict[str, np.ndarray]:
    try:
        with ZipFile(io.BytesIO(blob)) as archive:
            infos = archive.infolist()
            names = {row.filename for row in infos}
            if (
                len(infos) != len(names)
                or any(not name.endswith(".npy") for name in names)
                or sum(row.file_size for row in infos) > MAX_UNCOMPRESSED_BYTES
                or any(row.file_size > MAX_UNCOMPRESSED_BYTES for row in infos)
            ):
                raise RuntimeError("RSS_VNEXT_RAW_MESH_ARCHIVE_RESOURCE_OR_NAME_INVALID")
        with np.load(io.BytesIO(blob), allow_pickle=False) as payload:
            present = set(payload.files)
            if not required <= present or not present <= allowed:
                raise RuntimeError("RSS_VNEXT_RAW_MESH_ARCHIVE_FIELDS_DRIFT")
            arrays = {key: payload[key] for key in present}
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_ARCHIVE_DECODE_FAILED") from exc
    if any(
        arr.dtype.kind not in "fiu"
        or not np.isfinite(arr).all()
        for arr in arrays.values()
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_ARRAY_NONFINITE_OR_UNTYPED")
    return arrays


def _validate_generated_views(arrays: dict[str, np.ndarray]) -> None:
    canon, rgba, normals, cameras = (
        arrays["canonical_rgba"], arrays["generated_rgba"],
        arrays["generated_normals"], arrays["generated_virtual_cameras"],
    )
    if (
        canon.ndim != 5 or canon.shape[:3] != (1, 2, 4)
        or rgba.ndim != 5 or rgba.shape[:3] != (1, 6, 4)
        or normals.shape != (1, 6, 3, *rgba.shape[-2:])
        or canon.shape[-2:] != rgba.shape[-2:]
        or cameras.shape != (1, 6, 13)
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_GENERATED_VIEW_ABI_DRIFT")


def _validate_candidate(arrays: dict[str, np.ndarray], receipt: dict) -> dict[str, Any]:
    state = receipt.get("mesh_status")
    tokens = arrays["tessa_topology_token_ids"]
    if (
        tokens.ndim != 2 or tokens.shape[0] != 1
        or tokens.dtype.kind not in "iu"
        or not 1 <= tokens.shape[1] <= 131072
        or int(receipt.get("tessa_token_count") or 0) != tokens.shape[1]
        or state not in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE")
        or type(receipt.get("tessa_mesh_candidate_present")) is not bool
        or receipt["tessa_mesh_candidate_present"] != (
            state == "UNQUALIFIED_MESH_CANDIDATE"
        )
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_TESSA_STATUS_OR_TOKENS_DRIFT")
    has_mesh = state == "UNQUALIFIED_MESH_CANDIDATE"
    if (MESH_KEYS <= set(arrays)) != has_mesh or (
        (set(arrays) & MESH_KEYS) and not has_mesh
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_UNEXPECTED_OR_MISSING_SURFACE")
    if receipt.get("tessa_mesh_candidate_authority") != (
        "UNQUALIFIED_LEARNED_PROPOSAL_ONLY"
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_PREMATURE_CANONICAL_AUTHORITY")
    if not has_mesh:
        if (
            receipt.get("tessa_mesh_candidate_vertex_count") != 0
            or receipt.get("tessa_mesh_candidate_face_count") != 0
            or receipt.get("tessa_mesh_candidate_role_schema_id") is not None
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_ABSTAIN_FABRICATED_SURFACE")
        return {
            "mesh_state": "ABSTAIN",
            "topology_token_count": int(tokens.shape[1]),
            "candidate_vertices": 0,
            "candidate_faces": 0,
            "nondegenerate_faces": 0,
            "component_id_continuity": False,
        }

    xyz = arrays["tessa_candidate_rest_vertices"]
    base = arrays["tessa_candidate_base_vertices"]
    faces = arrays["tessa_candidate_faces"]
    vertex_components = arrays["tessa_candidate_vertex_components"]
    face_components = arrays["tessa_candidate_face_components"]
    if (
        xyz.ndim != 3 or xyz.shape[0] != 1 or xyz.shape[-1] != 3
        or xyz.shape != base.shape or not 3 <= xyz.shape[1] <= 2_000_000
        or faces.ndim != 2 or faces.shape[1] != 3
        or faces.dtype.kind not in "iu" or not 1 <= faces.shape[0] <= 2_000_000
        or int(faces.min()) < 0 or int(faces.max()) >= xyz.shape[1]
        or vertex_components.shape != (xyz.shape[1],)
        or face_components.shape != (faces.shape[0],)
        or vertex_components.dtype.kind not in "iu"
        or face_components.dtype.kind not in "iu"
        or np.any(vertex_components < 0) or np.any(face_components < 0)
        or receipt.get("tessa_mesh_candidate_vertex_count") != int(xyz.shape[1])
        or receipt.get("tessa_mesh_candidate_face_count") != int(faces.shape[0])
        or type(receipt.get("tessa_mesh_candidate_role_schema_id")) is not str
        or not receipt["tessa_mesh_candidate_role_schema_id"]
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_INDEXED_TOPOLOGY_DOMAIN_DRIFT")
    n = xyz.shape[1]
    for key in MESH_KEYS - {
        "tessa_candidate_base_vertices", "tessa_candidate_rest_vertices",
        "tessa_candidate_faces", "tessa_candidate_vertex_components",
        "tessa_candidate_face_components",
    }:
        a = arrays[key]
        if (
            a.ndim not in (2, 3)
            or (a.shape[1] if a.ndim == 3 else a.shape[0]) != n
            or (a.ndim == 3 and a.shape[0] != 1)
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_EXPERT_VERTEX_ADDRESS_DRIFT")
    if (
        arrays["tessa_candidate_normals"].shape != (1, n, 3)
        or any(
            arrays[key].shape != (1, n)
            for key in ("tessa_candidate_unsigned_distance", "tessa_candidate_sign_logits")
        )
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_GEOMETRY_FIELD_DOMAIN_DRIFT")
    triangles = xyz[0, faces]
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=-1,
    )
    legal = bool((area2 > 1e-12).all())
    component_continuity = bool(
        np.all(vertex_components[faces] == face_components[:, None])
    )
    return {
        "mesh_state": state,
        "topology_token_count": int(tokens.shape[1]),
        "candidate_vertices": int(n),
        "candidate_faces": int(faces.shape[0]),
        "nondegenerate_faces": int(np.count_nonzero(area2 > 1e-12)),
        "component_id_continuity": component_continuity,
        "geometric_face_area_positive": legal,
    }


def execute_rss_vnext_mesh_capability_v3(
    request: dict[str, Any], *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = request.get("capability_metadata") or {}
    if (
        type(request) is not dict or type(metadata) is not dict
        or request.get("capability_id") != "product.mesh"
        or request.get("kind") != "MODULE"
        or metadata.get("executor_contract") != EXECUTOR_CONTRACT
        or request.get("model_releases")
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_V3_ROLE_OR_MODEL_CONTRACT")
    refs = request.get("input_artifacts") or ()
    if type(refs) not in (tuple, list) or any(type(r) is not dict for r in refs):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_INPUT_PARENT_DOMAIN")
    raw = [r for r in refs if r.get("artifact_type") == RAW_SOURCE]
    if not raw:
        calibrated = {
            **request,
            "capability_metadata": {
                **metadata, "executor_contract": "RealSaS.ProductMeshCompile.v2",
            },
        }
        return execute_rss_vnext_mesh_capability_v2(
            calibrated, read_object=read_object, put_bytes=put_bytes,
        )
    if len(raw) not in (1, 2) or len(refs) != len(raw) + 3 or (
        {r.get("artifact_type") for r in refs} != RAW_TYPES
    ) or any(
        r.get("schema_version") != "v1" for r in refs
    ) or any(
        r.get(key) != raw[0].get(key)
        for r in raw[1:]
        for key in ("content_sha256", "size_bytes", "storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_EXACT_SOURCE_AND_THREE_CHILDREN_REQUIRED")
    def one(artifact_type: str) -> dict:
        rows = [r for r in refs if r["artifact_type"] == artifact_type]
        if len(rows) != 1:
            raise RuntimeError("RSS_VNEXT_RAW_MESH_CHILD_CARDINALITY")
        return rows[0]
    source = raw[0]
    original = decode_uncalibrated_two_image_source_v1(
        _cas(source, read_object),
    )
    scene_ref, views_ref, receipt_ref = (
        one(RAW_SCENE), one(RAW_VIEWS), one(RAW_RECEIPT),
    )
    try:
        receipt = json.loads(_cas(receipt_ref, read_object).decode("utf8"))
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_RECEIPT_DECODE_FAILED") from exc
    binding = request.get("shape_parent_binding")
    if (
        type(binding) is not dict or set(binding) != {
            "schema", "source_content_sha256", "checkpoint_sha256",
            "model_release_behavior_sha256",
        }
        or binding.get("schema") != PARENT_BINDING_SCHEMA
        or any(not _digest(binding[k]) for k in (
            "source_content_sha256", "checkpoint_sha256",
            "model_release_behavior_sha256",
        ))
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_SHAPE_PARENT_BINDING_REQUIRED")
    if (
        type(receipt) is not dict
        or receipt.get("schema") != "RealSaS.RSSvNext.UncalibratedConstructionReceipt.v1"
        or receipt.get("source_content_sha256") != source["content_sha256"]
        or receipt.get("source_binding_sha256") != original.source_binding_sha256
        or receipt.get("source_rights_claim_sha256") != original.source_rights_claim_sha256
        or receipt.get("scene_content_sha256") != scene_ref["content_sha256"]
        or receipt.get("generated_views_content_sha256") != views_ref["content_sha256"]
        or binding["source_content_sha256"] != source["content_sha256"]
        or binding["checkpoint_sha256"] != receipt.get("checkpoint_sha256")
        or binding["model_release_behavior_sha256"] != receipt.get(
            "model_release_behavior_identity_sha256"
        )
        or receipt.get("original_camera_calibrated") is not False
        or receipt.get("original_pixel_count") != 2
        or receipt.get("generated_view_count") != 6
        or receipt.get("source_ray_evidence_qualified") is not False
        or receipt.get("generated_views_as_original_source") is not False
        or receipt.get("canonical_mesh_qualified") is not False
        or receipt.get("product_revision_minted") is not False
        or receipt.get("source_rights_verified") is not False
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_SOURCE_OR_CHECKPOINT_PARENT_DRIFT")
    scene = _archive(
        _cas(scene_ref, read_object),
        required=SCENE_REQUIRED, allowed=SCENE_REQUIRED | MESH_KEYS,
    )
    views = _archive(
        _cas(views_ref, read_object),
        required=VIEWS_REQUIRED, allowed=VIEWS_REQUIRED,
    )
    _validate_generated_views(views)
    if (
        scene["scene_triplanes"].ndim != 5
        or scene["scene_triplanes"].shape[:2] != (1, 3)
        or scene["query_xyz"].ndim != 3
        or scene["query_xyz"].shape[0] != 1
        or scene["query_xyz"].shape[2] != 3
        or scene["local_features"].ndim != 3
        or scene["local_features"].shape[:2] != scene["query_xyz"].shape[:2]
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_SHARED_H_DOMAIN_DRIFT")
    measured = _validate_candidate(scene, receipt)
    diagnostics = {
        "schema": "RealSaS.RSSvNext.RawMeshCompilerCourt.v1",
        "source_content_sha256": source["content_sha256"],
        "scene_content_sha256": scene_ref["content_sha256"],
        "generated_views_content_sha256": views_ref["content_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_original_camera": "UNKNOWN_UNCALIBRATED",
        "generated_target_cameras_are_observations": False,
        "source_ray_replay_admitted": False,
        "source_visibility_and_appearance_admitted": False,
        "tessa_reinference_performed": False,
        "canonical_mesh_minted": False,
        "product_revision_minted": False,
        **measured,
    }
    # Shape already retained every candidate CAS artifact. No inferred
    # camera, candidate triangle or generated normal is an independently
    # qualified original-source position/ray/appearance correspondence.
    # This is a measured compiler boundary, NOT a second pseudo-product lane.
    return {
        "capability_id": "product.mesh",
        "status": "FAIL",
        "outputs": [],
        "diagnostics": diagnostics,
        "failure": {
            "code": "RSS_VNEXT_RAW_MESH_SOURCE_QUALIFICATION_PENDING",
            "class": "COMPILER_MESH",
            "reported_owner_module_id": "product.mesh",
            "diagnostics": diagnostics,
        },
    }


__all__ = [
    "EXECUTOR_CONTRACT", "RAW_SOURCE", "RAW_SCENE", "RAW_VIEWS",
    "RAW_RECEIPT", "execute_rss_vnext_mesh_capability_v3",
]
