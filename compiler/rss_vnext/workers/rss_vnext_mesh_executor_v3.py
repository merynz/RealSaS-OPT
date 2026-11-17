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
RAW_CONFIG = "RealSaS.RSSvNextUnifiedModelConfig"
RAW_DECODER = "RealSaS.RSSvNextEmbeddedMIRADecoder"
RAW_TYPES = {RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT}
RAW_RELEASE_TYPES = RAW_TYPES | {RAW_CONFIG, RAW_DECODER}
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
LAYER_EXTRA_KEYS = {"tessa_seed_vertex_layer_ids"}
SEEDED_EXTRA_KEYS = {
    "tessa_candidate_mechanics_local_features",
    "tessa_seed_original_faces",
    "tessa_seed_vertex_parent_indices", "tessa_seed_vertex_parent_weights",
}
RAW_GRAPHICS_KEYS = {
    "region_relation_valid_mask", "presentation_role_probabilities",
    "source_semantic_patch_logits", "source_semantic_log_uncertainty",
}

AXIS_MIRA_KEYS = {
    "axis_mira_dynamic_vertices", "axis_mira_soft_skin_weights",
    "axis_mira_discrete_skin_weights", "axis_mira_joint_hidden_states",
    "axis_mira_dynamic_displacement", "axis_mira_readiness_logits",
    "axis_mira_joint_valid_mask",
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
    seeded_abi = receipt.get("topology_candidate_abi") == "ONE_IRIS_SEEDED_MESH_V1"
    if (
        receipt.get("topology_candidate_abi") not in (
            None, "LEGACY_INDEPENDENT_TESSA_GRAMMAR",
            "ONE_IRIS_SEEDED_MESH_V1",
        )
        or receipt.get("tessa_mesh_candidate_authority") != (
            "UNQUALIFIED_IRIS_SEEDED_SURFACE_ONLY" if seeded_abi
            else "UNQUALIFIED_LEARNED_PROPOSAL_ONLY"
        )
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_PREMATURE_CANONICAL_AUTHORITY")
    if not has_mesh:
        if set(arrays) & (SEEDED_EXTRA_KEYS | LAYER_EXTRA_KEYS):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_ABSTAIN_FAKE_SEEDED_ATTRIBUTES")
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
    if seeded_abi:
        from hashlib import sha256 as _sha
        if not SEEDED_EXTRA_KEYS <= set(arrays):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_SEEDED_CONTEXT_OR_LINEAGE_REQUIRED")
        mh = arrays["tessa_candidate_mechanics_local_features"]
        p = arrays["tessa_seed_vertex_parent_indices"]
        w = arrays["tessa_seed_vertex_parent_weights"]
        src_faces = arrays["tessa_seed_original_faces"]
        original_n = receipt.get("tessa_seed_original_vertex_count", -1)
        original_f = receipt.get("tessa_seed_original_face_count", -1)
        if (
            mh.shape != arrays["tessa_candidate_local_features"].shape
            or p.shape != (n, 2) or w.shape != (n, 2)
            or p.dtype.kind not in "iu"
            or type(original_n) is not int
            or not 3 <= original_n <= n
            or type(original_f) is not int or original_f < 1
            or src_faces.shape != (original_f,3)
            or src_faces.dtype.kind not in "iu"
            or np.any(src_faces < 0) or np.any(src_faces >= original_n)
            or np.any(p < 0) or np.any(p >= original_n)
            or not np.allclose(w.sum(axis=-1), 1.0, atol=1e-7, rtol=0)
            or not np.all(
                (np.isclose(w[:, 0], 1.) & np.isclose(w[:, 1], 0.))
                | (np.isclose(w[:, 0], 0.5) & np.isclose(w[:, 1], 0.5))
            )
            or type(receipt.get("tessa_seed_surface_sha256")) is not str
            or not _digest(receipt.get("tessa_seed_surface_sha256"))
            or not _digest(receipt.get("tessa_seed_scene_sha256"))
            or receipt.get("tessa_seed_original_face_count", 0) < 1
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_SEEDED_PARENT_OR_CONTEXT_DRIFT")
        scene = arrays["scene_triplanes"]
        h = _sha()
        h.update(str((str(scene.dtype), scene.shape)).encode())
        h.update(scene.tobytes())
        if h.hexdigest() != receipt["tessa_seed_scene_sha256"]:
            raise RuntimeError("RSS_VNEXT_RAW_MESH_SEEDED_SCENE_HASH_DRIFT")
        def _tensor_hash(x: np.ndarray) -> str:
            d=_sha()
            d.update(str((str(x.dtype),x.shape)).encode())
            d.update(x.tobytes())
            return d.hexdigest()
        original_xyz=base[:, :original_n]
        original_components=vertex_components[:original_n]
        expected_surface=_sha((
            _tensor_hash(original_xyz)
            +_tensor_hash(src_faces)+_tensor_hash(original_components)
        ).encode()).hexdigest()
        if expected_surface != receipt["tessa_seed_surface_sha256"]:
            raise RuntimeError("RSS_VNEXT_RAW_MESH_SEEDED_SURFACE_HASH_DRIFT")
        if (
            not np.array_equal(p[:original_n],np.repeat(
                np.arange(original_n)[:,None],2,axis=1,
            ))
            or not np.allclose(
                w[:original_n],np.array([1.,0.])[None],
                rtol=0,atol=1e-7,
            )
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_ORIGINAL_VERTEX_ADDRESS_DRIFT")
        # Semantic layer partitions are model evidence, not attachment or
        # signed-source truth. Require byte-sealed face and parent addresses.
        fields=(
            "tessa_seed_layer_names","tessa_seed_layer_roles",
            "tessa_seed_layer_namespace_sha256",
            "tessa_seed_vertex_layer_sha256",
        )
        has_layer="tessa_seed_vertex_layer_ids" in arrays
        if has_layer != any(key in receipt for key in fields):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_LAYER_NAMESPACE_INCOMPLETE")
        if has_layer:
            names=receipt.get("tessa_seed_layer_names")
            roles=receipt.get("tessa_seed_layer_roles")
            layers=arrays["tessa_seed_vertex_layer_ids"]
            if (
                type(names) is not list or type(roles) is not list
                or not 2 <= len(names) <= 16 or len(names)!=len(roles)
                or any(type(name) is not str or not name for name in names)
                or len(set(names)) != len(names)
                or any(type(row) is not list or not row for row in roles)
                or any(type(role) is not str or not role
                       for row in roles for role in row)
                or len(set(role for row in roles for role in row)) != sum(
                    len(row) for row in roles
                )
                or layers.shape!=(n,) or layers.dtype.kind not in "iu"
                or np.any(layers<0) or np.any(layers>=len(names))
                or any(not np.any(layers[:original_n]==k)
                       for k in range(len(names)))
                or not np.all(layers[faces]==layers[faces[:,0:1]])
                or receipt.get("tessa_seed_vertex_layer_sha256")
                    != _tensor_hash(layers)
            ):
                raise RuntimeError("RSS_VNEXT_RAW_MESH_LAYER_ADDRESS_DRIFT")
            import json as _json
            expected_namespace=_sha(_json.dumps({
                "schema":"RealSaS.RSSvNext.SemanticLayerIndexedSurface.v1",
                "layers":[{"name":name,"keep_roles":role}
                          for name,role in zip(names,roles)],
            },separators=(",",":"),sort_keys=True).encode()).hexdigest()
            if expected_namespace != receipt.get(
                "tessa_seed_layer_namespace_sha256"
            ):
                raise RuntimeError("RSS_VNEXT_RAW_MESH_LAYER_NAMESPACE_HASH_DRIFT")
        for idx in range(original_n,n):
            a,b=map(int,p[idx])
            expected=0.5*(base[0,a]+base[0,b])
            if a==b or not np.allclose(
                base[0,idx],expected,atol=1e-6,rtol=0,
            ):
                raise RuntimeError("RSS_VNEXT_RAW_MESH_PARENT_POSITION_DRIFT")
            if has_layer and (
                layers[idx]!=layers[a] or layers[a]!=layers[b]
            ):
                raise RuntimeError("RSS_VNEXT_RAW_MESH_LAYER_PARENT_DRIFT")
    elif set(arrays) & (SEEDED_EXTRA_KEYS | LAYER_EXTRA_KEYS):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_LEGACY_HAS_SEEDED_ATTRIBUTES")
    for key in MESH_KEYS - {
        "tessa_candidate_base_vertices", "tessa_candidate_rest_vertices",
        "tessa_candidate_faces", "tessa_candidate_vertex_components",
        "tessa_candidate_face_components",
        "tessa_candidate_unsigned_distance", "tessa_candidate_sign_logits",
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
        "topology_candidate_abi": (
            "ONE_IRIS_SEEDED_MESH_V1" if seeded_abi
            else "LEGACY_INDEPENDENT_TESSA_GRAMMAR"
        ),
        "candidate_vertices": int(n),
        "candidate_faces": int(faces.shape[0]),
        "nondegenerate_faces": int(np.count_nonzero(area2 > 1e-12)),
        "component_id_continuity": component_continuity,
        "geometric_face_area_positive": legal,
    }


def _validate_raw_axis_mira(
    scene: dict[str, np.ndarray], receipt: dict,
    mesh: dict[str, Any],
) -> dict[str, Any]:
    """Consume exact AXIS/MIRA candidates rather than re-inferring rig/skin.

    The raw source has no observed camera; structural/control programs remain
    learned proposals with no canonical mechanics or contact authority.
    """
    from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
        _validate_neural_control_events,
    )
    status = receipt.get("axis_mira_control_status")
    fields = set(scene) & AXIS_MIRA_KEYS
    if status is None and not fields:
        return {"control_state": "LEGACY_NOT_GENERATED", "control_event_count": 0,
                "discrete_skin_present": False}
    events = receipt.get("axis_mira_control_events")
    count = receipt.get("axis_mira_control_event_count")
    digest = receipt.get("axis_mira_control_events_sha256")
    hard = receipt.get("axis_mira_discrete_skin_candidate_present")
    authority = receipt.get("axis_mira_control_authority")
    if type(events) is not list or type(count) is not int or len(events) != count:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_AXIS_EVENTS_CARDINALITY")
    if status == "ABSTAIN_NO_VALID_PREDICTED_MESH":
        if (
            mesh["mesh_state"] != "ABSTAIN" or count != 0 or digest is not None
            or hard is not False or fields or authority != "NO_CONTROL_CANDIDATE"
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_ABSTAIN_HAS_RIG_OR_SKIN")
        return {"control_state": status, "control_event_count": 0,
                "discrete_skin_present": False}
    if (
        status != "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE"
        or mesh["mesh_state"] != "UNQUALIFIED_MESH_CANDIDATE"
        or not 1 <= count <= 4096
        or authority != "LEARNED_CAUSAL_CONTROL_NOT_CANONICAL_RIG_SKIN"
        or type(hard) is not bool
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_AXIS_PROPOSAL_STATUS")
    _validate_neural_control_events(tuple(events))
    event_bytes = json.dumps(
        events, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if not _digest(digest) or sha256(event_bytes).hexdigest() != digest:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_AXIS_EVENTS_HASH_DRIFT")
    if hard is not bool(fields):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_MIRA_SKIN_PRESENCE_DRIFT")
    if hard and fields != AXIS_MIRA_KEYS:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_PARTIAL_MIRA_ADDRESSING")
    if not hard:
        return {"control_state": status, "control_event_count": count,
                "discrete_skin_present": False}
    n = mesh["candidate_vertices"]
    xyz, delta, quality = (
        scene["axis_mira_dynamic_vertices"],
        scene["axis_mira_dynamic_displacement"],
        scene["axis_mira_readiness_logits"],
    )
    soft = scene["axis_mira_soft_skin_weights"]
    discrete = scene["axis_mira_discrete_skin_weights"]
    hidden = scene["axis_mira_joint_hidden_states"]
    joint_mask = scene["axis_mira_joint_valid_mask"]
    j = soft.shape[-1] if soft.ndim == 3 else 0
    if (
        xyz.shape != (1, n, 3) or delta.shape != xyz.shape
        or quality.shape != (1, n, 4)
        or soft.shape != (1, n, j) or j < 1
        or discrete.shape != soft.shape
        or hidden.ndim != 3 or hidden.shape[:2] != (1, j)
        or joint_mask.shape != (1, j)
        or joint_mask.dtype.kind not in "iu"
        or not np.isin(joint_mask, [0, 1]).all()
        or not np.allclose(soft.sum(-1), 1, atol=1e-5)
        or not np.allclose(discrete.sum(-1), 1, atol=1e-5)
        or (soft < -1e-6).any() or (discrete < -1e-6).any()
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_MIRA_JOINT_OR_VERTEX_ADDRESS_DRIFT")
    return {
        "control_state": status,
        "control_event_count": count,
        "discrete_skin_present": True,
        "discrete_skin_joints": int(j),
        "discrete_skin_max_simplex_error": float(
            np.max(np.abs(discrete.sum(-1) - 1.)),
        ),
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
    parent_types = {r.get("artifact_type") for r in refs}
    pinned_decoder_present = parent_types == RAW_RELEASE_TYPES
    if len(raw) not in (1, 2) or (
        parent_types != RAW_TYPES and parent_types != RAW_RELEASE_TYPES
    ) or len(refs) != len(raw) + (5 if pinned_decoder_present else 3) or any(
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
    if pinned_decoder_present:
        config_ref, decoder_ref = one(RAW_CONFIG), one(RAW_DECODER)
        if (
            receipt.get("unified_model_config_content_sha256") != config_ref["content_sha256"]
            or receipt.get("embedded_mira_decoder_content_sha256") != decoder_ref["content_sha256"]
        ):
            raise RuntimeError("RSS_VNEXT_RAW_MESH_EMBEDDED_CODEC_RECEIPT_DRIFT")
        # Decoder weights are *not* loaded in Mesh. Rig/Skin reopen them
        # from exact CAS bytes and replay discrete SkinTokens.
        _cas(config_ref, read_object)
    elif (
        receipt.get("unified_model_config_content_sha256")
        or receipt.get("embedded_mira_decoder_content_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_MISSING_EXACT_CODEC_CHILD")
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
        required=SCENE_REQUIRED, allowed=SCENE_REQUIRED | MESH_KEYS | SEEDED_EXTRA_KEYS | LAYER_EXTRA_KEYS | AXIS_MIRA_KEYS | RAW_GRAPHICS_KEYS,
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
    controls_measured = _validate_raw_axis_mira(scene, receipt, measured)
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
        **controls_measured,
    }
    if (
        measured["mesh_state"] == "UNQUALIFIED_MESH_CANDIDATE"
        and measured["component_id_continuity"]
        and measured["geometric_face_area_positive"]
    ):
        from compiler.realsas_compiler_core.types import QualificationError
        from .rss_vnext_raw_mesh_admission_v1 import admit_raw_two_view_mesh_v1
        from .rss_vnext_mesh_executor_v2 import _store_json
        try:
            canonical, static, witness = admit_raw_two_view_mesh_v1(
                source=original, scene=scene, receipt=receipt,
                scene_digest=scene_ref["content_sha256"],
            )
        except (RuntimeError, ValueError, QualificationError) as exc:
            diagnostics["static_mesh_admission_blocker"] = str(exc)
            canonical, static, witness = None, None, None
        if witness is not None:
            diagnostics["real_source_static_witness"] = witness
        if canonical is not None and static is not None:
            outputs = [
                _store_json(
                    static.to_dict(),
                    role="rss_two_view_static_qualification",
                    artifact_type="RealSaS.StaticCanonicalMeshQualificationIR",
                    put_bytes=put_bytes,
                    authority_class="REAL_SOURCE_ALPHA_STATIC_WITH_INFERRED_CAMERA_NOT_ORIGINAL_RAY",
                ),
                _store_json(
                    canonical.to_dict(),
                    role="canonical_mesh",
                    artifact_type="RealSaS.CanonicalMeshIR",
                    put_bytes=put_bytes,
                    authority_class="COMPILER_CANONICAL_MESH_TWO_ORIGINAL_ALPHA_VIEWS",
                ),
            ]
            diagnostics["canonical_mesh_minted"] = True
            diagnostics["canonical_mesh_lineage_hash"] = canonical.mesh_lineage_hash
            diagnostics["original_camera_observed"] = False
            diagnostics["source_3d_geometry_observed"] = False
            diagnostics["source_appearance_qualified"] = False
            return {
                "capability_id": "product.mesh",
                "status": "PASS",
                "outputs": outputs,
                "diagnostics": diagnostics,
                "failure": None,
            }
    # A failed or absent measured source fit cannot be silently converted
    # into canonical geometry. Preserve the entire Shape scene CAS for repair.
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
