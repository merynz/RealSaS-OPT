from __future__ import annotations

"""RSS.vNext -> Compiler immutable *proposal* boundary.

This is deliberately NOT CanonicalMeshIR or ProductRevision. The Compiler
can accept a fully addressed and hash-sealed proposal for qualification, but
cannot promote its content without source, geometry, mechanics, appearance
and product courts. Source-ray colors are ray samples, never surface texture.
"""

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

_SCHEMA = "RealSaS.RSSvNext.NeuralInferenceEvidence.v1"
_AUTHORITY = "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"
_RAY_AUTHORITY = "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE"
_DEPTH_AUTHORITY = "UNKNOWN_NO_SOURCE_DEPTH"


def _reject(code: str) -> None:
    raise QualificationError(code)


def _hex(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _vector(value, count: int, code: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        _reject(code)
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            _reject(code)


def _rows(value, count: int, width: int | None, code: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        _reject(code)
    for row in value:
        if not isinstance(row, (list, tuple)) or (width is not None and len(row) != width):
            _reject(code)
        if not row:
            _reject(code)
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                _reject(code)


def _mesh(value: Mapping[str, Any]) -> int:
    expected = {
        "topology_tokens", "rest_xyz", "base_xyz", "local_features",
        "faces", "vertex_components",
        "face_components", "normals_xyz", "semantic_logits",
        "presentation_role_logits", "presentation_role_schema_id",
        "affordance_logits", "graphics_logits", "static_quality_logits",
        "unsigned_distance", "sign_logits", "source_rays",
        "source_authority", "signed_solid_authority", "canonical_mesh_authority",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _reject("RSS_VNEXT_EVIDENCE_MESH_FIELDS")
    if (
        value["source_authority"] != "LEARNED_PREDICTED_TOPOLOGY"
        or value["canonical_mesh_authority"] is not False
        or value["signed_solid_authority"] != "UNQUALIFIED"
    ):
        _reject("RSS_VNEXT_EVIDENCE_PREMATURE_AUTHORITY")
    tokens = value["topology_tokens"]
    if not isinstance(tokens, (tuple, list)) or not tokens or any(
        type(t) is not int or t < 0 for t in tokens
    ):
        _reject("RSS_VNEXT_EVIDENCE_TOPOLOGY_TOKENS")
    verts = value["rest_xyz"]
    if not isinstance(verts, (list, tuple)) or len(verts) < 3:
        _reject("RSS_VNEXT_EVIDENCE_VERTEX_COUNT")
    n = len(verts)
    _rows(verts, n, 3, "RSS_VNEXT_EVIDENCE_VERTEX_XYZ")
    _rows(value["base_xyz"], n, 3, "RSS_VNEXT_EVIDENCE_BASE_VERTEX_XYZ")
    _rows(value["local_features"], n, None, "RSS_VNEXT_EVIDENCE_LOCAL_FEATURES")
    if len({len(row) for row in value["local_features"]}) != 1:
        _reject("RSS_VNEXT_EVIDENCE_FEATURE_WIDTH_DRIFT")
    _rows(value["normals_xyz"], n, 3, "RSS_VNEXT_EVIDENCE_NORMAL_XYZ")
    for field_name in (
        "semantic_logits", "presentation_role_logits", "affordance_logits",
        "graphics_logits", "static_quality_logits",
    ):
        _rows(value[field_name], n, None, "RSS_VNEXT_EVIDENCE_FIELD:" + field_name)
        if len({len(row) for row in value[field_name]}) != 1:
            _reject("RSS_VNEXT_EVIDENCE_FIELD_WIDTH:" + field_name)
    _vector(value["unsigned_distance"], n, "RSS_VNEXT_EVIDENCE_UDF")
    if any(d < 0 for d in value["unsigned_distance"]):
        _reject("RSS_VNEXT_EVIDENCE_UDF_NEGATIVE")
    _vector(value["sign_logits"], n, "RSS_VNEXT_EVIDENCE_SIGN_LOGITS")
    if not isinstance(value["presentation_role_schema_id"], str) or not value["presentation_role_schema_id"]:
        _reject("RSS_VNEXT_EVIDENCE_PRESENTATION_SCHEMA_REQUIRED")
    for key, length in (("vertex_components", n), ("face_components", len(value["faces"]))):
        a = value[key]
        if not isinstance(a, (list, tuple)) or len(a) != length or any(
            type(x) is not int or x < 0 for x in a
        ):
            _reject("RSS_VNEXT_EVIDENCE_COMPONENT_ID_INVALID")
    faces = value["faces"]
    if not isinstance(faces, (tuple, list)) or not faces:
        _reject("RSS_VNEXT_EVIDENCE_FACES_REQUIRED")
    for i, face in enumerate(faces):
        if not isinstance(face, (tuple, list)) or len(face) != 3 or any(
            type(x) is not int or not 0 <= x < n for x in face
        ) or len(set(face)) != 3:
            _reject("RSS_VNEXT_EVIDENCE_FACE_INVALID")
        if (
            len({value["vertex_components"][x] for x in face}) != 1
            or value["vertex_components"][face[0]] != value["face_components"][i]
        ):
            _reject("RSS_VNEXT_EVIDENCE_CROSS_COMPONENT_FACE")
    ray = value["source_rays"]
    if not isinstance(ray, Mapping) or set(ray) != {
        "rgba", "valid", "alpha", "appearance_authority", "depth_authority",
    } or ray["appearance_authority"] != _RAY_AUTHORITY or ray["depth_authority"] != _DEPTH_AUTHORITY:
        _reject("RSS_VNEXT_EVIDENCE_SOURCE_RAY_AUTHORITY")
    if any(not isinstance(ray.get(key), (list, tuple)) or len(ray[key]) != n
           for key in ("rgba", "valid", "alpha")):
        _reject("RSS_VNEXT_EVIDENCE_RAY_VERTEX_BINDING")
    for index in range(n):
        if not all(len(ray[key][index]) == 2 for key in ("rgba", "valid", "alpha")):
            _reject("RSS_VNEXT_EVIDENCE_RAY_TWO_SOURCES")
        for view in range(2):
            _vector(ray["rgba"][index][view], 4, "RSS_VNEXT_EVIDENCE_RAY_RGBA")
            if any(not 0 <= channel <= 1 for channel in ray["rgba"][index][view]):
                _reject("RSS_VNEXT_EVIDENCE_RAY_RGBA_RANGE")
            if type(ray["valid"][index][view]) is not bool:
                _reject("RSS_VNEXT_EVIDENCE_RAY_VALID_TYPE")
            alpha = ray["alpha"][index][view]
            if (type(alpha) not in (int, float)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
                _reject("RSS_VNEXT_EVIDENCE_RAY_ALPHA")
            if abs(alpha - ray["rgba"][index][view][3]) > 1e-5:
                _reject("RSS_VNEXT_EVIDENCE_RAY_ALPHA_DRIFT")
    return n


def _dynamic(value: Mapping[str, Any], n: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "vertices", "weights", "displacement", "readiness_logits", "authority",
    } or value["authority"] != "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL":
        _reject("RSS_VNEXT_EVIDENCE_DYNAMIC_AUTHORITY")
    for k, w in (("vertices", 3), ("displacement", 3), ("readiness_logits", 4)):
        _rows(value[k], n, w, "RSS_VNEXT_EVIDENCE_DYNAMIC_SHAPE:" + k)
    weights = value["weights"]
    _rows(weights, n, None, "RSS_VNEXT_EVIDENCE_SKIN_SHAPE")
    if len({len(row) for row in weights}) != 1:
        _reject("RSS_VNEXT_EVIDENCE_SKIN_JOINT_ORDER_DRIFT")
    if any(w < -1e-6 or abs(sum(row) - 1.0) > 1e-5 for row in weights for w in row):
        _reject("RSS_VNEXT_EVIDENCE_SKIN_SIMPLEX")




def _scene(value: Mapping[str, Any]) -> None:
    """Validate shared semantic/Graphics relations without assigning artist authority."""
    # Existing sealed V1 research receipts have a relation-only scene. Keep
    # them parseable as incomplete, unqualified evidence. The richer producer
    # must emit the COMPLETE address-bound field trio together; partial fields
    # cannot be interpreted as drawable/graphics evidence.
    base_keys = {
        "source_semantic_patch_logits", "source_semantic_log_uncertainty",
        "region_relation_logits", "region_relation_valid_mask",
        "presentation_role_probabilities",
        "constructed_camera_features", "constructed_camera_angles",
        "authority",
    }
    extended_keys = {
        "region_query_xyz", "region_membership_logits", "region_graphics_logits",
    }
    if not isinstance(value, Mapping) or set(value) not in (
        base_keys, base_keys | extended_keys,
    ):
        _reject("RSS_VNEXT_EVIDENCE_SCENE_FIELDS")
    has_region_field = extended_keys.issubset(value)
    if value["authority"] != "PREDICTED_FROM_TWO_ORIGINAL_VIEWS":
        _reject("RSS_VNEXT_EVIDENCE_SCENE_AUTHORITY_DRIFT")

    def walk(raw, *, bool_allowed: bool = False):
        if isinstance(raw, (list, tuple)):
            if not raw:
                _reject("RSS_VNEXT_EVIDENCE_SCENE_EMPTY_TENSOR")
            for item in raw:
                walk(item, bool_allowed=bool_allowed)
        elif type(raw) is bool:
            if not bool_allowed:
                _reject("RSS_VNEXT_EVIDENCE_SCENE_BOOL_UNEXPECTED")
        elif not isinstance(raw, (float, int)) or not math.isfinite(raw):
            _reject("RSS_VNEXT_EVIDENCE_SCENE_NONFINITE")

    for name in (
        "source_semantic_patch_logits", "source_semantic_log_uncertainty",
        "region_relation_logits", "presentation_role_probabilities",
    ):
        walk(value[name])
    mask = value["region_relation_valid_mask"]
    if not isinstance(mask, (list, tuple)) or not mask:
        _reject("RSS_VNEXT_EVIDENCE_REGION_RELATION_MASK")
    walk(mask, bool_allowed=True)
    slots = len(mask)
    # Neither the anonymous region graph nor the field at H(x) can be used
    # downstream without the exact query addresses and an explicit UNKNOWN
    # membership channel. Reject unaddressed, untyped, or self-edge evidence.
    if has_region_field:
        xyz = value["region_query_xyz"]
        member = value["region_membership_logits"]
        graphics = value["region_graphics_logits"]
        role = value["presentation_role_probabilities"]
        if not all(isinstance(v, (list, tuple)) and v for v in (xyz, member, graphics, role)):
            _reject("RSS_VNEXT_EVIDENCE_REGION_FIELD_EMPTY")
        n = len(xyz)
        if len(member) != n or len(graphics) != n or len(role) != n:
            _reject("RSS_VNEXT_EVIDENCE_REGION_ADDRESS_DRIFT")
        for row in xyz:
            _vector(row, 3, "RSS_VNEXT_EVIDENCE_REGION_QUERY_XYZ")
        for row in member:
            _vector(row, slots + 1, "RSS_VNEXT_EVIDENCE_REGION_UNKNOWN_CHANNEL")
        for row in graphics:
            _vector(row, 5, "RSS_VNEXT_EVIDENCE_GRAPHICS_FIELD_FIVE_CHANNELS")
        role_dim = len(role[0])
        if role_dim < 2:
            _reject("RSS_VNEXT_EVIDENCE_ROLE_CHANNEL_INVALID")
        for row in role:
            _vector(row, role_dim, "RSS_VNEXT_EVIDENCE_ROLE_CHANNEL_DRIFT")
            if any(float(v) < 0 or float(v) > 1 for v in row) or abs(sum(row) - 1.0) > 1e-4:
                _reject("RSS_VNEXT_EVIDENCE_ROLE_SIMPLEX_INVALID")
    relations = value["region_relation_logits"]
    if len(relations) != slots:
        _reject("RSS_VNEXT_EVIDENCE_RELATION_SLOT_COUNT_DRIFT")
    for i in range(slots):
        if len(mask[i]) != slots or len(relations[i]) != slots:
            _reject("RSS_VNEXT_EVIDENCE_RELATION_GRAPH_SHAPE")
        for j in range(slots):
            if type(mask[i][j]) is not bool or mask[i][j] != (i != j):
                _reject("RSS_VNEXT_EVIDENCE_RELATION_SELF_EDGE_MASK_INVALID")
            _vector(relations[i][j], 3, "RSS_VNEXT_EVIDENCE_RELATION_CHANNEL_DRIFT")
    if len(value["source_semantic_patch_logits"]) != 2:
        _reject("RSS_VNEXT_EVIDENCE_SOURCE_PATCH_COUNT")
    camera = value["constructed_camera_features"]
    angles = value["constructed_camera_angles"]
    if (camera is None) != (angles is None):
        _reject("RSS_VNEXT_EVIDENCE_AUX_CAMERA_PAIR")
    if camera is not None:
        if len(camera) not in (4, 6) or len(angles) != len(camera):
            _reject("RSS_VNEXT_EVIDENCE_AUX_CAMERA_COUNT")
        for row in camera:
            _vector(row, 13, "RSS_VNEXT_EVIDENCE_AUX_CAMERA_13D")
        walk(angles)

@dataclass(frozen=True)
class RSSvNextInferenceEvidenceIRV1:
    checkpoint_sha256: str
    observed_front_sha256: str
    observed_back_sha256: str
    camera_binding_sha256: str
    mesh_status: str
    mesh: Mapping[str, Any] | None
    control_events: tuple[Mapping[str, Any], ...] = ()
    dynamic: Mapping[str, Any] | None = None
    scene: Mapping[str, Any] | None = None
    reason: str = ""
    observed_view_count: int = 2
    auxiliary_view_authority: str = "LEARNED_PREDICTED_ONLY"
    qualification_status: str = _AUTHORITY
    schema_version: str = _SCHEMA
    evidence_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def inference_evidence_hash_v1(value: RSSvNextInferenceEvidenceIRV1) -> str:
    payload = value.to_dict()
    payload.pop("evidence_hash", None)
    return content_sha256(payload)


def _validate_neural_control_events(events: tuple[Mapping[str, Any], ...]) -> None:
    """Validate actual AXIS event domains before canonical hashing and after read."""
    expected_event_fields = {
        "kind", "coord_indices", "pointer_a", "pointer_b", "pointer_c",
        "aux_type_id", "dof_mask", "frame_6d", "scalar_params",
        "skin_code_id", "anchor_parent_kind",
    }
    for ev in events:
        if not isinstance(ev, Mapping) or set(ev) != expected_event_fields:
            _reject("RSS_VNEXT_EVIDENCE_CONTROL_EVENT_FIELDS_DRIFT")
        if type(ev["kind"]) is not int or not 1 <= ev["kind"] <= 18:
            _reject("RSS_VNEXT_EVIDENCE_CONTROL_EVENT_KIND_INVALID")
        for key in (
            "pointer_a", "pointer_b", "pointer_c", "aux_type_id",
            "skin_code_id", "anchor_parent_kind",
        ):
            if type(ev[key]) is not int or ev[key] < -1:
                _reject("RSS_VNEXT_EVIDENCE_CONTROL_INTEGER_FIELD_INVALID")
        indices = ev["coord_indices"]
        if (not isinstance(indices, (list, tuple)) or len(indices) != 3
                or any(type(v) is not int or v < -1 for v in indices)):
            _reject("RSS_VNEXT_EVIDENCE_CONTROL_COORD_INVALID")
        for key, width in (("dof_mask", 9), ("frame_6d", 6), ("scalar_params", 4)):
            _vector(ev[key], width, "RSS_VNEXT_EVIDENCE_CONTROL_FIELD_INVALID:" + key)
        if any(value not in (0., 1.) for value in ev["dof_mask"]):
            _reject("RSS_VNEXT_EVIDENCE_CONTROL_DOF_NOT_BINARY")
        if ev["anchor_parent_kind"] not in (-1, 0, 1):
            _reject("RSS_VNEXT_EVIDENCE_CONTROL_ANCHOR_PARENT_INVALID")
    if len(events) > 4096:
        _reject("RSS_VNEXT_EVIDENCE_CONTROL_RESOURCE_LIMIT")


def validate_inference_evidence_v1(
    value: RSSvNextInferenceEvidenceIRV1,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> None:
    if type(value) is not RSSvNextInferenceEvidenceIRV1 or value.schema_version != _SCHEMA:
        _reject("RSS_VNEXT_EVIDENCE_SCHEMA_DRIFT")
    if (
        value.qualification_status != _AUTHORITY or value.observed_view_count != 2
        or value.auxiliary_view_authority != "LEARNED_PREDICTED_ONLY"
    ):
        _reject("RSS_VNEXT_EVIDENCE_AUTHORITY_DRIFT")
    for digest in (
        value.checkpoint_sha256, value.observed_front_sha256,
        value.observed_back_sha256, value.camera_binding_sha256,
    ):
        if not _hex(digest):
            _reject("RSS_VNEXT_EVIDENCE_SOURCE_DIGEST_REQUIRED")
    if expected_checkpoint_sha256 is not None and value.checkpoint_sha256 != expected_checkpoint_sha256:
        _reject("RSS_VNEXT_EVIDENCE_CHECKPOINT_BINDING_DRIFT")
    if value.scene is not None:
        _scene(value.scene)
    if value.mesh_status == "ABSTAIN":
        if value.mesh is not None or value.dynamic is not None or value.control_events or not value.reason:
            _reject("RSS_VNEXT_EVIDENCE_ABSTAIN_MUST_NOT_PROMOTE")
    elif value.mesh_status == "UNQUALIFIED_MESH_CANDIDATE":
        n = _mesh(value.mesh)
        if value.scene is not None and "region_query_xyz" in value.scene:
            # The candidate mesh queries the SAME region slot vocabulary as
            # H(x), at exact decoded vertex coordinates. Vocabulary width
            # mismatch cannot be silently repaired by Compiler.
            slot_count = len(value.scene["region_relation_valid_mask"])
            if any(len(row) != slot_count + 1 for row in value.mesh["semantic_logits"]):
                _reject("RSS_VNEXT_EVIDENCE_MESH_REGION_VOCABULARY_DRIFT")
            if any(len(row) != 5 for row in value.mesh["graphics_logits"]):
                _reject("RSS_VNEXT_EVIDENCE_MESH_GRAPHICS_VOCABULARY_DRIFT")
            roles = len(value.scene["presentation_role_probabilities"][0])
            if any(len(row) != roles for row in value.mesh["presentation_role_logits"]):
                _reject("RSS_VNEXT_EVIDENCE_MESH_PRESENTATION_ROLE_VOCABULARY_DRIFT")
        if value.dynamic is not None:
            if not value.control_events:
                _reject("RSS_VNEXT_EVIDENCE_DYNAMIC_WITHOUT_CONTROL")
            _dynamic(value.dynamic, n)
        # RSS model exports RigProgramEventV6 (one typed causal vocabulary).
        # Preserve unqualified status while refusing malformed/retagged neural
        # sequences *before* any model-to-Compiler qualification. Do not try to
        # infer missing pointers, quantize float codes or invent program tokens.
        _validate_neural_control_events(value.control_events)
    else:
        _reject("RSS_VNEXT_EVIDENCE_MESH_STATUS_INVALID")
    if value.evidence_hash != inference_evidence_hash_v1(value):
        _reject("RSS_VNEXT_EVIDENCE_HASH_DRIFT")


def candidate_face_region_readout_v1(
    value: RSSvNextInferenceEvidenceIRV1,
) -> tuple[dict[str, Any], ...]:
    """Deterministically read region evidence at EXACT candidate triangle IDs.

    This is NOT a drawable partition, joint label, source segmentation or
    independent-control proof.  The model explicitly predicts per-vertex
    R+UNKNOWN logits on its decoded topology.  Compiler may average those
    *named-by-index* probability vectors over each face without inventing a
    new semantic class or guessing source rays/depth.  Face identity stays in
    the neural mesh proposal's address domain; it must be requalified before
    any canonical mesh binding or persistent drawable identity is minted.
    """
    validate_inference_evidence_v1(value)
    if value.mesh_status != "UNQUALIFIED_MESH_CANDIDATE" or value.mesh is None:
        _reject("RSS_VNEXT_FACE_REGION_REQUIRES_MESH_PROPOSAL")
    if value.scene is None or "region_query_xyz" not in value.scene:
        _reject("RSS_VNEXT_FACE_REGION_REQUIRES_ADDRESSED_SCENE")
    mesh = value.mesh
    result = []
    for face_index, vertices in enumerate(mesh["faces"]):
        vector = [0.0] * len(mesh["semantic_logits"][0])
        for vertex_id in vertices:
            logits = mesh["semantic_logits"][vertex_id]
            highest = max(logits)
            exps = [math.exp(float(x) - highest) for x in logits]
            normalizer = sum(exps)
            for i, raw in enumerate(exps):
                vector[i] += raw / normalizer / 3.0
        result.append({
            "candidate_face_index": face_index,
            "candidate_vertex_indices": tuple(vertices),
            "candidate_component_id": mesh["face_components"][face_index],
            "candidate_region_probabilities": tuple(vector[:-1]),
            "unknown_probability": vector[-1],
            "candidate_evidence_hash": value.evidence_hash,
            "checkpoint_sha256": value.checkpoint_sha256,
            "authority": "UNQUALIFIED_FACE_POOL_OF_LEARNED_VERTEX_MEMBERSHIP",
            "source_surface_or_drawable_authority": False,
        })
    return tuple(result)


def seal_inference_evidence_v1(
    value: RSSvNextInferenceEvidenceIRV1,
) -> RSSvNextInferenceEvidenceIRV1:
    # Give a typed qualification error before canonical hashing encounters a
    # NaN/invalid field. The same validation is repeated after sealing.
    if value.mesh is not None:
        _mesh(value.mesh)
    # Admission must precede canonical hashing for every optional evidence
    # family, not just Mesh/AXIS. A poisoned semantic field or dynamic weight
    # otherwise reaches the JSON hasher before it can raise a typed error.
    if value.scene is not None:
        _scene(value.scene)
    if value.dynamic is not None and value.mesh is not None:
        _dynamic(value.dynamic, len(value.mesh["rest_xyz"]))
    # Reject invalid neural causal event fields before content hashing.
    _validate_neural_control_events(value.control_events)
    sealed = replace(value, evidence_hash=inference_evidence_hash_v1(value))
    validate_inference_evidence_v1(sealed)
    return sealed


__all__ = [
    "RSSvNextInferenceEvidenceIRV1",
    "inference_evidence_hash_v1",
    "seal_inference_evidence_v1",
    "validate_inference_evidence_v1",
    "candidate_face_region_readout_v1",
]
