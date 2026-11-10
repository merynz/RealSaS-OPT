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
        "topology_tokens", "rest_xyz", "faces", "vertex_components",
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
            if type(ray["valid"][index][view]) is not bool:
                _reject("RSS_VNEXT_EVIDENCE_RAY_VALID_TYPE")
            alpha = ray["alpha"][index][view]
            if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
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
    if value.mesh_status == "ABSTAIN":
        if value.mesh is not None or value.dynamic is not None or value.control_events or not value.reason:
            _reject("RSS_VNEXT_EVIDENCE_ABSTAIN_MUST_NOT_PROMOTE")
    elif value.mesh_status == "UNQUALIFIED_MESH_CANDIDATE":
        n = _mesh(value.mesh)
        if value.dynamic is not None:
            if not value.control_events:
                _reject("RSS_VNEXT_EVIDENCE_DYNAMIC_WITHOUT_CONTROL")
            _dynamic(value.dynamic, n)
        for ev in value.control_events:
            if not isinstance(ev, Mapping) or not ev or "kind" not in ev:
                _reject("RSS_VNEXT_EVIDENCE_CONTROL_EVENT_INVALID")
    else:
        _reject("RSS_VNEXT_EVIDENCE_MESH_STATUS_INVALID")
    if value.evidence_hash != inference_evidence_hash_v1(value):
        _reject("RSS_VNEXT_EVIDENCE_HASH_DRIFT")


def seal_inference_evidence_v1(
    value: RSSvNextInferenceEvidenceIRV1,
) -> RSSvNextInferenceEvidenceIRV1:
    sealed = replace(value, evidence_hash=inference_evidence_hash_v1(value))
    validate_inference_evidence_v1(sealed)
    return sealed


__all__ = [
    "RSSvNextInferenceEvidenceIRV1",
    "inference_evidence_hash_v1",
    "seal_inference_evidence_v1",
    "validate_inference_evidence_v1",
]
