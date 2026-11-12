from __future__ import annotations

"""Independent CAS read court for unified RSS.vNext product evidence.

Lightweight Compiler consumers validate the *index* and model JSON without
decompressing huge neural triplanes. Semantic, Mechanics or Graphics can open
the FULL same H(x) scene independently on demand. Every child stays separately
addressable and provenance-bound; no CanonicalMesh or ProductRevision is
ever inferred from receipt hashes alone.
"""

from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)

SCHEMA = "RealSaS.RSSvNext.ProductEvidenceFanout.v1"
INDEX_ARTIFACT_TYPE = "RealSaS.RSSvNextProductEvidenceFanout"
_MAX_INDEX_BYTES = 64 * 1024
_CHILDREN = {
    "original_source": ("RealSaS.RSSvNextOriginalSource", "v1"),
    "normalized_model_source": ("RealSaS.RSSvNextModelSourceTensors", "v1"),
    "neural_evidence": ("RealSaS.RSSvNextInferenceEvidence", "v1"),
    "neural_scene": ("RealSaS.RSSvNextNeuralSceneCarrier", "v1"),
    "inference_receipt": ("RealSaS.RSSvNextInferenceReceipt", "v2"),
}
_INDEX_FIELDS = {
    "schema", "source_artifact_sha256", "source_binding_sha256",
    "source_rights_claim_sha256", "unified_model_release_behavior_sha256",
    "unified_checkpoint_sha256", "unified_model_config_sha256",
    "evidence_hash", "mesh_status", "original_view_count",
    "original_source_rights_verified", "learned_evidence_is_not_source_truth",
    "canonical_mesh_authority", "product_revision_minted",
    "topology_abstain_retains_shared_carrier", "children",
}


def _hex(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def _cas_bytes(
    ref: Mapping[str, Any],
    payload: bytes,
    *,
    error_prefix: str,
) -> None:
    if (
        type(ref) is not dict or type(payload) is not bytes
        or ref.get("content_sha256") != sha256(payload).hexdigest()
        or type(ref.get("size_bytes")) is not int
        or ref["size_bytes"] != len(payload)
    ):
        raise RuntimeError(error_prefix + "_CAS_BYTES_DRIFT")


def validate_rss_vnext_fanout_index_v1(
    index_payload: bytes,
    *,
    expected_source_content_sha256: str,
    expected_source_binding_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_release_behavior_sha256: str,
) -> dict[str, Any]:
    if type(index_payload) is not bytes or not 1 <= len(index_payload) <= _MAX_INDEX_BYTES:
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_BYTES_INVALID")
    try:
        data = json.loads(index_payload.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_JSON_INVALID") from exc
    if type(data) is not dict or set(data) != _INDEX_FIELDS or data["schema"] != SCHEMA:
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_FIELDS_INVALID")
    expected = {
        "source_artifact_sha256": expected_source_content_sha256,
        "source_binding_sha256": expected_source_binding_sha256,
        "unified_checkpoint_sha256": expected_checkpoint_sha256,
        "unified_model_release_behavior_sha256": expected_model_release_behavior_sha256,
    }
    if any(
        not _hex(value) or data.get(name) != value
        for name, value in expected.items()
    ) or any(
        not _hex(data.get(name))
        for name in ("source_rights_claim_sha256", "unified_model_config_sha256", "evidence_hash")
    ):
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_SOURCE_OR_MODEL_BINDING_DRIFT")
    if (
        data.get("mesh_status") not in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE")
        or type(data.get("original_view_count")) is not int
        or data["original_view_count"] != 2
        or data.get("original_source_rights_verified") is not False
        or data.get("learned_evidence_is_not_source_truth") is not True
        or data.get("canonical_mesh_authority") is not False
        or data.get("product_revision_minted") is not False
        or data.get("topology_abstain_retains_shared_carrier") is not True
    ):
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_PREMATURE_AUTHORITY")
    children = data["children"]
    if type(children) is not list or len(children) != len(_CHILDREN):
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_CHILD_CARDINALITY")
    seen = set()
    digests = set()
    for row in children:
        if type(row) is not dict or set(row) != {
            "role", "artifact_type", "schema_version", "content_sha256",
            "size_bytes",
        }:
            raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_CHILD_FIELDS_INVALID")
        role = row["role"]
        if type(role) is not str or role not in _CHILDREN or role in seen:
            raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_CHILD_ROLE_INVALID")
        seen.add(role)
        if (
            (row["artifact_type"], row["schema_version"]) != _CHILDREN[role]
            or not _hex(row["content_sha256"])
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
        ):
            raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_CHILD_ABI_DRIFT")
        digests.add(row["content_sha256"])
    if seen != set(_CHILDREN) or len(digests) != len(_CHILDREN):
        raise RuntimeError("RSS_VNEXT_FANOUT_INDEX_CHILD_OMITTED_OR_ALIASED")
    return data


def bind_rss_vnext_fanout_refs_v1(
    index: Mapping[str, Any],
    *,
    input_artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    """Read-only source/child index court, no expensive neural-scene load."""
    if not isinstance(index, Mapping) or index.get("schema") != SCHEMA:
        raise RuntimeError("RSS_VNEXT_FANOUT_READ_UNVALIDATED_INDEX")
    by_role: dict[str, dict[str, Any]] = {}
    for row in input_artifacts:
        if type(row) is not dict:
            raise RuntimeError("RSS_VNEXT_FANOUT_READ_CHILD_TYPE")
        role = row.get("role")
        if role in _CHILDREN:
            if role in by_role:
                raise RuntimeError("RSS_VNEXT_FANOUT_READ_DUPLICATE_CHILD")
            by_role[role] = row
    if set(by_role) != set(_CHILDREN):
        raise RuntimeError("RSS_VNEXT_FANOUT_READ_MISSING_CHILD")
    for indexed in index["children"]:
        row = by_role[indexed["role"]]
        if any(row.get(key) != indexed[key] for key in (
            "role", "artifact_type", "schema_version", "content_sha256",
            "size_bytes",
        )) or not row.get("storage_key"):
            raise RuntimeError("RSS_VNEXT_FANOUT_READ_CHILD_BINDING_DRIFT")
    return by_role


def read_rss_vnext_neural_evidence_v1(
    index: Mapping[str, Any],
    *,
    child_refs: Mapping[str, dict[str, Any]],
    read_object: Callable[[dict[str, Any]], bytes],
) -> RSSvNextInferenceEvidenceIRV1:
    """The cheap, independently validatable model proposal before Mesh."""
    row = child_refs["neural_evidence"]
    payload = read_object(row)
    _cas_bytes(row, payload, error_prefix="RSS_VNEXT_NEURAL_EVIDENCE")
    try:
        decoded = json.loads(payload.decode("utf-8"))
        evidence = RSSvNextInferenceEvidenceIRV1(**decoded)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError("RSS_VNEXT_NEURAL_EVIDENCE_JSON_INVALID") from exc
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=index["unified_checkpoint_sha256"],
    )
    if (
        evidence.evidence_hash != index["evidence_hash"]
        or evidence.mesh_status != index["mesh_status"]
    ):
        raise RuntimeError("RSS_VNEXT_NEURAL_EVIDENCE_INDEX_DRIFT")
    return evidence


def read_rss_vnext_shared_neural_scene_v1(
    index: Mapping[str, Any],
    *,
    child_refs: Mapping[str, dict[str, Any]],
    read_object: Callable[[dict[str, Any]], bytes],
) -> dict[str, Any]:
    """Heavy on-demand consumer: full H/fields, same source and checkpoint."""
    from models.rss_vnext.neural_scene_archive_v1 import read_neural_scene_carrier_v1

    evidence = read_rss_vnext_neural_evidence_v1(
        index, child_refs=child_refs, read_object=read_object,
    )
    row = child_refs["neural_scene"]
    payload = read_object(row)
    _cas_bytes(row, payload, error_prefix="RSS_VNEXT_NEURAL_SCENE")
    return read_neural_scene_carrier_v1(payload, evidence=evidence)


__all__ = [
    "SCHEMA", "INDEX_ARTIFACT_TYPE",
    "validate_rss_vnext_fanout_index_v1", "bind_rss_vnext_fanout_refs_v1",
    "read_rss_vnext_neural_evidence_v1", "read_rss_vnext_shared_neural_scene_v1",
]
