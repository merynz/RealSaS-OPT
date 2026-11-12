from __future__ import annotations

"""Lossless-by-reference RSS.vNext Shape -> Mesh -> later expert transport.

This is a CAS lineage / continuation receipt, NOT a source-truth qualifier.
A candidate surface, or its ABSTAIN, must never become the only artifact
visible to later Graphics, Mechanics, source-ray or training-feedback consumers.
"""

from copy import deepcopy
from typing import Any, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from .rss_vnext_product_fanout_v1 import (
    SCHEMA as FANOUT_SCHEMA,
    bind_rss_vnext_fanout_refs_v1,
)

SCHEMA = "RealSaS.RSSvNext.EvidenceContinuity.v1"
ARTIFACT_TYPE = "RealSaS.RSSvNextEvidenceContinuity"
ROLE = "rss_evidence_continuity"
AUTHORITY = "IMMUTABLE_EVIDENCE_REFERENCES_NOT_PRODUCT_QUALIFICATION"
_CHILD_ROLES = frozenset((
    "original_source", "normalized_model_source", "neural_evidence",
    "neural_scene", "inference_receipt", "unified_model_config",
    "embedded_mira_decoder",
))
_DERIVED_ROLES = frozenset((
    "rss_computed_surface_candidate", "rss_sparse_face_relation_candidates",
    "rss_cross_expert_capability_court",
    "rss_relation_epistemic_court",
))
_FIELDS = frozenset((
    "schema", "source_ref", "fanout_index_ref", "preserved_children",
    "derived_candidates", "source_binding_sha256", "unified_checkpoint_sha256",
    "unified_model_release_behavior_sha256", "evidence_hash", "mesh_status",
    "neural_scene_replaced_by_candidate", "has_product_authority",
    "source_rights_verified", "authority_class", "continuity_hash",
))


def _reference(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError("RSS_VNEXT_CONTINUITY_REFERENCE_INVALID")
    required = {"artifact_type", "schema_version", "storage_key", "content_sha256", "size_bytes"}
    if not required.issubset(value):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_REFERENCE_FIELDS_MISSING")
    if (
        type(value["artifact_type"]) is not str or not value["artifact_type"]
        or type(value["schema_version"]) is not str or not value["schema_version"]
        or type(value["storage_key"]) is not str or not value["storage_key"]
        or type(value["content_sha256"]) is not str or len(value["content_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["content_sha256"])
        or type(value["size_bytes"]) is not int or value["size_bytes"] <= 0
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_REFERENCE_DOMAIN_INVALID")
    # Preserve all CAS-level reference fields; never replace the parent's
    # complete artifact identity with just a digest or guessed model label.
    return deepcopy(value)


def _hash(data: Mapping[str, Any]) -> str:
    return content_sha256({k: v for k, v in data.items() if k != "continuity_hash"})


def seal_rss_vnext_evidence_continuity_v1(
    *, index: Mapping[str, Any],
    source_ref: dict[str, Any], index_ref: dict[str, Any],
    child_refs: Mapping[str, dict[str, Any]],
    derived_refs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Bind all seven source/model/decoder children and current candidate output.

    No H(x) archive is opened, copied, quantized, converted to a mesh, or
    discarded. All later consumers can address the original CAS children.
    """
    if not isinstance(index, Mapping) or index.get("schema") != FANOUT_SCHEMA:
        raise RuntimeError("RSS_VNEXT_CONTINUITY_FANOUT_INDEX_REQUIRED")
    if (source_ref.get("artifact_type") != "RealSaS.RSSvNextTwoViewSource"
        or index_ref.get("artifact_type") != "RealSaS.RSSvNextProductEvidenceFanout"
        or index_ref.get("schema_version") != "v1"
        or index.get("source_artifact_sha256") != source_ref.get("content_sha256")
        or index.get("mesh_status") not in {"ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE"}
        or index.get("original_view_count") != 2
        or index.get("canonical_mesh_authority") is not False
        or index.get("product_revision_minted") is not False):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_PARENT_AUTHORITY_DRIFT")
    if set(child_refs) != _CHILD_ROLES:
        raise RuntimeError("RSS_VNEXT_CONTINUITY_ALL_SEVEN_EXPERT_CHILDREN_REQUIRED")
    bound = bind_rss_vnext_fanout_refs_v1(
        index, input_artifacts=list(child_refs.values()),
    )
    if set(bound) != _CHILD_ROLES:
        raise RuntimeError("RSS_VNEXT_CONTINUITY_CHILD_BINDING_DRIFT")
    if (
        type(derived_refs) is not tuple
        or len(derived_refs) not in (3, 4)
        or {r.get("role") for r in derived_refs} - _DERIVED_ROLES
        or derived_refs[0].get("role") != "rss_computed_surface_candidate"
        or derived_refs[-2].get("role") != "rss_relation_epistemic_court"
        or derived_refs[-1].get("role") != "rss_cross_expert_capability_court"
        or len({r.get("role") for r in derived_refs}) != len(derived_refs)
        or (index["mesh_status"] == "ABSTAIN" and len(derived_refs) != 3)
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_DERIVED_CANDIDATE_DOMAIN")
    children = [_reference(bound[role]) for role in sorted(_CHILD_ROLES)]
    derived = [_reference(ref) for ref in derived_refs]
    result = {
        "schema": SCHEMA,
        "source_ref": _reference(source_ref),
        "fanout_index_ref": _reference(index_ref),
        "preserved_children": children,
        "derived_candidates": derived,
        "source_binding_sha256": index["source_binding_sha256"],
        "unified_checkpoint_sha256": index["unified_checkpoint_sha256"],
        "unified_model_release_behavior_sha256": index["unified_model_release_behavior_sha256"],
        "evidence_hash": index["evidence_hash"],
        "mesh_status": index["mesh_status"],
        "neural_scene_replaced_by_candidate": False,
        "has_product_authority": False,
        "source_rights_verified": False,
        "authority_class": AUTHORITY,
    }
    result["continuity_hash"] = _hash(result)
    validate_rss_vnext_evidence_continuity_v1(
        result, index=index, source_ref=source_ref,
        index_ref=index_ref, child_refs=child_refs, derived_refs=derived_refs,
    )
    return result


def validate_rss_vnext_evidence_continuity_v1(
    continuity: Mapping[str, Any], *,
    index: Mapping[str, Any],
    source_ref: dict[str, Any], index_ref: dict[str, Any],
    child_refs: Mapping[str, dict[str, Any]],
    derived_refs: tuple[dict[str, Any], ...],
) -> None:
    """Fail closed on a lost/swapped parent, decoder, source or candidate."""
    if (
        type(continuity) is not dict or set(continuity) != _FIELDS
        or continuity.get("schema") != SCHEMA
        or continuity.get("authority_class") != AUTHORITY
        or continuity.get("neural_scene_replaced_by_candidate") is not False
        or continuity.get("has_product_authority") is not False
        or continuity.get("source_rights_verified") is not False
        or continuity.get("continuity_hash") != _hash(continuity)
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_RECEIPT_HASH_OR_AUTHORITY_DRIFT")
    if any(continuity.get(k) != index.get(ik) for k, ik in (
        ("source_binding_sha256", "source_binding_sha256"),
        ("unified_checkpoint_sha256", "unified_checkpoint_sha256"),
        ("unified_model_release_behavior_sha256", "unified_model_release_behavior_sha256"),
        ("evidence_hash", "evidence_hash"), ("mesh_status", "mesh_status"),
    )):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_PARENT_IDENTITY_DRIFT")
    if (
        continuity["source_ref"] != _reference(source_ref)
        or continuity["fanout_index_ref"] != _reference(index_ref)
        or index.get("source_artifact_sha256") != source_ref["content_sha256"]
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_SOURCE_OR_INDEX_REF_DRIFT")
    if set(child_refs) != _CHILD_ROLES:
        raise RuntimeError("RSS_VNEXT_CONTINUITY_CHILD_CARDINALITY_DRIFT")
    bound = bind_rss_vnext_fanout_refs_v1(
        index, input_artifacts=list(child_refs.values()),
    )
    children = continuity["preserved_children"]
    if (
        type(children) is not list or len(children) != len(_CHILD_ROLES)
        or children != [_reference(bound[role]) for role in sorted(_CHILD_ROLES)]
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_EXPERT_EVIDENCE_LOST_OR_SWAPPED")
    if (
        type(continuity["derived_candidates"]) is not list
        or continuity["derived_candidates"] != [_reference(r) for r in derived_refs]
        or not derived_refs
        or derived_refs[0]["role"] != "rss_computed_surface_candidate"
    ):
        raise RuntimeError("RSS_VNEXT_CONTINUITY_DERIVED_CANDIDATE_REF_DRIFT")


__all__ = [
    "SCHEMA", "ARTIFACT_TYPE", "ROLE", "AUTHORITY",
    "seal_rss_vnext_evidence_continuity_v1",
    "validate_rss_vnext_evidence_continuity_v1",
]
