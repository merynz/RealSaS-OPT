from __future__ import annotations

"""Exact two-view source/evidence identity court for RSS.vNext product consumers.

RealSaS/RealSaS++ use joint geometry, semantics, and appearance; a sampled image
ray is nevertheless not a source 3D point, an editable layer, or a verified
material. The court here answers a narrower question: did the inference receipt
actually name the two *original* RGBA tensors and cameras held by this CAS
source parent? It cannot certify silhouette fidelity, depth, ownership,
mechanical attachment, source rights, or a CanonicalMesh.

Match the exporter _source_tensor_digest (torch.float32, JSON shape/dtype
header, contiguous bytes) without importing Torch or a learned model. This
prevents a rehashed neural-evidence/index pair from being rebound to another
legitimate two-image CAS parent in a downstream Compiler consumer.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Any

import numpy as np

from .hashing import content_sha256
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    validate_inference_evidence_v1,
)
from .rss_vnext_source_observation_v1 import RSSvNextTwoViewSourceObservationV1
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.SourceEvidenceBindingCourt.v1"
_AUTHORITY = "EXACT_OBSERVED_TENSOR_IDENTITY_ONLY_NOT_SOURCE_FIDELITY"


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_SOURCE_EVIDENCE_BINDING_" + code)


def _exporter_tensor_sha256(values: np.ndarray) -> str:
    """Exactly reproduce inference_evidence_export_v1._source_tensor_digest."""
    tensor = np.ascontiguousarray(values, dtype=np.float32)
    header = json.dumps(
        {"dtype": "torch.float32", "shape": list(tensor.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return sha256(header + b"\x00" + tensor.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class RSSvNextSourceEvidenceBindingCourtV1:
    source_binding_sha256: str
    inference_evidence_hash: str
    checkpoint_sha256: str
    observed_front_sha256: str
    observed_back_sha256: str
    camera_binding_sha256: str
    court_hash: str
    schema: str = SCHEMA
    state: str = "EXACT_TWO_OBSERVED_VIEWS_AND_CAMERAS_BOUND"
    authority: str = _AUTHORITY
    source_surface_fidelity_qualified: bool = False
    source_visibility_qualified: bool = False
    correspondence_qualified: bool = False
    canonical_mesh_authority: bool = False
    executable_puppet_authority: bool = False
    source_rights_verified: bool = False


def validate_rss_vnext_source_evidence_binding_v1(
    *,
    source: RSSvNextTwoViewSourceObservationV1,
    evidence: RSSvNextInferenceEvidenceIRV1,
    expected_source_binding_sha256: str,
) -> RSSvNextSourceEvidenceBindingCourtV1:
    """Compare original CAS source *bytes*, not inferred pixel ownership.

    An index/receipt hash alone is no substitute for checking tensor content.
    The source object must already have been independently decoded from a
    CAS ref whose full byte hash was checked by the Platform adapter.
    """
    if type(source) is not RSSvNextTwoViewSourceObservationV1:
        _fail("ORIGINAL_CANONICAL_SOURCE_REQUIRED")
    if type(evidence) is not RSSvNextInferenceEvidenceIRV1:
        _fail("TYPED_INFERENCE_EVIDENCE_REQUIRED")
    validate_inference_evidence_v1(evidence)
    if (
        type(expected_source_binding_sha256) is not str
        or expected_source_binding_sha256 != source.source_binding_sha256
    ):
        _fail("CANONICAL_SOURCE_PARENT_DRIFT")

    # The producer normalizes uint8/255 in float32. No alpha compositing,
    # interpolation, resizing, unpremultiplication, or synthetic views.
    observed = source.native_rgba
    for view_index, declared in enumerate((
        evidence.observed_front_sha256, evidence.observed_back_sha256,
    )):
        normalized = np.ascontiguousarray(observed[view_index], dtype=np.float32)
        normalized /= np.float32(255.0)
        if _exporter_tensor_sha256(normalized) != declared:
            _fail("ORIGINAL_RGBA_TENSOR_MISMATCH_VIEW_" + str(view_index))

    camera_tensor = source.camera_features[np.newaxis, :, :]
    if _exporter_tensor_sha256(camera_tensor) != evidence.camera_binding_sha256:
        _fail("ORIGINAL_CAMERA_TENSOR_MISMATCH")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source_binding_sha256": source.source_binding_sha256,
        "inference_evidence_hash": evidence.evidence_hash,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "observed_front_sha256": evidence.observed_front_sha256,
        "observed_back_sha256": evidence.observed_back_sha256,
        "camera_binding_sha256": evidence.camera_binding_sha256,
        "state": "EXACT_TWO_OBSERVED_VIEWS_AND_CAMERAS_BOUND",
        "authority": _AUTHORITY,
        "source_surface_fidelity_qualified": False,
        "source_visibility_qualified": False,
        "correspondence_qualified": False,
        "canonical_mesh_authority": False,
        "executable_puppet_authority": False,
        "source_rights_verified": False,
    }
    return RSSvNextSourceEvidenceBindingCourtV1(
        source_binding_sha256=source.source_binding_sha256,
        inference_evidence_hash=evidence.evidence_hash,
        checkpoint_sha256=evidence.checkpoint_sha256,
        observed_front_sha256=evidence.observed_front_sha256,
        observed_back_sha256=evidence.observed_back_sha256,
        camera_binding_sha256=evidence.camera_binding_sha256,
        court_hash=content_sha256(payload),
    )


def validate_rss_vnext_source_evidence_binding_receipt_v1(
    value: RSSvNextSourceEvidenceBindingCourtV1,
) -> None:
    if (
        type(value) is not RSSvNextSourceEvidenceBindingCourtV1
        or value.schema != SCHEMA
        or value.state != "EXACT_TWO_OBSERVED_VIEWS_AND_CAMERAS_BOUND"
        or value.authority != _AUTHORITY
        or any((value.source_surface_fidelity_qualified,
                value.source_visibility_qualified,
                value.correspondence_qualified,
                value.canonical_mesh_authority,
                value.executable_puppet_authority,
                value.source_rights_verified))
    ):
        _fail("RECEIPT_AUTHORITY_DRIFT")
    fields = asdict(value)
    digest = fields.pop("court_hash")
    if content_sha256(fields) != digest:
        _fail("RECEIPT_CONTENT_DRIFT")


__all__ = [
    "SCHEMA", "RSSvNextSourceEvidenceBindingCourtV1",
    "validate_rss_vnext_source_evidence_binding_v1",
    "validate_rss_vnext_source_evidence_binding_receipt_v1",
]
