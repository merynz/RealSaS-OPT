from __future__ import annotations

"""Two original CAS views -> unqualified unified evidence, byte-for-byte.

The artificial pixels and camera here test binding only; no learned accuracy,
source 3D correspondence, source fidelity, or product release is asserted.
"""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    canonicalize_two_view_source_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    _exporter_tensor_sha256,
    validate_rss_vnext_source_evidence_binding_v1,
    validate_rss_vnext_source_evidence_binding_receipt_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext.inference_evidence_export_v1 import _source_tensor_digest


def _source():
    pixels = np.arange(2 * 4 * 14 * 14, dtype=np.uint32).reshape(2, 4, 14, 14)
    pixels = ((pixels * 37 + 19) % 255).astype(np.uint8)
    pixels[:, 3] = 255
    cam = np.zeros((2, 13), dtype=np.float32)
    cam[0, 3] = cam[0, 7] = cam[0, 11] = cam[0, 12] = 1.
    cam[1, 3] = -1.
    cam[1, 7] = 1.
    cam[1, 11] = -1.
    cam[1, 12] = 1.
    return canonicalize_two_view_source_v1(
        pixels, cam, source_rights_claim_sha256="e" * 64,
    )


def _evidence(source):
    rgba, camera = source.to_model_tensors()
    return seal_inference_evidence_v1(RSSvNextInferenceEvidenceIRV1(
        checkpoint_sha256="b" * 64,
        observed_front_sha256=_source_tensor_digest(rgba[0, 0]),
        observed_back_sha256=_source_tensor_digest(rgba[0, 1]),
        camera_binding_sha256=_source_tensor_digest(camera),
        mesh_status="ABSTAIN", mesh=None,
        reason="SYNTHETIC_NO_TOPOLOGY",
    ))


def _check(source, evidence, *, binding=None):
    return validate_rss_vnext_source_evidence_binding_v1(
        source=source,
        evidence=evidence,
        expected_source_binding_sha256=(
            source.source_binding_sha256 if binding is None else binding
        ),
    )


def test_exporter_and_compiler_reproduce_exact_byte_hashes_without_torch_dependency_in_consumer():
    source = _source()
    rgba, camera = source.to_model_tensors()
    assert _exporter_tensor_sha256(
        source.native_rgba[0].astype(np.float32) / np.float32(255.)
    ) == _source_tensor_digest(rgba[0, 0])
    assert _exporter_tensor_sha256(
        source.native_rgba[1].astype(np.float32) / np.float32(255.)
    ) == _source_tensor_digest(rgba[0, 1])
    assert _exporter_tensor_sha256(source.camera_features[None]) == _source_tensor_digest(camera)


def test_one_real_cas_source_and_one_evidence_bind_as_unqualified_exact_bytes():
    source = _source()
    evidence = _evidence(source)
    original = evidence.to_dict()
    receipt = _check(source, evidence)
    validate_rss_vnext_source_evidence_binding_receipt_v1(receipt)
    assert receipt.source_binding_sha256 == source.source_binding_sha256
    assert receipt.inference_evidence_hash == evidence.evidence_hash
    assert receipt.state == "EXACT_TWO_OBSERVED_VIEWS_AND_CAMERAS_BOUND"
    assert receipt.authority == "EXACT_OBSERVED_TENSOR_IDENTITY_ONLY_NOT_SOURCE_FIDELITY"
    assert not receipt.source_surface_fidelity_qualified
    assert not receipt.source_visibility_qualified
    assert not receipt.correspondence_qualified
    assert not receipt.canonical_mesh_authority
    assert not receipt.executable_puppet_authority
    assert not receipt.source_rights_verified
    assert evidence.to_dict() == original


@pytest.mark.parametrize("view_index", [0, 1])
def test_rehashed_neural_receipt_cannot_switch_observed_source_pixels(view_index):
    source = _source()
    evidence = _evidence(source)
    forged = replace(evidence,
                     observed_front_sha256=("c" * 64 if view_index == 0 else evidence.observed_front_sha256),
                     observed_back_sha256=("d" * 64 if view_index == 1 else evidence.observed_back_sha256),
                     evidence_hash="")
    forged = seal_inference_evidence_v1(forged)
    with pytest.raises(QualificationError, match=f"ORIGINAL_RGBA_TENSOR_MISMATCH_VIEW_{view_index}"):
        _check(source, forged)


def test_source_swap_with_self_consistent_cas_parent_cannot_claim_same_neural_evidence():
    source = _source()
    evidence = _evidence(source)
    swapped = source.native_rgba.copy()
    swapped[0, 0, 0, 0] ^= 1
    other = canonicalize_two_view_source_v1(
        swapped, source.camera_features.copy(),
        source_rights_claim_sha256=source.source_rights_claim_sha256,
    )
    with pytest.raises(QualificationError, match="ORIGINAL_RGBA_TENSOR_MISMATCH_VIEW_0"):
        _check(other, evidence)


def test_camera_change_does_not_get_hidden_by_identical_source_rgba():
    source = _source()
    evidence = _evidence(source)
    camera = source.camera_features.copy()
    camera[0, 0] = 0.125
    altered = canonicalize_two_view_source_v1(
        source.native_rgba.copy(), camera,
        source_rights_claim_sha256=source.source_rights_claim_sha256,
    )
    with pytest.raises(QualificationError, match="ORIGINAL_CAMERA_TENSOR_MISMATCH"):
        _check(altered, evidence)


def test_original_cas_source_parent_claim_is_checked_separately():
    source = _source()
    evidence = _evidence(source)
    with pytest.raises(QualificationError, match="CANONICAL_SOURCE_PARENT_DRIFT"):
        _check(source, evidence, binding="a" * 64)


def test_tampered_receipt_cannot_elevate_capability_even_when_rehashed():
    source = _source()
    evidence = _evidence(source)
    receipt = _check(source, evidence)
    with pytest.raises(QualificationError, match="RECEIPT_AUTHORITY_DRIFT"):
        validate_rss_vnext_source_evidence_binding_receipt_v1(replace(
            receipt, canonical_mesh_authority=True,
        ))
    with pytest.raises(QualificationError, match="RECEIPT_CONTENT_DRIFT"):
        validate_rss_vnext_source_evidence_binding_receipt_v1(replace(
            receipt, observed_front_sha256="f" * 64,
        ))


def test_source_binding_is_typed_and_does_not_accept_old_eight_view_input():
    source = _source()
    evidence = _evidence(source)
    with pytest.raises(QualificationError, match="ORIGINAL_CANONICAL_SOURCE_REQUIRED"):
        validate_rss_vnext_source_evidence_binding_v1(
            source={"views": 8}, evidence=evidence,
            expected_source_binding_sha256=source.source_binding_sha256,
        )
