from __future__ import annotations

"""Two genuine uncalibrated uploads -> pinned generative H(x) construction.

Original files remain byte-exact in upstream CAS. This is an unqualified
neural scene and AXIS/MIRA candidates, not a calibrated source,
a source-ray replay, or a product artifact. It uses the SAME two-stage sampler and six-view IRIS
construction as calibrated RSS.vNext inference and learned S-LRM training.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
from torch import Tensor

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
    decode_uncalibrated_two_image_source_v1,
    derive_uncalibrated_pair_conditioning_v1,
)
from .realsas_product_neural_bundle_v1 import RSSvNextPinnedNeuralBundleV1
from .realsas_two_source_construction_v1 import (
    RSSvNextTwoSourceConstructionV1,
    run_two_source_neural_construction_v1,
)
from .realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)

SCHEMA = "RealSaS.RSSvNext.UncalibratedSourceToNeuralScene.v1"


@dataclass(frozen=True)
class RSSvNextUncalibratedNeuralSceneCandidateV1:
    source: RSSvNextUncalibratedTwoImageSourceV1
    source_cas_sha256: str
    source_binding_sha256: str
    derived_conditioning_sha256: str
    neural_bundle_sha256: str
    construction: RSSvNextTwoSourceConstructionV1
    original_camera_features: None = None
    original_camera_observed: bool = False
    virtual_cameras_are_original: bool = False
    generated_pixels_are_original: bool = False
    source_ray_evidence_qualified: bool = False
    compiler_neural_mesh_admitted: bool = False
    product_revision_minted: bool = False
    schema: str = SCHEMA


def run_uncalibrated_two_image_neural_scene_v1(
    *, raw_cas_bytes: bytes,
    expected_raw_cas_sha256: str,
    neural_bundle: RSSvNextPinnedNeuralBundleV1,
    canonical_seed: int | None = None,
    multiview_seed: int | None = None,
    max_topology_tokens: int | None = None,
) -> RSSvNextUncalibratedNeuralSceneCandidateV1:
    """Decode two raw image files and actually run both learned stages.

    No original 13D source camera is accepted or supplied; the only virtual
    cameras in this path correspond to GENERATED canonical and multiview
    images. No original source-ray or source-depth proof may be inferred.
    """
    if (
        type(raw_cas_bytes) is not bytes or type(expected_raw_cas_sha256) is not str
        or sha256(raw_cas_bytes).hexdigest() != expected_raw_cas_sha256
    ):
        raise RuntimeError("RSS_VNEXT_RAW_NEURAL_SOURCE_CAS_PIN_DRIFT")
    if (
        type(neural_bundle) is not RSSvNextPinnedNeuralBundleV1
        or neural_bundle.model.training
        or not getattr(neural_bundle.model, "construction_requires_generated_multiview", False)
        or type(neural_bundle.checkpoint_sha256) is not str
        or len(neural_bundle.checkpoint_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in neural_bundle.checkpoint_sha256)
        or neural_bundle.canonical.checkpoint_sha256 is None
        or neural_bundle.multiview.checkpoint_sha256 is None
    ):
        raise RuntimeError("RSS_VNEXT_RAW_NEURAL_BUNDLE_NOT_PINNED_OR_PREPARED")
    source = decode_uncalibrated_two_image_source_v1(raw_cas_bytes)
    derived = derive_uncalibrated_pair_conditioning_v1(
        source, target_square_size=neural_bundle.image_size,
    )
    device = next(neural_bundle.model.parameters()).device
    pixels = torch.from_numpy(
        np.array(derived.rgba_uint8, copy=True),
    ).to(device=device, dtype=torch.float32).div_(255.).unsqueeze(0)
    if pixels.shape != (
        1, 2, 4, neural_bundle.image_size, neural_bundle.image_size,
    ):
        raise RuntimeError("RSS_VNEXT_RAW_NEURAL_TWO_VIEW_PIXEL_DOMAIN")
    canonical_cameras, six_cameras = target_canonical_and_six_cameras_v1(
        batch_size=1, device=str(device), dtype=pixels.dtype,
    )
    def _default_seed(label: str) -> int:
        return int.from_bytes(sha256((
            source.source_binding_sha256 + neural_bundle.checkpoint_sha256 + label
        ).encode("ascii")).digest()[:8], "big") % (2**63)
    if canonical_seed is None:
        canonical_seed = _default_seed("CANONICAL")
    if multiview_seed is None:
        multiview_seed = _default_seed("MULTIVIEW")
    if (
        type(canonical_seed) is not int or not 0 <= canonical_seed < 2**63
        or type(multiview_seed) is not int or not 0 <= multiview_seed < 2**63
    ):
        raise ValueError("RSS_VNEXT_RAW_NEURAL_SEED_DOMAIN")
    construction = run_two_source_neural_construction_v1(
        model=neural_bundle.model,
        observed_rgba=pixels,
        original_camera_features=None,
        source_binding_sha256=source.source_binding_sha256,
        canonical_cameras=canonical_cameras,
        six_view_cameras=six_cameras,
        canonical_sampler=neural_bundle.canonical,
        multiview_sampler=neural_bundle.multiview,
        image_size=neural_bundle.image_size,
        denoise_steps=neural_bundle.denoise_steps,
        canonical_seed=canonical_seed,
        multiview_seed=multiview_seed,
        max_topology_tokens=max_topology_tokens,
        generate_controls=True,
    )
    state = construction.proposal["state"]
    mesh = construction.proposal.get("mesh_candidate")
    if (
        state.camera_features is not None
        or state.source_camera_authority
        != "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
        or state.source_view_authority != (
            "SOURCE_OBSERVED_UNCALIBRATED_0",
            "SOURCE_OBSERVED_UNCALIBRATED_1",
        )
        or construction.original_camera_calibrated
        or (mesh is not None and mesh.source_ray_evidence is not None)
        or (mesh is None and construction.proposal.get("control_event_proposal"))
        or (mesh is not None and construction.proposal.get("control_status") != (
            "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE"
        ))
        or construction.product_revision_minted
    ):
        raise RuntimeError("RSS_VNEXT_RAW_NEURAL_FALSE_SOURCE_PROJECTION")
    return RSSvNextUncalibratedNeuralSceneCandidateV1(
        source=source,
        source_cas_sha256=expected_raw_cas_sha256,
        source_binding_sha256=source.source_binding_sha256,
        derived_conditioning_sha256=derived.conditioning_sha256,
        neural_bundle_sha256=neural_bundle.checkpoint_sha256,
        construction=construction,
    )


__all__ = [
    "SCHEMA", "RSSvNextUncalibratedNeuralSceneCandidateV1",
    "run_uncalibrated_two_image_neural_scene_v1",
]
