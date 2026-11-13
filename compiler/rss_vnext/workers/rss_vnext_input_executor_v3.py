from __future__ import annotations

"""Single product.input role for arbitrary two-image and calibrated source CAS.

All variants retain their exact source bytes. The raw-upload envelope may
contain UNKNOWN cameras and no segmented silhouette. Input cannot promote
derived pixels or virtual reconstruction cameras to original source evidence.
"""

from hashlib import sha256
from typing import Any, Callable

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    decode_uncalibrated_two_image_source_v1,
)
from .rss_vnext_source_codec_v1 import (
    ARTIFACT_TYPE as CALIBRATED_ARTIFACT_TYPE,
    SCHEMA_VERSION as CALIBRATED_SCHEMA_VERSION,
    validate_rss_vnext_source_ref_v1,
)

EXECUTOR_CONTRACT = "RealSaS.ProductInputTwoView.v3"
RAW_ARTIFACT_TYPE = "RealSaS.RSSvNextUncalibratedTwoImageSource"
RAW_SCHEMA_VERSION = "v1"


def execute_rss_vnext_input_capability_v3(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = request.get("capability_metadata") or {}
    if (
        type(request) is not dict or type(metadata) is not dict
        or request.get("capability_id") != "product.input"
        or request.get("kind") != "MODULE"
        or metadata.get("executor_contract") != EXECUTOR_CONTRACT
        or not str(request.get("execution_id") or "")
        or request.get("model_releases")
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V3_EXECUTOR_OR_MODEL_CONTRACT")
    refs = list(request.get("input_artifacts") or ())
    valid_types = {
        (RAW_ARTIFACT_TYPE, RAW_SCHEMA_VERSION),
        (CALIBRATED_ARTIFACT_TYPE, CALIBRATED_SCHEMA_VERSION),
    }
    if (
        len(refs) != 1 or type(refs[0]) is not dict
        or (refs[0].get("artifact_type"), refs[0].get("schema_version"))
        not in valid_types
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V3_EXACT_TWO_ORIGINAL_IMAGE_SOURCE")
    source_ref = refs[0]
    original = read_object(source_ref)
    if (
        type(original) is not bytes
        or source_ref.get("content_sha256") != sha256(original).hexdigest()
        or type(source_ref.get("size_bytes")) is not int
        or source_ref["size_bytes"] != len(original)
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V3_SOURCE_CAS_DRIFT")
    is_raw = source_ref["artifact_type"] == RAW_ARTIFACT_TYPE
    if is_raw:
        source = decode_uncalibrated_two_image_source_v1(original)
        binding = source.source_binding_sha256
        rights = source.source_rights_claim_sha256
    else:
        source = validate_rss_vnext_source_ref_v1(source_ref, original)
        binding = source.source_binding_sha256
        rights = source.source_rights_claim_sha256
    sealed = put_bytes(original)
    if (
        type(sealed) is not dict
        or sealed.get("content_sha256") != sha256(original).hexdigest()
        or sealed.get("size_bytes") != len(original)
        or not sealed.get("storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V3_OUTPUT_CAS_DRIFT")
    return {
        "capability_id": "product.input",
        "status": "PASS",
        "failure": None,
        "outputs": [{
            "role": "rss_vnext_original_two_image_source",
            "artifact_type": source_ref["artifact_type"],
            "schema_version": source_ref["schema_version"],
            "storage_key": sealed["storage_key"],
            "content_sha256": sealed["content_sha256"],
            "size_bytes": sealed["size_bytes"],
            "authority_class": (
                "RAW_TWO_USER_IMAGES_CAMERA_UNKNOWN_NOT_3D_SOURCE_TRUTH"
                if is_raw else
                "CALIBRATED_TWO_IMAGE_SOURCE_OBSERVATION_NO_RIGHTS_AUTHORITY"
            ),
        }],
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductInputDiagnostics.v3",
            "source_binding_sha256": binding,
            "source_rights_claim_sha256": rights,
            "original_observed_view_count": 2,
            "source_camera_calibrated": not is_raw,
            "generated_view_count": 0,
            "source_pixels_preserved_exact_bytes": True,
            "independent_model_inference_performed": False,
            "source_rights_verified": False,
            "source_foreground_or_depth_verified": False,
            "product_revision_minted": False,
        },
    }


__all__ = [
    "EXECUTOR_CONTRACT", "RAW_ARTIFACT_TYPE", "RAW_SCHEMA_VERSION",
    "execute_rss_vnext_input_capability_v3",
]
