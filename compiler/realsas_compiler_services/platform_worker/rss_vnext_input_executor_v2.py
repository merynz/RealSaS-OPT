from __future__ import annotations

"""Native two-observation Input capability for the unified RSS.vNext product.

Model-free, exactly byte-preserving Input -> Shape edge. Does not crop/matte,
rescale, invent back views, run independent DINO/IRIS or call legacy eight-view
ShapeInput. CAS deduplication is intentional; original evidence is never
serialized to a lossy feature-only proxy.
"""

from hashlib import sha256
from typing import Any, Callable

from .rss_vnext_source_codec_v1 import (
    ARTIFACT_TYPE, SCHEMA_VERSION,
    validate_rss_vnext_source_ref_v1,
)

EXECUTOR_CONTRACT = "RealSaS.ProductInputTwoView.v2"


def execute_rss_vnext_input_capability_v2(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if (
        request.get("capability_id") != "product.input"
        or request.get("kind") != "MODULE"
        or metadata.get("executor_contract") != EXECUTOR_CONTRACT
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V2_EXECUTOR_CONTRACT_DRIFT")
    if not str(request.get("execution_id") or ""):
        raise RuntimeError("RSS_VNEXT_INPUT_V2_EXECUTION_ID_REQUIRED")
    if request.get("model_releases"):
        raise RuntimeError("RSS_VNEXT_INPUT_V2_MODEL_RELEASE_FORBIDDEN")
    refs = list(request.get("input_artifacts") or ())
    if (
        len(refs) != 1 or not isinstance(refs[0], dict)
        or refs[0].get("artifact_type") != ARTIFACT_TYPE
        or refs[0].get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V2_EXACT_TWO_VIEW_SOURCE_REQUIRED")
    original = read_object(refs[0])
    source = validate_rss_vnext_source_ref_v1(refs[0], original)
    output = put_bytes(original)  # the *same* bytes; CAS can deduplicate.
    if (
        output.get("content_sha256") != sha256(original).hexdigest()
        or output.get("size_bytes") != len(original)
        or not output.get("storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_INPUT_V2_CAS_WRITE_DRIFT")
    return {
        "capability_id": "product.input",
        "status": "PASS",
        "failure": None,
        "outputs": [{
            "role": "rss_vnext_two_view_source",
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "storage_key": output["storage_key"],
            "content_sha256": output["content_sha256"],
            "size_bytes": output["size_bytes"],
            "authority_class": "LOSSLESS_ORIGINAL_TWO_OBSERVATIONS_NOT_SOURCE_TRUTH",
        }],
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductInputDiagnostics.v2",
            "source_binding_sha256": source.source_binding_sha256,
            "source_rights_claim_sha256": source.source_rights_claim_sha256,
            "original_view_count": 2,
            "no_reencoding_or_resampling": True,
            "independent_dino_inference_performed": False,
            "source_rights_verified": False,
            "model_evidence_qualified": False,
            "product_revision_minted": False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT", "execute_rss_vnext_input_capability_v2"]
