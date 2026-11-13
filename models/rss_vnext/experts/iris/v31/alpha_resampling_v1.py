from __future__ import annotations

"""Exact source-alpha coverage resampling selected by the bounded V3.1 court."""

import hashlib
import json

import torch
from torch.nn import functional as F


POLICY_SCHEMA_V1 = "RealSaS.IRISAlphaResamplingPolicy.v1"
COURT_SCHEMA_V1 = "RealSaS.V31AlphaResamplingPolicyCourt.v1"
COURT_RUN_ID_V1 = 40647553023
COURT_ARTIFACT_ID_V1 = 12798500687
COURT_ARTIFACT_DIGEST_V1 = "86952dba8de2a723721b5aea2d1e88c9c0516457e52c9bb60b85a859bc1bcb25"
COURT_HEAD_SHA_V1 = "5fee7f3d6def45d0935dacf786d4c96763e84cf2"

POLICY_V1 = {
    "schema": POLICY_SCHEMA_V1,
    "semantic": "SOURCE_ALPHA_COVERAGE",
    "input_resolution": 1024,
    "output_resolution": 518,
    "mode": "BILINEAR",
    "align_corners": False,
    "antialias": True,
    "compute_device": "CPU",
    "compute_dtype": "FLOAT32",
    "clamp": False,
    "court": {
        "schema": COURT_SCHEMA_V1,
        "run_id": COURT_RUN_ID_V1,
        "artifact_id": COURT_ARTIFACT_ID_V1,
        "artifact_digest_sha256": COURT_ARTIFACT_DIGEST_V1,
        "head_sha": COURT_HEAD_SHA_V1,
        "status": "V31_ALPHA_RESAMPLING_POLICY_SELECTED_BOUNDED_V1",
        "selected_policy": "BILINEAR_ANTIALIAS",
        "scenario_count": 173,
        "worst_coverage_abs_error": 0.00013733096420764923,
        "support_recall": 1.0,
        "minimum_output_alpha": 0.0,
        "maximum_output_alpha": 1.0,
    },
    "claims": {
        "observation_admission_minted": False,
        "training_authorized": False,
        "model_release_minted": False,
        "product_shape_activated": False,
        "fit8_claimed": False,
        "unseen_claimed": False,
    },
}


def policy_sha256_v1() -> str:
    payload = json.dumps(
        POLICY_V1,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float01_cpu(native_alpha: torch.Tensor) -> torch.Tensor:
    if native_alpha.ndim != 5 or native_alpha.shape[2] != 1:
        raise ValueError("IRIS_V31_ALPHA_SOURCE_SHAPE_INVALID")
    if tuple(native_alpha.shape[-2:]) != (1024, 1024) or min(native_alpha.shape[:2]) <= 0:
        raise ValueError("IRIS_V31_ALPHA_SOURCE_RESOLUTION_INVALID")
    if native_alpha.dtype == torch.uint8:
        value = native_alpha.to(device="cpu", dtype=torch.float32).div(255.0)
    elif torch.is_floating_point(native_alpha):
        value = native_alpha.detach().to(device="cpu", dtype=torch.float32)
    else:
        raise ValueError("IRIS_V31_ALPHA_SOURCE_DTYPE_INVALID")
    if not bool(torch.isfinite(value).all()) or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError("IRIS_V31_ALPHA_SOURCE_RANGE_INVALID")
    return value.contiguous()


def resample_source_alpha_v1(native_alpha: torch.Tensor) -> torch.Tensor:
    """Apply the selected no-clamp CPU-FP32 bilinear-antialias alpha policy."""

    source = _float01_cpu(native_alpha)
    batch, views = source.shape[:2]
    target = F.interpolate(
        source.reshape(batch * views, 1, 1024, 1024),
        size=(518, 518),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).reshape(batch, views, 1, 518, 518)
    if target.dtype != torch.float32 or target.device.type != "cpu":
        raise RuntimeError("IRIS_V31_ALPHA_RESAMPLING_NUMERICAL_POLICY_DRIFT")
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("IRIS_V31_ALPHA_RESAMPLING_NONFINITE")
    if float(target.min()) < 0.0 or float(target.max()) > 1.0:
        raise RuntimeError("IRIS_V31_ALPHA_RESAMPLING_RANGE_DRIFT")
    return target.contiguous()
