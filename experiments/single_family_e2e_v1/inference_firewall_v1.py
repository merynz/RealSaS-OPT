from __future__ import annotations

import inspect
from pathlib import Path

_FORBIDDEN_PARAMETER_TOKENS=("teacher","truth","primary_geometry","source_mesh","skeleton_path","skin_path","m5")
_FORBIDDEN_SOURCE_PATH_TOKENS=("primary_geometry.npz","skeleton_truth","skin_truth","m5_product_core","teacher_projection")


def assert_inference_callable_firewall_v1(fn):
    params=tuple(inspect.signature(fn).parameters)
    bad=[p for p in params if any(tok in p.lower() for tok in _FORBIDDEN_PARAMETER_TOKENS)]
    if bad: raise RuntimeError(f"INFERENCE_FIREWALL_FORBIDDEN_PARAMETERS:{bad}")
    return {"status":"PASS_INFERENCE_SIGNATURE_FIREWALL","parameters":params}


def assert_source_path_firewall_v1(path):
    text=Path(path).read_text(encoding='utf-8').lower(); bad=[tok for tok in _FORBIDDEN_SOURCE_PATH_TOKENS if tok.lower() in text]
    if bad: raise RuntimeError(f"INFERENCE_FIREWALL_FORBIDDEN_SOURCE_PATHS:{bad}")
    return {"status":"PASS_INFERENCE_SOURCE_PATH_FIREWALL","forbidden_matches":[]}
