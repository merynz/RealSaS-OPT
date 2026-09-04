from __future__ import annotations

"""Visible, hash-pinned authority loader for the canonical graph optimizer.

The historical compiler transport still supplies byte-preserved contract modules
needed by the optimizer, but the product-authoritative optimizer implementation is
ordinary reviewable repository source. Loading fails closed if those visible bytes
drift from the pre-existing BYTE_EXACT_V0_5 manifest authority.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys


EXPECTED_SHA256 = "b2fddb64753ca783e298be4f1078c70b9667fa9931c67de738f955976e2587c1"
EXPECTED_BYTES = 53544
SOURCE_PATH = Path(__file__).resolve().parents[1] / "vendor" / "realsas_synthesis" / "canonical_graph_optimizer.py"


def _load_visible_optimizer():
    if not SOURCE_PATH.is_file():
        raise ImportError(f"visible canonical graph optimizer missing:{SOURCE_PATH}")
    raw = SOURCE_PATH.read_bytes()
    if len(raw) != EXPECTED_BYTES:
        raise ImportError(f"visible canonical graph optimizer size mismatch:{len(raw)}")
    got = hashlib.sha256(raw).hexdigest()
    if got != EXPECTED_SHA256:
        raise ImportError(f"visible canonical graph optimizer SHA mismatch:{got}")
    name = "_realsas_visible_canonical_graph_optimizer_authority"
    spec = importlib.util.spec_from_file_location(name, SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct visible canonical graph optimizer module spec")
    module = importlib.util.module_from_spec(spec)
    # Dataclass decoration resolves postponed annotations through sys.modules while
    # the module body executes. Register the exact visible module before exec, just
    # like importlib's normal import path, and remove the partial module on failure.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    fn = getattr(module, "optimize_canonical_graph_v18_98", None)
    if not callable(fn):
        sys.modules.pop(name, None)
        raise ImportError("visible canonical graph optimizer entrypoint missing")
    return module, fn


MODULE, optimize_canonical_graph_v18_98 = _load_visible_optimizer()
SOURCE_SHA256 = EXPECTED_SHA256
SOURCE_BYTES = EXPECTED_BYTES

__all__ = [
    "optimize_canonical_graph_v18_98",
    "SOURCE_PATH",
    "SOURCE_SHA256",
    "SOURCE_BYTES",
]
