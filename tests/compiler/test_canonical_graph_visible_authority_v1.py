from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from compiler.realsas_compiler_core import canonical_graph_optimizer_authority as authority
from compiler.realsas_compiler_core import rig


EXPECTED_SHA256 = "b2fddb64753ca783e298be4f1078c70b9667fa9931c67de738f955976e2587c1"
EXPECTED_BYTES = 53544


def test_rig_uses_visible_hash_pinned_optimizer_authority() -> None:
    assert rig.optimize_canonical_graph_v18_98 is authority.optimize_canonical_graph_v18_98
    path = authority.SOURCE_PATH.resolve()
    assert path.name == "canonical_graph_optimizer.py"
    assert path.parent.name == "realsas_synthesis"
    assert path.parent.parent.name == "vendor"
    raw = path.read_bytes()
    assert len(raw) == EXPECTED_BYTES
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert authority.SOURCE_SHA256 == EXPECTED_SHA256
    assert authority.SOURCE_BYTES == EXPECTED_BYTES
    assert Path(inspect.getsourcefile(rig.optimize_canonical_graph_v18_98) or "").resolve() == path


def test_visible_optimizer_authority_fails_closed_on_byte_drift() -> None:
    with TemporaryDirectory() as td:
        bad = Path(td) / "canonical_graph_optimizer.py"
        raw = authority.SOURCE_PATH.read_bytes()
        bad.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with mock.patch.object(authority, "SOURCE_PATH", bad):
            with pytest.raises(ImportError, match="SHA mismatch"):
                authority._load_visible_optimizer()


def test_rig_source_does_not_import_hidden_optimizer_implementation_directly() -> None:
    source = Path(rig.__file__).read_text(encoding="utf-8")
    assert "from .canonical_graph_optimizer_authority import optimize_canonical_graph_v18_98" in source
    assert "from realsas_synthesis.canonical_graph_optimizer import optimize_canonical_graph_v18_98" not in source
