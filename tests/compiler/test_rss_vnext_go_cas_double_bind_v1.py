from __future__ import annotations

"""Go transport alias invariant for the real one-trigger RSS product graph.

The goal's original CAS artifact and product.input's byte-preserving output
may have different artifact UUIDs. Only the exact source content is allowed
two refs. Fanout children and canonical IR remain uniquely addressed.
"""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.realsas_compiler_services.platform_worker.rss_vnext_cas_parent_v1 import (
    select_rss_vnext_parent_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_mesh_executor_v2 import _parents
from compiler.realsas_compiler_services.platform_worker.rss_vnext_rig_executor_v3 import _one, _cas


SOURCE = "RealSaS.RSSvNextTwoViewSource"
INDEX = "RealSaS.RSSvNextProductEvidenceFanout"


def _row(kind, label):
    raw = label.encode()
    digest = sha256(raw).hexdigest()
    return {
        "id": f"uuid:{label}",
        "role": "input:0000",
        "artifact_type": kind,
        "schema_version": "v1",
        "storage_key": f"sha256/{digest}",
        "content_sha256": digest,
        "size_bytes": len(raw),
    }


def _envelope():
    source = _row(SOURCE, "source")
    children = [_row(f"RealSaS.RSSvNextChild{i}", f"child{i}") for i in range(7)]
    return {
        "input_artifacts": [source, _row(INDEX, "fanout"), *children],
    }


def test_go_distinct_artifact_ids_one_exact_source_cas_for_all_consumers():
    request = _envelope()
    source = request["input_artifacts"][0]
    assert _parents(request)[0] == source
    assert _one(request, SOURCE, "v1") == source

    duplicate = {**source, "id": "another-artifact-uuid", "role": "input:0009"}
    request["input_artifacts"].append(duplicate)
    selected, index, refs = _parents(request)
    assert len(refs) == 10
    assert selected["content_sha256"] == source["content_sha256"]
    assert index["artifact_type"] == INDEX
    assert _one(request, SOURCE, "v1")["content_sha256"] == source["content_sha256"]
    assert select_rss_vnext_parent_v1(request, SOURCE, "v1") == source


@pytest.mark.parametrize("key, replacement", [
    ("content_sha256", "f" * 64),
    ("size_bytes", 21),
    ("storage_key", "tampered"),
])
def test_same_artifact_type_different_source_cas_never_silently_deduplicates(key, replacement):
    request = _envelope()
    source = request["input_artifacts"][0]
    request["input_artifacts"].append({
        **source, "id": "other-uuid", key: replacement,
    })
    with pytest.raises(RuntimeError, match="REUSED_BYTES_DRIFT"):
        _one(request, SOURCE, "v1")
    with pytest.raises(RuntimeError, match="REUSED_BYTES_DRIFT"):
        _parents(request)


def test_three_source_refs_and_duplicate_fanout_are_unconditionally_rejected():
    request = _envelope()
    source = request["input_artifacts"][0]
    request["input_artifacts"].extend([
        {**source, "id": "source-2"},
        {**source, "id": "source-3"},
    ])
    with pytest.raises(RuntimeError, match="CARDINALITY"):
        _one(request, SOURCE, "v1")
    with pytest.raises(RuntimeError, match="EXPECTS_SOURCE_INDEX"):
        _parents(request)

    request = _envelope()
    request["input_artifacts"].append(deepcopy(request["input_artifacts"][1]))
    with pytest.raises(RuntimeError, match="CARDINALITY"):
        _one(request, INDEX, "v1")
    with pytest.raises(RuntimeError, match="SOURCE_OR_INDEX_CARDINALITY"):
        _parents(request)


def test_duplicate_alias_cannot_bypass_actual_cas_digest_read():
    request = _envelope()
    first = request["input_artifacts"][0]
    request["input_artifacts"].append({**first, "id": "different-uuid"})
    selected = _one(request, SOURCE, "v1")
    with pytest.raises(RuntimeError, match="CAS_BYTES_OR_HASH_DRIFT"):
        _cas(selected, lambda _: b"tampered source")
