from __future__ import annotations

"""Unambiguous CAS parent selection for a Go product capability envelope.

A goal source artifact and the BYTE_IDENTICAL_CAS_REUSE product.input output
may have different artifact UUIDs. Go deduplicates UUIDs, not CAS contents.
A downstream consumer may observe one or two source refs for the same bytes.
Only this one source artifact is allowed duplicate transport identities: no
silent deduplication of fanout children, inference receipts or canonical IR.
"""

from typing import Any, Mapping

SOURCE_KIND = "RealSaS.RSSvNextTwoViewSource"
SOURCE_VERSION = "v1"


def select_rss_vnext_parent_v1(
    request: Mapping[str, Any], kind: str, version: str,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise RuntimeError("RSS_VNEXT_PARENT_REQUEST_REQUIRED")
    refs = request.get("input_artifacts")
    if not isinstance(refs, (list, tuple)) or any(type(x) is not dict for x in refs):
        raise RuntimeError("RSS_VNEXT_PARENT_INPUT_REFS_REQUIRED")
    matches = [r for r in refs if r.get("artifact_type") == kind
               and r.get("schema_version") == version]
    source = kind == SOURCE_KIND and version == SOURCE_VERSION
    if len(matches) == 1:
        return dict(matches[0])
    if not source or len(matches) != 2:
        raise RuntimeError(f"RSS_VNEXT_PARENT_CARDINALITY:{kind}:{version}:{len(matches)}")
    first, second = matches
    for row in matches:
        if (
            type(row.get("storage_key")) is not str
            or not row["storage_key"]
            or type(row.get("content_sha256")) is not str
            or len(row["content_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["content_sha256"])
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
        ):
            raise RuntimeError("RSS_VNEXT_SOURCE_PARENT_CAS_IDENTITY_INVALID")
    for key in ("storage_key", "content_sha256", "size_bytes"):
        if first[key] != second[key]:
            raise RuntimeError("RSS_VNEXT_SOURCE_PARENT_REUSED_BYTES_DRIFT")
    # Neither artifact UUID, ordinal nor mutable transport role can redefine
    # source truth. The selected ref's bytes are still read and SHA-checked.
    return dict(first)


__all__ = [
    "SOURCE_KIND", "SOURCE_VERSION", "select_rss_vnext_parent_v1",
]
