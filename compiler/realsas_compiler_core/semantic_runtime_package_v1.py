"""Semantic runtime package contract for the V7 presentation consumer.

Stage42 compiles semantic slot order into an effective depth sort key. Those
numbers are deliberately not physical camera depth. The generic RSS builder is
reused for byte layout only, then this module replaces the legacy physical-depth
label with the truthful semantic sort-key contract before the package is sealed.
Runtime may consume the key; it may not infer order from it or call it canonical
camera depth.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

from .types import QualificationError

CANONICAL_DEPTH_CONTRACT = "CANONICAL_CAMERA_DEPTH_ASCENDING__UNRESOLVED_TIES_FAIL_V1"
SEMANTIC_EFFECTIVE_DEPTH_CONTRACT = "COMPILED_SEMANTIC_EFFECTIVE_DEPTH_ASCENDING_V1"
SEMANTIC_ORDER_CONTRACT = "SEMANTIC_SLOT_ORDER_THEN_CANONICAL_DEPTH_WITHIN_SLOT_V1"
SEMANTIC_ORDER_OPERATOR_ID = "SEMANTIC_SLOT_ORDER_WITH_INTRA_SLOT_CANONICAL_DEPTH_V1"
SEMANTIC_ORDER_ENCODING = "EFFECTIVE_DEPTH_SORT_KEY_V1"


def _manifest_rows(payload: bytes) -> tuple[list[str], dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("SEMANTIC_RUNTIME_MANIFEST_UTF8_INVALID") from exc
    rows = [row for row in text.splitlines() if row]
    values: dict[str, str] = {}
    for row in rows:
        if "=" not in row:
            raise QualificationError("SEMANTIC_RUNTIME_MANIFEST_ROW_INVALID")
        key, value = row.split("=", 1)
        if not key or key in values:
            raise QualificationError("SEMANTIC_RUNTIME_MANIFEST_KEY_DUPLICATE")
        values[key] = value
    return rows, values


def seal_semantic_runtime_entries(entries: Mapping[str, bytes]) -> OrderedDict[str, bytes]:
    """Return RSS entries with a truthful semantic sort-key depth contract.

    The base source-owned packer has already verified and serialized the Stage42
    arrays. This function changes no array bytes. It only changes the manifest
    authority label after verifying that the explicit V7 relation contract is
    present and complete.
    """
    if "manifest.txt" not in entries:
        raise QualificationError("SEMANTIC_RUNTIME_MANIFEST_MISSING")
    rows, values = _manifest_rows(bytes(entries["manifest.txt"]))
    required = {
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
        "semantic_order_operator_id": SEMANTIC_ORDER_OPERATOR_ID,
        "semantic_order_encoding": SEMANTIC_ORDER_ENCODING,
        "depth_ownership_contract": CANONICAL_DEPTH_CONTRACT,
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise QualificationError("SEMANTIC_RUNTIME_CONTRACT_DRIFT:" + key)
    for key in ("qualified_contact_contract_hash", "presentation_relations_hash"):
        if len(values.get(key, "")) != 64:
            raise QualificationError("SEMANTIC_RUNTIME_RELATION_HASH_INVALID:" + key)

    rewritten: list[str] = []
    for row in rows:
        key, _ = row.split("=", 1)
        if key == "depth_ownership_contract":
            rewritten.append(
                "depth_ownership_contract=" + SEMANTIC_EFFECTIVE_DEPTH_CONTRACT
            )
        else:
            rewritten.append(row)
    rewritten.extend(
        (
            "semantic_physical_depth_final_authority=false",
            "semantic_sort_key_compiled_upstream=true",
            "runtime_semantic_order_inference=false",
        )
    )
    result = OrderedDict((str(key), bytes(value)) for key, value in entries.items())
    result["manifest.txt"] = ("\n".join(rewritten) + "\n").encode("utf-8")
    _, check = _manifest_rows(result["manifest.txt"])
    if (
        check.get("depth_ownership_contract") != SEMANTIC_EFFECTIVE_DEPTH_CONTRACT
        or check.get("semantic_physical_depth_final_authority") != "false"
        or check.get("semantic_sort_key_compiled_upstream") != "true"
        or check.get("runtime_semantic_order_inference") != "false"
    ):
        raise QualificationError("SEMANTIC_RUNTIME_MANIFEST_REWRITE_FAILED")
    return result
