from __future__ import annotations

"""Read-only evidence inventory for the shared corpus truth families.

Evidence inventory is deliberately weaker than truth qualification. A legacy PASS,
a file's existence, or a successful producer/court pair may be valuable provenance,
but none is silently translated into the new CorpusTruthAuthority contract. The
output answers only: which exact evidence exists, what did it historically claim,
and what still needs a current independent family court?
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable

from tools.corpus.truth_authority_matrix_v1 import load_contract, new_pending_matrix


SCHEMA = "RealSaS.CorpusTruthFamilyInventory.v1"
ROW_SCHEMA = "RealSaS.CorpusTruthFamilyInventoryRow.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def verified_ref(path: Path, expected_sha256: str | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("CORPUS_TRUTH_INVENTORY_HASH_DRIFT:" + str(path))
    return {"path": str(path), "sha256": actual}


def bound_ref(row: dict | None, code: str) -> tuple[dict, dict]:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return {"path": str(path), "sha256": expected}, load_json(path, code + "_JSON")


def evidence_descriptor(
    label: str,
    ref: dict,
    payload: dict | None = None,
    *,
    role: str = "LEGACY_OR_EXISTING_EVIDENCE",
) -> dict:
    payload = dict(payload or {})
    return {
        "label": label,
        "role": role,
        "path": str(ref["path"]),
        "sha256": str(ref["sha256"]),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "qualification_minted": payload.get("qualification_minted"),
        "training_admission_claimed": payload.get("training_admission_claimed"),
    }


def make_family_row(family: str, *, reason: str = "NO_CURRENT_FAMILY_AUTHORITY_COURT") -> dict:
    return {
        "schema": ROW_SCHEMA,
        "family": family,
        "evidence_status": "NONE",
        "evidence": [],
        "legacy_pass_like_claim_count": 0,
        "legacy_fail_like_claim_count": 0,
        "authority_state": "PENDING",
        "authority_reason": reason,
    }


def _status_class(value) -> str:
    text = str(value or "").upper()
    if not text:
        return "NONE"
    if text.startswith("PASS") or text in {"SEALED", "READY", "QUALIFIED"}:
        return "PASS_LIKE"
    if any(token in text for token in ("FAIL", "ERROR", "INVALID", "INCOMPLETE", "REJECT")):
        return "FAIL_LIKE"
    return "OTHER"


def add_evidence(row: dict, descriptor: dict) -> None:
    if row.get("authority_state") != "PENDING":
        raise ValueError("CORPUS_TRUTH_INVENTORY_MAY_NOT_MINT_AUTHORITY")
    key = (descriptor["path"], descriptor["sha256"], descriptor["label"])
    existing = {
        (item["path"], item["sha256"], item["label"])
        for item in row["evidence"]
    }
    if key in existing:
        return
    row["evidence"].append(descriptor)
    category = _status_class(descriptor.get("status"))
    if category == "PASS_LIKE":
        row["legacy_pass_like_claim_count"] += 1
    elif category == "FAIL_LIKE":
        row["legacy_fail_like_claim_count"] += 1
    row["evidence_status"] = "PRESENT_UNQUALIFIED"


def inventory_subject(
    *,
    subject_id: str,
    source_sha256: str,
    family_evidence: dict[str, Iterable[dict]],
    contract: dict | None = None,
) -> dict:
    contract = contract or load_contract()
    matrix = new_pending_matrix(
        subject_id=subject_id,
        source_sha256=source_sha256,
        contract=contract,
    )
    rows = {family: make_family_row(family) for family in contract["family_order"]}
    unknown = set(family_evidence) - set(rows)
    if unknown:
        raise ValueError("CORPUS_TRUTH_INVENTORY_UNKNOWN_FAMILY:" + ",".join(sorted(unknown)))
    for family, descriptors in family_evidence.items():
        for descriptor in descriptors:
            add_evidence(rows[family], dict(descriptor))

    ordered = [rows[family] for family in contract["family_order"]]
    payload = {
        "schema": SCHEMA,
        "subject_id": subject_id,
        "source_sha256": source_sha256,
        "source_mutated": False,
        "training_authorized": False,
        "final_disposition_minted": False,
        "split_membership": "PENDING_UNTIL_FINAL_ADMISSION",
        "authority_matrix": matrix,
        "families": ordered,
        "evidence_present_family_count": sum(
            row["evidence_status"] == "PRESENT_UNQUALIFIED" for row in ordered
        ),
        "family_count": len(ordered),
    }
    payload["inventory_identity_sha256"] = digest(payload)
    return payload


def validate_inventory(value: dict, contract: dict | None = None) -> None:
    contract = contract or load_contract()
    if value.get("schema") != SCHEMA:
        raise ValueError("CORPUS_TRUTH_INVENTORY_SCHEMA")
    if value.get("source_mutated") is not False:
        raise ValueError("CORPUS_TRUTH_INVENTORY_SOURCE_MUTATION")
    if value.get("training_authorized") is not False:
        raise ValueError("CORPUS_TRUTH_INVENTORY_TRAINING_AUTHORITY")
    if value.get("final_disposition_minted") is not False:
        raise ValueError("CORPUS_TRUTH_INVENTORY_DISPOSITION_AUTHORITY")
    if value.get("split_membership") != "PENDING_UNTIL_FINAL_ADMISSION":
        raise ValueError("CORPUS_TRUTH_INVENTORY_PREMATURE_SPLIT")
    rows = list(value.get("families") or ())
    if [row.get("family") for row in rows] != list(contract["family_order"]):
        raise ValueError("CORPUS_TRUTH_INVENTORY_FAMILY_ORDER")
    for row in rows:
        if row.get("authority_state") != "PENDING":
            raise ValueError("CORPUS_TRUTH_INVENTORY_MINTED_AUTHORITY")
        expected = "PRESENT_UNQUALIFIED" if row.get("evidence") else "NONE"
        if row.get("evidence_status") != expected:
            raise ValueError("CORPUS_TRUTH_INVENTORY_EVIDENCE_STATUS:" + row["family"])
        for evidence in row.get("evidence") or ():
            if len(str(evidence.get("sha256") or "")) != 64:
                raise ValueError("CORPUS_TRUTH_INVENTORY_EVIDENCE_SHA:" + row["family"])
