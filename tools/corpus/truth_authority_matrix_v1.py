from __future__ import annotations

"""Deterministic per-subject truth authority matrix for the shared RealSaS corpus.

This module never qualifies measurements by itself. It validates already-produced
family authority rows and deterministically derives which downstream consumers may
use the subject. The important invariant is that one weak truth family cannot erase
independent exact truth for another model.

Unknown, unavailable and masked truth remain explicit. They are never converted to
negative targets. Final split membership is forbidden until final admission has been
minted outside this helper by independent family courts.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


SCHEMA = "RealSaS.CorpusSubjectTruthAuthority.v1"
REPORT_SCHEMA = "RealSaS.CorpusSubjectTruthAuthorityReport.v1"
PENDING_SPLIT = "PENDING_UNTIL_FINAL_ADMISSION"


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_contract(path: Path | None = None) -> dict:
    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "canonical"
            / "CORPUS_TRUTH_AUTHORITY_CONTRACT_V1.json"
        )
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "RealSaS.CorpusTruthAuthorityContract.v1":
        raise ValueError("CORPUS_TRUTH_AUTHORITY_CONTRACT_SCHEMA")
    return value


def _hex64(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def pending_family_row(family: str, reason: str = "PENDING_INDEPENDENT_FAMILY_AUDIT") -> dict:
    return {
        "family": family,
        "authority": "PENDING",
        "coverage_fraction": None,
        "validity_mask_sha256": None,
        "producer_evidence_sha256": None,
        "court_evidence_sha256": None,
        "reason": reason,
    }


def new_pending_matrix(*, subject_id: str, source_sha256: str, contract: dict | None = None) -> dict:
    contract = contract or load_contract()
    if not subject_id:
        raise ValueError("CORPUS_TRUTH_SUBJECT_ID_REQUIRED")
    if not _hex64(source_sha256):
        raise ValueError("CORPUS_TRUTH_SOURCE_SHA256_REQUIRED")
    return {
        "schema": SCHEMA,
        "subject_id": subject_id,
        "source_sha256": source_sha256,
        "source_mutated": False,
        "split_membership": PENDING_SPLIT,
        "subject_disposition": "PENDING_AUTHORITY_AUDIT",
        "families": [pending_family_row(name) for name in contract["family_order"]],
    }


def _validate_contract(contract: dict) -> None:
    order = list(contract.get("family_order") or ())
    families = dict(contract.get("families") or {})
    states = set(contract.get("family_authority_states") or ())
    dispositions = set(contract.get("subject_dispositions") or ())
    consumers = dict(contract.get("consumers") or {})
    if not order or len(order) != len(set(order)) or set(order) != set(families):
        raise ValueError("CORPUS_TRUTH_CONTRACT_FAMILY_SET")
    if not states or not dispositions or not consumers:
        raise ValueError("CORPUS_TRUTH_CONTRACT_ENUMS")
    for consumer, spec in consumers.items():
        requirements = list((spec or {}).get("requirements") or ())
        if not requirements:
            raise ValueError("CORPUS_TRUTH_CONTRACT_CONSUMER_EMPTY:" + consumer)
        seen = set()
        for requirement in requirements:
            family = str(requirement.get("family") or "")
            if family not in families or family in seen:
                raise ValueError("CORPUS_TRUTH_CONTRACT_CONSUMER_FAMILY:" + consumer)
            seen.add(family)
            if not isinstance(requirement.get("required"), bool):
                raise ValueError("CORPUS_TRUTH_CONTRACT_REQUIRED_FLAG:" + consumer)
            if not isinstance(requirement.get("mask_allowed"), bool):
                raise ValueError("CORPUS_TRUTH_CONTRACT_MASK_FLAG:" + consumer)


def _validate_family_row(row: dict, *, family: str, states: set[str]) -> None:
    if set(row) != {
        "family",
        "authority",
        "coverage_fraction",
        "validity_mask_sha256",
        "producer_evidence_sha256",
        "court_evidence_sha256",
        "reason",
    }:
        raise ValueError("CORPUS_TRUTH_FAMILY_FIELDS:" + family)
    if row.get("family") != family:
        raise ValueError("CORPUS_TRUTH_FAMILY_ORDER:" + family)
    authority = row.get("authority")
    if authority not in states:
        raise ValueError("CORPUS_TRUTH_FAMILY_AUTHORITY:" + family)
    reason = row.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("CORPUS_TRUTH_FAMILY_REASON:" + family)

    coverage = row.get("coverage_fraction")
    mask = row.get("validity_mask_sha256")
    producer = row.get("producer_evidence_sha256")
    court = row.get("court_evidence_sha256")

    if authority == "QUALIFIED_CORE":
        if coverage != 1.0 or mask is not None:
            raise ValueError("CORPUS_TRUTH_CORE_COVERAGE:" + family)
        if not _hex64(producer) or not _hex64(court):
            raise ValueError("CORPUS_TRUTH_CORE_EVIDENCE:" + family)
    elif authority == "QUALIFIED_MASKED":
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
            raise ValueError("CORPUS_TRUTH_MASKED_COVERAGE:" + family)
        coverage = float(coverage)
        if not (0.0 < coverage < 1.0):
            raise ValueError("CORPUS_TRUTH_MASKED_COVERAGE:" + family)
        if not _hex64(mask) or not _hex64(producer) or not _hex64(court):
            raise ValueError("CORPUS_TRUTH_MASKED_EVIDENCE:" + family)
    else:
        if coverage is not None or mask is not None:
            raise ValueError("CORPUS_TRUTH_NONQUALIFIED_TARGET_LEAK:" + family)
        if producer is not None and not _hex64(producer):
            raise ValueError("CORPUS_TRUTH_PRODUCER_HASH:" + family)
        if court is not None and not _hex64(court):
            raise ValueError("CORPUS_TRUTH_COURT_HASH:" + family)


def _family_map(matrix: dict, contract: dict) -> dict[str, dict]:
    rows = list(matrix.get("families") or ())
    order = list(contract["family_order"])
    if len(rows) != len(order):
        raise ValueError("CORPUS_TRUTH_FAMILY_CARDINALITY")
    mapping = {}
    states = set(contract["family_authority_states"])
    for expected, row in zip(order, rows):
        if not isinstance(row, dict):
            raise ValueError("CORPUS_TRUTH_FAMILY_ROW")
        _validate_family_row(row, family=expected, states=states)
        if expected in mapping:
            raise ValueError("CORPUS_TRUTH_FAMILY_DUPLICATE:" + expected)
        mapping[expected] = row
    return mapping


def derive_consumer_eligibility(matrix: dict, contract: dict | None = None) -> dict[str, dict]:
    contract = contract or load_contract()
    _validate_contract(contract)
    families = _family_map(matrix, contract)
    result = {}

    for consumer, spec in contract["consumers"].items():
        blockers = []
        masked = []
        optional_absent = []
        for requirement in spec["requirements"]:
            family = requirement["family"]
            required = bool(requirement["required"])
            mask_allowed = bool(requirement["mask_allowed"])
            authority = families[family]["authority"]

            if authority == "QUALIFIED_CORE":
                continue
            if authority == "QUALIFIED_MASKED":
                if mask_allowed:
                    masked.append(family)
                elif required:
                    blockers.append({"family": family, "reason": "MASK_FORBIDDEN"})
                else:
                    optional_absent.append(family)
                continue
            if not required and authority in {"PENDING", "UNAVAILABLE", "UNSUPPORTED"}:
                optional_absent.append(family)
                continue
            if authority == "PENDING":
                blockers.append({"family": family, "reason": "PENDING"})
            elif authority in {"UNAVAILABLE", "UNSUPPORTED"}:
                blockers.append({"family": family, "reason": "MISSING_REQUIRED_TRUTH"})
            elif authority == "QUARANTINE_REPAIRABLE":
                blockers.append({"family": family, "reason": "QUARANTINED"})
            elif authority == "DROP_UNSUITABLE":
                blockers.append({"family": family, "reason": "UNSUITABLE"})
            else:
                blockers.append({"family": family, "reason": "UNKNOWN_AUTHORITY"})

        if blockers:
            reasons = {row["reason"] for row in blockers}
            if "UNSUITABLE" in reasons:
                status = "INELIGIBLE_UNSUITABLE"
            elif "QUARANTINED" in reasons:
                status = "QUARANTINED"
            elif "PENDING" in reasons:
                status = "PENDING"
            elif "MASK_FORBIDDEN" in reasons:
                status = "INELIGIBLE_MASK_FORBIDDEN"
            else:
                status = "INELIGIBLE_MISSING_TRUTH"
        elif masked:
            status = "ELIGIBLE_MASKED"
        else:
            status = "ELIGIBLE_CORE"

        result[consumer] = {
            "status": status,
            "masked_families": sorted(masked),
            "optional_absent_families": sorted(optional_absent),
            "blockers": blockers,
        }
    return result


def derive_subject_disposition(matrix: dict, contract: dict | None = None) -> str:
    contract = contract or load_contract()
    families = _family_map(matrix, contract)
    authorities = [row["authority"] for row in families.values()]
    eligibility = derive_consumer_eligibility(matrix, contract)

    if "DROP_UNSUITABLE" in authorities:
        return "DROP_UNSUITABLE"
    if "QUARANTINE_REPAIRABLE" in authorities:
        return "QUARANTINE_REPAIRABLE"
    if "PENDING" in authorities:
        return "PENDING_AUTHORITY_AUDIT"

    usable = [
        row["status"]
        for row in eligibility.values()
        if row["status"] in {"ELIGIBLE_CORE", "ELIGIBLE_MASKED"}
    ]
    if not usable:
        return "DROP_UNSUITABLE"
    if (
        any(value == "QUALIFIED_MASKED" for value in authorities)
        or any(value in {"UNAVAILABLE", "UNSUPPORTED"} for value in authorities)
        or any(value == "ELIGIBLE_MASKED" for value in usable)
        or len(usable) != len(eligibility)
    ):
        return "ADMIT_MASKED"
    return "ADMIT_CORE"


def validate_subject_matrix(matrix: dict, contract: dict | None = None) -> dict:
    contract = contract or load_contract()
    _validate_contract(contract)
    if matrix.get("schema") != SCHEMA:
        raise ValueError("CORPUS_TRUTH_MATRIX_SCHEMA")
    if set(matrix) != {
        "schema",
        "subject_id",
        "source_sha256",
        "source_mutated",
        "split_membership",
        "subject_disposition",
        "families",
    }:
        raise ValueError("CORPUS_TRUTH_MATRIX_FIELDS")
    if not isinstance(matrix.get("subject_id"), str) or not matrix["subject_id"]:
        raise ValueError("CORPUS_TRUTH_MATRIX_SUBJECT")
    if not _hex64(matrix.get("source_sha256")):
        raise ValueError("CORPUS_TRUTH_MATRIX_SOURCE")
    if matrix.get("source_mutated") is not False:
        raise ValueError("CORPUS_TRUTH_SOURCE_MUTATION_FORBIDDEN")
    if matrix.get("split_membership") != PENDING_SPLIT:
        raise ValueError("CORPUS_TRUTH_PREMATURE_SPLIT")
    if matrix.get("subject_disposition") not in set(contract["subject_dispositions"]):
        raise ValueError("CORPUS_TRUTH_MATRIX_DISPOSITION")

    _family_map(matrix, contract)
    eligibility = derive_consumer_eligibility(matrix, contract)
    derived = derive_subject_disposition(matrix, contract)
    declared = matrix["subject_disposition"]
    if declared != derived:
        raise ValueError(
            "CORPUS_TRUTH_DISPOSITION_MISMATCH:" + declared + "!=" + derived
        )

    payload = {
        "schema": REPORT_SCHEMA,
        "subject_id": matrix["subject_id"],
        "source_sha256": matrix["source_sha256"],
        "subject_disposition": derived,
        "consumer_eligibility": eligibility,
        "split_membership": PENDING_SPLIT,
        "training_authorized": False,
        "authority_contract_sha256": _sha256_json(contract),
        "matrix_identity_sha256": _sha256_json(matrix),
    }
    payload["report_identity_sha256"] = _sha256_json(payload)
    return payload


def with_family_authority(
    matrix: dict,
    *,
    family: str,
    authority: str,
    reason: str,
    coverage_fraction=None,
    validity_mask_sha256=None,
    producer_evidence_sha256=None,
    court_evidence_sha256=None,
    contract: dict | None = None,
) -> dict:
    """Return a copy with one explicit family decision and refreshed disposition."""
    contract = contract or load_contract()
    result = deepcopy(matrix)
    order = list(contract["family_order"])
    if family not in order:
        raise ValueError("CORPUS_TRUTH_UNKNOWN_FAMILY:" + family)
    index = order.index(family)
    result["families"][index] = {
        "family": family,
        "authority": authority,
        "coverage_fraction": coverage_fraction,
        "validity_mask_sha256": validity_mask_sha256,
        "producer_evidence_sha256": producer_evidence_sha256,
        "court_evidence_sha256": court_evidence_sha256,
        "reason": reason,
    }
    result["subject_disposition"] = derive_subject_disposition(result, contract)
    return result
