from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = "RealSaS.FReferenceInputBundleContract.v1"
SYNTHETIC_SMOKE = "SYNTHETIC_SMOKE"
DIAGNOSTIC_FIT8 = "DIAGNOSTIC_FIT8"
QUALIFIED_FIT8 = "QUALIFIED_FIT8"
AUTHORITY_STATES = {SYNTHETIC_SMOKE, DIAGNOSTIC_FIT8, QUALIFIED_FIT8}
ZERO_SHA256 = "0" * 64


def _validate_sha(name: str, value: str, *, allow_zero: bool) -> None:
    if len(value) != 64:
        raise ValueError(f"F_REFERENCE_INPUT_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"F_REFERENCE_INPUT_SHA_NONHEX:{name}") from exc
    if not allow_zero and value == ZERO_SHA256:
        raise ValueError(f"F_REFERENCE_INPUT_PLACEHOLDER_SHA_FORBIDDEN:{name}")


@dataclass(frozen=True)
class FReferenceInputBundleManifestV1:
    subject_key: str
    arm: str
    authority_state: str
    res: tuple[int, int, int]
    global_sign_authority: bool
    support_clipping_required: bool
    source_sha256: str
    teacher_bundle_sha256: str
    extraction_truth_sha256: str
    canonical_camera_contract_sha256: str
    extraction_scale_manifest_sha256: str
    measurement_protocol_sha256: str
    bundle_npz_sha256: str
    provenance: str
    metadata: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_REFERENCE_INPUT_SCHEMA_INVALID")
        if not self.subject_key or not self.arm:
            raise ValueError("F_REFERENCE_INPUT_IDENTITY_MISSING")
        if self.authority_state not in AUTHORITY_STATES:
            raise ValueError("F_REFERENCE_INPUT_AUTHORITY_STATE_INVALID")
        if len(self.res) != 3 or any(int(value) <= 0 for value in self.res):
            raise ValueError("F_REFERENCE_INPUT_RES_INVALID")
        if not self.provenance:
            raise ValueError("F_REFERENCE_INPUT_PROVENANCE_MISSING")

        qualified = self.authority_state == QUALIFIED_FIT8
        diagnostic = self.authority_state == DIAGNOSTIC_FIT8
        for name in (
            "source_sha256",
            "teacher_bundle_sha256",
            "extraction_truth_sha256",
            "canonical_camera_contract_sha256",
            "extraction_scale_manifest_sha256",
            "measurement_protocol_sha256",
            "bundle_npz_sha256",
        ):
            _validate_sha(name, getattr(self, name), allow_zero=not qualified)

        if qualified:
            if self.subject_key.startswith("SYNTHETIC_"):
                raise ValueError("F_REFERENCE_INPUT_QUALIFIED_SUBJECT_SYNTHETIC")
            if not self.metadata.get("source_exact_bytes_verified", False):
                raise ValueError("F_REFERENCE_INPUT_QUALIFIED_SOURCE_NOT_VERIFIED")
            if not self.metadata.get("teacher_bundle_validated", False):
                raise ValueError("F_REFERENCE_INPUT_QUALIFIED_TEACHER_NOT_VALIDATED")
            if not self.metadata.get("extraction_truth_validated", False):
                raise ValueError("F_REFERENCE_INPUT_QUALIFIED_EXTRACTION_NOT_VALIDATED")
            if not self.metadata.get("measurement_protocol_preregistered", False):
                raise ValueError("F_REFERENCE_INPUT_QUALIFIED_PROTOCOL_NOT_PREREGISTERED")
        elif diagnostic:
            if self.source_sha256 == ZERO_SHA256:
                raise ValueError("F_REFERENCE_INPUT_DIAGNOSTIC_SOURCE_SHA_REQUIRED")
        else:
            if not self.subject_key.startswith("SYNTHETIC_"):
                raise ValueError("F_REFERENCE_INPUT_SMOKE_SUBJECT_MUST_BE_SYNTHETIC")

    @property
    def matched_court_eligible(self) -> bool:
        self.validate()
        return self.authority_state == QUALIFIED_FIT8

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["res"] = list(self.res)
        return payload

    def manifest_sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


__all__ = [
    "AUTHORITY_STATES",
    "DIAGNOSTIC_FIT8",
    "FReferenceInputBundleManifestV1",
    "QUALIFIED_FIT8",
    "SCHEMA_VERSION",
    "SYNTHETIC_SMOKE",
    "ZERO_SHA256",
]
