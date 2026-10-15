from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from tools.corpus.fit8_f_reference_input_bundle_contract_v1 import (
    AUTHORITY_STATES,
    DIAGNOSTIC_FIT8,
    QUALIFIED_FIT8,
    SYNTHETIC_SMOKE,
    ZERO_SHA256,
)

SCHEMA_VERSION = "RealSaS.FReferenceInputBundleContract.v2"


def _sha(name: str, value: str, *, allow_zero: bool) -> None:
    if len(value) != 64:
        raise ValueError(f"F_REFERENCE_INPUT_V2_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"F_REFERENCE_INPUT_V2_SHA_NONHEX:{name}") from exc
    if not allow_zero and value == ZERO_SHA256:
        raise ValueError(f"F_REFERENCE_INPUT_V2_PLACEHOLDER_SHA_FORBIDDEN:{name}")


@dataclass(frozen=True)
class FReferenceInputBundleManifestV2:
    """F backend input manifest with geometry-specific teacher authority.

    V1 named a dependency `teacher_bundle_sha256`, which accidentally coupled an
    F geometry backend run to materialization of unrelated appearance/mechanics/
    dynamic teacher channels. V2 binds the exact `FGeometryTeacherSlice.v1`
    instead. The slice itself records whether a parent full-truth bundle is bound
    or intentionally deferred, so modular geometry courts do not masquerade as a
    full corpus closure.
    """

    subject_key: str
    arm: str
    authority_state: str
    res: tuple[int, int, int]
    global_sign_authority: bool
    support_clipping_required: bool
    source_sha256: str
    source_carrier_sha256: str
    geometry_teacher_slice_sha256: str
    canonical_camera_contract_sha256: str
    measurement_protocol_sha256: str
    dense_grid_authority_sha256: str
    array_content_sha256: str
    bundle_npz_sha256: str
    provenance: str
    metadata: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_REFERENCE_INPUT_V2_SCHEMA_INVALID")
        if not self.subject_key or not self.arm:
            raise ValueError("F_REFERENCE_INPUT_V2_IDENTITY_MISSING")
        if self.authority_state not in AUTHORITY_STATES:
            raise ValueError("F_REFERENCE_INPUT_V2_AUTHORITY_STATE_INVALID")
        if len(self.res) != 3 or any(int(v) <= 0 for v in self.res):
            raise ValueError("F_REFERENCE_INPUT_V2_RES_INVALID")
        if not self.provenance:
            raise ValueError("F_REFERENCE_INPUT_V2_PROVENANCE_MISSING")

        qualified = self.authority_state == QUALIFIED_FIT8
        diagnostic = self.authority_state == DIAGNOSTIC_FIT8
        for name in (
            "source_sha256",
            "source_carrier_sha256",
            "geometry_teacher_slice_sha256",
            "canonical_camera_contract_sha256",
            "measurement_protocol_sha256",
            "dense_grid_authority_sha256",
            "array_content_sha256",
            "bundle_npz_sha256",
        ):
            _sha(name, getattr(self, name), allow_zero=not qualified)

        if qualified:
            if self.subject_key.startswith("SYNTHETIC_"):
                raise ValueError("F_REFERENCE_INPUT_V2_QUALIFIED_SUBJECT_SYNTHETIC")
            required_flags = (
                "source_exact_bytes_verified",
                "source_carrier_verified",
                "geometry_teacher_slice_validated",
                "canonical_camera_contract_validated",
                "measurement_protocol_preregistered",
                "dense_grid_authority_validated",
                "array_content_validated",
                "outside_shell_qualified",
            )
            for flag in required_flags:
                if not self.metadata.get(flag, False):
                    raise ValueError(f"F_REFERENCE_INPUT_V2_QUALIFICATION_FLAG_FALSE:{flag}")
        elif diagnostic:
            if self.source_sha256 == ZERO_SHA256:
                raise ValueError("F_REFERENCE_INPUT_V2_DIAGNOSTIC_SOURCE_SHA_REQUIRED")
        elif self.authority_state == SYNTHETIC_SMOKE:
            if not self.subject_key.startswith("SYNTHETIC_"):
                raise ValueError("F_REFERENCE_INPUT_V2_SMOKE_SUBJECT_MUST_BE_SYNTHETIC")

    @property
    def matched_geometry_court_eligible(self) -> bool:
        self.validate()
        return self.authority_state == QUALIFIED_FIT8

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["res"] = list(self.res)
        return payload

    def sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


__all__ = [
    "FReferenceInputBundleManifestV2",
    "SCHEMA_VERSION",
]
