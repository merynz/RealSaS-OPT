from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

SCHEMA_VERSION = "RealSaS.FGeometryTeacherSlice.v1"
PARENT_BOUND = "PARENT_FULL_TRUTH_BOUND"
PARENT_DEFERRED = "PARENT_FULL_TRUTH_DEFERRED"
PARENT_STATES = {PARENT_BOUND, PARENT_DEFERRED}
ZERO_SHA256 = "0" * 64


def _sha(name: str, value: str, *, allow_zero: bool = False) -> None:
    if len(value) != 64:
        raise ValueError(f"F_GEOMETRY_SLICE_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"F_GEOMETRY_SLICE_SHA_NONHEX:{name}") from exc
    if not allow_zero and value == ZERO_SHA256:
        raise ValueError(f"F_GEOMETRY_SLICE_ZERO_SHA_FORBIDDEN:{name}")


@dataclass(frozen=True)
class FGeometryTeacherSliceV1:
    """Minimal subject authority consumed by F geometry/extraction courts.

    This is a typed child slice of the eventual full FIT8 teacher bundle. It lets
    geometry research proceed without pretending that unrelated appearance,
    mechanics or dynamic channels have already been materialized. A geometry-only
    court may be scientifically qualified against this slice; it is not a full
    FIT8 teacher-bundle PASS until `parent_state == PARENT_FULL_TRUTH_BOUND`.
    """

    subject_key: str
    source_exact_bytes_sha256: str
    source_carrier_sha256: str
    source_geometry_frame_sha256: str
    topology_qualification_sha256: str
    self_intersection_qualification_sha256: str
    extraction_truth_sha256: str
    extraction_scale_manifest_sha256: str
    field_policy_sha256: str
    parent_state: str
    parent_full_truth_bundle_sha256: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_GEOMETRY_SLICE_SCHEMA_INVALID")
        if not self.subject_key or self.subject_key.startswith("SYNTHETIC_"):
            raise ValueError("F_GEOMETRY_SLICE_SUBJECT_INVALID")
        for name in (
            "source_exact_bytes_sha256",
            "source_carrier_sha256",
            "source_geometry_frame_sha256",
            "topology_qualification_sha256",
            "self_intersection_qualification_sha256",
            "extraction_truth_sha256",
            "extraction_scale_manifest_sha256",
            "field_policy_sha256",
        ):
            _sha(name, getattr(self, name))
        if self.parent_state not in PARENT_STATES:
            raise ValueError("F_GEOMETRY_SLICE_PARENT_STATE_INVALID")
        if self.parent_state == PARENT_BOUND:
            _sha("parent_full_truth_bundle_sha256", self.parent_full_truth_bundle_sha256)
        else:
            _sha(
                "parent_full_truth_bundle_sha256",
                self.parent_full_truth_bundle_sha256,
                allow_zero=True,
            )
            if self.parent_full_truth_bundle_sha256 != ZERO_SHA256:
                raise ValueError("F_GEOMETRY_SLICE_DEFERRED_PARENT_MUST_BE_ZERO")
        if not self.provenance:
            raise ValueError("F_GEOMETRY_SLICE_PROVENANCE_MISSING")

    @property
    def full_truth_lineage_closed(self) -> bool:
        self.validate()
        return self.parent_state == PARENT_BOUND

    @property
    def geometry_court_eligible(self) -> bool:
        self.validate()
        return True

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


__all__ = [
    "FGeometryTeacherSliceV1",
    "PARENT_BOUND",
    "PARENT_DEFERRED",
    "PARENT_STATES",
    "SCHEMA_VERSION",
    "ZERO_SHA256",
]
