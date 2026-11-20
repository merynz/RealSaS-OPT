from __future__ import annotations

"""Fail-closed 300–400 *real* source pilot with user-authored split membership.

This is an admission contract, NOT a dataset creator or a claim that 300–400
licensed sources are already available. All fields must come from independently
audited source/native/render records; SHA-shaped strings alone are not proof of
file possession. The separate evidence verification gate checks local receipts.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json

SCHEMA = "RealSaS.RSSvNext.Real400PilotCohort.v1"
SPLITS = ("TRAIN", "VAL", "UNSEEN_TEST")
REQUIRED_CORE = frozenset({
    "TWO_ORIGINAL_RGBA", "CANONICAL_RGB", "SIX_RGB_NORMAL",
    "NATIVE_MESH", "NATIVE_CAMERA",
})
OPTIONAL_TEACHERS = frozenset({
    "NATIVE_POSE",  # optional per asset; missing is UNKNOWN, not negative
    "SOURCE_SLR", "SOURCE_GRAPHICS", "SOURCE_SKIN",
    "SOURCE_AXIS_SEQUENCE", "SOURCE_POSED_SURFACE",
    "SOURCE_ANCHOR", "SOURCE_LAYER_CLEARANCE",
    "SOURCE_CONTROL_INTERVENTION", "SOURCE_OPEN_HEM",
    "SOURCE_PIN_SUPPORT",
})
VALID_TEACHERS = REQUIRED_CORE | OPTIONAL_TEACHERS
RECEIPT_TYPES = ("license", "native", "render")
EVIDENCE_SCHEMA = "RealSaS.RSSvNext.SourcePilotEvidence.v1"


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _json_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RealPilotAssetV1:
    source_id: str
    source_asset_sha256: str
    original_front_sha256: str
    original_back_sha256: str
    # Shared mesh, derivative, pack, or original scene must carry SAME key.
    lineage_group: str
    split: str
    license_receipt_sha256: str
    native_receipt_sha256: str
    render_receipt_sha256: str
    teacher_families: tuple[str, ...]
    license_decision: str = "COMMERCIAL_TRAINING_APPROVED"
    noai_review: str = "NO_EXPLICIT_NOAI_CONFIRMED"
    native_qualification: str = "SOURCE_NATIVE_VERIFIED"

    def validate(self) -> None:
        if (
            type(self.source_id) is not str or not self.source_id.strip()
            or len(self.source_id) > 128
            or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in self.source_id)
            or type(self.lineage_group) is not str
            or not self.lineage_group.strip() or len(self.lineage_group) > 256
            or self.split not in SPLITS
            or self.license_decision != "COMMERCIAL_TRAINING_APPROVED"
            or self.noai_review != "NO_EXPLICIT_NOAI_CONFIRMED"
            or self.native_qualification != "SOURCE_NATIVE_VERIFIED"
            or any(not _sha(x) for x in (
                self.source_asset_sha256,
                self.original_front_sha256,
                self.original_back_sha256,
                self.license_receipt_sha256,
                self.native_receipt_sha256,
                self.render_receipt_sha256,
            ))
            or self.original_front_sha256 == self.original_back_sha256
            or type(self.teacher_families) is not tuple
            or len(set(self.teacher_families)) != len(self.teacher_families)
            or not set(self.teacher_families).issubset(VALID_TEACHERS)
            or not REQUIRED_CORE.issubset(self.teacher_families)
        ):
            raise ValueError(
                "RSS_VNEXT_REAL_PILOT_SOURCE_LICENSE_NATIVE_OR_CORE_TRUTH_INCOMPLETE:"
                + str(self.source_id)
            )


@dataclass(frozen=True)
class RealPilotCohortReceiptV1:
    schema: str
    cohort_sha256: str
    total_assets: int
    train_assets: int
    val_assets: int
    unseen_test_assets: int
    lineage_groups: int
    teacher_family_coverage_train: dict[str, int]
    source_evidence_bytes_verified: bool
    full_pilot_launch_qualified: bool = False


def qualify_real_pilot_cohort_v1(
    assets: tuple[RealPilotAssetV1, ...],
) -> RealPilotCohortReceiptV1:
    """Validate 300..400 independently qualified identities without picking splits.

    Every split comes exclusively from the user-supplied source manifest.
    We enforce only disjointness, separate validation/test and train-only
    optimizer access. No asset, especially SK190/Pieta, is automatically
    assigned to a split; no 80/10/10 distribution is imposed.
    """
    if type(assets) is not tuple or not 300 <= len(assets) <= 400:
        raise ValueError("RSS_VNEXT_REAL_PILOT_REQUIRES_300_TO_400_SOURCES")
    ids: set[str] = set()
    asset_sha: set[str] = set()
    images: set[str] = set()
    groups: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}
    coverage = {family: 0 for family in sorted(VALID_TEACHERS)}
    for item in assets:
        if type(item) is not RealPilotAssetV1:
            raise TypeError("RSS_VNEXT_REAL_PILOT_ASSET_TYPE")
        item.validate()
        if (
            item.source_id in ids or item.source_asset_sha256 in asset_sha
            or item.original_front_sha256 in images
            or item.original_back_sha256 in images
        ):
            raise ValueError("RSS_VNEXT_REAL_PILOT_DUPLICATE_SOURCE_OR_IMAGE")
        ids.add(item.source_id)
        asset_sha.add(item.source_asset_sha256)
        images.update((item.original_front_sha256,item.original_back_sha256))
        previous = groups.setdefault(item.lineage_group, item.split)
        if previous != item.split:
            raise ValueError("RSS_VNEXT_REAL_PILOT_LINEAGE_LEAK_BETWEEN_SPLITS")
        counts[item.split] += 1
        if item.split == "TRAIN":
            for family in item.teacher_families:
                coverage[family] += 1
    # These are evaluation protocol requirements, NOT a suggested split ratio.
    # Which asset enters which split remains exclusively user-controlled.
    if any(counts[split] < 1 for split in SPLITS):
        raise ValueError("RSS_VNEXT_REAL_PILOT_SPLIT_REQUIRED_NONEMPTY")
    total = len(assets)
    content = [asdict(x) for x in sorted(assets, key=lambda x: x.source_id)]
    return RealPilotCohortReceiptV1(
        schema=SCHEMA,
        cohort_sha256=_json_digest({"schema":SCHEMA,"assets":content}),
        total_assets=total,
        train_assets=counts["TRAIN"],
        val_assets=counts["VAL"],
        unseen_test_assets=counts["UNSEEN_TEST"],
        lineage_groups=len(groups),
        teacher_family_coverage_train=coverage,
        source_evidence_bytes_verified=False,
    )


def verify_pilot_source_evidence_v1(
    assets: tuple[RealPilotAssetV1, ...],
    receipt_root: str | Path,
    cohort: RealPilotCohortReceiptV1,
) -> RealPilotCohortReceiptV1:
    """Inspect ALL 3*300..400 locally mounted source audit receipts.

    Each JSON file is immutable and content-addressed by SHA; the issuer and
    observation truth still need independent external authorization. This
    guarantees the run cannot be admitted on SHA-shaped placeholders.
    """
    if cohort != qualify_real_pilot_cohort_v1(assets):
        raise ValueError("RSS_VNEXT_REAL_PILOT_COHORT_IDENTITY_DRIFT")
    root = Path(receipt_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("RSS_VNEXT_REAL_PILOT_RECEIPT_ROOT_MISSING")
    for item in assets:
        for kind in RECEIPT_TYPES:
            path = root / item.source_id / (kind + ".json")
            if path.is_symlink() or not path.is_file():
                raise ValueError("RSS_VNEXT_REAL_PILOT_SOURCE_EVIDENCE_MISSING")
            raw = path.read_bytes()
            expected = getattr(item,kind + "_receipt_sha256")
            if sha256(raw).hexdigest() != expected:
                raise ValueError("RSS_VNEXT_REAL_PILOT_SOURCE_EVIDENCE_SHA_DRIFT")
            if len(raw) > 512 * 1024:
                raise ValueError("RSS_VNEXT_REAL_PILOT_EVIDENCE_TOO_LARGE")
            try:
                data = json.loads(raw)
            except (ValueError,UnicodeDecodeError) as exc:
                raise ValueError("RSS_VNEXT_REAL_PILOT_EVIDENCE_NOT_JSON") from exc
            if (
                type(data) is not dict
                or data.get("schema") != EVIDENCE_SCHEMA
                or data.get("kind") != kind.upper()
                or data.get("source_id") != item.source_id
                or data.get("source_asset_sha256") != item.source_asset_sha256
                or data.get("decision") != "APPROVED_SOURCE_VERIFICATION"
                or type(data.get("reviewer")) is not str
                or not data["reviewer"].strip()
                or type(data.get("reviewed_at")) is not str
                or not data["reviewed_at"].strip()
                or (
                    kind == "render" and
                    sorted(data.get("qualified_teacher_families", []))
                    != sorted(item.teacher_families)
                )
            ):
                raise ValueError("RSS_VNEXT_REAL_PILOT_EVIDENCE_QUALIFICATION_DRIFT")
    from dataclasses import replace
    return replace(cohort,source_evidence_bytes_verified=True)


def assert_train_only_source_v1(
    asset: RealPilotAssetV1, cohort: RealPilotCohortReceiptV1,
) -> None:
    """No VAL / UNSEEN_TEST item may be routed to optimizer training."""
    asset.validate()
    if (
        asset.split != "TRAIN" or
        not cohort.source_evidence_bytes_verified
    ):
        # Real per-source receipts are verified separately by the trainer;
        # a cohort-wide flag is not sufficient evidence of teacher truth.
        raise ValueError("RSS_VNEXT_REAL_PILOT_TRAIN_SOURCE_NOT_ADMITTED")


__all__ = [
    "SCHEMA", "SPLITS", "REQUIRED_CORE", "OPTIONAL_TEACHERS",
    "RealPilotAssetV1", "RealPilotCohortReceiptV1",
    "qualify_real_pilot_cohort_v1","verify_pilot_source_evidence_v1",
    "assert_train_only_source_v1",
]
