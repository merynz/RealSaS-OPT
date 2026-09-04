from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

from canonical.architecture_freeze_gate_v2 import require_family_selection_authority
from experiments.family_selection_v1.prefit_family_truth_eligibility_v1 import (
    POLICY_ID as TRUTH_ELIGIBILITY_POLICY_ID,
    PrefitFamilyTruthEligibilityV1,
)

SCHEMA = "RealSaS.PostFreezeFamilySelector.v1"
POLICY_ID = "FIT8_PREFIT_BLINDED_HASH_ORDER_V2"
RANK_DOMAIN = "RealSaS/FIT8/pre-fit/blinded/hash-order/v2"
FORBIDDEN_POSTFIT_KEYS = frozenset({
    "fit_loss", "train_loss", "val_loss", "test_loss", "checkpoint", "checkpoint_sha256",
    "iris_score", "iris_metrics", "geppetto_score", "geppetto_metrics",
    "arachne_score", "arachne_metrics", "product_score", "product_pass",
    "e2e_pass", "runtime_pass", "selected_checkpoint", "optimizer_steps",
})


@dataclass(frozen=True)
class PrefitFamilyCandidateV1:
    asset_id: str
    source_family_id: str
    master_native_1024: bool
    master_fit_admit: bool
    iris_truth_capable: bool
    geppetto_truth_capable: bool
    arachne_truth_capable: bool
    single_pose_core_eligible: bool
    exact_camera_raster_authority: bool
    explicit_source_textured_rgba: bool
    prefit_truth_eligibility_pass: bool
    prefit_truth_eligibility_sha256: str
    visual_character_only_pass: bool
    visual_no_render_artifact_pass: bool
    visual_no_dominant_geometric_block_pass: bool
    visual_clear_character_silhouette_pass: bool
    visual_audit_id: str
    visual_audit_sha256: str
    candidate_authority_sha256: str

    def validate(self) -> None:
        for field in ("asset_id", "source_family_id", "prefit_truth_eligibility_sha256", "visual_audit_id", "visual_audit_sha256", "candidate_authority_sha256"):
            if not getattr(self, field):
                raise ValueError(f"EMPTY_REQUIRED_FIELD:{field}")
        for field in ("prefit_truth_eligibility_sha256", "visual_audit_sha256", "candidate_authority_sha256"):
            value = str(getattr(self, field))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"AUTHORITY_HASH_MUST_BE_SHA256_HEX:{field}")

    @property
    def eligible(self) -> bool:
        self.validate()
        return all((
            self.master_native_1024, self.master_fit_admit, self.iris_truth_capable,
            self.geppetto_truth_capable, self.arachne_truth_capable, self.single_pose_core_eligible,
            self.exact_camera_raster_authority, self.explicit_source_textured_rgba,
            self.prefit_truth_eligibility_pass, self.visual_character_only_pass,
            self.visual_no_render_artifact_pass, self.visual_no_dominant_geometric_block_pass,
            self.visual_clear_character_silhouette_pass,
        ))


@dataclass(frozen=True)
class SelectedFamilyV1:
    ordinal: int
    asset_id: str
    source_family_id: str
    blinded_rank_sha256: str
    candidate_authority_sha256: str
    prefit_truth_eligibility_sha256: str
    visual_audit_id: str
    visual_audit_sha256: str


@dataclass(frozen=True)
class FamilySelectionResultV1:
    schema: str
    policy_id: str
    architecture_freeze_fingerprint_sha256: str
    selection_apparatus_fingerprint_sha256: str
    requested_count: int
    eligible_count: int
    selected: tuple[SelectedFamilyV1, ...]
    candidate_set_sha256: str
    selection_sha256: str
    scientific_fit_steps_before_selection: int
    fit_metrics_consumed: bool
    selection_basis: str

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _rank(asset_id: str) -> str:
    return sha256(f"{RANK_DOMAIN}\0{asset_id}".encode("utf-8")).hexdigest()


def _reject_postfit_fields(raw_records: Iterable[Mapping[str, object]]) -> None:
    for index, record in enumerate(raw_records):
        bad = FORBIDDEN_POSTFIT_KEYS.intersection(record.keys())
        if bad:
            raise ValueError(f"POSTFIT_INFORMATION_FORBIDDEN_BEFORE_SELECTION:row={index}:keys={sorted(bad)}")


def candidate_from_mapping_v1(record: Mapping[str, object]) -> PrefitFamilyCandidateV1:
    _reject_postfit_fields((record,))
    allowed = set(PrefitFamilyCandidateV1.__dataclass_fields__)
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f"UNKNOWN_PREFIT_CANDIDATE_FIELDS:{sorted(unknown)}")
    candidate = PrefitFamilyCandidateV1(**record)
    candidate.validate()
    return candidate


def _validate_truth_eligibility_authority(candidate: PrefitFamilyCandidateV1, report: PrefitFamilyTruthEligibilityV1 | None) -> None:
    if report is None:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_REPORT_REQUIRED:{candidate.asset_id}")
    if report.policy_id != TRUTH_ELIGIBILITY_POLICY_ID:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_POLICY_DRIFT:{candidate.asset_id}:{report.policy_id}")
    if report.asset_id != candidate.asset_id:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_ASSET_MISMATCH:{candidate.asset_id}:{report.asset_id}")
    if report.scientific_fit_steps != 0 or report.postfit_information_consumed is not False:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_POSTFIT_CONTAMINATION:{candidate.asset_id}")
    if report.source_mesh_used_for_model_input or report.teacher_truth_used_for_model_input:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_MODEL_INPUT_FIREWALL_VIOLATION:{candidate.asset_id}")
    if report.pass_prefit_truth_eligibility is not True:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_NOT_PASS:{candidate.asset_id}")
    if candidate.prefit_truth_eligibility_pass is not True:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_CANDIDATE_FLAG_NOT_PASS:{candidate.asset_id}")
    if report.eligibility_sha256 != candidate.prefit_truth_eligibility_sha256:
        raise ValueError(f"PREFIT_TRUTH_ELIGIBILITY_SHA_MISMATCH:{candidate.asset_id}")


def select_prefit_families_v1(
    repo_root: str | Path,
    candidates: Iterable[PrefitFamilyCandidateV1],
    *,
    truth_eligibility_reports: Mapping[str, PrefitFamilyTruthEligibilityV1],
    count: int = 8,
) -> FamilySelectionResultV1:
    if count < 1:
        raise ValueError("selection count must be positive")
    seal = require_family_selection_authority(Path(repo_root).resolve())
    rows = tuple(candidates)
    if not rows:
        raise ValueError("candidate set is empty")
    seen_assets: set[str] = set(); seen_families: set[str] = set()
    for row in rows:
        row.validate()
        if row.asset_id in seen_assets:
            raise ValueError(f"DUPLICATE_ASSET_ID:{row.asset_id}")
        if row.source_family_id in seen_families:
            raise ValueError(f"DUPLICATE_SOURCE_FAMILY_ID:{row.source_family_id}")
        seen_assets.add(row.asset_id); seen_families.add(row.source_family_id)
        if row.prefit_truth_eligibility_pass:
            _validate_truth_eligibility_authority(row, truth_eligibility_reports.get(row.asset_id))
    eligible = tuple(row for row in rows if row.eligible)
    if len(eligible) < count:
        raise RuntimeError(f"INSUFFICIENT_PREFIT_ELIGIBLE_FAMILIES:need={count}:have={len(eligible)}")
    chosen = sorted(eligible, key=lambda row: (_rank(row.asset_id), row.asset_id))[:count]
    selected = tuple(SelectedFamilyV1(i, row.asset_id, row.source_family_id, _rank(row.asset_id), row.candidate_authority_sha256, row.prefit_truth_eligibility_sha256, row.visual_audit_id, row.visual_audit_sha256) for i, row in enumerate(chosen))
    candidate_set_sha = _canonical_hash([asdict(row) for row in sorted(rows, key=lambda r: r.asset_id)])
    living_fp = str(seal["living_source_fingerprint_sha256"])
    selection_fp = str(seal["selection_apparatus_fingerprint_sha256"])
    hash_payload = {
        "schema": SCHEMA, "policy_id": POLICY_ID,
        "architecture_freeze_fingerprint_sha256": living_fp,
        "selection_apparatus_fingerprint_sha256": selection_fp,
        "requested_count": count, "eligible_count": len(eligible),
        "selected": [asdict(row) for row in selected], "candidate_set_sha256": candidate_set_sha,
        "scientific_fit_steps_before_selection": 0, "fit_metrics_consumed": False,
        "selection_basis": "TYPED_PREFIT_TRUTH_ELIGIBILITY_PLUS_SEALED_VISUAL_AUDIT_PLUS_BLINDED_SHA256_ORDER",
    }
    return FamilySelectionResultV1(
        SCHEMA, POLICY_ID, living_fp, selection_fp, count, len(eligible), selected,
        candidate_set_sha, _canonical_hash(hash_payload), 0, False,
        "TYPED_PREFIT_TRUTH_ELIGIBILITY_PLUS_SEALED_VISUAL_AUDIT_PLUS_BLINDED_SHA256_ORDER",
    )


def load_candidate_ledger_v1(path: str | Path) -> tuple[PrefitFamilyCandidateV1, ...]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != "RealSaS.PrefitFamilyCandidateLedger.v1":
        raise ValueError("candidate ledger schema mismatch")
    raw = obj.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("candidate ledger candidates must be a list")
    _reject_postfit_fields(raw)
    return tuple(candidate_from_mapping_v1(row) for row in raw)
