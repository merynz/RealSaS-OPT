from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

from canonical.architecture_freeze_gate_v1 import require_family_selection_authority


SCHEMA = "RealSaS.PostFreezeFamilySelector.v1"
POLICY_ID = "FIT8_PREFIT_BLINDED_HASH_ORDER_V1"
RANK_DOMAIN = "RealSaS/FIT8/pre-fit/blinded/hash-order/v1"

# These fields encode results that can only exist after model fitting/evaluation. A
# candidate ledger containing any of them is rejected rather than silently ignored.
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
    visual_character_only_pass: bool
    visual_no_render_artifact_pass: bool
    visual_no_dominant_geometric_block_pass: bool
    visual_clear_character_silhouette_pass: bool
    visual_audit_id: str
    visual_audit_sha256: str
    candidate_authority_sha256: str

    def validate(self) -> None:
        for field in ("asset_id", "source_family_id", "visual_audit_id", "visual_audit_sha256", "candidate_authority_sha256"):
            if not getattr(self, field):
                raise ValueError(f"EMPTY_REQUIRED_FIELD:{field}")
        if len(self.visual_audit_sha256) != 64 or len(self.candidate_authority_sha256) != 64:
            raise ValueError("AUTHORITY_HASH_MUST_BE_SHA256_HEX")

    @property
    def eligible(self) -> bool:
        self.validate()
        return all((
            self.master_native_1024,
            self.master_fit_admit,
            self.iris_truth_capable,
            self.geppetto_truth_capable,
            self.arachne_truth_capable,
            self.single_pose_core_eligible,
            self.exact_camera_raster_authority,
            self.explicit_source_textured_rgba,
            self.visual_character_only_pass,
            self.visual_no_render_artifact_pass,
            self.visual_no_dominant_geometric_block_pass,
            self.visual_clear_character_silhouette_pass,
        ))


@dataclass(frozen=True)
class SelectedFamilyV1:
    ordinal: int
    asset_id: str
    source_family_id: str
    blinded_rank_sha256: str
    candidate_authority_sha256: str
    visual_audit_id: str
    visual_audit_sha256: str


@dataclass(frozen=True)
class FamilySelectionResultV1:
    schema: str
    policy_id: str
    architecture_freeze_fingerprint_sha256: str
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


def select_prefit_families_v1(
    repo_root: str | Path,
    candidates: Iterable[PrefitFamilyCandidateV1],
    *,
    count: int = 8,
) -> FamilySelectionResultV1:
    """Select a deterministic pre-fit panel only after architecture freeze.

    Selection uses no model output. The caller must provide a sealed visual audit made
    before fitting; among fully eligible candidates, membership is determined only by
    a fixed domain-separated SHA-256 order. Changing source architecture invalidates
    the architecture seal and blocks this function before candidate ranking.
    """
    if count < 1:
        raise ValueError("selection count must be positive")
    root = Path(repo_root).resolve()
    seal = require_family_selection_authority(root)
    rows = tuple(candidates)
    if not rows:
        raise ValueError("candidate set is empty")
    seen_assets: set[str] = set()
    seen_families: set[str] = set()
    for row in rows:
        row.validate()
        if row.asset_id in seen_assets:
            raise ValueError(f"DUPLICATE_ASSET_ID:{row.asset_id}")
        if row.source_family_id in seen_families:
            raise ValueError(f"DUPLICATE_SOURCE_FAMILY_ID:{row.source_family_id}")
        seen_assets.add(row.asset_id)
        seen_families.add(row.source_family_id)

    eligible = tuple(row for row in rows if row.eligible)
    if len(eligible) < count:
        raise RuntimeError(f"INSUFFICIENT_PREFIT_ELIGIBLE_FAMILIES:need={count}:have={len(eligible)}")
    ranked = sorted(eligible, key=lambda row: (_rank(row.asset_id), row.asset_id))
    chosen = ranked[:count]
    selected = tuple(
        SelectedFamilyV1(
            ordinal=i,
            asset_id=row.asset_id,
            source_family_id=row.source_family_id,
            blinded_rank_sha256=_rank(row.asset_id),
            candidate_authority_sha256=row.candidate_authority_sha256,
            visual_audit_id=row.visual_audit_id,
            visual_audit_sha256=row.visual_audit_sha256,
        )
        for i, row in enumerate(chosen)
    )
    candidate_payload = [asdict(row) for row in sorted(rows, key=lambda r: r.asset_id)]
    candidate_set_sha = _canonical_hash(candidate_payload)
    selection_hash_payload = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "architecture_freeze_fingerprint_sha256": seal["generic_source_fingerprint_sha256"],
        "requested_count": count,
        "eligible_count": len(eligible),
        "selected": [asdict(row) for row in selected],
        "candidate_set_sha256": candidate_set_sha,
        "scientific_fit_steps_before_selection": 0,
        "fit_metrics_consumed": False,
        "selection_basis": "SEALED_PREFIT_ELIGIBILITY_PLUS_DOMAIN_SEPARATED_SHA256_ORDER",
    }
    selection_sha = _canonical_hash(selection_hash_payload)
    return FamilySelectionResultV1(
        schema=SCHEMA,
        policy_id=POLICY_ID,
        architecture_freeze_fingerprint_sha256=seal["generic_source_fingerprint_sha256"],
        requested_count=count,
        eligible_count=len(eligible),
        selected=selected,
        candidate_set_sha256=candidate_set_sha,
        selection_sha256=selection_sha,
        scientific_fit_steps_before_selection=0,
        fit_metrics_consumed=False,
        selection_basis="SEALED_PREFIT_ELIGIBILITY_PLUS_DOMAIN_SEPARATED_SHA256_ORDER",
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
