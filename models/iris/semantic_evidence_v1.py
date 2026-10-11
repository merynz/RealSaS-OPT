from __future__ import annotations

"""Provider-neutral frozen semantic evidence contract for IRIS research.

This module deliberately does not import or mutate the canonical IRIS geometry
model.  External/open-vocabulary segmentation models are evidence producers,
not authority.  Compiler qualification and final-M addressing happen downstream.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "RealSaS.IRISFrozenSemanticMaskEvidence.v1"
UNKNOWN_LABEL = "unknown"


def _norm_label(value: str) -> str:
    value = "_".join(str(value).strip().lower().replace("-", " ").split())
    if not value:
        raise ValueError("semantic label must be non-empty")
    return value


def _array_hash(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class FrozenSemanticMaskEvidenceV1:
    """Modal per-view mask proposals emitted by any frozen external provider.

    `proposal_masks` is [P,H,W] bool.  A proposal is intentionally not called a
    canonical owner: multiple proposals may refer to one owner, one proposal may
    be wrong, and hidden surfaces may have no proposal at all.
    """

    labels: tuple[str, ...]
    proposal_masks: np.ndarray
    proposal_view_index: np.ndarray
    proposal_label_index: np.ndarray
    proposal_score: np.ndarray
    proposal_ids: tuple[str, ...]
    provider_id: str
    source_identity: str
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        labels = tuple(_norm_label(x) for x in self.labels)
        masks = np.asarray(self.proposal_masks)
        views = np.asarray(self.proposal_view_index)
        li = np.asarray(self.proposal_label_index)
        scores = np.asarray(self.proposal_score)
        p = len(masks)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("semantic labels must be unique and non-empty")
        if masks.ndim != 3 or masks.dtype != np.bool_ or p == 0:
            raise ValueError("proposal_masks must be non-empty bool [P,H,W]")
        if masks.shape[1] <= 0 or masks.shape[2] <= 0:
            raise ValueError("proposal masks require positive image dimensions")
        if views.shape != (p,) or li.shape != (p,) or scores.shape != (p,):
            raise ValueError("proposal metadata shape mismatch")
        if len(self.proposal_ids) != p or len(set(self.proposal_ids)) != p:
            raise ValueError("proposal_ids must be unique and match masks")
        if np.any(views < 0) or np.any(views >= 8):
            raise ValueError("proposal view index must be in [0,7]")
        if np.any(li < 0) or np.any(li >= len(labels)):
            raise ValueError("proposal label index outside vocabulary")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("proposal scores must be finite in [0,1]")
        if not self.provider_id or not self.source_identity:
            raise ValueError("provider_id and source_identity are required")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "proposal_masks", masks)
        object.__setattr__(self, "proposal_view_index", views.astype(np.int8, copy=False))
        object.__setattr__(self, "proposal_label_index", li.astype(np.int16, copy=False))
        object.__setattr__(self, "proposal_score", scores.astype(np.float32, copy=False))

    @property
    def proposal_count(self) -> int:
        return int(len(self.proposal_masks))

    @property
    def height(self) -> int:
        return int(self.proposal_masks.shape[1])

    @property
    def width(self) -> int:
        return int(self.proposal_masks.shape[2])

    def label_for(self, proposal_index: int) -> str:
        return self.labels[int(self.proposal_label_index[proposal_index])]

    def identity_payload(self) -> dict:
        return {
            "schema": self.schema_version,
            "provider_id": self.provider_id,
            "source_identity": self.source_identity,
            "labels": list(self.labels),
            "proposal_ids": list(self.proposal_ids),
            "proposal_view_index_hash": _array_hash(self.proposal_view_index),
            "proposal_label_index_hash": _array_hash(self.proposal_label_index),
            "proposal_score_hash": _array_hash(self.proposal_score),
            "proposal_masks_hash": _array_hash(self.proposal_masks),
            "shape": list(self.proposal_masks.shape),
        }

    def evidence_hash(self) -> str:
        raw = json.dumps(self.identity_payload(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(raw).hexdigest()


def build_mask_evidence_v1(
    proposals: Iterable[dict],
    *,
    labels: Iterable[str],
    provider_id: str,
    source_identity: str,
    image_shape: tuple[int, int],
) -> FrozenSemanticMaskEvidenceV1:
    """Build a deterministic bundle from provider proposal dictionaries.

    Each proposal requires `view_index`, `label`, `score`, and bool `mask`.
    Unknown provider labels fail closed instead of being silently remapped.
    """

    vocab = tuple(_norm_label(x) for x in labels)
    if len(set(vocab)) != len(vocab):
        raise ValueError("duplicate normalized semantic labels")
    label_index = {x: i for i, x in enumerate(vocab)}
    rows = []
    h, w = map(int, image_shape)
    for ordinal, row in enumerate(proposals):
        label = _norm_label(row["label"])
        if label not in label_index:
            raise ValueError(f"provider label outside declared vocabulary:{label}")
        mask = np.asarray(row["mask"], dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"proposal mask shape mismatch:{mask.shape}!={(h, w)}")
        score = float(row["score"])
        view = int(row["view_index"])
        pid = str(row.get("proposal_id") or f"V{view}:{label}:{ordinal:04d}")
        rows.append((view, label_index[label], score, pid, mask))
    if not rows:
        raise ValueError("no semantic proposals supplied")
    rows.sort(key=lambda x: (x[0], x[1], -x[2], x[3]))
    return FrozenSemanticMaskEvidenceV1(
        labels=vocab,
        proposal_masks=np.stack([x[4] for x in rows]).astype(bool),
        proposal_view_index=np.asarray([x[0] for x in rows], dtype=np.int8),
        proposal_label_index=np.asarray([x[1] for x in rows], dtype=np.int16),
        proposal_score=np.asarray([x[2] for x in rows], dtype=np.float32),
        proposal_ids=tuple(x[3] for x in rows),
        provider_id=str(provider_id),
        source_identity=str(source_identity),
    )


def write_mask_evidence_v1(path: str | Path, evidence: FrozenSemanticMaskEvidenceV1) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = evidence.identity_payload()
    np.savez_compressed(
        path,
        proposal_masks=evidence.proposal_masks,
        proposal_view_index=evidence.proposal_view_index,
        proposal_label_index=evidence.proposal_label_index,
        proposal_score=evidence.proposal_score,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "evidence_hash": evidence.evidence_hash(),
        "schema": SCHEMA,
    }


def load_mask_evidence_v1(path: str | Path, *, expected_sha256: str | None = None) -> FrozenSemanticMaskEvidenceV1:
    path = Path(path)
    actual = sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("semantic evidence bytes drift")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        if meta.get("schema") != SCHEMA:
            raise ValueError("semantic evidence schema drift")
        evidence = FrozenSemanticMaskEvidenceV1(
            labels=tuple(meta["labels"]),
            proposal_masks=data["proposal_masks"].astype(bool),
            proposal_view_index=data["proposal_view_index"],
            proposal_label_index=data["proposal_label_index"],
            proposal_score=data["proposal_score"],
            proposal_ids=tuple(meta["proposal_ids"]),
            provider_id=str(meta["provider_id"]),
            source_identity=str(meta["source_identity"]),
        )
    if evidence.identity_payload() != meta:
        raise ValueError("semantic evidence metadata/hash mismatch")
    return evidence
