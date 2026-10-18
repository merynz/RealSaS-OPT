from __future__ import annotations

"""Focused post-V8 relation trainer (V9).

V9 is deliberately narrower than the historical V8 curriculum.  The exact
completed V8 model is restored, its relation head is replaced by the reviewed
V8->GraphicsRelationHeadV2 zero-step migration artifact, and only the
camera-conditioned relation view branch is trainable.  The shared scene carrier,
mechanical/appearance/structure heads, and the relation embedding/static path are
sealed.  Truth comes exclusively from Relation Truth V2 sidecars with explicit
validity; static same/cut/attachment remain abstentions.

The held-out split is defined over the *base two-hit ray pair*, never individual
orientations.  The two emitted orientations of one ray therefore cannot leak
across train/held-out partitions.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .cohort_v1 import SUBJECT_IDS, VNextCheckpointStoreV1
from .contracts_v1 import IRISVNextConfigV1
from .graphics_truth_v2 import STATIC_RELATION_KEYS_V2, VIEW_RELATION_KEYS_V2
from .heads_v2 import GraphicsRelationHeadV2
from .model_v1 import IRISVNextV1
from .training_v2 import graphics_relation_loss_v2


V9_ARCHITECTURE_ID = "RealSaS.IRISVNextFocusedRelation.v9"
V9_CHECKPOINT_SCHEMA = "RealSaS.IRISVNextFocusedRelationCheckpoint.v9"
V9_BINDING_SCHEMA = "RealSaS.IRISVNextFocusedRelationBinding.v1"
V9_SPLIT_POLICY = "ORIENTED_TWO_HIT_PAIR_GROUP_HASH_RANK_V1"
V9_SPLIT_SEED = 26092357
V9_HOLDOUT_FRACTION = 0.20
V8_SOURCE_COMMIT = "5083c9915533ec8ec6530c7d229c635887abfea7"
V8_CHECKPOINT_SHA256 = "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
V8_BINDING_SHA256 = "735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2"
V8_GLOBAL_STEP = 12000
_SOURCE_KEYS = ("image_tokens", "token_xy", "camera_features")
_ALL_KEYS = STATIC_RELATION_KEYS_V2 + VIEW_RELATION_KEYS_V2


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _value_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_V9_JSON_OBJECT_REQUIRED:{path}")
    return value


def _module_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(cpu.dtype).encode())
        h.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode())
        h.update(cpu.numpy().tobytes(order="C"))
    return h.hexdigest()


def _pair_rank(subject_id: str, view_index: int, pair_index: int) -> bytes:
    return hashlib.sha256(
        f"{V9_SPLIT_POLICY}|{V9_SPLIT_SEED}|{subject_id}|{view_index}|{pair_index}".encode()
    ).digest()


@dataclass(frozen=True)
class FocusedRelationPolicyV9:
    relation_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    maximum_optimizer_steps: int
    evaluation_interval: int
    checkpoint_interval: int
    batch_pair_groups_per_subject: int

    @classmethod
    def from_preregistration(cls, value: Mapping[str, Any]) -> "FocusedRelationPolicyV9":
        if value.get("schema") != "RealSaS.IRISVNextFocusedRelationPreregistration.v1":
            raise ValueError("VNEXT_V9_PREREGISTRATION_SCHEMA_INVALID")
        held = value.get("held_out") or {}
        if held.get("split_policy") != V9_SPLIT_POLICY:
            raise ValueError("VNEXT_V9_SPLIT_POLICY_DRIFT")
        if int(held.get("split_seed", -1)) != V9_SPLIT_SEED:
            raise ValueError("VNEXT_V9_SPLIT_SEED_DRIFT")
        if abs(float(held.get("holdout_fraction", -1.0)) - V9_HOLDOUT_FRACTION) > 1.0e-12:
            raise ValueError("VNEXT_V9_HOLDOUT_FRACTION_DRIFT")
        if held.get("no_training_example_leakage") is not True:
            raise ValueError("VNEXT_V9_HELDOUT_LEAKAGE_GUARD_REQUIRED")

        optimizer = value.get("optimizer") or {}
        if float(optimizer.get("scene_core_learning_rate", -1.0)) != 0.0:
            raise ValueError("VNEXT_V9_SCENE_CORE_LR_MUST_BE_ZERO")
        lr = float(optimizer["relation_head_learning_rate"])
        wd = float(optimizer["weight_decay"])
        betas_raw = optimizer.get("betas")
        if not isinstance(betas_raw, list) or len(betas_raw) != 2:
            raise ValueError("VNEXT_V9_BETAS_REQUIRED")
        betas = (float(betas_raw[0]), float(betas_raw[1]))

        schedule = value.get("schedule") or {}
        policy = cls(
            relation_learning_rate=lr,
            weight_decay=wd,
            betas=betas,
            maximum_optimizer_steps=int(schedule["maximum_optimizer_steps"]),
            evaluation_interval=int(schedule["evaluation_interval"]),
            checkpoint_interval=int(schedule["checkpoint_interval"]),
            batch_pair_groups_per_subject=int(schedule["batch_pair_groups_per_subject"]),
        )
        policy.validate()
        if value.get("thresholds_locked_before_training") is not True:
            raise ValueError("VNEXT_V9_THRESHOLDS_MUST_BE_LOCKED")
        if value.get("scientific_fit_started") is not False:
            raise ValueError("VNEXT_V9_PREREG_ALREADY_STARTED")
        return policy

    def validate(self) -> None:
        if not math.isfinite(self.relation_learning_rate) or self.relation_learning_rate <= 0:
            raise ValueError("VNEXT_V9_RELATION_LR_INVALID")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("VNEXT_V9_WEIGHT_DECAY_INVALID")
        if len(self.betas) != 2 or any(not 0.0 <= x < 1.0 for x in self.betas):
            raise ValueError("VNEXT_V9_BETAS_INVALID")
        if min(
            self.maximum_optimizer_steps,
            self.evaluation_interval,
            self.checkpoint_interval,
            self.batch_pair_groups_per_subject,
        ) <= 0:
            raise ValueError("VNEXT_V9_SCHEDULE_NONPOSITIVE")
        if self.evaluation_interval > self.maximum_optimizer_steps:
            raise ValueError("VNEXT_V9_EVAL_INTERVAL_EXCEEDS_SCHEDULE")


class V9RelationCorpus:
    """Prepared source tensors + versioned Relation Truth V2 sidecars."""

    def __init__(
        self,
        *,
        prepared_root: Path,
        prepared_manifest_path: Path,
        relation_root: Path,
        relation_manifest_path: Path,
        device: torch.device,
    ) -> None:
        self.prepared_root = prepared_root.resolve()
        self.relation_root = relation_root.resolve()
        self.device = torch.device(device)
        prepared_manifest = _load_json(prepared_manifest_path.resolve())
        relation_manifest = _load_json(relation_manifest_path.resolve())
        if prepared_manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
            raise ValueError("VNEXT_V9_PREPARED_MANIFEST_SCHEMA_INVALID")
        if relation_manifest.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
            raise ValueError("VNEXT_V9_RELATION_MANIFEST_SCHEMA_INVALID")
        if tuple(relation_manifest.get("subject_order") or ()) != SUBJECT_IDS:
            raise ValueError("VNEXT_V9_RELATION_SUBJECT_ORDER_DRIFT")

        prepared_rows = list(prepared_manifest.get("subjects") or ())
        relation_rows = list(relation_manifest.get("subjects") or ())
        if tuple(row.get("subject_id") for row in prepared_rows) != SUBJECT_IDS:
            raise ValueError("VNEXT_V9_PREPARED_SUBJECT_ORDER_DRIFT")
        if tuple(row.get("subject_id") for row in relation_rows) != SUBJECT_IDS:
            raise ValueError("VNEXT_V9_RELATION_ROW_ORDER_DRIFT")

        source_rows = []
        for row in prepared_rows:
            path = (self.prepared_root / str(row["shard_file"])).resolve()
            if not path.is_relative_to(self.prepared_root) or _sha(path) != str(row["shard_sha256"]):
                raise ValueError("VNEXT_V9_PREPARED_SHARD_HASH_OR_PATH_DRIFT")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if str(payload.get("subject_id")) != str(row["subject_id"]):
                raise ValueError("VNEXT_V9_PREPARED_SHARD_SUBJECT_DRIFT")
            source = {}
            for key in _SOURCE_KEYS:
                tensor = payload.get(key)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("VNEXT_V9_SOURCE_FIELD_MISSING:" + key)
                if tensor.ndim > 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                source[key] = tensor
            source_rows.append(source)

        self.source: dict[str, torch.Tensor] = {}
        for key in _SOURCE_KEYS:
            values = [row[key] for row in source_rows]
            if len({tuple(x.shape) for x in values}) != 1:
                raise ValueError("VNEXT_V9_SOURCE_BATCH_SHAPE_DRIFT:" + key)
            tensor = torch.stack(values, dim=0).contiguous()
            self.source[key] = tensor.to(self.device)
        if self.source["camera_features"].ndim != 3 or self.source["camera_features"].shape[:2] != (8, 8):
            raise ValueError("VNEXT_V9_CAMERA_FEATURE_SHAPE_INVALID")

        self.truth: list[dict[str, Any]] = []
        self.train_pairs: list[list[torch.Tensor]] = []
        self.heldout_pairs: list[list[torch.Tensor]] = []
        for subject_index, row in enumerate(relation_rows):
            path = (self.relation_root / str(row["file"])).resolve()
            if not path.is_relative_to(self.relation_root) or _sha(path) != str(row["sha256"]):
                raise ValueError("VNEXT_V9_RELATION_SIDECAR_HASH_OR_PATH_DRIFT")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            subject_id = SUBJECT_IDS[subject_index]
            if payload.get("schema") != "RealSaS.IRISVNextFit8RelationTruth.v2" or payload.get("subject_id") != subject_id:
                raise ValueError("VNEXT_V9_RELATION_SIDECAR_IDENTITY_DRIFT")
            views = list(payload.get("views") or ())
            if len(views) != 8:
                raise ValueError("VNEXT_V9_RELATION_VIEW_COUNT_DRIFT")
            train_views: list[torch.Tensor] = []
            held_views: list[torch.Tensor] = []
            for view_index, view in enumerate(views):
                pair_count = self._validate_oriented_view(subject_id, view_index, view)
                ranked = sorted(range(pair_count), key=lambda i: _pair_rank(subject_id, view_index, i))
                holdout_count = int(round(pair_count * V9_HOLDOUT_FRACTION))
                holdout_count = max(1, min(pair_count - 1, holdout_count))
                held = torch.tensor(sorted(ranked[:holdout_count]), dtype=torch.long)
                train = torch.tensor(sorted(ranked[holdout_count:]), dtype=torch.long)
                if len(train) == 0 or len(held) == 0:
                    raise ValueError("VNEXT_V9_SPLIT_EMPTY_PARTITION")
                train_views.append(train)
                held_views.append(held)
            self.truth.append(payload)
            self.train_pairs.append(train_views)
            self.heldout_pairs.append(held_views)

    @staticmethod
    def _validate_oriented_view(subject_id: str, view_index: int, view: Mapping[str, Any]) -> int:
        if view.get("schema") != "RealSaS.IRISVNextFit8RelationTruth.v2":
            raise ValueError("VNEXT_V9_VIEW_SCHEMA_INVALID")
        point_a = view.get("point_a")
        point_b = view.get("point_b")
        if not isinstance(point_a, torch.Tensor) or not isinstance(point_b, torch.Tensor):
            raise ValueError("VNEXT_V9_VIEW_POINT_TENSOR_REQUIRED")
        if point_a.shape != point_b.shape or point_a.ndim != 2 or point_a.shape[-1] != 3:
            raise ValueError("VNEXT_V9_VIEW_POINT_SHAPE_INVALID")
        count = int(point_a.shape[0])
        if count < 4 or count % 2:
            raise ValueError("VNEXT_V9_ORIENTED_ROW_COUNT_INVALID")
        if not torch.equal(point_a[0::2], point_b[1::2]) or not torch.equal(point_b[0::2], point_a[1::2]):
            raise ValueError(f"VNEXT_V9_PAIR_ORIENTATION_POINT_DRIFT:{subject_id}:V{view_index}")

        static_truth = view.get("static_truth") or {}
        static_validity = view.get("static_validity") or {}
        view_truth = view.get("view_truth") or {}
        view_validity = view.get("view_validity") or {}
        for name in STATIC_RELATION_KEYS_V2:
            if name not in static_truth or name not in static_validity:
                raise ValueError("VNEXT_V9_STATIC_TARGET_MISSING:" + name)
            valid = torch.as_tensor(static_validity[name], dtype=torch.bool)
            if tuple(valid.shape) != (count,) or bool(valid.any()):
                raise ValueError("VNEXT_V9_STATIC_RELATION_MUST_ABSTAIN:" + name)
        for name in VIEW_RELATION_KEYS_V2:
            if name not in view_truth or name not in view_validity:
                raise ValueError("VNEXT_V9_VIEW_TARGET_MISSING:" + name)
            target = torch.as_tensor(view_truth[name])
            valid = torch.as_tensor(view_validity[name], dtype=torch.bool)
            if tuple(target.shape) != (count,) or tuple(valid.shape) != (count,) or not bool(valid.all()):
                raise ValueError("VNEXT_V9_VIEW_TARGET_VALIDITY_DRIFT:" + name)

        front = torch.as_tensor(view_truth["front_probability"])
        visa = torch.as_tensor(view_truth["visibility_probability_a"])
        visb = torch.as_tensor(view_truth["visibility_probability_b"])
        occ = torch.as_tensor(view_truth["occlusion_probability"])
        if not torch.equal(front[0::2], torch.ones_like(front[0::2])) or not torch.equal(front[1::2], torch.zeros_like(front[1::2])):
            raise ValueError("VNEXT_V9_FRONT_ORIENTATION_DRIFT")
        if not torch.equal(visa, front) or not torch.equal(visb, 1.0 - front):
            raise ValueError("VNEXT_V9_VISIBILITY_ORIENTATION_DRIFT")
        if not torch.equal(occ, torch.ones_like(occ)):
            raise ValueError("VNEXT_V9_OCCLUSION_CONTRACT_DRIFT")
        for name in (
            "projected_alpha_boundary_probability",
            "projected_appearance_boundary_probability",
        ):
            target = torch.as_tensor(view_truth[name])
            if not torch.equal(target[0::2], target[1::2]):
                raise ValueError("VNEXT_V9_PROJECTED_BOUNDARY_PAIR_DRIFT:" + name)
        return count // 2

    @staticmethod
    def _rows_from_pairs(pair_indices: torch.Tensor) -> torch.Tensor:
        rows = torch.empty(len(pair_indices) * 2, dtype=torch.long)
        rows[0::2] = pair_indices * 2
        rows[1::2] = pair_indices * 2 + 1
        return rows

    def minimum_train_pair_groups(self) -> int:
        return min(int(len(self.train_pairs[s][v])) for s in range(8) for v in range(8))

    def training_batch(self, *, step: int, view_index: int, pair_groups_per_subject: int) -> dict[str, Any]:
        points_a, points_b = [], []
        truth: dict[str, list[torch.Tensor]] = {name: [] for name in _ALL_KEYS}
        validity: dict[str, list[torch.Tensor]] = {name: [] for name in _ALL_KEYS}
        for subject_index, subject_id in enumerate(SUBJECT_IDS):
            available = self.train_pairs[subject_index][view_index]
            if len(available) < pair_groups_per_subject:
                raise ValueError(
                    f"VNEXT_V9_TRAIN_SUPPORT_TOO_SMALL:{subject_id}:V{view_index}:"
                    f"have={len(available)}:need={pair_groups_per_subject}"
                )
            generator = torch.Generator(device="cpu").manual_seed(
                V9_SPLIT_SEED + int(step) * 1000003 + subject_index * 10007 + view_index * 101
            )
            choose = available[torch.randperm(len(available), generator=generator)[:pair_groups_per_subject]]
            rows = self._rows_from_pairs(choose)
            view = self.truth[subject_index]["views"][view_index]
            points_a.append(view["point_a"][rows])
            points_b.append(view["point_b"][rows])
            for name in STATIC_RELATION_KEYS_V2:
                truth[name].append(view["static_truth"][name][rows])
                validity[name].append(view["static_validity"][name][rows])
            for name in VIEW_RELATION_KEYS_V2:
                truth[name].append(view["view_truth"][name][rows])
                validity[name].append(view["view_validity"][name][rows])
        return {
            "point_a": torch.stack(points_a).to(self.device),
            "point_b": torch.stack(points_b).to(self.device),
            "truth": {name: torch.stack(values).to(self.device) for name, values in truth.items()},
            "validity": {name: torch.stack(values).to(self.device) for name, values in validity.items()},
        }

    def heldout_view_batch(self, view_index: int) -> dict[str, Any]:
        rows_by_subject = [
            self._rows_from_pairs(self.heldout_pairs[s][view_index]) for s in range(8)
        ]
        max_rows = max(len(rows) for rows in rows_by_subject)
        point_a = torch.zeros((8, max_rows, 3), dtype=torch.float32)
        point_b = torch.zeros_like(point_a)
        truth = {name: torch.zeros((8, max_rows), dtype=torch.float32) for name in _ALL_KEYS}
        validity = {name: torch.zeros((8, max_rows), dtype=torch.bool) for name in _ALL_KEYS}
        real = torch.zeros((8, max_rows), dtype=torch.bool)
        for subject_index, rows in enumerate(rows_by_subject):
            n = len(rows)
            view = self.truth[subject_index]["views"][view_index]
            point_a[subject_index, :n] = view["point_a"][rows]
            point_b[subject_index, :n] = view["point_b"][rows]
            real[subject_index, :n] = True
            for name in STATIC_RELATION_KEYS_V2:
                truth[name][subject_index, :n] = view["static_truth"][name][rows]
                validity[name][subject_index, :n] = view["static_validity"][name][rows]
            for name in VIEW_RELATION_KEYS_V2:
                truth[name][subject_index, :n] = view["view_truth"][name][rows]
                validity[name][subject_index, :n] = view["view_validity"][name][rows]
        return {
            "point_a": point_a.to(self.device),
            "point_b": point_b.to(self.device),
            "truth": {name: value.to(self.device) for name, value in truth.items()},
            "validity": {name: value.to(self.device) for name, value in validity.items()},
            "real": real.to(self.device),
        }


def bootstrap_v9_model(
    *,
    v8_checkpoint_path: Path,
    migration_artifact_path: Path,
    migration_receipt_path: Path,
    device: torch.device,
    config: IRISVNextConfigV1 = IRISVNextConfigV1(),
) -> IRISVNextV1:
    checkpoint_path = v8_checkpoint_path.resolve()
    if _sha(checkpoint_path) != V8_CHECKPOINT_SHA256:
        raise ValueError("VNEXT_V9_V8_CHECKPOINT_SHA_DRIFT")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("step", -1)) != V8_GLOBAL_STEP or payload.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("VNEXT_V9_V8_CHECKPOINT_IDENTITY_DRIFT")
    binding = payload.get("binding") or {}
    if binding.get("schema") != "RealSaS.IRISVNextFit8Binding.v10":
        raise ValueError("VNEXT_V9_V8_BINDING_SCHEMA_INVALID")
    if not str((binding.get("post_m1_parameter_authority") or {}).get("scene_core", "")).startswith("SEALED_LR_ZERO"):
        raise ValueError("VNEXT_V9_V8_SCENE_CORE_SEAL_MISSING")

    model = IRISVNextV1(config)
    model.load_state_dict(payload["model"], strict=True)
    model.relations = GraphicsRelationHeadV2(config)

    receipt = _load_json(migration_receipt_path.resolve())
    if receipt.get("schema") != "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1" or receipt.get("status") != "PASS":
        raise ValueError("VNEXT_V9_MIGRATION_RECEIPT_INVALID")
    if int(receipt.get("training_steps_performed", -1)) != 0:
        raise ValueError("VNEXT_V9_MIGRATION_ALREADY_TRAINED")
    if (receipt.get("source_v8") or {}).get("source_commit_sha") != V8_SOURCE_COMMIT:
        raise ValueError("VNEXT_V9_MIGRATION_V8_SOURCE_DRIFT")
    artifact_path = migration_artifact_path.resolve()
    if _sha(artifact_path) != (receipt.get("target") or {}).get("sha256"):
        raise ValueError("VNEXT_V9_MIGRATION_ARTIFACT_SHA_DRIFT")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "RealSaS.IRISGraphicsRelationHeadV2Artifact.v1":
        raise ValueError("VNEXT_V9_MIGRATION_ARTIFACT_SCHEMA_INVALID")
    if artifact.get("source_v8_checkpoint_sha256") != V8_CHECKPOINT_SHA256:
        raise ValueError("VNEXT_V9_MIGRATION_ARTIFACT_V8_DRIFT")
    if artifact.get("scene_core_included") is not False or artifact.get("optimizer_state_included") is not False:
        raise ValueError("VNEXT_V9_MIGRATION_ARTIFACT_SCOPE_INVALID")
    if int(artifact.get("training_steps_performed", -1)) != 0:
        raise ValueError("VNEXT_V9_MIGRATION_ARTIFACT_ALREADY_TRAINED")
    model.relations.load_state_dict(artifact["state_dict"], strict=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.relations.view, model.relations.view_pair):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.to(device)
    return model


def _family_metrics(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    p = probability.detach().float()[valid].clamp(1.0e-6, 1.0 - 1.0e-6)
    t = target.detach().float()[valid]
    if len(p) == 0:
        return {
            "valid": 0, "positive": 0, "negative": 0,
            "bce_mean": None, "bce_p50": None, "bce_p95": None,
            "accuracy": None, "balanced_accuracy": None, "brier_mean": None,
        }
    bce = F.binary_cross_entropy(p, t, reduction="none")
    pred = p >= 0.5
    truth = t >= 0.5
    positive = int(truth.sum().item())
    negative = int(len(truth) - positive)
    accuracy = float((pred == truth).float().mean().item())
    balanced = None
    if positive and negative:
        tpr = float(pred[truth].float().mean().item())
        tnr = float((~pred[~truth]).float().mean().item())
        balanced = 0.5 * (tpr + tnr)
    return {
        "valid": int(len(p)),
        "positive": positive,
        "negative": negative,
        "bce_mean": float(bce.mean().item()),
        "bce_p50": float(torch.quantile(bce, 0.50).item()),
        "bce_p95": float(torch.quantile(bce, 0.95).item()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "brier_mean": float(torch.mean((p - t) ** 2).item()),
    }


class IRISVNextFocusedRelationTrainerV9:
    architecture_id = V9_ARCHITECTURE_ID

    def __init__(
        self,
        *,
        prepared_root: Path,
        prepared_manifest_path: Path,
        relation_root: Path,
        relation_manifest_path: Path,
        v8_checkpoint_path: Path,
        migration_artifact_path: Path,
        migration_receipt_path: Path,
        materialization_receipt_path: Path,
        preregistration_path: Path,
        readiness_receipt_path: Path,
        checkpoint_root: Path,
        expected_source_sha: str,
        device: str | torch.device,
        repo_root: Path | None = None,
        config: IRISVNextConfigV1 = IRISVNextConfigV1(),
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("VNEXT_V9_REQUIRES_CUDA_A100_CLASS_EXECUTION")
        source_sha = str(expected_source_sha).strip().lower()
        if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
            raise ValueError("VNEXT_V9_SOURCE_SHA_INVALID")
        self.expected_source_sha = source_sha
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()

        prereg_path = preregistration_path.resolve()
        prereg = _load_json(prereg_path)
        self.policy = FocusedRelationPolicyV9.from_preregistration(prereg)
        materialization_path = materialization_receipt_path.resolve()
        materialization = _load_json(materialization_path)
        if materialization.get("schema") != "RealSaS.IRISVNextFit8PretrainingMaterializationReceipt.v1":
            raise ValueError("VNEXT_V9_MATERIALIZATION_SCHEMA_INVALID")
        if materialization.get("status") != "MATERIALIZED_AND_BYTE_VERIFIED__TRAINING_NOT_STARTED":
            raise ValueError("VNEXT_V9_MATERIALIZATION_NOT_READY")
        if materialization.get("source_commit_sha") != source_sha or materialization.get("candidate_roundtrip_verified") is not True:
            raise ValueError("VNEXT_V9_MATERIALIZATION_SOURCE_OR_ROUNDTRIP_DRIFT")

        relation_manifest_path = relation_manifest_path.resolve()
        migration_receipt_path = migration_receipt_path.resolve()
        exact = prereg.get("exact_artifact_bindings") or {}
        required_bindings = {
            "source_commit_sha": source_sha,
            "materialization_receipt_sha256": _sha(materialization_path),
            "migration_receipt_sha256": _sha(migration_receipt_path),
            "relation_truth_v2_manifest_sha256": _sha(relation_manifest_path),
        }
        for key, expected in required_bindings.items():
            if str(exact.get(key) or "").lower() != expected.lower():
                raise ValueError("VNEXT_V9_PREREGISTRATION_BINDING_DRIFT:" + key)

        readiness_path = readiness_receipt_path.resolve()
        readiness = _load_json(readiness_path)
        if readiness.get("schema") != "RealSaS.IRISVNextFocusedRelationPretrainingReadiness.v1":
            raise ValueError("VNEXT_V9_READINESS_SCHEMA_INVALID")
        if readiness.get("status") != "READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED":
            raise ValueError("VNEXT_V9_READINESS_NOT_PASS")
        if readiness.get("source_commit_sha") != source_sha or readiness.get("training_steps_executed") != 0:
            raise ValueError("VNEXT_V9_READINESS_SOURCE_OR_STEP_DRIFT")
        if readiness.get("training_entry_authorized") is not True:
            raise ValueError("VNEXT_V9_READINESS_NOT_AUTHORIZED")
        if readiness.get("preregistration_sha256") != _sha(prereg_path):
            raise ValueError("VNEXT_V9_READINESS_PREREGISTRATION_DRIFT")

        self.corpus = V9RelationCorpus(
            prepared_root=prepared_root,
            prepared_manifest_path=prepared_manifest_path,
            relation_root=relation_root,
            relation_manifest_path=relation_manifest_path,
            device=self.device,
        )
        if self.corpus.minimum_train_pair_groups() < self.policy.batch_pair_groups_per_subject:
            raise ValueError(
                "VNEXT_V9_PREREGISTERED_BATCH_EXCEEDS_MINIMUM_SUPPORT:"
                f"have={self.corpus.minimum_train_pair_groups()}:"
                f"need={self.policy.batch_pair_groups_per_subject}"
            )
        self.model = bootstrap_v9_model(
            v8_checkpoint_path=v8_checkpoint_path,
            migration_artifact_path=migration_artifact_path,
            migration_receipt_path=migration_receipt_path,
            device=self.device,
            config=config,
        )

        self.frozen_hashes = {
            "scene_core": _module_hash(self.model.scene_core),
            "mechanical_head": _module_hash(self.model.mechanical_demand),
            "appearance_head": _module_hash(self.model.appearance),
            "structure_head": _module_hash(self.model.structure),
            "relation_embedding": _module_hash(self.model.relations.embedding),
            "relation_static_pair": _module_hash(self.model.relations.static_pair),
        }
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        expected_trainable = {
            id(p)
            for module in (self.model.relations.view, self.model.relations.view_pair)
            for p in module.parameters()
        }
        if {id(p) for p in trainable} != expected_trainable:
            raise RuntimeError("VNEXT_V9_TRAINABLE_PARAMETER_SCOPE_DRIFT")
        self.optimizer = torch.optim.AdamW(
            [{"params": trainable, "lr": self.policy.relation_learning_rate, "name": "relation_view_branch"}],
            weight_decay=self.policy.weight_decay,
            betas=self.policy.betas,
            fused=True,
        )

        random.seed(V9_SPLIT_SEED)
        np.random.seed(V9_SPLIT_SEED)
        torch.manual_seed(V9_SPLIT_SEED)
        torch.cuda.manual_seed_all(V9_SPLIT_SEED)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Frozen H is deterministic for V9: encode once in eval mode and detach.
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.scene_planes = self.model.encode_scene(
                self.corpus.source["image_tokens"],
                self.corpus.source["token_xy"],
                self.corpus.source["camera_features"],
            ).detach()

        self.binding = {
            "schema": V9_BINDING_SCHEMA,
            "architecture_id": V9_ARCHITECTURE_ID,
            "source_commit_sha": source_sha,
            "materialization_receipt_sha256": _sha(materialization_path),
            "preregistration_sha256": _sha(prereg_path),
            "readiness_receipt_sha256": _sha(readiness_path),
            "prepared_manifest_sha256": _sha(prepared_manifest_path.resolve()),
            "relation_truth_v2_manifest_sha256": _sha(relation_manifest_path),
            "migration_receipt_sha256": _sha(migration_receipt_path),
            "migration_artifact_sha256": _sha(migration_artifact_path.resolve()),
            "v8_checkpoint_sha256": _sha(v8_checkpoint_path.resolve()),
            "split": {
                "policy": V9_SPLIT_POLICY,
                "seed": V9_SPLIT_SEED,
                "holdout_fraction": V9_HOLDOUT_FRACTION,
                "orientation_grouped": True,
            },
            "optimization": asdict(self.policy),
            "trainable_authority": "RELATION_VIEW_AND_VIEW_PAIR_ONLY",
            "scene_core": "SEALED__PRECOMPUTED_DETERMINISTIC_H",
            "static_relation_path": "SEALED__NO_AUTHORITY_TO_REWRITE",
            "teacher_inputs_in_inference_forward": False,
        }
        self.binding_hash = _value_hash(self.binding)
        self.store = VNextCheckpointStoreV1(checkpoint_root, self.binding_hash)
        self.step = 0
        self.started = time.monotonic()
        previous = self.store.load()
        if previous is not None:
            self._restore(previous)
        else:
            self.save()

    def _assert_frozen_non_regression(self) -> dict[str, bool]:
        current = {
            "scene_core": _module_hash(self.model.scene_core),
            "mechanical_head": _module_hash(self.model.mechanical_demand),
            "appearance_head": _module_hash(self.model.appearance),
            "structure_head": _module_hash(self.model.structure),
            "relation_embedding": _module_hash(self.model.relations.embedding),
            "relation_static_pair": _module_hash(self.model.relations.static_pair),
        }
        result = {name: current[name] == expected for name, expected in self.frozen_hashes.items()}
        if not all(result.values()):
            failed = ",".join(name for name, ok in result.items() if not ok)
            raise RuntimeError("VNEXT_V9_FROZEN_PARAMETER_DRIFT:" + failed)
        return result

    def _payload(self) -> dict[str, Any]:
        self._assert_frozen_non_regression()
        return {
            "schema": V9_CHECKPOINT_SCHEMA,
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "relation_state": self.model.relations.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "training_scope": "RELATION_VIEW_AND_VIEW_PAIR_ONLY",
            "frozen_hashes": dict(self.frozen_hashes),
        }

    def save(self):
        return self.store.save(self._payload())

    def _restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != V9_CHECKPOINT_SCHEMA:
            raise ValueError("VNEXT_V9_RESUME_SCHEMA_INVALID")
        if payload.get("binding_sha256") != self.binding_hash:
            raise ValueError("VNEXT_V9_RESUME_BINDING_DRIFT")
        if payload.get("frozen_hashes") != self.frozen_hashes:
            raise ValueError("VNEXT_V9_RESUME_FROZEN_HASH_DRIFT")
        self.model.relations.load_state_dict(payload["relation_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self._assert_frozen_non_regression()

    def _predict(self, point_a: torch.Tensor, point_b: torch.Tensor, view_index: int) -> dict[str, torch.Tensor]:
        # H is sealed: no gradient graph may reach scene_core or static relation path.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feature_a = self.model.scene_core.query(self.scene_planes, point_a)["H"].detach()
            feature_b = self.model.scene_core.query(self.scene_planes, point_b)["H"].detach()
        cameras = self.corpus.source["camera_features"][:, int(view_index), None, :].expand(
            -1, feature_a.shape[1], -1
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model.relations.relation_probabilities(feature_a, feature_b, cameras)

    def one_step(self) -> dict[str, Any]:
        if self.step >= self.policy.maximum_optimizer_steps:
            raise StopIteration("VNEXT_V9_FOCUSED_RELATION_SCHEDULE_COMPLETE")
        next_step = self.step + 1
        view_index = next_step % 8
        batch = self.corpus.training_batch(
            step=next_step,
            view_index=view_index,
            pair_groups_per_subject=self.policy.batch_pair_groups_per_subject,
        )
        self.model.relations.view.train()
        self.model.relations.view_pair.train()
        self.optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in self.optimizer.param_groups[0]["params"]]
        predicted = self._predict(batch["point_a"], batch["point_b"], view_index)
        loss = graphics_relation_loss_v2(predicted, batch["truth"], batch["validity"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("VNEXT_V9_NONFINITE_LOSS")
        loss.backward()

        gradient_sq = 0.0
        parameter_sq = 0.0
        for parameter in self.optimizer.param_groups[0]["params"]:
            p = parameter.detach().float()
            parameter_sq += float(torch.sum(p * p).item())
            if parameter.grad is not None:
                g = parameter.grad.detach().float()
                gradient_sq += float(torch.sum(g * g).item())
        self.optimizer.step()
        update_sq = 0.0
        for parameter, old in zip(self.optimizer.param_groups[0]["params"], before):
            delta = parameter.detach().float() - old.to(self.device, dtype=torch.float32)
            update_sq += float(torch.sum(delta * delta).item())
        self.step = next_step
        frozen = self._assert_frozen_non_regression()
        if self.step % self.policy.checkpoint_interval == 0:
            self.save()

        families = {
            name: _family_metrics(predicted[name], batch["truth"][name], batch["validity"][name])
            for name in _ALL_KEYS
        }
        parameter_l2 = math.sqrt(parameter_sq)
        update_l2 = math.sqrt(update_sq)
        return {
            "global_step": self.step,
            "view_index": view_index,
            "loss": float(loss.detach().item()),
            "family_metrics": families,
            "telemetry": {
                "group": "relation_view_branch",
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "parameter_l2": parameter_l2,
                "gradient_l2": math.sqrt(gradient_sq),
                "update_l2": update_l2,
                "update_parameter_ratio": update_l2 / max(parameter_l2, 1.0e-30),
            },
            "frozen_non_regression": frozen,
            "batch_subjects": 8,
            "pair_groups_per_subject": self.policy.batch_pair_groups_per_subject,
            "optimizer_steps": 1,
            "elapsed_seconds": time.monotonic() - self.started,
        }

    @torch.no_grad()
    def evaluate_heldout(self) -> dict[str, Any]:
        self.model.eval()
        family_probability: dict[str, list[torch.Tensor]] = {name: [] for name in _ALL_KEYS}
        family_truth: dict[str, list[torch.Tensor]] = {name: [] for name in _ALL_KEYS}
        family_validity: dict[str, list[torch.Tensor]] = {name: [] for name in _ALL_KEYS}
        for view_index in range(8):
            batch = self.corpus.heldout_view_batch(view_index)
            predicted = self._predict(batch["point_a"], batch["point_b"], view_index)
            for name in _ALL_KEYS:
                valid = batch["validity"][name] & batch["real"]
                family_probability[name].append(predicted[name].detach().float().cpu())
                family_truth[name].append(batch["truth"][name].detach().float().cpu())
                family_validity[name].append(valid.detach().cpu())
        metrics = {}
        for name in _ALL_KEYS:
            p = torch.cat([x.reshape(-1) for x in family_probability[name]])
            t = torch.cat([x.reshape(-1) for x in family_truth[name]])
            v = torch.cat([x.reshape(-1) for x in family_validity[name]])
            metrics[name] = _family_metrics(p, t, v)
        frozen = self._assert_frozen_non_regression()
        return {
            "schema": "RealSaS.IRISVNextFocusedRelationHeldOutMetrics.v1",
            "global_step": self.step,
            "split_policy": V9_SPLIT_POLICY,
            "split_seed": V9_SPLIT_SEED,
            "holdout_fraction": V9_HOLDOUT_FRACTION,
            "families": metrics,
            "frozen_non_regression": frozen,
            "training_steps_executed": self.step,
            "scientific_fit_pass_claimed": False,
        }
