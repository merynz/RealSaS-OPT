"""Truth139 TESSA topology trainer over sealed student conditioning and target-only topology.

Every epoch exposes every TRAIN subject exactly once.  The selected topology window cycles by
`epoch % subject_window_count`, so source-subject weighting is equal while the fixed recipe spans
multiple complete passes over the most complex topology.  Conditioning and teacher topology remain
physically separate inputs and are joined only by sealed subject/source identities inside this
training worker.  No unseen artifact is accepted.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from models.iris.training.cohort_v1 import AtomicCheckpointStore, file_hash, value_hash
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import (
    TESSATrainingWindowV1,
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
)
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from .dual_stream_v31 import TESSADualStreamStructuralizerV31


TEACHER_SCHEMA = "RealSaS.Truth139TESSATeacherTrainManifest.v1"
CONDITIONING_SCHEMA = "RealSaS.Truth139TESSAConditioningIndex.v1"
RECIPE_SCHEMA = "RealSaS.Truth139TESSATrainingRecipe.v1"
CHECKPOINT_SCHEMA = "RealSaS.Truth139TESSACheckpoint.v1"
RESULT_SCHEMA = "RealSaS.Truth139TESSATrainingResult.v1"


def _require(value, code: str) -> None:
    if not value:
        raise ValueError(code)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _codec_config(cfg: TESSAStructuralizerConfigV2) -> TESSAConfigV1:
    return TESSAConfigV1(
        coordinate_bins=cfg.coordinate_bins,
        max_faces=cfg.max_faces,
        max_vertices=cfg.max_vertices,
        local_attention_window=cfg.local_attention_window,
        query_chunk_size=cfg.query_chunk_size,
        surface_latent_count=cfg.surface_memory_tokens,
    )


class _TESSASubjectV1:
    def __init__(
        self,
        *,
        subject_id: str,
        conditioning_row: dict,
        teacher_row: dict,
        config: TESSAStructuralizerConfigV2,
        target_tokens: int,
    ) -> None:
        self.subject_id = subject_id
        if str(conditioning_row.get("subject_id")) != subject_id or str(teacher_row.get("subject_id")) != subject_id:
            raise ValueError("TRUTH139_TESSA_SUBJECT_JOIN_ID")
        if str(conditioning_row.get("source_sha256")) != str(teacher_row.get("source_sha256")):
            raise ValueError("TRUTH139_TESSA_SUBJECT_JOIN_SOURCE_SHA")

        cache_ref = dict(conditioning_row.get("conditioning_cache") or {})
        cache_path = Path(str(cache_ref.get("path") or "")).resolve()
        _require(cache_path.is_file() and file_hash(cache_path) == str(cache_ref.get("sha256") or ""),
                 "TRUTH139_TESSA_CONDITIONING_CACHE_DRIFT:" + subject_id)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        _require(cache.get("schema") == "RealSaS.Truth139TESSAConditioningCache.v1",
                 "TRUTH139_TESSA_CONDITIONING_CACHE_SCHEMA")
        _require(cache.get("subject_id") == subject_id and cache.get("teacher_topology_used") is False,
                 "TRUTH139_TESSA_CONDITIONING_CACHE_AUTHORITY")
        self.scene_source = cache["scene_source"].float().contiguous()
        self.surface_source = cache["surface_source"].float().contiguous()
        _require(self.scene_source.ndim == 3 and self.scene_source.shape[0] == 1,
                 "TRUTH139_TESSA_SCENE_CACHE_SHAPE")
        _require(self.surface_source.ndim == 3 and self.surface_source.shape[0] == 1,
                 "TRUTH139_TESSA_SURFACE_CACHE_SHAPE")

        mesh_ref = dict(teacher_row.get("teacher_mesh") or {})
        mesh_path = Path(str(mesh_ref.get("path") or "")).resolve()
        _require(mesh_path.is_file() and file_hash(mesh_path) == str(mesh_ref.get("sha256") or ""),
                 "TRUTH139_TESSA_TEACHER_MESH_DRIFT:" + subject_id)
        with np.load(mesh_path, allow_pickle=False) as z:
            vertices = np.asarray(z["vertices_normalized"])
            faces = np.asarray(z["faces"])
        sequence = build_teacher_asset_sequence_v1(vertices, faces, cfg=_codec_config(config))
        self.windows = build_truncated_teacher_windows_v1(
            sequence,
            cfg=_codec_config(config),
            target_tokens=int(target_tokens),
        )
        _require(self.windows, "TRUTH139_TESSA_TEACHER_WINDOWS_EMPTY:" + subject_id)

    def window(self, epoch: int) -> TESSATrainingWindowV1:
        return self.windows[int(epoch) % len(self.windows)]


class Truth139TESSATrainerV1:
    def __init__(
        self,
        *,
        conditioning_index: dict,
        teacher_manifest: dict,
        recipe: dict,
        checkpoint_root,
        output_root,
        device: str = "cuda",
    ) -> None:
        _require(conditioning_index.get("schema") == CONDITIONING_SCHEMA and conditioning_index.get("scope") == "TRAIN",
                 "TRUTH139_TESSA_TRAIN_CONDITIONING_SCHEMA")
        _require(teacher_manifest.get("schema") == TEACHER_SCHEMA, "TRUTH139_TESSA_TRAIN_TEACHER_SCHEMA")
        _require(recipe.get("schema") == RECIPE_SCHEMA, "TRUTH139_TESSA_TRAIN_RECIPE_SCHEMA")
        split = str(recipe.get("split_identity_sha256") or "")
        _require(split and split == conditioning_index.get("split_identity_sha256") == teacher_manifest.get("split_identity_sha256"),
                 "TRUTH139_TESSA_TRAIN_SPLIT_DRIFT")
        _require(recipe.get("architecture_id") == TESSADualStreamStructuralizerV31.architecture_id,
                 "TRUTH139_TESSA_TRAIN_ARCHITECTURE")
        _require(recipe.get("attention_semantics") == TESSADualStreamStructuralizerV31.attention_semantics,
                 "TRUTH139_TESSA_TRAIN_ATTENTION_SEMANTICS")
        _require(recipe.get("presentation_stream_present") is False,
                 "TRUTH139_TESSA_TRAIN_PRESENTATION_STREAM_FORBIDDEN")
        _require(conditioning_index.get("teacher_topology_used") is False,
                 "TRUTH139_TESSA_TRAIN_CONDITIONING_TEACHER_LEAKAGE")
        _require(teacher_manifest.get("contains_unseen_subject_ids") is False and teacher_manifest.get("contains_unseen_file_refs") is False,
                 "TRUTH139_TESSA_TRAIN_UNSEEN_FIREWALL")
        _require(recipe["checkpoint"]["selection_rule"] == "FIXED_FINAL_ENDPOINT__NO_CHECKPOINT_SHOPPING",
                 "TRUTH139_TESSA_TRAIN_SELECTION_RULE")

        conditioning_rows = {str(row["subject_id"]): row for row in conditioning_index.get("subjects") or ()}
        teacher_rows = {str(row["subject_id"]): row for row in teacher_manifest.get("subjects") or ()}
        _require(conditioning_rows and set(conditioning_rows) == set(teacher_rows),
                 "TRUTH139_TESSA_TRAIN_SUBJECT_ROSTER")
        self.ids = sorted(conditioning_rows)

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "TRUTH139_TESSA_TRAIN_CUDA_REQUIRED")
        runtime = recipe["runtime"]
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False
        self.seed = 261029731
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed & 0xFFFFFFFF)

        config = TESSAStructuralizerConfigV2(**dict(recipe["model_config"]))
        config.validate()
        self.config = config
        self.model = TESSADualStreamStructuralizerV31(
            scene_input_dim=int(recipe["scene_input_dim"]),
            surface_input_dim=int(recipe["surface_input_dim"]),
            config=config,
            backend=str(runtime["attention_backend"]),
        ).to(self.device).train()
        self.weights = TESSAObjectiveWeightsV2(**dict(recipe["objective"]))
        self.weights.validate()
        target_tokens = int(recipe["teacher_windows"]["target_tokens_per_window"])
        self.subjects = {
            subject: _TESSASubjectV1(
                subject_id=subject,
                conditioning_row=conditioning_rows[subject],
                teacher_row=teacher_rows[subject],
                config=config,
                target_tokens=target_tokens,
            )
            for subject in self.ids
        }
        expected_windows = {str(k): int(v) for k, v in recipe["teacher_windows"]["windows_by_subject"].items()}
        actual_windows = {subject: len(self.subjects[subject].windows) for subject in self.ids}
        _require(actual_windows == expected_windows, "TRUTH139_TESSA_TRAIN_WINDOW_COUNT_DRIFT")

        self.recipe = recipe
        self.conditioning_index = conditioning_index
        self.teacher_manifest = teacher_manifest
        binding = {
            "schema": "RealSaS.Truth139TESSATrainingBinding.v1",
            "split_identity_sha256": split,
            "conditioning_index_identity_sha256": conditioning_index["index_identity_sha256"],
            "teacher_manifest_identity_sha256": teacher_manifest["manifest_identity_sha256"],
            "recipe_sha256": value_hash(recipe),
            "architecture_id": self.model.architecture_id,
            "attention_semantics": self.model.attention_semantics,
            "subject_ids": self.ids,
            "iris_checkpoint_sha256": recipe["iris_checkpoint_sha256"],
            "iris_foundation_identity": recipe["iris_foundation_identity"],
        }
        self.binding = binding
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root), self.binding_hash)
        self.output = Path(output_root).resolve(); self.output.mkdir(parents=True, exist_ok=True)

        opt = recipe["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(opt["learning_rate"]), weight_decay=float(opt["weight_decay"])
        )
        epochs = int(recipe["teacher_windows"]["epochs"])
        groups = math.ceil(len(self.ids) / int(opt["subjects_per_optimizer_step"]))
        self.total_steps = epochs * groups
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.total_steps), eta_min=float(opt["minimum_learning_rate"])
        )
        self.completed_epoch = 0
        self.step = 0
        self.exposures = {subject: 0 for subject in self.ids}
        previous = self.store.load()
        if previous:
            self.model.load_state_dict(previous["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(previous["optimizer_state_dict"])
            self.scheduler.load_state_dict(previous["scheduler_state_dict"])
            self.completed_epoch = int(previous["completed_epoch"])
            self.step = int(previous["step"])
            self.exposures = {str(k): int(v) for k, v in previous["exposures"].items()}

    def _order(self, epoch: int) -> list[str]:
        return sorted(self.ids, key=lambda subject: hashlib.sha256(
            f"{self.seed}|{int(epoch)}|{subject}".encode()
        ).hexdigest())

    def _subject_loss(self, subject: _TESSASubjectV1, epoch: int):
        window = subject.window(epoch)
        scene = subject.scene_source.to(self.device, non_blocking=True)
        surface = subject.surface_source.to(self.device, non_blocking=True)
        input_ids = torch.tensor([window.input_ids], device=self.device, dtype=torch.long)
        labels = torch.tensor([window.labels], device=self.device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=bool(self.recipe["runtime"]["bf16_autocast"])):
            output = self.model(
                scene_source=scene,
                surface_source=surface,
                input_ids=input_ids,
                labels=labels,
                sequence_position_offset=int(window.sequence_position_offset),
                objective_weights=self.weights,
            )
        _require(output.loss is not None and output.objective is not None, "TRUTH139_TESSA_TRAIN_OBJECTIVE_MISSING")
        return output.objective

    def _save(self) -> None:
        self.store.save({
            "step": int(self.step),
            "completed_epoch": int(self.completed_epoch),
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "exposures": self.exposures,
            "binding_sha256": self.binding_hash,
        })

    def train(self) -> dict:
        started = time.monotonic()
        epochs = int(self.recipe["teacher_windows"]["epochs"])
        group_size = int(self.recipe["optimizer"]["subjects_per_optimizer_step"])
        checkpoint_every = int(self.recipe["checkpoint"]["interval_epochs"])
        _require(group_size >= 1 and checkpoint_every >= 1, "TRUTH139_TESSA_TRAIN_RUNTIME_POLICY")
        while self.completed_epoch < epochs:
            epoch = self.completed_epoch
            terms: dict[str, list[float]] = {}
            order = self._order(epoch)
            for start in range(0, len(order), group_size):
                group = order[start:start + group_size]
                self.optimizer.zero_grad(set_to_none=True)
                for subject_id in group:
                    objective = self._subject_loss(self.subjects[subject_id], epoch)
                    (objective.total / len(group)).backward()
                    self.exposures[subject_id] += 1
                    for key, value in (
                        ("total", objective.total),
                        ("token_cross_entropy", objective.token_cross_entropy),
                        ("coordinate_metric", objective.coordinate_metric),
                        ("structural_special_token", objective.structural_special_token),
                        ("coordinate_expected_abs_error_bins", objective.coordinate_expected_abs_error_bins),
                    ):
                        terms.setdefault(key, []).append(float(value.detach().cpu()))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=float(self.recipe["optimizer"]["gradient_clip_norm"]))
                self.optimizer.step(); self.scheduler.step(); self.step += 1
            self.completed_epoch += 1
            print(json.dumps({
                "event": "TRUTH139_TESSA_EPOCH", "epoch": self.completed_epoch, "epochs": epochs,
                "optimizer_step": self.step, "train_subject_count": len(self.ids),
                "loss": {key: float(np.mean(values)) for key, values in terms.items()},
                "unseen": "NOT_MOUNTED"
            }, sort_keys=True), flush=True)
            if self.completed_epoch % checkpoint_every == 0 or self.completed_epoch == epochs:
                self._save()

        _require(all(count == epochs for count in self.exposures.values()),
                 "TRUTH139_TESSA_TRAIN_SUBJECT_EXPOSURE_IMBALANCE")
        selected = self.output / "selected_final_tessa_checkpoint.pt"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": self.model.architecture_id,
            "attention_semantics": self.model.attention_semantics,
            "config": asdict(self.config),
            "scene_input_dim": int(self.recipe["scene_input_dim"]),
            "surface_input_dim": int(self.recipe["surface_input_dim"]),
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
        }
        staged = selected.with_suffix(".pt.partial"); torch.save(payload, staged)
        torch.load(staged, map_location="cpu", weights_only=True); staged.replace(selected)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "TRAINING_COMPLETE__UNSEEN_NOT_OPENED__FIXED_FINAL_ENDPOINT",
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "split_identity_sha256": self.recipe["split_identity_sha256"],
            "conditioning_index_identity_sha256": self.conditioning_index["index_identity_sha256"],
            "teacher_manifest_identity_sha256": self.teacher_manifest["manifest_identity_sha256"],
            "iris_checkpoint_sha256": self.recipe["iris_checkpoint_sha256"],
            "iris_foundation_identity": self.recipe["iris_foundation_identity"],
            "train_subject_count": len(self.ids),
            "epochs": epochs,
            "subject_exposures": self.exposures,
            "optimizer_steps": self.step,
            "selected_checkpoint_sha256": file_hash(selected),
            "selection_rule": self.recipe["checkpoint"]["selection_rule"],
            "unseen_opened": False,
            "unseen_gradient_updates": False,
            "unseen_checkpoint_selection": False,
            "tessa_pass_claimed": False,
            "compiler_m_pass_claimed": False,
            "product_pass_claimed": False,
            "wall_seconds": time.monotonic() - started,
        }
        _atomic_json(self.output / "TESSA_TRAINING_RESULT.json", result)
        return result
