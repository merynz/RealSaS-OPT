"""Real FIT8 7->1 LOFO trainer for IRIS V3.1.

Scientific update semantics are fold-level and explicit: one global update equals
one equally weighted gradient contribution from every one of the seven training
subjects. Physical microbatching is one subject at a time and cannot change that
meaning. The held-out subject is not loaded until a train-only checkpoint has
been selected and frozen.
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

from models.iris.training.apparatus_v1 import targets as field_targets
from models.iris.training.cohort_v1 import AtomicCheckpointStore, file_hash, value_hash
from models.iris.v5.direct_fstar_fit_v5 import DirectFStarFitPolicyV5, direct_fstar_loss_v5, direct_fstar_metrics_v5
from models.iris.v31.checkpoint_v1 import write_checkpoint_v1
from models.iris.v31.model_v31 import IRISConfigV31, IRISV31, admit_observations_v31
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
    appearance_rgba_loss_v1,
    heteroscedastic_mechanical_loss_v1,
    pair_probe_loss_v1,
)
from models.training_runtime.semantic_batch_v1 import SemanticExample, execute_semantic_step


def _require(value, code):
    if not value:
        raise ValueError(code)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _stateless_indices(count: int, take: int, seed: int) -> np.ndarray:
    if count <= 0 or take <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    return rng.integers(0, count, size=min(take, count), endpoint=False, dtype=np.int64)


def _partition(subject: str, family: str, count: int, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    _require(0.0 < train_fraction < 1.0, "IRIS_V31_LOFO_PRESENTATION_PARTITION_FRACTION")
    train, val = [], []
    for index in range(count):
        token = f"{seed}|{subject}|{family}|{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(1 << 64)
        (train if bucket < train_fraction else val).append(index)
    if count >= 2 and not val:
        val.append(train.pop())
    if count >= 2 and not train:
        train.append(val.pop())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class FoldSubject:
    def __init__(self, row: dict, device: torch.device, recipe: dict):
        self.row = row
        self.subject_id = str(row["subject_id"])
        self.device = device
        geometry = Path(row["geometry_shard"]["path"])
        presentation = Path(row["presentation_targets"]["path"])
        features = Path(row["frozen_feature_cache"]["features_path"])
        for path, expected, code in (
            (geometry, row["geometry_shard"]["sha256"], "GEOMETRY"),
            (presentation, row["presentation_targets"]["sha256"], "PRESENTATION"),
            (features, row["frozen_feature_cache"]["features_sha256"], "FEATURE"),
        ):
            _require(path.is_file() and file_hash(path) == expected, f"IRIS_V31_LOFO_{code}_BYTES_DRIFT:{self.subject_id}")
        self.geometry = torch.load(geometry, map_location="cpu", weights_only=True)
        self.features = torch.load(features, map_location="cpu", weights_only=True)
        with np.load(presentation, allow_pickle=False) as z:
            self.presentation = {name: np.asarray(z[name]).copy() for name in z.files}
        _require(self.geometry.get("subject_id") == self.subject_id, "IRIS_V31_LOFO_SUBJECT_GEOMETRY_DRIFT")
        _require(set(self.features) == {"final", "layer2", "alpha"}, "IRIS_V31_LOFO_FEATURE_FIELDS")
        self.cameras = self.geometry["camera_features"].float()
        admit_observations_v31(
            self.features["final"].float(), self.features["layer2"].float(), self.features["alpha"].float(), self.cameras
        )
        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        seed = int(recipe["sampling"]["seed"])
        self.partitions = {}
        for family, key in (("mechanics", "mechanical_valid"), ("appearance", "appearance_valid"), ("pair", "pair_valid")):
            valid = np.flatnonzero(np.asarray(self.presentation[key], dtype=np.bool_))
            train_local, val_local = _partition(self.subject_id, family, len(valid), seed, fraction)
            self.partitions[family] = (valid[train_local], valid[val_local])

    def observation(self):
        final = self.features["final"].to(self.device, non_blocking=True).float()
        layer2 = self.features["layer2"].to(self.device, non_blocking=True).float()
        alpha = self.features["alpha"].to(self.device, non_blocking=True).float()
        cameras = self.cameras.to(self.device, non_blocking=True).float()
        return final, layer2, alpha, cameras

    def field_teacher(self):
        keys = (
            "signed_2d", "center_xyz", "normalization_half_extent", "camera_origins",
            "camera_right", "camera_screen_up", "camera_half_extent", "resolution",
        )
        return {key: self.geometry[key].to(self.device) if isinstance(self.geometry[key], torch.Tensor) else self.geometry[key] for key in keys}


class IrisV31LofoTrainer:
    def __init__(self, *, manifest: dict, recipe: dict, fold_id: str, checkpoint_root, output_root, device="cuda"):
        _require(manifest.get("schema") == "RealSaS.IRISV31LofoInputManifest.v1", "IRIS_V31_LOFO_INPUT_SCHEMA")
        _require(manifest.get("heldout_selection_blind") is True and manifest.get("train_subjects_per_fold") == 7, "IRIS_V31_LOFO_INPUT_PARTITION")
        _require(recipe.get("schema") == "RealSaS.IRISV31LofoRecipe.v1", "IRIS_V31_LOFO_RECIPE_SCHEMA")
        folds = [row for row in manifest["folds"] if row["fold_id"] == fold_id]
        _require(len(folds) == 1, "IRIS_V31_LOFO_FOLD_ID")
        self.fold = folds[0]
        _require(self.fold.get("checkpoint_selection_may_read_heldout") is False and self.fold.get("heldout_gradient_updates") is False, "IRIS_V31_LOFO_HELDOUT_FIREWALL")
        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "IRIS_V31_LOFO_CUDA_REQUIRED")
        torch.set_float32_matmul_precision(recipe["optimizer"]["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(recipe["optimizer"]["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(recipe["optimizer"]["tf32"])
        torch.backends.cudnn.deterministic = bool(recipe["optimizer"]["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False
        seed = int(recipe["sampling"]["seed"]) + int(self.fold["fold_index"]) * 1000003
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        self.seed = seed
        config_row = dict(recipe["model_config"])
        config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
        config_row["pair_channels"] = tuple(config_row["pair_channels"])
        self.config = IRISConfigV31(**config_row)
        self.config.validate()
        _require(self.config.mechanical_channels == MECHANICAL_CHANNELS_V31_PRESENTATION, "IRIS_V31_LOFO_MECHANICAL_CHANNEL_DRIFT")
        _require(self.config.pair_channels == PAIR_CHANNELS_V31_PRESENTATION, "IRIS_V31_LOFO_PAIR_CHANNEL_DRIFT")
        self.model = IRISV31(self.config).to(self.device).train()
        self.recipe = recipe
        self.manifest = manifest
        rows = {row["subject_id"]: row for row in manifest["subjects"]}
        _require(set(self.fold["train_subjects"]) <= set(rows) and self.fold["heldout_subject"] in rows, "IRIS_V31_LOFO_FOLD_SUBJECT_DRIFT")
        # Critical firewall: heldout row is retained only as inert metadata here.
        self.train_subjects = {subject: FoldSubject(rows[subject], self.device, recipe) for subject in self.fold["train_subjects"]}
        self.heldout_row = rows[self.fold["heldout_subject"]]
        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema": "RealSaS.IRISV31LofoTrainingBinding.v1",
            "input_manifest_sha256": value_hash(manifest),
            "recipe_sha256": value_hash(recipe),
            "fold": self.fold,
            "config": asdict(self.config),
        }
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root) / fold_id, self.binding_hash)
        optimizer = recipe["optimizer"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(optimizer["learning_rate"]), weight_decay=float(optimizer["weight_decay"]))
        total_steps = sum(int(row["steps"]) for row in recipe["schedule"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=float(optimizer["minimum_learning_rate"]))
        self.step = 0
        self.best = None
        prior = self.store.load()
        if prior:
            self.model.load_state_dict(prior["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(prior["optimizer_state_dict"])
            self.scheduler.load_state_dict(prior["scheduler_state_dict"])
            self.step = int(prior["step"])
            self.best = prior.get("best")

    def _stage(self, step: int) -> dict:
        cursor = 0
        for stage in self.recipe["schedule"]:
            cursor += int(stage["steps"])
            if step < cursor:
                return stage
        return self.recipe["schedule"][-1]

    def _sample_geometry(self, subject: FoldSubject, step: int):
        total = int(self.recipe["sampling"]["geometry_queries_per_subject_step"])
        counts = {
            "replay": total // 2,
            "near": total // 4,
            "straddle": total - total // 2 - total // 4,
        }
        chunks, targets = [], []
        for offset, (name, key) in enumerate((("replay", "replay_points"), ("near", "near_points"), ("straddle", "straddle_points"))):
            source = subject.geometry[key]
            ids = _stateless_indices(len(source), counts[name], self.seed + step * 1009 + offset * 97 + int(hashlib.sha256(subject.subject_id.encode()).hexdigest()[:8], 16))
            q = source[torch.from_numpy(ids)].to(self.device).float()
            chunks.append(q)
            if name == "replay":
                t = subject.geometry["replay_targets"][torch.from_numpy(ids)].to(self.device).float()
            else:
                t = field_targets(subject.field_teacher(), q.detach().cpu().numpy(), device=self.device.type, torch_authority=True)
                t = torch.from_numpy(t).to(self.device).float()
            targets.append(t)
        return torch.cat(chunks)[None], torch.cat(targets)[None]

    def _sample_presentation(self, subject: FoldSubject, family: str, step: int, train: bool):
        pool = subject.partitions[family][0 if train else 1]
        if not len(pool):
            return None
        budget_key = {"mechanics": "mechanical_queries_per_subject_step", "appearance": "appearance_queries_per_subject_step", "pair": "pair_queries_per_subject_step"}[family]
        budget = int(self.recipe["sampling"][budget_key]) if train else min(len(pool), max(64, int(self.recipe["sampling"][budget_key])))
        ids = pool[_stateless_indices(len(pool), budget, self.seed + step * 8191 + {"mechanics": 11, "appearance": 23, "pair": 37}[family])]
        return ids

    def _subject_loss(self, subject: FoldSubject, step: int, stage: dict, *, train: bool):
        final, layer2, alpha, cameras = subject.observation()
        admit_observations_v31(final, layer2, alpha, cameras)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(self.recipe["optimizer"]["bf16_autocast"])):
            carrier = self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras)
            q, target = self._sample_geometry(subject, step)
            gm, _ = self.model.query_states(carrier, q)
            pred_field = self.model.geometry(gm)[..., 0].float()
            geometry_loss = direct_fstar_loss_v5(pred_field, target, policy=DirectFStarFitPolicyV5(field_clamp=float(self.recipe["sampling"]["field_clamp"]), huber_beta=float(self.recipe["sampling"]["field_huber_beta"]))) ["total"]
            terms = {"geometry": geometry_loss}
            weights = stage["loss_weights"]
            for family in ("mechanics", "appearance", "pair"):
                ids = self._sample_presentation(subject, family, step, train)
                if ids is None:
                    continue
                if family == "mechanics":
                    xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][ids]).to(self.device)[None]
                    target_m = torch.from_numpy(subject.presentation["mechanical_target"][ids]).to(self.device)[None]
                    valid_m = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
                    gm_m, _ = self.model.query_states(carrier, xyz)
                    terms["mechanics"] = heteroscedastic_mechanical_loss_v1(self.model.mechanics(gm_m).float(), target_m, valid_m)
                elif family == "appearance":
                    xyz = torch.from_numpy(subject.presentation["appearance_xyz"][ids]).to(self.device)[None]
                    view = torch.from_numpy(subject.presentation["appearance_camera_index"][ids]).to(self.device).long()
                    target_a = torch.from_numpy(subject.presentation["appearance_rgba"][ids]).to(self.device)[None]
                    _, gp = self.model.query_states(carrier, xyz)
                    cam = cameras[:, view, :]
                    rgba = self.model.appearance(torch.cat((gp, cam), -1))[..., :4].sigmoid().float()
                    terms["appearance"] = appearance_rgba_loss_v1(rgba, target_a, torch.ones((1, len(ids)), device=self.device, dtype=torch.bool))
                else:
                    xa = torch.from_numpy(subject.presentation["pair_xyz_a"][ids]).to(self.device)[None]
                    xb = torch.from_numpy(subject.presentation["pair_xyz_b"][ids]).to(self.device)[None]
                    view = torch.from_numpy(subject.presentation["pair_camera_index"][ids]).to(self.device).long()
                    target_p = torch.from_numpy(subject.presentation["pair_target"][ids]).to(self.device)[None]
                    _, ga = self.model.query_states(carrier, xa)
                    _, gb = self.model.query_states(carrier, xb)
                    raw = self.model.pair(ga, gb, xa, xb, cameras[:, view, :]).float()
                    terms["occlusion_attenuation"] = pair_probe_loss_v1(raw, target_p, torch.ones((1, len(ids)), device=self.device, dtype=torch.bool))
            total = geometry_loss.new_zeros(())
            for name, weight in weights.items():
                if float(weight) > 0 and name in terms:
                    total = total + float(weight) * terms[name]
        return total.float(), {name: float(value.detach().cpu()) for name, value in terms.items()}

    @torch.no_grad()
    def _train_validation(self, step: int):
        self.model.eval()
        rows = []
        stage = self._stage(max(0, step - 1))
        for subject in self.train_subjects.values():
            geometry = subject.geometry
            count = min(int(self.recipe["evaluation"]["geometry_validation_points_per_train_subject"]), len(geometry["heldout_points"]))
            ids = np.arange(count, dtype=np.int64)
            final, layer2, alpha, cameras = subject.observation()
            carrier = self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras)
            q = geometry["heldout_points"][torch.from_numpy(ids)].to(self.device).float()[None]
            t = geometry["heldout_targets"][torch.from_numpy(ids)].to(self.device).float()[None]
            gm, _ = self.model.query_states(carrier, q)
            pred = self.model.geometry(gm)[..., 0].float()
            metrics = direct_fstar_metrics_v5(pred, t, near_zero_band=0.03)
            # Reuse exactly the preregistered presentation validation partition.
            presentation_score = 0.0
            presentation_terms = 0
            for family, weight_name in (("mechanics", "mechanics"), ("appearance", "appearance"), ("pair", "occlusion_attenuation")):
                if float(stage["loss_weights"].get(weight_name, 0.0)) <= 0 or not len(subject.partitions[family][1]):
                    continue
                loss, _ = self._subject_loss(subject, step + {"mechanics": 101, "appearance": 103, "pair": 107}[family], stage, train=False)
                # _subject_loss includes geometry; use only as stable train-only composite selector.
                presentation_score += float(loss.cpu())
                presentation_terms += 1
            score = float(metrics["mae"]) + (presentation_score / presentation_terms if presentation_terms else 0.0)
            rows.append({"subject_id": subject.subject_id, "score": score, "geometry": metrics})
        self.model.train()
        return {"step": step, "subjects": rows, "worst_subject_score": max(row["score"] for row in rows), "mean_subject_score": float(np.mean([row["score"] for row in rows]))}

    def _save_training_state(self):
        payload = {
            "binding_sha256": self.binding_hash,
            "step": self.step,
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best": self.best,
        }
        return self.store.save(payload)

    def train(self):
        total_steps = sum(int(row["steps"]) for row in self.recipe["schedule"])
        interval = int(self.recipe["evaluation"]["interval_steps"])
        examples = [SemanticExample(subject, 1.0, 1.0, 1) for subject in self.fold["train_subjects"]]
        started = time.monotonic()
        while self.step < total_steps:
            stage = self._stage(self.step)
            step_index = self.step
            def loss_sums(batch):
                values = []
                for example in batch:
                    loss, _ = self._subject_loss(self.train_subjects[example.key], step_index, stage, train=True)
                    values.append(loss)
                return torch.stack(values)
            execute_semantic_step(
                examples,
                loss_sums=loss_sums,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                parameters=self.model.parameters(),
                max_padded_cost=1,
                max_examples=1,
                clip_norm=float(self.recipe["optimizer"]["gradient_clip_norm"]),
            )
            self.step += 1
            if self.step % interval == 0 or self.step == total_steps:
                validation = self._train_validation(self.step)
                key = (validation["worst_subject_score"], validation["mean_subject_score"], self.step)
                if self.best is None or tuple(self.best["selection_key"]) > key:
                    self.best = {
                        "step": self.step,
                        "selection_key": list(key),
                        "validation": validation,
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                    }
                self._save_training_state()
                print(f"[IRIS_V31_LOFO] fold={self.fold['fold_id']} step={self.step}/{total_steps} stage={stage['id']} worst={validation['worst_subject_score']:.6f} mean={validation['mean_subject_score']:.6f}", flush=True)
        _require(self.best is not None, "IRIS_V31_LOFO_NO_CHECKPOINT_CANDIDATE")
        selected_hash = value_hash({"step": self.best["step"], "selection_key": self.best["selection_key"], "state": {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in self.best["model_state_dict"].items()}})
        # Only after this immutable selection identity exists may heldout truth be loaded.
        heldout = FoldSubject(self.heldout_row, self.device, self.recipe)
        self.model.load_state_dict(self.best["model_state_dict"], strict=True)
        self.model.eval()
        heldout_eval = self._evaluate_heldout(heldout)
        self.model.requires_grad_(False)
        checkpoint_path = write_checkpoint_v1(self.output / "selected_checkpoint.pt", self.model)
        result = {
            "schema": "RealSaS.IRISV31LofoFoldResult.v1",
            "fold_id": self.fold["fold_id"],
            "train_subjects": self.fold["train_subjects"],
            "heldout_subject": self.fold["heldout_subject"],
            "selected_step": self.best["step"],
            "selected_checkpoint_identity_before_heldout_open": selected_hash,
            "train_selection": self.best["validation"],
            "heldout": heldout_eval,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "wall_seconds": time.monotonic() - started,
            "heldout_used_for_gradient": False,
            "heldout_used_for_checkpoint_selection": False,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "status": "PASS_EXECUTED_LOFO_FOLD__SCIENTIFIC_QUALIFICATION_SEPARATE",
        }
        _atomic_json(self.output / "result.json", result)
        return result

    @torch.no_grad()
    def _evaluate_heldout(self, subject: FoldSubject):
        final, layer2, alpha, cameras = subject.observation()
        carrier = self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras)
        geometry = subject.geometry
        q = geometry["heldout_points"].to(self.device).float()[None]
        t = geometry["heldout_targets"].to(self.device).float()[None]
        gm, _ = self.model.query_states(carrier, q)
        metrics = direct_fstar_metrics_v5(self.model.geometry(gm)[..., 0].float(), t, near_zero_band=0.03)
        presentation = {}
        for family in ("mechanics", "appearance", "pair"):
            ids = np.flatnonzero(np.asarray(subject.presentation[{"mechanics": "mechanical_valid", "appearance": "appearance_valid", "pair": "pair_valid"}[family]], dtype=np.bool_))
            if not len(ids):
                presentation[family] = {"count": 0, "loss": None}
                continue
            stage = self.recipe["schedule"][-1]
            loss, terms = self._subject_loss(subject, self.step + 100000 + {"mechanics": 1, "appearance": 2, "pair": 3}[family], stage, train=False)
            presentation[family] = {"count": int(len(ids)), "composite_loss": float(loss.cpu()), "terms": terms}
        return {"geometry": metrics, "presentation": presentation}


def run_fold(*, input_manifest_path, recipe_path, fold_id, checkpoint_root, output_root, device="cuda"):
    input_manifest_path = Path(input_manifest_path).resolve()
    recipe_path = Path(recipe_path).resolve()
    manifest = json.loads(input_manifest_path.read_text())
    recipe = json.loads(recipe_path.read_text())
    trainer = IrisV31LofoTrainer(manifest=manifest, recipe=recipe, fold_id=fold_id, checkpoint_root=checkpoint_root, output_root=output_root, device=device)
    result = trainer.train()
    return result
