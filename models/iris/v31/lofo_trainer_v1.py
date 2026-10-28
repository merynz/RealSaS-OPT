"""Real FIT8 7->1 LOFO trainer for IRIS V3.1.

One global optimizer update is the equally weighted accumulation of exactly the
seven training subjects. Physical microbatching is one subject at a time. The
held-out subject is not instantiated until a train-only checkpoint identity has
been selected and frozen.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
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


def _indices(count: int, take: int, seed: int) -> np.ndarray:
    if count <= 0 or take <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    return rng.integers(0, count, size=min(int(take), count), endpoint=False, dtype=np.int64)


def presentation_partition_v1(subject: str, family: str, count: int, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    _require(0.0 < float(train_fraction) < 1.0, "IRIS_V31_LOFO_PRESENTATION_PARTITION_FRACTION")
    train, val = [], []
    for index in range(int(count)):
        token = f"{seed}|{subject}|{family}|{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(1 << 64)
        (train if bucket < float(train_fraction) else val).append(index)
    if count >= 2 and not val:
        val.append(train.pop())
    if count >= 2 and not train:
        train.append(val.pop())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class FoldSubject:
    def __init__(self, row: dict, device: torch.device, recipe: dict):
        self.subject_id = str(row["subject_id"])
        self.device = device
        geometry_path = Path(row["geometry_shard"]["path"])
        target_path = Path(row["presentation_targets"]["path"])
        feature_path = Path(row["frozen_feature_cache"]["features_path"])
        for path, expected, code in (
            (geometry_path, row["geometry_shard"]["sha256"], "GEOMETRY"),
            (target_path, row["presentation_targets"]["sha256"], "PRESENTATION"),
            (feature_path, row["frozen_feature_cache"]["features_sha256"], "FEATURE"),
        ):
            _require(path.is_file() and file_hash(path) == expected, f"IRIS_V31_LOFO_{code}_BYTES_DRIFT:{self.subject_id}")
        self.geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
        cache_payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        _require(isinstance(cache_payload, dict) and set(cache_payload) == {"identity", "features"}, "IRIS_V31_LOFO_FEATURE_CACHE_ENVELOPE")
        self.features = cache_payload["features"]
        _require(set(self.features) == {"final", "layer2", "alpha"}, "IRIS_V31_LOFO_FEATURE_FIELDS")
        with np.load(target_path, allow_pickle=False) as z:
            self.presentation = {name: np.asarray(z[name]).copy() for name in z.files}
        _require(self.geometry.get("subject_id") == self.subject_id, "IRIS_V31_LOFO_SUBJECT_GEOMETRY_DRIFT")
        self.cameras = self.geometry["camera_features"].float()
        admit_observations_v31(self.features["final"].float(), self.features["layer2"].float(), self.features["alpha"].float(), self.cameras)
        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        seed = int(recipe["sampling"]["seed"])
        self.partitions = {}
        for family, key in (("mechanics", "mechanical_valid"), ("appearance", "appearance_valid"), ("pair", "pair_valid")):
            valid = np.flatnonzero(np.asarray(self.presentation[key], dtype=np.bool_))
            train_local, val_local = presentation_partition_v1(self.subject_id, family, len(valid), seed, fraction)
            self.partitions[family] = (valid[train_local], valid[val_local])

    def observation(self):
        return (
            self.features["final"].to(self.device, non_blocking=True).float(),
            self.features["layer2"].to(self.device, non_blocking=True).float(),
            self.features["alpha"].to(self.device, non_blocking=True).float(),
            self.cameras.to(self.device, non_blocking=True).float(),
        )

    def field_teacher(self):
        keys = (
            "signed_2d", "center_xyz", "normalization_half_extent", "camera_origins",
            "camera_right", "camera_screen_up", "camera_half_extent", "resolution",
        )
        return {key: self.geometry[key] for key in keys}


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
        self.seed = int(recipe["sampling"]["seed"]) + int(self.fold["fold_index"]) * 1000003
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed & 0xFFFFFFFF)
        config_row = dict(recipe["model_config"])
        config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
        config_row["pair_channels"] = tuple(config_row["pair_channels"])
        self.config = IRISConfigV31(**config_row)
        self.config.validate()
        _require(self.config.mechanical_channels == MECHANICAL_CHANNELS_V31_PRESENTATION, "IRIS_V31_LOFO_MECHANICAL_CHANNEL_DRIFT")
        _require(self.config.pair_channels == PAIR_CHANNELS_V31_PRESENTATION, "IRIS_V31_LOFO_PAIR_CHANNEL_DRIFT")
        self.model = IRISV31(self.config).to(self.device).train()
        self.recipe, self.manifest = recipe, manifest
        rows = {row["subject_id"]: row for row in manifest["subjects"]}
        _require(set(self.fold["train_subjects"]) <= set(rows) and self.fold["heldout_subject"] in rows, "IRIS_V31_LOFO_FOLD_SUBJECT_DRIFT")
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
        opt = recipe["optimizer"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(opt["learning_rate"]), weight_decay=float(opt["weight_decay"]))
        total_steps = sum(int(row["steps"]) for row in recipe["schedule"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=float(opt["minimum_learning_rate"]))
        self.step, self.best = 0, None
        prior = self.store.load()
        if prior:
            self.model.load_state_dict(prior["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(prior["optimizer_state_dict"])
            self.scheduler.load_state_dict(prior["scheduler_state_dict"])
            self.step, self.best = int(prior["step"]), prior.get("best")

    def _stage(self, step: int) -> dict:
        cursor = 0
        for stage in self.recipe["schedule"]:
            cursor += int(stage["steps"])
            if step < cursor:
                return stage
        return self.recipe["schedule"][-1]

    def _geometry_batch(self, subject: FoldSubject, step: int):
        total = int(self.recipe["sampling"]["geometry_queries_per_subject_step"])
        counts = (total // 2, total // 4, total - total // 2 - total // 4)
        rows = (("replay_points", "replay_targets"), ("near_points", None), ("straddle_points", None))
        q_parts, t_parts = [], []
        subject_seed = int(hashlib.sha256(subject.subject_id.encode()).hexdigest()[:8], 16)
        for offset, ((point_key, target_key), count) in enumerate(zip(rows, counts)):
            points = subject.geometry[point_key]
            ids = _indices(len(points), count, self.seed + step * 1009 + offset * 97 + subject_seed)
            q = points[torch.from_numpy(ids)].to(self.device).float()
            q_parts.append(q)
            if target_key:
                target = subject.geometry[target_key][torch.from_numpy(ids)].to(self.device).float()
            else:
                target_np = field_targets(subject.field_teacher(), q.detach().cpu().numpy(), device=self.device.type, torch_authority=True)
                target = torch.from_numpy(target_np).to(self.device).float()
            t_parts.append(target)
        return torch.cat(q_parts)[None], torch.cat(t_parts)[None]

    def _presentation_ids(self, subject: FoldSubject, family: str, step: int, *, train: bool, all_rows: bool = False):
        if all_rows:
            key = {"mechanics": "mechanical_valid", "appearance": "appearance_valid", "pair": "pair_valid"}[family]
            pool = np.flatnonzero(np.asarray(subject.presentation[key], dtype=np.bool_))
        else:
            pool = subject.partitions[family][0 if train else 1]
        if not len(pool):
            return None
        if all_rows:
            return pool
        budget_key = {"mechanics": "mechanical_queries_per_subject_step", "appearance": "appearance_queries_per_subject_step", "pair": "pair_queries_per_subject_step"}[family]
        budget = int(self.recipe["sampling"][budget_key]) if train else min(len(pool), max(64, int(self.recipe["sampling"][budget_key])))
        return pool[_indices(len(pool), budget, self.seed + step * 8191 + {"mechanics": 11, "appearance": 23, "pair": 37}[family])]

    def _carrier(self, subject: FoldSubject):
        final, layer2, alpha, cameras = subject.observation()
        admit_observations_v31(final, layer2, alpha, cameras)
        return self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras), cameras

    def _presentation_loss(self, subject: FoldSubject, carrier, cameras, family: str, ids: np.ndarray):
        if family == "mechanics":
            xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            target = torch.from_numpy(subject.presentation["mechanical_target"][ids]).to(self.device)[None]
            gm, _ = self.model.query_states(carrier, xyz)
            return heteroscedastic_mechanical_loss_v1(self.model.mechanics(gm).float(), target, torch.ones((1, len(ids)), device=self.device, dtype=torch.bool))
        if family == "appearance":
            xyz = torch.from_numpy(subject.presentation["appearance_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            view = torch.from_numpy(subject.presentation["appearance_camera_index"][ids]).to(self.device).long()
            target = torch.from_numpy(subject.presentation["appearance_rgba"][ids]).to(self.device)[None]
            _, gp = self.model.query_states(carrier, xyz)
            cam = cameras[:, view, :].to(gp.dtype)
            rgba = self.model.appearance(torch.cat((gp, cam), -1))[..., :4].sigmoid().float()
            return appearance_rgba_loss_v1(rgba, target, torch.ones((1, len(ids)), device=self.device, dtype=torch.bool))
        xa = torch.from_numpy(subject.presentation["pair_xyz_a"][ids]).to(self.device)[None].to(carrier.dtype)
        xb = torch.from_numpy(subject.presentation["pair_xyz_b"][ids]).to(self.device)[None].to(carrier.dtype)
        view = torch.from_numpy(subject.presentation["pair_camera_index"][ids]).to(self.device).long()
        target = torch.from_numpy(subject.presentation["pair_target"][ids]).to(self.device)[None]
        _, ga = self.model.query_states(carrier, xa)
        _, gb = self.model.query_states(carrier, xb)
        raw = self.model.pair(ga, gb, xa, xb, cameras[:, view, :].to(ga.dtype)).float()
        return pair_probe_loss_v1(raw, target, torch.ones((1, len(ids)), device=self.device, dtype=torch.bool))

    def _subject_loss(self, subject: FoldSubject, step: int, stage: dict, *, train: bool):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(self.recipe["optimizer"]["bf16_autocast"])):
            carrier, cameras = self._carrier(subject)
            q, target = self._geometry_batch(subject, step)
            gm, _ = self.model.query_states(carrier, q.to(carrier.dtype))
            pred = self.model.geometry(gm)[..., 0].float()
            geometry = direct_fstar_loss_v5(pred, target, policy=DirectFStarFitPolicyV5(field_clamp=float(self.recipe["sampling"]["field_clamp"]), huber_beta=float(self.recipe["sampling"]["field_huber_beta"]))) ["total"]
            terms = {"geometry": geometry}
            for family, weight_name in (("mechanics", "mechanics"), ("appearance", "appearance"), ("pair", "occlusion_attenuation")):
                if float(stage["loss_weights"].get(weight_name, 0.0)) <= 0:
                    continue
                ids = self._presentation_ids(subject, family, step, train=train)
                if ids is not None:
                    terms[weight_name] = self._presentation_loss(subject, carrier, cameras, family, ids)
            total = geometry.new_zeros(())
            for name, weight in stage["loss_weights"].items():
                if float(weight) > 0 and name in terms:
                    total = total + float(weight) * terms[name]
        return total.float(), {name: float(value.detach().cpu()) for name, value in terms.items()}

    @torch.no_grad()
    def _evaluate_subject(self, subject: FoldSubject, stage: dict, *, all_presentation: bool) -> dict:
        self.model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(self.recipe["optimizer"]["bf16_autocast"])):
            carrier, cameras = self._carrier(subject)
            points, targets = subject.geometry["heldout_points"], subject.geometry["heldout_targets"]
            limit = len(points) if all_presentation else min(int(self.recipe["evaluation"]["geometry_validation_points_per_train_subject"]), len(points))
            q = points[:limit].to(self.device).float()[None]
            t = targets[:limit].to(self.device).float()[None]
            gm, _ = self.model.query_states(carrier, q.to(carrier.dtype))
            geometry = direct_fstar_metrics_v5(self.model.geometry(gm)[..., 0].float(), t, near_zero_band=0.03)
            losses = {}
            for family, weight_name in (("mechanics", "mechanics"), ("appearance", "appearance"), ("pair", "occlusion_attenuation")):
                if float(stage["loss_weights"].get(weight_name, 0.0)) <= 0:
                    continue
                ids = self._presentation_ids(subject, family, self.step, train=False, all_rows=all_presentation)
                if ids is not None:
                    losses[weight_name] = float(self._presentation_loss(subject, carrier, cameras, family, ids).float().cpu())
        score = float(geometry["mae"])
        for name, weight in stage["loss_weights"].items():
            if name != "geometry" and name in losses:
                score += float(weight) * losses[name]
        return {"subject_id": subject.subject_id, "score": score, "geometry": geometry, "presentation_losses": losses}

    @torch.no_grad()
    def _train_validation(self):
        stage = self._stage(max(0, self.step - 1))
        rows = [self._evaluate_subject(subject, stage, all_presentation=False) for subject in self.train_subjects.values()]
        self.model.train()
        return {
            "step": self.step,
            "subjects": rows,
            "worst_subject_score": max(row["score"] for row in rows),
            "mean_subject_score": float(np.mean([row["score"] for row in rows])),
        }

    def _save(self):
        return self.store.save({
            "binding_sha256": self.binding_hash,
            "step": self.step,
            "model_state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best": self.best,
        })

    def train(self):
        total_steps = sum(int(row["steps"]) for row in self.recipe["schedule"])
        interval = int(self.recipe["evaluation"]["interval_steps"])
        examples = [SemanticExample(subject, 1.0, 1.0, 1) for subject in self.fold["train_subjects"]]
        started = time.monotonic()
        while self.step < total_steps:
            stage, step_index = self._stage(self.step), self.step

            def loss_sums(batch):
                return torch.stack([self._subject_loss(self.train_subjects[row.key], step_index, stage, train=True)[0] for row in batch])

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
                validation = self._train_validation()
                key = (validation["worst_subject_score"], validation["mean_subject_score"], self.step)
                if self.best is None or tuple(self.best["selection_key"]) > key:
                    self.best = {
                        "step": self.step,
                        "selection_key": list(key),
                        "validation": validation,
                        "model_state_dict": {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()},
                    }
                self._save()
                print(f"[IRIS_V31_LOFO] fold={self.fold['fold_id']} step={self.step}/{total_steps} stage={stage['id']} worst={validation['worst_subject_score']:.6f} mean={validation['mean_subject_score']:.6f}", flush=True)

        _require(self.best is not None, "IRIS_V31_LOFO_NO_CHECKPOINT_CANDIDATE")
        state_hashes = {name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for name, tensor in self.best["model_state_dict"].items()}
        selected_identity = value_hash({"step": self.best["step"], "selection_key": self.best["selection_key"], "state_sha256": state_hashes})
        heldout = FoldSubject(self.heldout_row, self.device, self.recipe)
        self.model.load_state_dict(self.best["model_state_dict"], strict=True)
        self.model.eval()
        heldout_eval = self._evaluate_subject(heldout, self.recipe["schedule"][-1], all_presentation=True)
        self.model.requires_grad_(False)
        checkpoint_path = write_checkpoint_v1(self.output / "selected_checkpoint.pt", self.model)
        result = {
            "schema": "RealSaS.IRISV31LofoFoldResult.v1",
            "fold_id": self.fold["fold_id"],
            "train_subjects": self.fold["train_subjects"],
            "heldout_subject": self.fold["heldout_subject"],
            "selected_step": self.best["step"],
            "selected_checkpoint_identity_before_heldout_open": selected_identity,
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


def run_fold(*, input_manifest_path, recipe_path, fold_id, checkpoint_root, output_root, device="cuda"):
    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31LofoTrainer(
        manifest=manifest,
        recipe=recipe,
        fold_id=fold_id,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
