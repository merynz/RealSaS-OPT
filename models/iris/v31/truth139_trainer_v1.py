"""Truth139 shared dense-IRIS trainer with physical train/unseen separation.

Training chronology:
1. fit the dense geometry foundation on TRAIN subjects only using direct F*, near-zero sign
   stabilization and visible source-normal evidence;
2. byte-seal/freeze that foundation;
3. create a fresh optimizer and fit private mechanics/appearance adapters only.

The large cohort uses deterministic balanced subject epochs rather than accumulating every subject
into every optimizer update.  One physical microbatch is one subject; gradients are accumulated
across a fixed number of subjects before each optimizer step.  Every completed epoch exposes every
TRAIN subject exactly once, so no morphology receives hidden weighting.
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
from models.iris.v31.evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2
from models.iris.v31.fixed_snake_capacity_pilot_v1 import mechanics_target_log1p_v1
from models.iris.v31.geometry_supervision_v1 import truth139_geometry_objective_v1
from models.iris.v31.lofo_trainer_v1 import _indices, presentation_partition_v1
from models.iris.v31.mechanics_fixed_scale_student_t_v4 import (
    fixed_scale_student_t_mechanical_loss_v4,
    robust_fixed_scale_v4,
)
from models.iris.v31.model_v31 import admit_observations_v31
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
    appearance_rgba_loss_v1,
)
from models.iris.v31.fstar_teacher_v31 import exact_fstar_targets_v31
from models.iris.v31.truth139_model_v1 import IRISV31Truth139V1


TRAIN_MANIFEST_SCHEMA = "RealSaS.IRISV31Truth139TrainManifest.v1"
RECIPE_SCHEMA = "RealSaS.IRISV31Truth139TrainingRecipe.v1"
RESULT_SCHEMA = "RealSaS.IRISV31Truth139TrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.IRISV31Truth139Checkpoint.v1"
FOUNDATION_STAGE = "GEOMETRY_FOUNDATION"
EVIDENCE_STAGE = "EVIDENCE_ADAPTER_ACQUISITION"


def _require(value, code: str) -> None:
    if not value:
        raise ValueError(code)


def _tensor_hash(parameter: torch.Tensor) -> str:
    value = parameter.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _normal_partition(subject: str, count: int, seed: int, fraction: float):
    return presentation_partition_v1(subject, "surface_normal", count, seed, fraction)


class Truth139SubjectV1:
    def __init__(self, row: dict, *, device: torch.device, recipe: dict):
        self.subject_id = str(row["subject_id"])
        self.device = device
        for role in (
            "geometry_shard",
            "presentation_targets",
            "surface_normal_truth",
            "frozen_feature_cache",
        ):
            ref = dict(row.get(role) or {})
            path = Path(str(ref.get("path") or "")).resolve()
            _require(path.is_file() and file_hash(path) == ref.get("sha256"),
                     "TRUTH139_SUBJECT_REF_DRIFT:" + self.subject_id + ":" + role)

        self.geometry = torch.load(
            Path(row["geometry_shard"]["path"]), map_location="cpu", weights_only=True
        )
        _require(self.geometry.get("subject_id") == self.subject_id,
                 "TRUTH139_GEOMETRY_SUBJECT_DRIFT:" + self.subject_id)
        cache = torch.load(
            Path(row["frozen_feature_cache"]["path"]), map_location="cpu", weights_only=True
        )
        _require(isinstance(cache, dict) and set(cache) == {"identity", "features"},
                 "TRUTH139_FEATURE_CACHE_ENVELOPE")
        self.features = cache["features"]
        _require(set(self.features) == {"final", "layer2", "alpha"},
                 "TRUTH139_FEATURE_FIELDS")
        self.cameras = self.geometry["camera_features"].float()
        admit_observations_v31(
            self.features["final"].float(),
            self.features["layer2"].float(),
            self.features["alpha"].float(),
            self.cameras,
        )

        with np.load(Path(row["presentation_targets"]["path"]), allow_pickle=False) as z:
            self.presentation = {name: np.asarray(z[name]).copy() for name in z.files}
        with np.load(Path(row["surface_normal_truth"]["path"]), allow_pickle=False) as z:
            self.normal = {name: np.asarray(z[name]).copy() for name in z.files}
        required_normal = {"xyz", "normal_world", "view_index", "face_index", "valid"}
        _require(required_normal <= set(self.normal), "TRUTH139_NORMAL_TRUTH_FIELDS")
        count = len(self.normal["xyz"])
        _require(
            self.normal["xyz"].shape == (count, 3)
            and self.normal["normal_world"].shape == (count, 3)
            and self.normal["valid"].shape == (count,),
            "TRUTH139_NORMAL_TRUTH_LAYOUT",
        )

        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        seed = int(recipe["sampling"]["seed"])
        self.partitions = {}
        for family, key in (("mechanics", "mechanical_valid"), ("appearance", "appearance_valid")):
            valid = np.flatnonzero(np.asarray(self.presentation[key], dtype=np.bool_))
            local_train, local_val = presentation_partition_v1(
                self.subject_id, family, len(valid), seed, fraction
            )
            self.partitions[family] = (valid[local_train], valid[local_val])
        normal_valid = np.flatnonzero(np.asarray(self.normal["valid"], dtype=np.bool_))
        ntrain, nval = _normal_partition(self.subject_id, len(normal_valid), seed, fraction)
        self.partitions["surface_normal"] = (normal_valid[ntrain], normal_valid[nval])

    def observation(self):
        return (
            self.features["final"].to(self.device, non_blocking=True).float(),
            self.features["layer2"].to(self.device, non_blocking=True).float(),
            self.features["alpha"].to(self.device, non_blocking=True).float(),
            self.cameras.to(self.device, non_blocking=True).float(),
        )

    def field_teacher(self) -> dict:
        keys = (
            "signed_2d",
            "center_xyz",
            "normalization_half_extent",
            "camera_origins",
            "camera_right",
            "camera_screen_up",
            "camera_half_extent",
            "resolution",
        )
        return {key: self.geometry[key] for key in keys}


class Truth139TrainerV1:
    def __init__(
        self,
        *,
        train_manifest: dict,
        recipe: dict,
        checkpoint_root,
        output_root,
        device: str = "cuda",
    ) -> None:
        _require(train_manifest.get("schema") == TRAIN_MANIFEST_SCHEMA, "TRUTH139_TRAIN_MANIFEST_SCHEMA")
        _require(recipe.get("schema") == RECIPE_SCHEMA, "TRUTH139_RECIPE_SCHEMA")
        _require(train_manifest.get("contains_unseen_subject_ids") is False, "TRUTH139_TRAIN_UNSEEN_IDS_PRESENT")
        _require(train_manifest.get("contains_unseen_file_refs") is False, "TRUTH139_TRAIN_UNSEEN_REFS_PRESENT")
        _require(train_manifest.get("optimizer_scope") == "TRAIN_ONLY", "TRUTH139_TRAIN_SCOPE")
        rows = list(train_manifest.get("subjects") or ())
        ids = [str(row.get("subject_id") or "") for row in rows]
        _require(ids and len(ids) == len(set(ids)) == int(train_manifest["subject_count"]),
                 "TRUTH139_TRAIN_SUBJECT_CARDINALITY")
        _require(int(recipe["train_subject_count"]) == len(ids), "TRUTH139_RECIPE_SUBJECT_COUNT_DRIFT")
        _require(recipe.get("split_identity_sha256") == train_manifest.get("split_identity_sha256"),
                 "TRUTH139_RECIPE_SPLIT_DRIFT")
        _require(recipe.get("selection_rule") == "FIXED_FINAL_STAGE_ENDPOINT__NO_CHECKPOINT_SHOPPING",
                 "TRUTH139_SELECTION_RULE")

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "TRUTH139_CUDA_REQUIRED")
        runtime = recipe["runtime"]
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False

        self.recipe = recipe
        self.manifest = train_manifest
        self.ids = ids
        self.seed = int(recipe["sampling"]["seed"])
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed & 0xFFFFFFFF)

        config_row = dict(recipe["model_config"])
        config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
        config_row["pair_channels"] = tuple(config_row["pair_channels"])
        self.config = IRISEvidenceAdapterConfigV2(**config_row)
        self.config.validate()
        _require(self.config.mechanical_channels == MECHANICAL_CHANNELS_V31_PRESENTATION,
                 "TRUTH139_MECHANICAL_CHANNEL_DRIFT")
        _require(self.config.pair_channels == PAIR_CHANNELS_V31_PRESENTATION,
                 "TRUTH139_PAIR_CHANNEL_DRIFT")
        self.model = IRISV31Truth139V1(self.config).to(self.device).train()
        self.subjects = {
            row["subject_id"]: Truth139SubjectV1(row, device=self.device, recipe=recipe)
            for row in rows
        }
        self.fixed_mechanics_scale = self._compute_fixed_mechanics_scale()

        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema": "RealSaS.IRISV31Truth139TrainingBinding.v1",
            "train_manifest_identity_sha256": train_manifest["manifest_identity_sha256"],
            "recipe_sha256": value_hash(recipe),
            "split_identity_sha256": train_manifest["split_identity_sha256"],
            "subject_ids": ids,
            "unseen_subject_set_commitment_sha256": train_manifest["unseen_subject_set_commitment_sha256"],
            "architecture_id": self.model.architecture_id,
            "config": asdict(self.config),
            "fixed_mechanics_scale": list(self.fixed_mechanics_scale),
            "optimizer_update_semantics": recipe["optimizer_update_semantics"],
        }
        self.binding = binding
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root), self.binding_hash)

        self.stage_index = 0
        self.completed_epoch = 0
        self.global_optimizer_step = 0
        self.foundation_identity = None
        self.exposure_counts = {
            stage["id"]: {subject_id: 0 for subject_id in self.ids}
            for stage in recipe["schedule"]
        }
        self.optimizer = None
        self.scheduler = None
        previous = self.store.load()
        if previous:
            self.model.load_state_dict(previous["model_state_dict"], strict=True)
            self.stage_index = int(previous["stage_index"])
            self.completed_epoch = int(previous["completed_epoch"])
            self.global_optimizer_step = int(previous["step"])
            self.foundation_identity = previous.get("foundation_identity")
            self.exposure_counts = previous["exposure_counts"]
        self._configure_stage()
        self._build_optimizer()
        if previous and int(previous["stage_index"]) == self.stage_index:
            if previous.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(previous["optimizer_state_dict"])
                self.scheduler.load_state_dict(previous["scheduler_state_dict"])

    @property
    def stage(self) -> dict:
        return self.recipe["schedule"][self.stage_index]

    def _foundation_identity(self) -> str:
        return value_hash({name: _tensor_hash(parameter) for name, parameter in self.model.foundation_named_parameters()})

    def _compute_fixed_mechanics_scale(self) -> tuple[float, float]:
        targets, masks = [], []
        for subject in self.subjects.values():
            ids = subject.partitions["mechanics"][0]
            if not len(ids):
                continue
            value = torch.from_numpy(subject.presentation["mechanical_target"][ids]).float()
            transformed = mechanics_target_log1p_v1(value)
            targets.append(transformed)
            masks.append(torch.ones(len(transformed), dtype=torch.bool))
        _require(targets, "TRUTH139_MECHANICS_SCALE_NO_TRAIN_ROWS")
        target = torch.cat(targets, dim=0)
        valid = torch.cat(masks, dim=0)
        return robust_fixed_scale_v4(
            target,
            valid,
            minimum_scale=float(self.recipe["mechanics"]["minimum_fixed_scale"]),
        )

    def _configure_stage(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.stage["id"] == FOUNDATION_STAGE:
            for _, parameter in self.model.foundation_named_parameters():
                parameter.requires_grad_(True)
        elif self.stage["id"] == EVIDENCE_STAGE:
            _require(self.foundation_identity is not None, "TRUTH139_FOUNDATION_IDENTITY_REQUIRED")
            _require(self._foundation_identity() == self.foundation_identity, "TRUTH139_FOUNDATION_DRIFT")
            for _, parameter in self.model.evidence_named_parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError("TRUTH139_STAGE_UNKNOWN:" + str(self.stage["id"]))

    def _stage_optimizer_updates(self) -> int:
        per_step = int(self.recipe["optimizer"]["subjects_per_optimizer_step"])
        return int(self.stage["epochs"]) * math.ceil(len(self.ids) / per_step)

    def _build_optimizer(self) -> None:
        opt = self.stage["optimizer"]
        parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
        _require(parameters, "TRUTH139_EMPTY_TRAINABLE_SET")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self._stage_optimizer_updates()),
            eta_min=float(opt["minimum_learning_rate"]),
        )

    def _subject_order(self, epoch: int) -> list[str]:
        stage_id = self.stage["id"]
        return sorted(
            self.ids,
            key=lambda subject: hashlib.sha256(
                f"{self.seed}|{stage_id}|{int(epoch)}|{subject}".encode()
            ).hexdigest(),
        )

    def _carrier(self, subject: Truth139SubjectV1):
        final, layer2, alpha, cameras = subject.observation()
        return self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras), cameras

    def _geometry_batch(self, subject: Truth139SubjectV1, epoch: int):
        total = int(self.recipe["sampling"]["geometry_queries_per_subject_exposure"])
        counts = (total // 2, total // 4, total - total // 2 - total // 4)
        rows = (("replay_points", "replay_targets"), ("near_points", None), ("straddle_points", None))
        subject_seed = int(hashlib.sha256(subject.subject_id.encode()).hexdigest()[:8], 16)
        q_parts, t_parts = [], []
        for offset, ((point_key, target_key), count) in enumerate(zip(rows, counts)):
            points = subject.geometry[point_key]
            ids = _indices(len(points), count, self.seed + epoch * 1009 + offset * 97 + subject_seed)
            q = points[torch.from_numpy(ids)].to(self.device).float()
            q_parts.append(q)
            if target_key:
                target = subject.geometry[target_key][torch.from_numpy(ids)].to(self.device).float()
            else:
                target_np = exact_fstar_targets_v31(
                    subject.field_teacher(), q.detach().cpu().numpy(), device=self.device.type
                )
                target = torch.from_numpy(target_np).to(self.device).float()
            t_parts.append(target)
        return torch.cat(q_parts)[None], torch.cat(t_parts)[None]

    def _normal_batch(self, subject: Truth139SubjectV1, epoch: int):
        pool = subject.partitions["surface_normal"][0]
        _require(len(pool), "TRUTH139_NORMAL_TRAIN_PARTITION_EMPTY:" + subject.subject_id)
        budget = int(self.recipe["sampling"]["normal_queries_per_subject_exposure"])
        subject_seed = int(hashlib.sha256((subject.subject_id + "|normal").encode()).hexdigest()[:8], 16)
        pick = pool[_indices(len(pool), budget, self.seed + epoch * 4099 + subject_seed)]
        xyz = torch.from_numpy(subject.normal["xyz"][pick]).to(self.device)[None].float()
        target = torch.from_numpy(subject.normal["normal_world"][pick]).to(self.device)[None].float()
        valid = torch.ones((1, len(pick)), device=self.device, dtype=torch.bool)
        return xyz, target, valid

    def _presentation_ids(self, subject: Truth139SubjectV1, family: str, epoch: int):
        pool = subject.partitions[family][0]
        if not len(pool):
            return None
        budget_key = {
            "mechanics": "mechanical_queries_per_subject_exposure",
            "appearance": "appearance_queries_per_subject_exposure",
        }[family]
        budget = int(self.recipe["sampling"][budget_key])
        salt = 8191 if family == "mechanics" else 12289
        subject_seed = int(hashlib.sha256((subject.subject_id + "|" + family).encode()).hexdigest()[:8], 16)
        return pool[_indices(len(pool), budget, self.seed + epoch * salt + subject_seed)]

    def _geometry_loss(self, subject: Truth139SubjectV1, carrier, epoch: int):
        q, target = self._geometry_batch(subject, epoch)
        gg, _ = self.model.query_states(carrier, q.to(carrier.dtype))
        predicted_field = self.model.geometry(gg)[..., 0].float()
        normal_xyz, normal_target, normal_valid = self._normal_batch(subject, epoch)
        predicted_normal = self.model.query_surface_normal(carrier, normal_xyz.to(carrier.dtype)).float()
        policy = self.recipe["geometry"]
        return truth139_geometry_objective_v1(
            predicted_field=predicted_field,
            target_fstar=target,
            predicted_normal=predicted_normal,
            target_normal=normal_target,
            normal_valid=normal_valid,
            field_clamp=float(self.recipe["sampling"]["field_clamp"]),
            huber_beta=float(self.recipe["sampling"]["field_huber_beta"]),
            near_zero_band=float(policy["near_zero_band"]),
            sign_weight=float(policy["sign_weight"]),
            normal_weight=float(policy["normal_weight"]),
        )

    def _evidence_loss(self, subject: Truth139SubjectV1, carrier, cameras, epoch: int):
        stage = self.stage
        total = carrier.new_zeros(())
        terms = {}
        mechanics_ids = self._presentation_ids(subject, "mechanics", epoch)
        if mechanics_ids is not None:
            xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][mechanics_ids]).to(self.device)[None].to(carrier.dtype)
            target = torch.from_numpy(subject.presentation["mechanical_target"][mechanics_ids]).to(self.device)[None].float()
            transformed = mechanics_target_log1p_v1(target)
            state = self.model.query_mechanics_state(carrier, xyz)
            raw = self.model.mechanics(state).float()
            valid = torch.ones((1, len(mechanics_ids)), device=self.device, dtype=torch.bool)
            mechanics = fixed_scale_student_t_mechanical_loss_v4(
                raw,
                transformed,
                valid,
                fixed_scale=self.fixed_mechanics_scale,
                degrees_of_freedom=float(self.recipe["mechanics"]["degrees_of_freedom"]),
            )
            terms["mechanics"] = mechanics
            total = total + float(stage["loss_weights"]["mechanics"]) * mechanics

        appearance_ids = self._presentation_ids(subject, "appearance", epoch)
        if appearance_ids is not None:
            xyz = torch.from_numpy(subject.presentation["appearance_xyz"][appearance_ids]).to(self.device)[None].to(carrier.dtype)
            view = torch.from_numpy(subject.presentation["appearance_camera_index"][appearance_ids]).to(self.device).long()
            target = torch.from_numpy(subject.presentation["appearance_rgba"][appearance_ids]).to(self.device)[None].float()
            _, gp = self.model.query_states(carrier, xyz)
            cam = cameras[:, view, :].to(gp.dtype)
            rgba = self.model.appearance(torch.cat((gp, cam), dim=-1))[..., :4].sigmoid().float()
            valid = torch.ones((1, len(appearance_ids)), device=self.device, dtype=torch.bool)
            appearance = appearance_rgba_loss_v1(rgba, target, valid)
            terms["appearance"] = appearance
            total = total + float(stage["loss_weights"]["appearance"]) * appearance
        _require(terms, "TRUTH139_EVIDENCE_NO_VALID_ROWS:" + subject.subject_id)
        return {"total": total, **terms}

    def _subject_loss(self, subject: Truth139SubjectV1, epoch: int):
        autocast = bool(self.recipe["runtime"]["bf16_autocast"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            carrier, cameras = self._carrier(subject)
            if self.stage["id"] == FOUNDATION_STAGE:
                report = self._geometry_loss(subject, carrier, epoch)
            else:
                report = self._evidence_loss(subject, carrier, cameras, epoch)
        return report

    def _save_resume(self) -> None:
        self.store.save({
            "step": int(self.global_optimizer_step),
            "stage_index": int(self.stage_index),
            "completed_epoch": int(self.completed_epoch),
            "foundation_identity": self.foundation_identity,
            "model_state_dict": {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "exposure_counts": self.exposure_counts,
            "binding_sha256": self.binding_hash,
        })

    def _seal_foundation(self) -> None:
        identity = self._foundation_identity()
        if self.foundation_identity is not None:
            _require(identity == self.foundation_identity, "TRUTH139_FOUNDATION_RESEAL_DRIFT")
            return
        self.foundation_identity = identity
        payload = {
            "schema": "RealSaS.IRISV31Truth139FoundationSeal.v1",
            "architecture_id": self.model.architecture_id,
            "foundation_identity": identity,
            "parameters": {name: value.detach().cpu().clone() for name, value in self.model.foundation_named_parameters()},
            "train_manifest_identity_sha256": self.manifest["manifest_identity_sha256"],
            "split_identity_sha256": self.manifest["split_identity_sha256"],
        }
        path = self.output / "geometry_foundation.pt"
        temp = path.with_suffix(".pt.partial")
        torch.save(payload, temp)
        torch.load(temp, map_location="cpu", weights_only=True)
        temp.replace(path)
        _atomic_json(self.output / "geometry_foundation_receipt.json", {
            "schema": "RealSaS.IRISV31Truth139FoundationReceipt.v1",
            "foundation_identity": identity,
            "checkpoint_sha256": file_hash(path),
            "train_subject_count": len(self.ids),
            "unseen_opened": False,
        })

    def _advance_stage(self) -> bool:
        if self.completed_epoch < int(self.stage["epochs"]):
            return False
        if self.stage["id"] == FOUNDATION_STAGE:
            self._seal_foundation()
        if self.stage_index + 1 >= len(self.recipe["schedule"]):
            return False
        self.stage_index += 1
        self.completed_epoch = 0
        self._configure_stage()
        self._build_optimizer()
        self._save_resume()
        return True

    def train(self) -> dict:
        started = time.monotonic()
        batch_size = int(self.recipe["optimizer"]["subjects_per_optimizer_step"])
        _require(batch_size >= 1, "TRUTH139_SUBJECT_BATCH_INVALID")
        checkpoint_interval = int(self.recipe["checkpoint"]["interval_epochs"])
        _require(checkpoint_interval >= 1, "TRUTH139_CHECKPOINT_INTERVAL_INVALID")

        while True:
            if self.completed_epoch >= int(self.stage["epochs"]):
                if not self._advance_stage():
                    break
                continue
            epoch = self.completed_epoch
            order = self._subject_order(epoch)
            epoch_terms: dict[str, list[float]] = {}
            for start in range(0, len(order), batch_size):
                group = order[start : start + batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                for subject_id in group:
                    report = self._subject_loss(self.subjects[subject_id], epoch)
                    (report["total"] / len(group)).backward()
                    self.exposure_counts[self.stage["id"]][subject_id] += 1
                    for key, value in report.items():
                        epoch_terms.setdefault(key, []).append(float(value.detach().cpu()))
                parameters = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=float(self.stage["optimizer"]["gradient_clip_norm"])
                )
                self.optimizer.step()
                self.scheduler.step()
                self.global_optimizer_step += 1
            self.completed_epoch += 1
            means = {key: float(np.mean(values)) for key, values in epoch_terms.items()}
            print(json.dumps({
                "event": "TRUTH139_EPOCH",
                "stage": self.stage["id"],
                "epoch": self.completed_epoch,
                "epochs": int(self.stage["epochs"]),
                "optimizer_step": self.global_optimizer_step,
                "train_subject_count": len(self.ids),
                "loss": means,
                "foundation": "SEALED" if self.foundation_identity else "TRAINING",
                "unseen": "NOT_MOUNTED",
            }, sort_keys=True), flush=True)
            if self.completed_epoch % checkpoint_interval == 0 or self.completed_epoch == int(self.stage["epochs"]):
                self._save_resume()

        _require(self.foundation_identity is not None, "TRUTH139_FOUNDATION_NOT_SEALED_AT_END")
        _require(self._foundation_identity() == self.foundation_identity, "TRUTH139_FOUNDATION_CHANGED_DURING_EVIDENCE")
        for stage in self.recipe["schedule"]:
            counts = self.exposure_counts[stage["id"]]
            expected = int(stage["epochs"])
            _require(set(counts) == set(self.ids) and all(int(value) == expected for value in counts.values()),
                     "TRUTH139_SUBJECT_EXPOSURE_IMBALANCE:" + stage["id"])

        checkpoint = self.output / "selected_final_checkpoint.pt"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": self.model.architecture_id,
            "config": asdict(self.config),
            "foundation_identity": self.foundation_identity,
            "fixed_mechanics_scale": list(self.fixed_mechanics_scale),
            "model_state_dict": {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
        }
        temp = checkpoint.with_suffix(".pt.partial")
        torch.save(payload, temp)
        torch.load(temp, map_location="cpu", weights_only=True)
        temp.replace(checkpoint)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "TRAINING_COMPLETE__UNSEEN_NOT_OPENED__EVALUATION_PENDING",
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "split_identity_sha256": self.manifest["split_identity_sha256"],
            "train_manifest_identity_sha256": self.manifest["manifest_identity_sha256"],
            "unseen_subject_set_commitment_sha256": self.manifest["unseen_subject_set_commitment_sha256"],
            "train_subject_count": len(self.ids),
            "foundation_identity": self.foundation_identity,
            "fixed_mechanics_scale": list(self.fixed_mechanics_scale),
            "subject_exposures": self.exposure_counts,
            "actual_optimizer_steps": self.global_optimizer_step,
            "selected_checkpoint_sha256": file_hash(checkpoint),
            "selection_rule": self.recipe["selection_rule"],
            "heldout_or_unseen_used_for_gradient": False,
            "heldout_or_unseen_used_for_checkpoint_selection": False,
            "unseen_opened": False,
            "generalization_claimed": False,
            "product_pass_claimed": False,
            "wall_seconds": time.monotonic() - started,
        }
        _atomic_json(self.output / "TRAINING_RESULT.json", result)
        return result
