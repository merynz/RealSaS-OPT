"""Fixed-Snake A100 capacity pilot for IRIS V3.1 evidence learning.

This pilot keeps the sealed FIT8 inputs, seven-subject equal-weight update,
held-out firewall, exact F* teacher, pair-family ABSTAIN, and source-equivalent
appearance supervision. It changes only:
1) a preregistered larger research capacity,
2) mechanics NLL parameterization to log1p(nonnegative magnitude),
3) execution scope to the already-existing Snake-held-out LOFO_00 fold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.iris.training.cohort_v1 import AtomicCheckpointStore, file_hash, value_hash
from models.iris.v31.lofo_trainer_v1 import FoldSubject, _atomic_json, _require
from models.iris.v31.lofo_trainer_v2 import pair_group_partition_v2
from models.iris.v31.lofo_trainer_v3 import IrisV31LofoTrainerV3
from models.iris.v31.model_v31 import IRISV31
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
    heteroscedastic_mechanical_loss_v1,
)
from models.training_runtime.semantic_batch_v1 import SemanticExample, execute_semantic_step

TRAINER_REVISION = "IRIS_V31_FIXED_SNAKE_CAPACITY_PILOT_V1"
ARCHITECTURE_ID = "RealSaS.IRIS.DenseObservationEvidence.v3_1.capacity_pilot_v1"
RECIPE_SCHEMA = "RealSaS.IRISV31FixedSnakeCapacityPilotRecipe.v1"
CHECKPOINT_SCHEMA = "RealSaS.IRISDenseObservationEvidenceCheckpoint.v3_1.capacity_pilot_v1"
FIXED_FOLD_ID = "LOFO_00_fit8_snake_v1"
FIXED_HELDOUT = "fit8_snake_v1"
MECHANICS_TARGET_TRANSFORM = "LOG1P_NONNEGATIVE_MAGNITUDES"


@dataclass(frozen=True)
class IRISCapacityPilotConfigV1:
    carrier_resolution: int
    carrier_channels: int
    mechanical_channels: tuple[str, ...]
    pair_channels: tuple[str, ...]
    query_width: int
    query_layers: int
    fourier_bands: int
    private_bottleneck: int
    lift_chunk: int
    delta_bound: float

    def validate(self) -> None:
        if min(
            self.carrier_resolution,
            self.carrier_channels,
            self.query_width,
            self.query_layers,
            self.fourier_bands,
            self.private_bottleneck,
            self.lift_chunk,
        ) <= 0:
            raise ValueError("IRIS_CAPACITY_PILOT_DIMENSION_INVALID")
        if self.query_layers != 3 or self.fourier_bands != 4:
            raise ValueError("IRIS_CAPACITY_PILOT_STRUCTURAL_DRIFT")
        if not np.isfinite(self.delta_bound) or self.delta_bound <= 0:
            raise ValueError("IRIS_CAPACITY_PILOT_DELTA_BOUND_INVALID")
        if tuple(self.mechanical_channels) != MECHANICAL_CHANNELS_V31_PRESENTATION:
            raise ValueError("IRIS_CAPACITY_PILOT_MECHANICAL_CHANNEL_DRIFT")
        if tuple(self.pair_channels) != PAIR_CHANNELS_V31_PRESENTATION:
            raise ValueError("IRIS_CAPACITY_PILOT_PAIR_CHANNEL_DRIFT")


def mechanics_target_log1p_v1(target: torch.Tensor) -> torch.Tensor:
    """Invertible robust coordinate for nonnegative mechanics magnitudes."""
    if target.shape[-1] != 2 or not bool(torch.isfinite(target).all()):
        raise ValueError("IRIS_CAPACITY_PILOT_MECHANICS_TARGET_INVALID")
    if bool((target < 0).any()):
        raise ValueError("IRIS_CAPACITY_PILOT_MECHANICS_TARGET_NEGATIVE")
    return torch.log1p(target)


def _pair_abstain_contract(recipe: dict, manifest: dict) -> None:
    qualification = dict(manifest.get("student_projection_qualification") or {})
    _require(
        manifest.get("student_training_authorized") is True
        and manifest.get("status") == "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT",
        "IRIS_CAPACITY_PILOT_INPUT_NOT_AUTHORIZED",
    )
    _require(
        qualification.get("schema") == "RealSaS.IRISV31StudentProjectionQualification.v1"
        and qualification.get("status") == "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED"
        and qualification.get("teacher_inputs_in_forward") is False
        and qualification.get("semantic_part_taxonomy_used") is False
        and qualification.get("pair_training_authorized") is False
        and int(qualification.get("pair_evidence_subject_count", -1)) == 0
        and int(qualification.get("pair_evidence_valid_rows", -1)) == 0,
        "IRIS_CAPACITY_PILOT_PROJECTION_DRIFT",
    )
    _require(
        int(recipe["sampling"]["pair_queries_per_subject_step"]) == 0
        and all(float(row["loss_weights"].get("occlusion_attenuation", 0.0)) == 0.0 for row in recipe["schedule"])
        and all("OCCLUSION_ATTENUATION" not in set(row.get("families") or ()) for row in recipe["schedule"]),
        "IRIS_CAPACITY_PILOT_PAIR_ABSTAIN_RECIPE_REQUIRED",
    )


def _write_checkpoint(path: Path, model: IRISV31, config: IRISCapacityPilotConfigV1) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "config": asdict(config),
        "model_state_dict": state,
    }
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    torch.load(tmp, map_location="cpu", weights_only=True)
    tmp.replace(target)
    return target


class IrisV31FixedSnakeCapacityPilotTrainer(IrisV31LofoTrainerV3):
    def __init__(self, *, manifest: dict, recipe: dict, fold_id: str, checkpoint_root, output_root, device="cuda"):
        _require(recipe.get("schema") == RECIPE_SCHEMA, "IRIS_CAPACITY_PILOT_RECIPE_SCHEMA")
        _require(recipe.get("trainer_revision") == TRAINER_REVISION, "IRIS_CAPACITY_PILOT_TRAINER_REVISION")
        pilot = dict(recipe.get("pilot") or {})
        _require(
            fold_id == FIXED_FOLD_ID
            and pilot.get("fixed_fold_id") == FIXED_FOLD_ID
            and pilot.get("fixed_heldout_subject") == FIXED_HELDOUT
            and pilot.get("lofo_expansion_authorized") is False,
            "IRIS_CAPACITY_PILOT_FIXED_HELDOUT_DRIFT",
        )
        _require(
            recipe.get("mechanics_target_transform") == MECHANICS_TARGET_TRANSFORM,
            "IRIS_CAPACITY_PILOT_MECHANICS_TRANSFORM_DRIFT",
        )
        _pair_abstain_contract(recipe, manifest)

        _require(manifest.get("schema") == "RealSaS.IRISV31LofoInputManifest.v1", "IRIS_CAPACITY_PILOT_INPUT_SCHEMA")
        folds = [row for row in manifest["folds"] if row["fold_id"] == fold_id]
        _require(len(folds) == 1, "IRIS_CAPACITY_PILOT_FOLD_ID")
        self.fold = folds[0]
        _require(
            self.fold["heldout_subject"] == FIXED_HELDOUT
            and self.fold.get("checkpoint_selection_may_read_heldout") is False
            and self.fold.get("heldout_gradient_updates") is False
            and len(self.fold["train_subjects"]) == 7,
            "IRIS_CAPACITY_PILOT_HELDOUT_FIREWALL",
        )

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "IRIS_CAPACITY_PILOT_CUDA_REQUIRED")
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
        self.config = IRISCapacityPilotConfigV1(**config_row)
        self.config.validate()
        self.model = IRISV31(self.config).to(self.device).train()

        self.recipe, self.manifest = recipe, manifest
        rows = {row["subject_id"]: row for row in manifest["subjects"]}
        _require(set(self.fold["train_subjects"]) <= set(rows) and FIXED_HELDOUT in rows, "IRIS_CAPACITY_PILOT_SUBJECT_DRIFT")
        self.train_subjects = {
            subject: FoldSubject(rows[subject], self.device, recipe)
            for subject in self.fold["train_subjects"]
        }
        self.heldout_row = rows[FIXED_HELDOUT]

        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        partition_seed = int(recipe["sampling"]["seed"])
        for subject in self.train_subjects.values():
            valid = np.flatnonzero(np.asarray(subject.presentation["pair_valid"], dtype=np.bool_))
            subject.partitions["pair"] = pair_group_partition_v2(
                subject.subject_id, valid, partition_seed, fraction
            )

        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema": "RealSaS.IRISV31FixedSnakeCapacityPilotBinding.v1",
            "input_manifest_sha256": value_hash(manifest),
            "recipe_sha256": value_hash(recipe),
            "fold": self.fold,
            "config": asdict(self.config),
            "architecture_id": ARCHITECTURE_ID,
            "mechanics_target_transform": MECHANICS_TARGET_TRANSFORM,
        }
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root) / fold_id, self.binding_hash)
        opt = recipe["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
        )
        total_steps = sum(int(row["steps"]) for row in recipe["schedule"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=float(opt["minimum_learning_rate"])
        )
        self.step, self.best = 0, None
        prior = self.store.load()
        if prior:
            self.model.load_state_dict(prior["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(prior["optimizer_state_dict"])
            self.scheduler.load_state_dict(prior["scheduler_state_dict"])
            self.step, self.best = int(prior["step"]), prior.get("best")

        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        _require(count == int(recipe["capacity_policy"]["expected_trainable_parameters"]), "IRIS_CAPACITY_PILOT_PARAMETER_COUNT_DRIFT")

    def _presentation_loss(self, subject: FoldSubject, carrier, cameras, family: str, ids: np.ndarray):
        if family != "mechanics":
            return super()._presentation_loss(subject, carrier, cameras, family, ids)
        xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
        target = torch.from_numpy(subject.presentation["mechanical_target"][ids]).to(self.device)[None].float()
        transformed = mechanics_target_log1p_v1(target)
        gm, _ = self.model.query_states(carrier, xyz)
        raw = self.model.mechanics(gm).float()
        valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
        return heteroscedastic_mechanical_loss_v1(raw, transformed, valid)

    def train(self):
        total_steps = sum(int(row["steps"]) for row in self.recipe["schedule"])
        interval = int(self.recipe["evaluation"]["interval_steps"])
        examples = [SemanticExample(subject, 1.0, 1.0, 1) for subject in self.fold["train_subjects"]]
        started = time.monotonic()

        while self.step < total_steps:
            stage, step_index = self._stage(self.step), self.step

            def loss_sums(batch):
                return torch.stack([
                    self._subject_loss(self.train_subjects[row.key], step_index, stage, train=True)[0]
                    for row in batch
                ])

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
                        "model_state_dict": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in self.model.state_dict().items()
                        },
                    }
                self._save()
                print(
                    f"[IRIS_CAPACITY_PILOT] fold={self.fold['fold_id']} "
                    f"step={self.step}/{total_steps} stage={stage['id']} "
                    f"worst={validation['worst_subject_score']:.6f} "
                    f"mean={validation['mean_subject_score']:.6f}",
                    flush=True,
                )

        _require(self.best is not None, "IRIS_CAPACITY_PILOT_NO_CHECKPOINT_CANDIDATE")
        state_hashes = {
            name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            for name, tensor in self.best["model_state_dict"].items()
        }
        selected_identity = value_hash({
            "step": self.best["step"],
            "selection_key": self.best["selection_key"],
            "state_sha256": state_hashes,
        })

        heldout = FoldSubject(self.heldout_row, self.device, self.recipe)
        self.model.load_state_dict(self.best["model_state_dict"], strict=True)
        self.model.eval()
        heldout_eval = self._evaluate_subject(
            heldout, self.recipe["schedule"][-1], all_presentation=True
        )
        self.model.requires_grad_(False)
        checkpoint_path = _write_checkpoint(
            self.output / "selected_checkpoint.pt", self.model, self.config
        )
        result = {
            "schema": "RealSaS.IRISV31FixedSnakeCapacityPilotResult.v1",
            "fold_id": self.fold["fold_id"],
            "train_subjects": self.fold["train_subjects"],
            "heldout_subject": FIXED_HELDOUT,
            "selected_step": self.best["step"],
            "selected_checkpoint_identity_before_heldout_open": selected_identity,
            "train_selection": self.best["validation"],
            "heldout": heldout_eval,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "wall_seconds": time.monotonic() - started,
            "heldout_used_for_gradient": False,
            "heldout_used_for_checkpoint_selection": False,
            "mechanics_target_transform": MECHANICS_TARGET_TRANSFORM,
            "lofo_expansion_authorized": False,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "status": "PASS_EXECUTED_FIXED_SNAKE_CAPACITY_PILOT__LOFO_EXPANSION_SEPARATE",
        }
        _atomic_json(self.output / "result.json", result)
        return result


def run_fixed_snake_pilot(*, input_manifest_path, recipe_path, checkpoint_root, output_root, device="cuda"):
    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31FixedSnakeCapacityPilotTrainer(
        manifest=manifest,
        recipe=recipe,
        fold_id=FIXED_FOLD_ID,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
