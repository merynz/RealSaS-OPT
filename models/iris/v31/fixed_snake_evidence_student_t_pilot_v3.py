"""Fixed-Snake V3.1 evidence pilot with Student-t mechanics likelihood.

This experiment imports the byte-sealed geometry foundation from the preceding
protected-adapter V2 run, initializes fresh evidence adapters, and changes only
the mechanics residual likelihood from Gaussian to fixed-df Student-t.  Truth,
log1p target coordinates, row partitions, model architecture, appearance loss,
and held-out firewall remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.iris.training.cohort_v1 import AtomicCheckpointStore, file_hash, value_hash
from models.iris.v5.direct_fstar_fit_v5 import direct_fstar_metrics_v5
from models.iris.v31.evidence_adapter_model_v2 import (
    IRISEvidenceAdapterConfigV2,
    IRISV31EvidenceAdapterV2,
)
from models.iris.v31.fixed_snake_capacity_pilot_v1 import (
    FIXED_FOLD_ID,
    FIXED_HELDOUT,
    _pair_abstain_contract,
    mechanics_target_log1p_v1,
)
from models.iris.v31.fixed_snake_evidence_adapter_pilot_v2 import (
    EVIDENCE_STAGE,
    IrisV31FixedSnakeEvidenceAdapterPilotV2,
)
from models.iris.v31.lofo_trainer_v1 import FoldSubject, _atomic_json, _require
from models.iris.v31.lofo_trainer_v2 import pair_group_partition_v2
from models.iris.v31.mechanics_student_t_v3 import student_t_mechanical_loss_v3
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
    appearance_rgba_loss_v1,
)
from models.training_runtime.semantic_batch_v1 import SemanticExample, execute_semantic_step

TRAINER_REVISION_V3 = "IRIS_V31_FIXED_SNAKE_EVIDENCE_STUDENT_T_PILOT_V3"
RECIPE_SCHEMA_V3 = "RealSaS.IRISV31FixedSnakeEvidenceStudentTPilotRecipe.v3"
CHECKPOINT_SCHEMA_V3 = "RealSaS.IRISDenseObservationEvidenceCheckpoint.v3_1.student_t_pilot_v3"
OBJECTIVE_ID_V3 = "STUDENT_T_NLL__LOG1P_NONNEGATIVE_MAGNITUDES__FIXED_DF_3"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _foundation_identity(model: IRISV31EvidenceAdapterV2) -> str:
    return value_hash({
        name: _tensor_sha256(parameter)
        for name, parameter in model.foundation_named_parameters()
    })


def _load_v2_foundation_payload(root: Path) -> tuple[dict, dict, Path]:
    root = Path(root).resolve() / FIXED_FOLD_ID
    latest = root / "LATEST.json"
    _require(latest.is_file(), "IRIS_ST_V3_V2_FOUNDATION_POINTER_MISSING")
    pointer = json.loads(latest.read_text())
    _require(
        pointer.get("schema") == "RealSaS.SharedFitCheckpointPointer.v1",
        "IRIS_ST_V3_V2_FOUNDATION_POINTER_SCHEMA",
    )
    name = str(pointer.get("file") or "")
    _require(Path(name).name == name and name, "IRIS_ST_V3_V2_FOUNDATION_PATH_ESCAPE")
    path = root / name
    _require(
        path.is_file() and file_hash(path) == pointer.get("sha256"),
        "IRIS_ST_V3_V2_FOUNDATION_BYTES_DRIFT",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(
        payload.get("binding_sha256") == pointer.get("binding_sha256")
        and int(payload.get("step", -1)) == int(pointer.get("step", -2))
        and int(payload.get("step", -1)) >= 1000
        and payload.get("optimizer_stage_id") == EVIDENCE_STAGE
        and isinstance(payload.get("foundation_identity"), str)
        and len(payload["foundation_identity"]) == 64,
        "IRIS_ST_V3_V2_FOUNDATION_PAYLOAD_DRIFT",
    )
    return pointer, payload, path


def _write_checkpoint_v3(path: Path, model: IRISV31EvidenceAdapterV2, config, objective: dict) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA_V3,
        "architecture_id": model.architecture_id,
        "config": asdict(config),
        "training_objective": objective,
        "model_state_dict": {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        },
    }
    temp = target.with_name(target.name + ".tmp")
    torch.save(payload, temp)
    torch.load(temp, map_location="cpu", weights_only=True)
    temp.replace(target)
    return target


def _distribution_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("IRIS_ST_V3_MECHANICS_AUDIT_EMPTY")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("IRIS_ST_V3_MECHANICS_AUDIT_INVALID")

    def channel(x: np.ndarray) -> dict:
        order = np.sort(x)
        return {
            "count": int(len(x)),
            "q50": float(np.quantile(x, 0.50)),
            "q90": float(np.quantile(x, 0.90)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "max": float(order[-1]),
            "top5": [float(v) for v in order[-5:][::-1]],
        }

    transformed = np.log1p(values)
    return {
        "raw_nonrigid": channel(values[:, 0]),
        "raw_energy": channel(values[:, 1]),
        "log1p_nonrigid": channel(transformed[:, 0]),
        "log1p_energy": channel(transformed[:, 1]),
    }


class IrisV31FixedSnakeEvidenceStudentTPilotV3(IrisV31FixedSnakeEvidenceAdapterPilotV2):
    """Fresh evidence fit over a sealed V2 geometry foundation."""

    def __init__(
        self,
        *,
        manifest: dict,
        recipe: dict,
        fold_id: str,
        foundation_checkpoint_root,
        checkpoint_root,
        output_root,
        device="cuda",
    ):
        _require(recipe.get("schema") == RECIPE_SCHEMA_V3, "IRIS_ST_V3_RECIPE_SCHEMA")
        _require(recipe.get("trainer_revision") == TRAINER_REVISION_V3, "IRIS_ST_V3_TRAINER_REVISION")
        _require(recipe.get("mechanics_objective", {}).get("id") == OBJECTIVE_ID_V3, "IRIS_ST_V3_OBJECTIVE_ID")
        pilot = dict(recipe.get("pilot") or {})
        _require(
            fold_id == FIXED_FOLD_ID
            and pilot.get("fixed_fold_id") == FIXED_FOLD_ID
            and pilot.get("fixed_heldout_subject") == FIXED_HELDOUT
            and pilot.get("selection_rule") == "FIXED_FINAL_EVIDENCE_STEP__NO_CHECKPOINT_SHOPPING"
            and int(pilot.get("selected_evidence_step", -1)) == 5000
            and pilot.get("lofo_expansion_authorized") is False,
            "IRIS_ST_V3_FIXED_PILOT_DRIFT",
        )
        _pair_abstain_contract(recipe, manifest)
        _require(
            recipe.get("mechanics_target_transform") == "LOG1P_NONNEGATIVE_MAGNITUDES"
            and recipe.get("partition_policy") == "UNCHANGED_HASHED_ROW_80_20_V1",
            "IRIS_ST_V3_MECHANICS_CONTRACT_DRIFT",
        )
        _require(manifest.get("schema") == "RealSaS.IRISV31LofoInputManifest.v1", "IRIS_ST_V3_INPUT_SCHEMA")

        folds = [row for row in manifest["folds"] if row["fold_id"] == fold_id]
        _require(len(folds) == 1, "IRIS_ST_V3_FOLD_ID")
        self.fold = folds[0]
        _require(
            self.fold["heldout_subject"] == FIXED_HELDOUT
            and self.fold.get("checkpoint_selection_may_read_heldout") is False
            and self.fold.get("heldout_gradient_updates") is False
            and len(self.fold["train_subjects"]) == 7,
            "IRIS_ST_V3_HELDOUT_FIREWALL",
        )

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "IRIS_ST_V3_CUDA_REQUIRED")
        runtime = recipe["runtime"]
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False

        self.seed = int(recipe["sampling"]["seed"]) + int(self.fold["fold_index"]) * 1000003
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed & 0xFFFFFFFF)

        config_row = dict(recipe["model_config"])
        config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
        config_row["pair_channels"] = tuple(config_row["pair_channels"])
        self.config = IRISEvidenceAdapterConfigV2(**config_row)
        self.config.validate()
        _require(self.config.mechanical_channels == MECHANICAL_CHANNELS_V31_PRESENTATION, "IRIS_ST_V3_MECHANICAL_CHANNEL_DRIFT")
        _require(self.config.pair_channels == PAIR_CHANNELS_V31_PRESENTATION, "IRIS_ST_V3_PAIR_CHANNEL_DRIFT")
        self.model = IRISV31EvidenceAdapterV2(self.config).to(self.device).train()

        self.recipe, self.manifest = recipe, manifest
        rows = {row["subject_id"]: row for row in manifest["subjects"]}
        _require(set(self.fold["train_subjects"]) <= set(rows) and FIXED_HELDOUT in rows, "IRIS_ST_V3_SUBJECT_DRIFT")
        self.train_subjects = {
            sid: FoldSubject(rows[sid], self.device, recipe)
            for sid in self.fold["train_subjects"]
        }
        self.heldout_row = rows[FIXED_HELDOUT]
        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        partition_seed = int(recipe["sampling"]["seed"])
        for subject in self.train_subjects.values():
            valid = np.flatnonzero(np.asarray(subject.presentation["pair_valid"], dtype=np.bool_))
            subject.partitions["pair"] = pair_group_partition_v2(
                subject.subject_id, valid, partition_seed, fraction
            )

        total = sum(p.numel() for p in self.model.parameters())
        foundation = sum(p.numel() for _, p in self.model.foundation_named_parameters())
        evidence = sum(p.numel() for _, p in self.model.evidence_named_parameters())
        capacity = recipe["capacity_policy"]
        _require(total == int(capacity["expected_total_parameters"]), "IRIS_ST_V3_TOTAL_PARAMETER_COUNT_DRIFT")
        _require(foundation == int(capacity["expected_foundation_parameters"]), "IRIS_ST_V3_FOUNDATION_PARAMETER_COUNT_DRIFT")
        _require(evidence == int(capacity["expected_evidence_parameters"]), "IRIS_ST_V3_EVIDENCE_PARAMETER_COUNT_DRIFT")

        # Import only the protected geometry foundation. Evidence adapters and
        # heads retain their fresh deterministic initialization from this run.
        pointer, source_payload, source_path = _load_v2_foundation_payload(
            Path(foundation_checkpoint_root)
        )
        source_state = source_payload["model_state_dict"]
        with torch.no_grad():
            for name, parameter in self.model.foundation_named_parameters():
                _require(name in source_state and source_state[name].shape == parameter.shape, "IRIS_ST_V3_FOUNDATION_STATE_MISSING:" + name)
                parameter.copy_(source_state[name].to(parameter.device, dtype=parameter.dtype))
        self.foundation_identity = _foundation_identity(self.model)
        _require(
            self.foundation_identity == source_payload["foundation_identity"],
            "IRIS_ST_V3_IMPORTED_FOUNDATION_IDENTITY_DRIFT",
        )
        self.foundation_source = {
            "pointer_sha256": file_hash(Path(foundation_checkpoint_root).resolve() / FIXED_FOLD_ID / "LATEST.json"),
            "payload_sha256": file_hash(source_path),
            "payload_step": int(source_payload["step"]),
            "foundation_identity": self.foundation_identity,
        }

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for _, parameter in self.model.evidence_named_parameters():
            parameter.requires_grad_(True)

        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema": "RealSaS.IRISV31FixedSnakeEvidenceStudentTPilotBinding.v3",
            "input_manifest_sha256": value_hash(manifest),
            "recipe_sha256": value_hash(recipe),
            "fold": self.fold,
            "config": asdict(self.config),
            "foundation_source": self.foundation_source,
            "architecture_id": self.model.architecture_id,
            "objective_id": OBJECTIVE_ID_V3,
        }
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root) / fold_id, self.binding_hash)

        stage = recipe["schedule"][0]
        opt = stage["optimizer"]
        trainable = tuple(p for p in self.model.parameters() if p.requires_grad)
        _require(trainable, "IRIS_ST_V3_EMPTY_TRAINABLE_SET")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(stage["steps"]),
            eta_min=float(opt["minimum_learning_rate"]),
        )
        self.step = 0
        prior = self.store.load()
        if prior:
            self.model.load_state_dict(prior["model_state_dict"], strict=True)
            self.step = int(prior["step"])
            _require(_foundation_identity(self.model) == self.foundation_identity, "IRIS_ST_V3_RESUME_FOUNDATION_DRIFT")
            self.optimizer.load_state_dict(prior["optimizer_state_dict"])
            self.scheduler.load_state_dict(prior["scheduler_state_dict"])

    def _mechanics_objective(self, raw: torch.Tensor, transformed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        objective = self.recipe["mechanics_objective"]
        return student_t_mechanical_loss_v3(
            raw,
            transformed,
            valid,
            degrees_of_freedom=float(objective["degrees_of_freedom"]),
            minimum_log_variance=float(objective["minimum_log_variance"]),
            maximum_log_variance=float(objective["maximum_log_variance"]),
        )

    def _presentation_loss(self, subject: FoldSubject, carrier, cameras, family: str, ids: np.ndarray):
        if family == "mechanics":
            xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            target = torch.from_numpy(subject.presentation["mechanical_target"][ids]).to(self.device)[None].float()
            transformed = mechanics_target_log1p_v1(target)
            state = self.model.query_mechanics_state(carrier, xyz)
            raw = self.model.mechanics(state).float()
            valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
            return self._mechanics_objective(raw, transformed, valid)
        if family == "appearance":
            xyz = torch.from_numpy(subject.presentation["appearance_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            view = torch.from_numpy(subject.presentation["appearance_camera_index"][ids]).to(self.device).long()
            target = torch.from_numpy(subject.presentation["appearance_rgba"][ids]).to(self.device)[None]
            _, gp = self.model.query_states(carrier, xyz)
            cam = cameras[:, view, :].to(gp.dtype)
            rgba = self.model.appearance(torch.cat((gp, cam), -1))[..., :4].sigmoid().float()
            valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
            return appearance_rgba_loss_v1(rgba, target, valid)
        raise RuntimeError("IRIS_ST_V3_UNAUTHORIZED_PRESENTATION_FAMILY:" + family)

    def _subject_loss(self, subject: FoldSubject, step: int, stage: dict, *, train: bool):
        autocast = bool(self.recipe["runtime"]["bf16_autocast"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            carrier, cameras = self._carrier(subject)
            terms = {}
            total = carrier.new_zeros(())
            for family, name in (("mechanics", "mechanics"), ("appearance", "appearance")):
                ids = self._presentation_ids(subject, family, step, train=train)
                if ids is not None:
                    loss = self._presentation_loss(subject, carrier, cameras, family, ids)
                    terms[name] = loss
                    total = total + float(stage["loss_weights"][name]) * loss
        return total.float(), {name: float(value.detach().cpu()) for name, value in terms.items()}

    @torch.no_grad()
    def _evaluate_subject(self, subject: FoldSubject, stage: dict, *, all_presentation: bool) -> dict:
        self.model.eval()
        autocast = bool(self.recipe["runtime"]["bf16_autocast"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            carrier, cameras = self._carrier(subject)
            points, targets = subject.geometry["heldout_points"], subject.geometry["heldout_targets"]
            limit = len(points) if all_presentation else min(
                int(self.recipe["evaluation"]["geometry_validation_points_per_train_subject"]), len(points)
            )
            q = points[:limit].to(self.device).float()[None]
            t = targets[:limit].to(self.device).float()[None]
            gg, _ = self.model.query_states(carrier, q.to(carrier.dtype))
            geometry = direct_fstar_metrics_v5(
                self.model.geometry(gg)[..., 0].float(), t, near_zero_band=0.03
            )
            losses = {}
            for family, name in (("mechanics", "mechanics"), ("appearance", "appearance")):
                ids = self._presentation_ids(
                    subject, family, self.step, train=False, all_rows=all_presentation
                )
                if ids is not None:
                    losses[name] = float(
                        self._presentation_loss(subject, carrier, cameras, family, ids).float().cpu()
                    )
        self.model.train()
        return {"subject_id": subject.subject_id, "geometry": geometry, "presentation_losses": losses}

    def _train_validation(self, stage: dict) -> dict:
        return {
            "step": self.step,
            "stage": stage["id"],
            "subjects": [
                self._evaluate_subject(subject, stage, all_presentation=False)
                for subject in self.train_subjects.values()
            ],
        }

    def _partition_audit(self) -> dict:
        subjects = {}
        for sid, subject in self.train_subjects.items():
            train_ids, val_ids = subject.partitions["mechanics"]
            subjects[sid] = {
                "train": _distribution_stats(subject.presentation["mechanical_target"][train_ids]),
                "validation": _distribution_stats(subject.presentation["mechanical_target"][val_ids]),
            }
        return {
            "schema": "RealSaS.IRISV31MechanicsPartitionAudit.v3",
            "partition_policy": self.recipe["partition_policy"],
            "heldout_visible": False,
            "subjects": subjects,
        }

    def _save(self) -> None:
        self.store.save({
            "binding_sha256": self.binding_hash,
            "step": self.step,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "foundation_identity": self.foundation_identity,
            "objective_id": OBJECTIVE_ID_V3,
        })

    def train(self):
        stage = self.recipe["schedule"][0]
        total_steps = int(stage["steps"])
        _require(total_steps == int(self.recipe["pilot"]["selected_evidence_step"]), "IRIS_ST_V3_TOTAL_STEP_DRIFT")
        interval = int(self.recipe["evaluation"]["interval_steps"])
        examples = [SemanticExample(subject, 1.0, 1.0, 1) for subject in self.fold["train_subjects"]]
        started = time.monotonic()

        audit_path = self.output / "MECHANICS_PARTITION_AUDIT_V3.json"
        if not audit_path.is_file():
            audit = self._partition_audit()
            _atomic_json(audit_path, audit)
            shark = audit["subjects"].get("fit8_shark_v1")
            if shark:
                print(
                    "[MECHANICS_PARTITION_AUDIT] shark "
                    f"train_raw_top5={shark['train']['raw_nonrigid']['top5']} "
                    f"val_raw_top5={shark['validation']['raw_nonrigid']['top5']} "
                    "heldout=SEALED",
                    flush=True,
                )

        if self.step == 0:
            baseline = self._train_validation(stage)
            summary = self._validation_summary(baseline)
            _atomic_json(self.output / "EVIDENCE_STUDENT_T_BASELINE.json", {
                "schema": "RealSaS.IRISV31EvidenceStudentTBaseline.v3",
                "step": 0,
                "foundation_identity": self.foundation_identity,
                "validation": baseline,
                "summary": summary,
                "heldout_visible": False,
            })
            print(
                "[STUDENT_T_HANDOFF] "
                f"foundation={self.foundation_identity[:12]} geometry_mae={summary['geometry_mae']:.6g} "
                f"mechanics_student_t_nll={summary['mechanics']:.6g} appearance={summary['appearance']:.6g} "
                "evidence=FRESH heldout=SEALED",
                flush=True,
            )

        while self.step < total_steps:
            step_index = self.step

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
                parameters=tuple(p for p in self.model.parameters() if p.requires_grad),
                max_padded_cost=1,
                max_examples=1,
                clip_norm=float(stage["optimizer"]["gradient_clip_norm"]),
            )
            self.step += 1

            _require(_foundation_identity(self.model) == self.foundation_identity, "IRIS_ST_V3_FOUNDATION_DRIFT")
            if self.step % interval == 0 or self.step == total_steps:
                validation = self._train_validation(stage)
                summary = self._validation_summary(validation)
                self._save()
                print(
                    "[GATE_ST_V3] "
                    f"evidence_step={self.step}/{total_steps} geometry_mae={summary['geometry_mae']:.6g} "
                    f"sign_err={summary['geometry_sign']:.6g} p95_abs={summary['geometry_p95']:.6g} "
                    f"mech_student_t_nll={summary['mechanics']:.6g} rgba_l1={summary['appearance']:.6g} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e} "
                    f"peak_vram_gib={torch.cuda.max_memory_allocated(self.device)/(2**30):.2f} "
                    "foundation=SEALED heldout=SEALED",
                    flush=True,
                )

        _require(self.step == total_steps, "IRIS_ST_V3_FINAL_STEP_DRIFT")
        _require(_foundation_identity(self.model) == self.foundation_identity, "IRIS_ST_V3_FINAL_FOUNDATION_DRIFT")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

        checkpoint_path = _write_checkpoint_v3(
            self.output / "selected_checkpoint.pt",
            self.model,
            self.config,
            self.recipe["mechanics_objective"],
        )
        checkpoint_sha = file_hash(checkpoint_path)
        selected_identity = value_hash({
            "evidence_step": self.step,
            "checkpoint_sha256": checkpoint_sha,
            "foundation_identity": self.foundation_identity,
            "selection_rule": self.recipe["pilot"]["selection_rule"],
            "objective_id": OBJECTIVE_ID_V3,
        })
        print(
            f"[CHECKPOINT_FREEZE_ST_V3] evidence_step={self.step} checkpoint_sha256={checkpoint_sha} "
            f"foundation_identity={self.foundation_identity} heldout=STILL_SEALED",
            flush=True,
        )

        # Held-out Snake is instantiated only after final bytes and selection
        # identity are immutable.
        heldout = FoldSubject(self.heldout_row, self.device, self.recipe)
        heldout_eval = self._evaluate_subject(heldout, stage, all_presentation=True)
        final_train = self._train_validation(stage)
        result = {
            "schema": "RealSaS.IRISV31FixedSnakeEvidenceStudentTPilotResult.v3",
            "fold_id": self.fold["fold_id"],
            "train_subjects": self.fold["train_subjects"],
            "heldout_subject": FIXED_HELDOUT,
            "selected_evidence_step": self.step,
            "selection_rule": self.recipe["pilot"]["selection_rule"],
            "selected_checkpoint_identity_before_heldout_open": selected_identity,
            "foundation_identity": self.foundation_identity,
            "foundation_source": self.foundation_source,
            "foundation_frozen_during_evidence": True,
            "mechanics_objective": self.recipe["mechanics_objective"],
            "partition_policy": self.recipe["partition_policy"],
            "train_final": final_train,
            "heldout": heldout_eval,
            "checkpoint_sha256": checkpoint_sha,
            "wall_seconds": time.monotonic() - started,
            "heldout_used_for_gradient": False,
            "heldout_used_for_checkpoint_selection": False,
            "truth_rows_deleted": False,
            "target_clipping": False,
            "subject_specific_filtering": False,
            "semantic_taxonomy_used": False,
            "pair_training_authorized": False,
            "lofo_expansion_authorized": False,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "status": "PASS_EXECUTED_FIXED_SNAKE_STUDENT_T_PILOT__SCIENTIFIC_QUALIFICATION_SEPARATE",
        }
        _atomic_json(self.output / "result.json", result)
        return result


def run_fixed_snake_evidence_student_t_pilot_v3(
    *,
    input_manifest_path,
    recipe_path,
    foundation_checkpoint_root,
    checkpoint_root,
    output_root,
    device="cuda",
):
    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31FixedSnakeEvidenceStudentTPilotV3(
        manifest=manifest,
        recipe=recipe,
        fold_id=FIXED_FOLD_ID,
        foundation_checkpoint_root=foundation_checkpoint_root,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
