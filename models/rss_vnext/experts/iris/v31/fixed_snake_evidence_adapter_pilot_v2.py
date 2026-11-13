"""Fixed-Snake V3.1 pilot with a protected geometry foundation.

Training is deliberately split into two optimizer domains:
1) geometry foundation acquisition;
2) mechanics/appearance adapter acquisition over a byte-frozen foundation.

There is no full shared-trunk joint unfreeze in this experiment.  The design is
informed by successful semantic-aware reconstruction practice: establish a
reconstruction foundation, preserve it, and bind new evidence families through
separate adapters/decoders before considering broader shared adaptation.
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
from models.iris.v5.direct_fstar_fit_v5 import (
    DirectFStarFitPolicyV5,
    direct_fstar_loss_v5,
    direct_fstar_metrics_v5,
)
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
from models.iris.v31.lofo_trainer_v1 import FoldSubject, _atomic_json, _require
from models.iris.v31.lofo_trainer_v2 import pair_group_partition_v2
from models.iris.v31.lofo_trainer_v3 import IrisV31LofoTrainerV3
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
    appearance_rgba_loss_v1,
    heteroscedastic_mechanical_loss_v1,
)
from models.training_runtime.semantic_batch_v1 import SemanticExample, execute_semantic_step

TRAINER_REVISION_V2 = "IRIS_V31_FIXED_SNAKE_EVIDENCE_ADAPTER_PILOT_V2"
RECIPE_SCHEMA_V2 = "RealSaS.IRISV31FixedSnakeEvidenceAdapterPilotRecipe.v2"
ARCHITECTURE_ID_V2 = IRISV31EvidenceAdapterV2.architecture_id
CHECKPOINT_SCHEMA_V2 = "RealSaS.IRISDenseObservationEvidenceCheckpoint.v3_1.evidence_adapter_v2"
FOUNDATION_STAGE = "GEOMETRY_FOUNDATION"
EVIDENCE_STAGE = "EVIDENCE_ADAPTER_ACQUISITION"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _write_checkpoint_v2(path: Path, model: IRISV31EvidenceAdapterV2, config) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA_V2,
        "architecture_id": ARCHITECTURE_ID_V2,
        "config": asdict(config),
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


class IrisV31FixedSnakeEvidenceAdapterPilotV2(IrisV31LofoTrainerV3):
    def __init__(
        self,
        *,
        manifest: dict,
        recipe: dict,
        fold_id: str,
        checkpoint_root,
        output_root,
        device="cuda",
    ):
        _require(recipe.get("schema") == RECIPE_SCHEMA_V2, "IRIS_EA_V2_RECIPE_SCHEMA")
        _require(recipe.get("trainer_revision") == TRAINER_REVISION_V2, "IRIS_EA_V2_TRAINER_REVISION")
        pilot = dict(recipe.get("pilot") or {})
        _require(
            fold_id == FIXED_FOLD_ID
            and pilot.get("fixed_fold_id") == FIXED_FOLD_ID
            and pilot.get("fixed_heldout_subject") == FIXED_HELDOUT
            and pilot.get("selection_rule") == "FIXED_FINAL_STEP__NO_CROSS_STAGE_SCALAR_CHECKPOINT_COMPETITION"
            and int(pilot.get("selected_step", -1)) == 6000
            and pilot.get("lofo_expansion_authorized") is False,
            "IRIS_EA_V2_FIXED_PILOT_DRIFT",
        )
        _pair_abstain_contract(recipe, manifest)
        _require(
            recipe.get("mechanics_target_transform") == "LOG1P_NONNEGATIVE_MAGNITUDES",
            "IRIS_EA_V2_MECHANICS_TRANSFORM_DRIFT",
        )
        _require(manifest.get("schema") == "RealSaS.IRISV31LofoInputManifest.v1", "IRIS_EA_V2_INPUT_SCHEMA")

        folds = [row for row in manifest["folds"] if row["fold_id"] == fold_id]
        _require(len(folds) == 1, "IRIS_EA_V2_FOLD_ID")
        self.fold = folds[0]
        _require(
            self.fold["heldout_subject"] == FIXED_HELDOUT
            and self.fold.get("checkpoint_selection_may_read_heldout") is False
            and self.fold.get("heldout_gradient_updates") is False
            and len(self.fold["train_subjects"]) == 7,
            "IRIS_EA_V2_HELDOUT_FIREWALL",
        )

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "IRIS_EA_V2_CUDA_REQUIRED")
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
        _require(self.config.mechanical_channels == MECHANICAL_CHANNELS_V31_PRESENTATION, "IRIS_EA_V2_MECHANICAL_CHANNEL_DRIFT")
        _require(self.config.pair_channels == PAIR_CHANNELS_V31_PRESENTATION, "IRIS_EA_V2_PAIR_CHANNEL_DRIFT")
        self.model = IRISV31EvidenceAdapterV2(self.config).to(self.device).train()

        self.recipe, self.manifest = recipe, manifest
        rows = {row["subject_id"]: row for row in manifest["subjects"]}
        _require(set(self.fold["train_subjects"]) <= set(rows) and FIXED_HELDOUT in rows, "IRIS_EA_V2_SUBJECT_DRIFT")
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
        _require(total == int(capacity["expected_trainable_parameters_total"]), "IRIS_EA_V2_TOTAL_PARAMETER_COUNT_DRIFT")
        _require(foundation == int(capacity["expected_foundation_graph_parameters"]), "IRIS_EA_V2_FOUNDATION_PARAMETER_COUNT_DRIFT")
        _require(evidence == int(capacity["expected_evidence_adapter_parameters"]), "IRIS_EA_V2_EVIDENCE_PARAMETER_COUNT_DRIFT")

        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema": "RealSaS.IRISV31FixedSnakeEvidenceAdapterPilotBinding.v2",
            "input_manifest_sha256": value_hash(manifest),
            "recipe_sha256": value_hash(recipe),
            "fold": self.fold,
            "config": asdict(self.config),
            "architecture_id": ARCHITECTURE_ID_V2,
        }
        self.binding_hash = value_hash(binding)
        self.store = AtomicCheckpointStore(Path(checkpoint_root) / fold_id, self.binding_hash)

        self.step = 0
        self.foundation_identity = None
        self.optimizer_stage_id = None
        self.optimizer = None
        self.scheduler = None
        prior = self.store.load()
        if prior:
            self.model.load_state_dict(prior["model_state_dict"], strict=True)
            self.step = int(prior["step"])
            self.foundation_identity = prior.get("foundation_identity")

        stage = self._stage(self.step)
        self._configure_stage(stage)
        self._build_optimizer(stage)
        if prior and prior.get("optimizer_stage_id") == stage["id"]:
            self.optimizer.load_state_dict(prior["optimizer_state_dict"])
            self.scheduler.load_state_dict(prior["scheduler_state_dict"])

    def _foundation_identity(self) -> str:
        rows = {
            name: _tensor_sha256(parameter)
            for name, parameter in self.model.foundation_named_parameters()
        }
        return value_hash(rows)

    def _configure_stage(self, stage: dict) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if stage["id"] == FOUNDATION_STAGE:
            for _, parameter in self.model.foundation_named_parameters():
                parameter.requires_grad_(True)
        elif stage["id"] == EVIDENCE_STAGE:
            _require(self.foundation_identity is not None, "IRIS_EA_V2_FOUNDATION_IDENTITY_REQUIRED")
            for _, parameter in self.model.evidence_named_parameters():
                parameter.requires_grad_(True)
        else:
            raise RuntimeError("IRIS_EA_V2_UNKNOWN_STAGE:" + str(stage["id"]))
        self.optimizer_stage_id = stage["id"]

    def _build_optimizer(self, stage: dict) -> None:
        opt = stage["optimizer"]
        parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
        _require(parameters, "IRIS_EA_V2_EMPTY_TRAINABLE_SET")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(stage["steps"]),
            eta_min=float(opt["minimum_learning_rate"]),
        )

    def _switch_stage_if_needed(self, stage: dict) -> bool:
        if self.optimizer_stage_id == stage["id"]:
            return False
        if stage["id"] == EVIDENCE_STAGE:
            _require(self.foundation_identity is not None, "IRIS_EA_V2_FOUNDATION_NOT_SEALED")
            _require(self._foundation_identity() == self.foundation_identity, "IRIS_EA_V2_FOUNDATION_DRIFT_BEFORE_EVIDENCE")
        self._configure_stage(stage)
        self._build_optimizer(stage)
        return True

    def _presentation_loss(self, subject: FoldSubject, carrier, cameras, family: str, ids: np.ndarray):
        if family == "mechanics":
            xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            target = torch.from_numpy(subject.presentation["mechanical_target"][ids]).to(self.device)[None].float()
            transformed = mechanics_target_log1p_v1(target)
            state = self.model.query_mechanics_state(carrier, xyz)
            raw = self.model.mechanics(state).float()
            valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
            return heteroscedastic_mechanical_loss_v1(raw, transformed, valid)
        if family == "appearance":
            xyz = torch.from_numpy(subject.presentation["appearance_xyz"][ids]).to(self.device)[None].to(carrier.dtype)
            view = torch.from_numpy(subject.presentation["appearance_camera_index"][ids]).to(self.device).long()
            target = torch.from_numpy(subject.presentation["appearance_rgba"][ids]).to(self.device)[None]
            _, gp = self.model.query_states(carrier, xyz)
            cam = cameras[:, view, :].to(gp.dtype)
            rgba = self.model.appearance(torch.cat((gp, cam), -1))[..., :4].sigmoid().float()
            valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
            return appearance_rgba_loss_v1(rgba, target, valid)
        raise RuntimeError("IRIS_EA_V2_UNAUTHORIZED_PRESENTATION_FAMILY:" + family)

    def _subject_loss(self, subject: FoldSubject, step: int, stage: dict, *, train: bool):
        autocast = bool(self.recipe["runtime"]["bf16_autocast"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            carrier, cameras = self._carrier(subject)
            if stage["id"] == FOUNDATION_STAGE:
                q, target = self._geometry_batch(subject, step)
                gg, _ = self.model.query_states(carrier, q.to(carrier.dtype))
                pred = self.model.geometry(gg)[..., 0].float()
                geometry = direct_fstar_loss_v5(
                    pred,
                    target,
                    policy=DirectFStarFitPolicyV5(
                        field_clamp=float(self.recipe["sampling"]["field_clamp"]),
                        huber_beta=float(self.recipe["sampling"]["field_huber_beta"]),
                    ),
                )["total"]
                total = geometry
                terms = {"geometry": geometry}
            else:
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
                int(self.recipe["evaluation"]["geometry_validation_points_per_train_subject"]),
                len(points),
            )
            q = points[:limit].to(self.device).float()[None]
            t = targets[:limit].to(self.device).float()[None]
            gg, _ = self.model.query_states(carrier, q.to(carrier.dtype))
            geometry = direct_fstar_metrics_v5(
                self.model.geometry(gg)[..., 0].float(), t, near_zero_band=0.03
            )
            losses = {}
            if stage["id"] == EVIDENCE_STAGE:
                for family, name in (("mechanics", "mechanics"), ("appearance", "appearance")):
                    ids = self._presentation_ids(
                        subject,
                        family,
                        self.step,
                        train=False,
                        all_rows=all_presentation,
                    )
                    if ids is not None:
                        losses[name] = float(
                            self._presentation_loss(subject, carrier, cameras, family, ids).float().cpu()
                        )
        self.model.train()
        return {
            "subject_id": subject.subject_id,
            "geometry": geometry,
            "presentation_losses": losses,
        }

    def _train_validation(self, stage: dict) -> dict:
        rows = [
            self._evaluate_subject(subject, stage, all_presentation=False)
            for subject in self.train_subjects.values()
        ]
        return {"step": self.step, "stage": stage["id"], "subjects": rows}

    def _validation_summary(self, validation: dict) -> dict:
        rows = validation["subjects"]
        def worst(path):
            values = []
            for row in rows:
                value = path(row)
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            return max(values) if values else None
        return {
            "geometry_mae": worst(lambda row: row["geometry"].get("mae")),
            "geometry_sign": worst(lambda row: row["geometry"].get("sign_disagreement_fraction")),
            "geometry_p95": worst(lambda row: row["geometry"].get("p95_abs")),
            "mechanics": worst(lambda row: row["presentation_losses"].get("mechanics")),
            "appearance": worst(lambda row: row["presentation_losses"].get("appearance")),
        }

    def _save(self) -> None:
        self.store.save({
            "binding_sha256": self.binding_hash,
            "step": self.step,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "optimizer_stage_id": self.optimizer_stage_id,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "foundation_identity": self.foundation_identity,
        })

    def train(self):
        total_steps = sum(int(row["steps"]) for row in self.recipe["schedule"])
        _require(total_steps == int(self.recipe["pilot"]["selected_step"]), "IRIS_EA_V2_TOTAL_STEP_DRIFT")
        interval = int(self.recipe["evaluation"]["interval_steps"])
        foundation_steps = int(self.recipe["foundation_policy"]["foundation_stage_steps"])
        examples = [SemanticExample(subject, 1.0, 1.0, 1) for subject in self.fold["train_subjects"]]
        started = time.monotonic()
        handoff_written = (self.output / "EVIDENCE_HANDOFF_BASELINE.json").is_file()

        while self.step < total_steps:
            stage = self._stage(self.step)
            switched = self._switch_stage_if_needed(stage)
            if switched and stage["id"] == EVIDENCE_STAGE and not handoff_written:
                baseline = self._train_validation(stage)
                summary = self._validation_summary(baseline)
                _atomic_json(self.output / "EVIDENCE_HANDOFF_BASELINE.json", {
                    "schema": "RealSaS.IRISV31EvidenceHandoffBaseline.v2",
                    "step": self.step,
                    "foundation_identity": self.foundation_identity,
                    "validation": baseline,
                    "summary": summary,
                    "heldout_visible": False,
                })
                print(
                    "[EVIDENCE_HANDOFF] "
                    f"step={self.step} foundation={self.foundation_identity[:12]} "
                    f"geometry_mae={summary['geometry_mae']:.6g} "
                    f"mechanics={summary['mechanics']:.6g} appearance={summary['appearance']:.6g} "
                    "foundation=FROZEN heldout=SEALED",
                    flush=True,
                )
                handoff_written = True

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

            if self.step == foundation_steps and self.foundation_identity is None:
                self.foundation_identity = self._foundation_identity()
                print(
                    f"[FOUNDATION_SEAL] step={self.step} identity={self.foundation_identity} heldout=SEALED",
                    flush=True,
                )

            if self.step % interval == 0 or self.step == total_steps:
                completed_stage = self._stage(max(0, self.step - 1))
                if completed_stage["id"] == EVIDENCE_STAGE:
                    _require(
                        self._foundation_identity() == self.foundation_identity,
                        "IRIS_EA_V2_FOUNDATION_DRIFT_DURING_EVIDENCE",
                    )
                validation = self._train_validation(completed_stage)
                summary = self._validation_summary(validation)
                self._save()
                mech = "NA" if summary["mechanics"] is None else f"{summary['mechanics']:.6g}"
                app = "NA" if summary["appearance"] is None else f"{summary['appearance']:.6g}"
                print(
                    "[GATE_EA_V2] "
                    f"step={self.step} stage={completed_stage['id']} "
                    f"geometry_mae={summary['geometry_mae']:.6g} "
                    f"sign_err={summary['geometry_sign']:.6g} p95_abs={summary['geometry_p95']:.6g} "
                    f"mech_log1p_nll={mech} rgba_l1={app} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e} "
                    f"peak_vram_gib={torch.cuda.max_memory_allocated(self.device)/(2**30):.2f} "
                    f"foundation={'SEALED' if self.foundation_identity else 'TRAINING'} heldout=SEALED",
                    flush=True,
                )

        _require(self.step == total_steps, "IRIS_EA_V2_FINAL_STEP_DRIFT")
        _require(self.foundation_identity is not None, "IRIS_EA_V2_FOUNDATION_IDENTITY_MISSING")
        _require(self._foundation_identity() == self.foundation_identity, "IRIS_EA_V2_FINAL_FOUNDATION_DRIFT")

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        checkpoint_path = _write_checkpoint_v2(
            self.output / "selected_checkpoint.pt", self.model, self.config
        )
        checkpoint_sha = file_hash(checkpoint_path)
        selected_identity = value_hash({
            "step": self.step,
            "checkpoint_sha256": checkpoint_sha,
            "foundation_identity": self.foundation_identity,
            "selection_rule": self.recipe["pilot"]["selection_rule"],
        })
        print(
            f"[CHECKPOINT_FREEZE] step={self.step} checkpoint_sha256={checkpoint_sha} "
            f"foundation_identity={self.foundation_identity} heldout=STILL_SEALED",
            flush=True,
        )

        # The held-out subject is instantiated only after final checkpoint bytes,
        # selection identity and foundation identity are immutable.
        heldout = FoldSubject(self.heldout_row, self.device, self.recipe)
        heldout_eval = self._evaluate_subject(
            heldout, self.recipe["schedule"][-1], all_presentation=True
        )
        final_train = self._train_validation(self.recipe["schedule"][-1])
        result = {
            "schema": "RealSaS.IRISV31FixedSnakeEvidenceAdapterPilotResult.v2",
            "fold_id": self.fold["fold_id"],
            "train_subjects": self.fold["train_subjects"],
            "heldout_subject": FIXED_HELDOUT,
            "selected_step": self.step,
            "selection_rule": self.recipe["pilot"]["selection_rule"],
            "selected_checkpoint_identity_before_heldout_open": selected_identity,
            "foundation_identity": self.foundation_identity,
            "foundation_frozen_during_evidence": True,
            "train_final": final_train,
            "heldout": heldout_eval,
            "checkpoint_sha256": checkpoint_sha,
            "wall_seconds": time.monotonic() - started,
            "heldout_used_for_gradient": False,
            "heldout_used_for_checkpoint_selection": False,
            "shared_foundation_jointly_finetuned_with_evidence": False,
            "semantic_taxonomy_used": False,
            "pair_training_authorized": False,
            "lofo_expansion_authorized": False,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "status": "PASS_EXECUTED_FIXED_SNAKE_EVIDENCE_ADAPTER_PILOT__SCIENTIFIC_QUALIFICATION_SEPARATE",
        }
        _atomic_json(self.output / "result.json", result)
        return result


def run_fixed_snake_evidence_adapter_pilot_v2(
    *, input_manifest_path, recipe_path, checkpoint_root, output_root, device="cuda"
):
    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31FixedSnakeEvidenceAdapterPilotV2(
        manifest=manifest,
        recipe=recipe,
        fold_id=FIXED_FOLD_ID,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
