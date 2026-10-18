from __future__ import annotations

"""IRIS vNext Fit8 trainer v3: field-anchored G2 + post-update surface court.

This authority is intentionally migration-aware. It can ingest the known failed
v2 step-1601 checkpoint only through the explicit surgical field rollback
procedure proven by the Fit8 recovery court. The deleted original step-1600
checkpoint is not reconstructed or claimed.
"""

from dataclasses import asdict, replace
from pathlib import Path
import math
import tempfile
import time

import torch
import torch.nn.functional as F

from .cohort_v1 import (
    VNextCheckpointStoreV1,
    VNextFit8PolicyV1,
    _REQUIRED_APPEARANCE,
    _REQUIRED_MECH,
    _REQUIRED_RELATION,
    _sha,
    _vh,
)
from .cohort_v2 import IRISVNextFit8TrainerV2, _source_identity_v2
from .contracts_v1 import IRISVNextConfigV1
from .surface_extraction_v1 import extract_surface_batch_v1, extracted_surface_loss_v1
from .training_v1 import graphics_relation_loss_v1, mechanical_demand_loss_v1


FAILED_V2_STEP1601_SHA256 = "d6e8f045c1833ccf7b5df64177b121597f2022c94407b05e7a55c7f6b579dde8"
FIELD_PREFIX = "scene_core.field."


def vnext_fit8_policy_v2() -> VNextFit8PolicyV1:
    """Return the court-selected curriculum while preserving total 12k steps."""
    base = VNextFit8PolicyV1()
    stages = list(base.stages)
    g2 = stages[1]
    if g2.stage_id != "G2_FLEXICUBES_SURFACE" or g2.steps != 1200:
        raise RuntimeError("VNEXT_G2_BASE_POLICY_DRIFT")
    stages[1] = replace(
        g2,
        scene_lr_scale=0.25,
        evidence_lr_scale=0.5,
        field_weight=0.5,
        extraction_weight=1.0,
    )
    policy = replace(base, stages=tuple(stages))
    policy.validate()
    if policy.total_steps != 12000:
        raise RuntimeError("VNEXT_G2_REPAIR_TOTAL_STEP_DRIFT")
    return policy


def _source_identity_v3(root: Path):
    value = dict(_source_identity_v2(root))
    value["models/iris/vnext/cohort_v3.py"] = _sha(root / "models/iris/vnext/cohort_v3.py")
    return value


class IRISVNextFit8TrainerV3(IRISVNextFit8TrainerV2):
    """v2 architecture with repaired G2 and a fail-closed post-update surface gate."""

    architecture_id = "RealSaS.IRISVNextFit8Trainer.v3"

    def __init__(
        self,
        *,
        data_root,
        manifest,
        checkpoint_root,
        device,
        model_config=IRISVNextConfigV1(),
        policy=None,
        repo_root=None,
        flexicubes_root=None,
    ):
        policy = vnext_fit8_policy_v2() if policy is None else policy
        policy.validate()
        target_root = Path(checkpoint_root)
        with tempfile.TemporaryDirectory(prefix="iris_vnext_v3_bootstrap_") as bootstrap:
            super().__init__(
                data_root=data_root,
                manifest=manifest,
                checkpoint_root=bootstrap,
                device=device,
                model_config=model_config,
                policy=policy,
                repo_root=repo_root,
                flexicubes_root=flexicubes_root,
            )

        self.binding = {
            **self.binding,
            "schema": "RealSaS.IRISVNextFit8Binding.v5",
            "source": _source_identity_v3(self.repo_root),
            "policy": asdict(policy),
            "optimizer_semantics": (
                "ONE_BATCH8_SCENE_ENCODE__NO_SUBJECT_SERIAL_MODEL_FORWARD__"
                "ONE_BACKWARD__ONE_STEP__POST_UPDATE_SURFACE_FAIL_CLOSED"
            ),
            "curriculum_semantics": (
                "G1__G2_FIELD_ANCHORED_FLEXICUBES__M1__A1__R1__T1__J1"
            ),
            "g2_repair_semantics": {
                "field_weight": 0.5,
                "scene_lr_scale": 0.25,
                "evidence_lr_scale": 0.5,
                "post_optimizer_surface_gate": True,
                "court": "SURGICAL_FIELD_ROLLBACK_G2_STABILITY_COURT_V2",
            },
        }
        self.binding_hash = _vh(self.binding)
        self.store = VNextCheckpointStoreV1(target_root, self.binding_hash)

        previous = self.store.load()
        if previous:
            self._restore(previous)
        else:
            self.step = 0
            self.stage_index = 0
            self.stage_step = 0
            self._apply_stage_lrs()

    def _surface_gate(self) -> dict:
        if self.extractor is None:
            raise RuntimeError("VNEXT_POST_UPDATE_SURFACE_GATE_EXTRACTOR_MISSING")
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        was_training = self.model.training
        try:
            self.model.train()
            source = self.cohort.tensors
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                planes = self.model.encode_scene(
                    source["image_tokens"], source["token_xy"], source["camera_features"]
                )
                extracted = extract_surface_batch_v1(
                    self.model,
                    planes,
                    self.extractor,
                    resolution=self.model.config.extraction_training_resolution,
                    witness_count=self.model.config.surface_witness_count,
                    training=True,
                )
            vertices = tuple(map(int, extracted.extracted_vertex_counts))
            faces = tuple(map(int, extracted.extracted_face_counts))
            if len(vertices) != 8 or len(faces) != 8:
                raise RuntimeError("VNEXT_POST_UPDATE_SURFACE_GATE_BATCH_DRIFT")
            if any(v <= 0 for v in vertices) or any(f <= 0 for f in faces):
                raise RuntimeError(
                    f"VNEXT_POST_UPDATE_SURFACE_GATE_EMPTY:V={vertices}:F={faces}"
                )
            return {
                "schema": "RealSaS.IRISVNextPostUpdateSurfaceCourt.v1",
                "phase": "POST_OPTIMIZER_PRE_COMMIT",
                "vertices": vertices,
                "faces": faces,
                "pass": True,
            }
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.model.train(was_training)

    def migrate_failed_v2_step1601(self, payload: dict, *, source_checkpoint_sha256: str):
        if self.step != 0 or self.stage_index != 0 or self.stage_step != 0:
            raise RuntimeError("VNEXT_V3_MIGRATION_TARGET_NOT_FRESH")
        if source_checkpoint_sha256 != FAILED_V2_STEP1601_SHA256:
            raise RuntimeError("VNEXT_V3_MIGRATION_SOURCE_SHA_DRIFT")
        if int(payload.get("step", -1)) != 1601:
            raise RuntimeError("VNEXT_V3_MIGRATION_SOURCE_STEP_DRIFT")
        if int(payload.get("stage_index", -1)) != 1 or int(payload.get("stage_step", -1)) != 1:
            raise RuntimeError("VNEXT_V3_MIGRATION_SOURCE_STAGE_DRIFT")
        if payload.get("schema") != "RealSaS.IRISVNextFit8Checkpoint.v4":
            raise RuntimeError("VNEXT_V3_MIGRATION_SOURCE_SCHEMA_DRIFT")

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

        name_of = {id(parameter): name for name, parameter in self.model.named_parameters()}
        rolled = []
        reset = []

        with torch.no_grad():
            for group in self.optimizer.param_groups:
                lr = float(group["lr"])
                weight_decay = float(group.get("weight_decay", 0.0))
                beta1, beta2 = map(float, group["betas"])
                eps = float(group["eps"])
                if bool(group.get("amsgrad", False)) or bool(group.get("maximize", False)):
                    raise RuntimeError("VNEXT_V3_MIGRATION_UNSUPPORTED_ADAMW_MODE")
                decay = 1.0 - lr * weight_decay
                for parameter in group["params"]:
                    name = name_of[id(parameter)]
                    if not name.startswith(FIELD_PREFIX):
                        continue
                    state = self.optimizer.state.get(parameter, {})
                    if not state or not all(k in state for k in ("step", "exp_avg", "exp_avg_sq")):
                        continue
                    t = int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                    if t <= 0:
                        continue
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add(eps)
                    adam_term = (lr / bc1) * (exp_avg / denom)
                    parameter.copy_((parameter + adam_term) / decay)
                    rolled.append(name)

        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                name = name_of[id(parameter)]
                if name.startswith(FIELD_PREFIX):
                    self.optimizer.state[parameter].clear()
                    reset.append(name)

        self.step = 1601
        self.stage_index = 1
        self.stage_step = 1
        self._apply_stage_lrs()
        self.optimizer.zero_grad(set_to_none=True)

        gate = self._surface_gate()
        receipt = {
            "schema": "RealSaS.IRISVNextFit8MigrationReceipt.v1",
            "status": "PASS",
            "migration": "FAILED_V2_STEP1601__SURGICAL_FIELD_ROLLBACK__FIELD_ADAM_RESET",
            "source_checkpoint_step": 1601,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_binding_sha256": payload.get("binding_sha256"),
            "target_binding_sha256": self.binding_hash,
            "target_binding_schema": self.binding["schema"],
            "retained_global_step": 1601,
            "retained_stage_index": 1,
            "retained_stage_step": 1,
            "field_parameter_rollback_count": len(rolled),
            "field_optimizer_state_reset_count": len(reset),
            "exact_deleted_step1600_recovered": False,
            "post_migration_surface_court": gate,
        }
        pointer = self.save()
        receipt["migrated_checkpoint_pointer"] = pointer
        return receipt

    def one_step(self):
        if self._advance_stage_if_needed():
            raise StopIteration("VNEXT_FIT8_COMPLETE")
        stage = self.stage
        next_step = self.step + 1
        view_index = next_step % 8
        source = self.cohort.tensors
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            planes = self.model.encode_scene(
                source["image_tokens"], source["token_xy"], source["camera_features"]
            )
            zero = planes.sum() * 0.0
            losses = {
                "field": zero,
                "extraction": zero,
                "extraction_chamfer": zero,
                "extraction_l_dev": zero,
                "mechanical": zero,
                "appearance": zero,
                "relations": zero,
                "topology": zero,
            }

            if stage.field_weight > 0:
                points, target = self.cohort.geometry_batch(self.policy, next_step)
                predicted = self.model.query_evidence(planes, points)
                losses["field"] = F.smooth_l1_loss(
                    predicted["field"].float(), target.float()
                )

            extracted = None
            if stage.extraction_weight > 0 or stage.topology_weight > 0:
                if self.extractor is None:
                    raise RuntimeError("VNEXT_FIT8_EXTRACTOR_MISSING")
                extracted = extract_surface_batch_v1(
                    self.model,
                    planes,
                    self.extractor,
                    resolution=self.model.config.extraction_training_resolution,
                    witness_count=self.model.config.surface_witness_count,
                    training=True,
                )
                if stage.extraction_weight > 0:
                    extraction = extracted_surface_loss_v1(
                        extracted,
                        source["teacher_surface_points_iris"],
                        l_dev_weight=self.policy.extraction_l_dev_weight,
                        flexible_weight_l1=self.policy.extraction_flexible_weight_l1,
                    )
                    losses["extraction"] = extraction["total"]
                    losses["extraction_chamfer"] = extraction["surface_chamfer"]
                    losses["extraction_l_dev"] = extraction["l_dev"]

            if stage.mechanical_weight > 0:
                batch = self.cohort.mechanical_batch(self.policy, next_step)
                predicted = self.model.query_evidence(
                    planes, batch["mechanical_points"]
                )["mechanical_demand"]
                losses["mechanical"] = mechanical_demand_loss_v1(
                    predicted, self._mechanical_truth(batch)
                )

            if stage.appearance_weight > 0:
                batch = self.cohort.appearance_batch(self.policy, next_step, view_index)
                local = self.model.query_evidence(planes, batch["appearance_points"])
                predicted = self.model.query_appearance(
                    local["H"], source["camera_features"][:, view_index]
                )
                rgba = F.smooth_l1_loss(
                    predicted["rgba_candidate"].float(), batch["appearance_rgba"].float()
                )
                alpha_boundary = F.binary_cross_entropy(
                    predicted["alpha_boundary_probability"].clamp(1e-6, 1 - 1e-6),
                    batch["appearance_alpha_boundary"],
                )
                appearance_boundary = F.binary_cross_entropy(
                    predicted["appearance_boundary_probability"].clamp(1e-6, 1 - 1e-6),
                    batch["appearance_boundary"],
                )
                losses["appearance"] = rgba + 0.25 * (
                    alpha_boundary + appearance_boundary
                )

            if stage.relation_weight > 0:
                batch = self.cohort.relation_batch(self.policy, next_step, view_index)
                feature_a = self.model.query_evidence(planes, batch["relation_point_a"])["H"]
                feature_b = self.model.query_evidence(planes, batch["relation_point_b"])["H"]
                static = self.model.relations.static_relations(feature_a, feature_b)
                cameras = source["camera_features"][:, view_index, None, :].expand(
                    -1, feature_a.shape[1], -1
                )
                viewed = self.model.relations.view_relations(feature_a, feature_b, cameras)
                predicted = {**static, **viewed}
                truth = {
                    key.removeprefix("relation_"): batch[key]
                    for key in _REQUIRED_RELATION[2:]
                }
                losses["relations"] = graphics_relation_loss_v1(predicted, truth)

            topology_offsets = None
            if stage.topology_weight > 0:
                if extracted is None:
                    raise RuntimeError("VNEXT_FIT8_MODEL_SURFACE_WITNESS_MISSING")
                input_ids, labels, topology_offsets = self.cohort.topology_batch(next_step)
                witness = self.model.query_evidence(planes, extracted.witness_xyz_iris)
                output = self.model.decode_topology(
                    scene_planes=planes,
                    witness_xyz=extracted.witness_xyz_tessa,
                    witness_normal=extracted.witness_normal,
                    witness_h=witness["H"],
                    witness_log_uncertainty=witness["log_uncertainty"],
                    witness_mechanical=witness["mechanical_demand"],
                    input_ids=input_ids.long(),
                    labels=labels.long(),
                    sequence_position_offset=topology_offsets.long(),
                )
                if output.loss is None:
                    raise RuntimeError("VNEXT_FIT8_TOPOLOGY_LOSS_MISSING")
                losses["topology"] = output.loss

            loss = (
                stage.field_weight * losses["field"]
                + stage.extraction_weight * losses["extraction"]
                + stage.mechanical_weight * losses["mechanical"]
                + stage.appearance_weight * losses["appearance"]
                + stage.relation_weight * losses["relations"]
                + stage.topology_weight * losses["topology"]
            )

        if not bool(torch.isfinite(loss)):
            raise RuntimeError("VNEXT_FIT8_NONFINITE_LOSS")
        loss.backward()
        self.optimizer.step()

        post_surface = None
        if stage.extraction_weight > 0 or stage.topology_weight > 0:
            try:
                post_surface = self._surface_gate()
            except Exception as exc:
                raise RuntimeError(
                    f"VNEXT_POST_UPDATE_SURFACE_GATE_FAILURE_AT_NEXT_STEP_{next_step}:{exc}"
                ) from exc

        self.step += 1
        self.stage_step += 1
        if self.step % self.policy.checkpoint_interval == 0:
            self.save()

        result = {
            "global_step": self.step,
            "stage": stage.stage_id,
            "stage_step": self.stage_step,
            "loss": float(loss.detach()),
            **{f"{key}_loss": float(value.detach()) for key, value in losses.items()},
            "view_index": view_index,
            "batch_subjects": 8,
            "shared_scene_encodes": 1,
            "subject_serial_model_forward": False,
            "optimizer_steps": 1,
            "elapsed_seconds": time.monotonic() - self.started,
            "surface_telemetry_phase": "POST_OPTIMIZER" if post_surface else None,
        }
        if topology_offsets is not None:
            result["topology_position_offsets"] = tuple(
                map(int, topology_offsets.detach().cpu().tolist())
            )
        if post_surface is not None:
            result["extracted_vertex_counts"] = post_surface["vertices"]
            result["extracted_face_counts"] = post_surface["faces"]
            result["post_update_surface_court"] = post_surface
        elif extracted is not None:
            result["extracted_vertex_counts"] = extracted.extracted_vertex_counts
            result["extracted_face_counts"] = extracted.extracted_face_counts
        return result
