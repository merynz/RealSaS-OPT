from __future__ import annotations

"""Active IRIS vNext Fit8 trainer.

V2 keeps the V1 whole-cohort optimizer/curriculum semantics and adds one missing
scientific contract: authored TESSA sequences may have different lengths.  Each
subject therefore owns a padded window bank.  One valid window per subject is
selected deterministically, including its global position offset, and all eight
windows are decoded in one B=8 forward.
"""

from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .cohort_v1 import (
    SUBJECT_IDS,
    VNextCheckpointStoreV1,
    VNextFit8CohortV1,
    VNextFit8PolicyV1,
    _REQUIRED_APPEARANCE,
    _REQUIRED_MECH,
    _REQUIRED_RELATION,
    _sha,
    _source_identity,
    _vh,
)
from .contracts_v1 import IRISVNextConfigV1
from .flexicubes_v1 import PinnedFlexiCubesV1, flexicubes_root_from_environment_v1
from .model_v1 import IRISVNextV1
from .surface_extraction_v1 import extract_surface_batch_v1, extracted_surface_loss_v1
from .training_v1 import (
    IRISVNextOptimizationPolicyV1,
    build_vnext_optimizer_v1,
    graphics_relation_loss_v1,
    mechanical_demand_loss_v1,
)


class VNextFit8CohortV2(VNextFit8CohortV1):
    """V1 tensor banks plus per-subject topology-window scheduling metadata."""

    def __init__(self, data_root, manifest, device):
        super().__init__(data_root, manifest, device)
        rows = tuple(self.manifest["subjects"])
        extra = {"topology_window_offsets": [], "topology_window_count": []}
        for row in rows:
            path = (self.root / str(row["shard_file"])).resolve()
            value = torch.load(path, map_location="cpu", weights_only=True)
            for key in extra:
                if key not in value or not isinstance(value[key], torch.Tensor):
                    raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_FIELD_MISSING:" + key)
                extra[key].append(value[key])
        for key, values in extra.items():
            if len({tuple(value.shape) for value in values}) != 1:
                raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_SHAPE_DRIFT:" + key)
            tensor = torch.stack(values, dim=0).contiguous()
            if self.device.type == "cuda":
                tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            else:
                tensor = tensor.to(self.device)
            self.tensors[key] = tensor
        self._validate_topology_windows()

    def _validate_topology_windows(self):
        ids = self.tensors["topology_input_ids"]
        labels = self.tensors["topology_labels"]
        offsets = self.tensors["topology_window_offsets"]
        counts = self.tensors["topology_window_count"]
        if ids.ndim != 3 or labels.shape != ids.shape:
            raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_BANK_SHAPE_INVALID")
        if offsets.shape != ids.shape[:2] or counts.shape != (8,):
            raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_METADATA_SHAPE_INVALID")
        if bool(torch.any(counts <= 0)) or bool(torch.any(counts > ids.shape[1])):
            raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_COUNT_INVALID")
        if bool(torch.any(offsets < 0)):
            raise ValueError("VNEXT_FIT8_TOPOLOGY_WINDOW_OFFSET_NEGATIVE")

    def topology_batch(self, step: int):
        ids = self.tensors["topology_input_ids"]
        labels = self.tensors["topology_labels"]
        offsets = self.tensors["topology_window_offsets"]
        counts = self.tensors["topology_window_count"].long()
        subject = torch.arange(8, device=self.device, dtype=torch.long)
        # Different coprime phase per subject; deterministic and checkpoint-free.
        window = (int(step) + subject * 17) % counts
        row = subject
        return ids[row, window], labels[row, window], offsets[row, window]


def _source_identity_v2(root: Path):
    value = dict(_source_identity(root))
    for path in (
        "models/iris/vnext/cohort_v2.py",
        "models/iris/topology/head_v1.py",
        "tools/iris_vnext/blender_authored_teacher_export_v1.py",
        "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
    ):
        value[path] = _sha(root / path)
    return value


class IRISVNextFit8TrainerV2:
    """One B=8 scene encode + one B=8 topology window forward + one update."""

    architecture_id = "RealSaS.IRISVNextFit8Trainer.v2"

    def __init__(
        self,
        *,
        data_root,
        manifest,
        checkpoint_root,
        device,
        model_config=IRISVNextConfigV1(),
        policy=VNextFit8PolicyV1(),
        repo_root=None,
        flexicubes_root=None,
    ):
        policy.validate()
        model_config.validate()
        self.device = torch.device(device)
        self.policy = policy
        self.cohort = VNextFit8CohortV2(data_root, manifest, self.device)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3])

        needs_surface = any(
            stage.extraction_weight > 0 or stage.topology_weight > 0
            for stage in policy.stages
        )
        self.extractor = None
        flexi_identity = None
        if needs_surface:
            root = flexicubes_root or flexicubes_root_from_environment_v1()
            self.extractor = PinnedFlexiCubesV1(
                root,
                device=self.device,
                weight_scale=model_config.flexicubes_weight_scale,
            )
            flexi_identity = self.extractor.identity

        self.binding = {
            "schema": "RealSaS.IRISVNextFit8Binding.v4",
            "source": _source_identity_v2(self.repo_root),
            "manifest_sha256": _vh(manifest),
            "policy": asdict(policy),
            "model_config": asdict(model_config),
            "flexicubes": flexi_identity,
            "optimizer_semantics": "ONE_BATCH8_SCENE_ENCODE__NO_SUBJECT_SERIAL_MODEL_FORWARD__ONE_BACKWARD__ONE_STEP",
            "topology_window_semantics": "ONE_VALID_WINDOW_PER_SUBJECT__BATCH8_DECODER_FORWARD__PER_SUBJECT_GLOBAL_POSITION_OFFSET",
            "surface_witness_semantics": "MODEL_FIELD_TO_PINNED_FLEXICUBES_TO_TESSA_CANONICAL_XYZ",
            "curriculum_semantics": "G1__G2_FLEXICUBES__M1__A1__R1__T1__J1",
            "teacher_inputs_in_inference_forward": False,
            "direct_flexicubes_parameter_teacher_targets": False,
            "teacher_surface_as_topology_witness": False,
            "legacy_5400_optimizer_state_reused": False,
        }
        self.binding_hash = _vh(self.binding)
        random.seed(policy.seed)
        np.random.seed(policy.seed)
        torch.manual_seed(policy.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(policy.seed)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.model = IRISVNextV1(model_config).to(self.device)
        self.optimizer = build_vnext_optimizer_v1(
            self.model,
            IRISVNextOptimizationPolicyV1(
                scene_learning_rate=policy.scene_learning_rate,
                evidence_learning_rate=policy.evidence_learning_rate,
                structure_learning_rate=policy.structure_learning_rate,
                weight_decay=policy.weight_decay,
                expected_cohort_size=8,
            ),
        )
        self.store = VNextCheckpointStoreV1(checkpoint_root, self.binding_hash)
        self.step = 0
        self.stage_index = 0
        self.stage_step = 0
        self.started = time.monotonic()
        self._apply_stage_lrs()
        previous = self.store.load()
        if previous:
            self._restore(previous)
        else:
            self.save()

    @property
    def stage(self):
        return self.policy.stages[self.stage_index]

    def _apply_stage_lrs(self):
        scales = {
            "scene_core": self.stage.scene_lr_scale,
            "evidence_heads": self.stage.evidence_lr_scale,
            "structure": self.stage.structure_lr_scale,
        }
        base = {
            "scene_core": self.policy.scene_learning_rate,
            "evidence_heads": self.policy.evidence_learning_rate,
            "structure": self.policy.structure_learning_rate,
        }
        for group in self.optimizer.param_groups:
            group["lr"] = base[group["name"]] * scales[group["name"]]

    def _payload(self):
        return {
            "schema": "RealSaS.IRISVNextFit8Checkpoint.v4",
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "stage_index": self.stage_index,
            "stage_step": self.stage_step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def save(self):
        return self.store.save(self._payload())

    def _restore(self, payload):
        self.step = int(payload["step"])
        self.stage_index = int(payload["stage_index"])
        self.stage_step = int(payload["stage_step"])
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self._apply_stage_lrs()

    def _advance_stage_if_needed(self):
        if self.stage_step < self.stage.steps:
            return False
        if self.stage_index + 1 >= len(self.policy.stages):
            return True
        self.stage_index += 1
        self.stage_step = 0
        self._apply_stage_lrs()
        return False

    @staticmethod
    def _mechanical_truth(batch):
        return {
            "stretch_major": batch["mechanical_stretch_major"],
            "stretch_minor": batch["mechanical_stretch_minor"],
            "tangent_direction_2theta": batch["mechanical_tangent_direction_2theta"],
            "bend_demand": batch["mechanical_bend_demand"],
            "rigidity_probability": batch["mechanical_rigidity_probability"],
        }

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
        }
        if topology_offsets is not None:
            result["topology_position_offsets"] = tuple(
                map(int, topology_offsets.detach().cpu().tolist())
            )
        if extracted is not None:
            result["extracted_vertex_counts"] = extracted.extracted_vertex_counts
            result["extracted_face_counts"] = extracted.extracted_face_counts
        return result
