from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

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

SUBJECT_IDS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _vh(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_identity(root: Path):
    paths = (
        "models/iris/vnext/contracts_v1.py",
        "models/iris/vnext/scene_core_v1.py",
        "models/iris/vnext/flexicubes_v1.py",
        "models/iris/vnext/surface_extraction_v1.py",
        "models/iris/vnext/heads_v1.py",
        "models/iris/vnext/structure_v1.py",
        "models/iris/vnext/model_v1.py",
        "models/iris/vnext/teacher_truth_v1.py",
        "models/iris/vnext/graphics_truth_v1.py",
        "models/iris/vnext/training_v1.py",
        "models/iris/vnext/cohort_v1.py",
        "canonical/VNEXT_SINGLE_CARRIER_CONTRACT_V1.json",
        "canonical/VNEXT_EXECUTION_GRAPH_V1.json",
    )
    return {path: _sha(root / path) for path in paths}


@dataclass(frozen=True)
class VNextStageV1:
    stage_id: str
    steps: int
    scene_lr_scale: float = 1.0
    evidence_lr_scale: float = 1.0
    structure_lr_scale: float = 0.0
    field_weight: float = 0.0
    extraction_weight: float = 0.0
    mechanical_weight: float = 0.0
    appearance_weight: float = 0.0
    relation_weight: float = 0.0
    topology_weight: float = 0.0
    minimum_lr_ratio: float = 0.1

    def validate(self):
        values = (
            self.scene_lr_scale,
            self.evidence_lr_scale,
            self.structure_lr_scale,
            self.field_weight,
            self.extraction_weight,
            self.mechanical_weight,
            self.appearance_weight,
            self.relation_weight,
            self.topology_weight,
            self.minimum_lr_ratio,
        )
        if self.steps <= 0 or min(values) < 0 or self.minimum_lr_ratio > 1:
            raise ValueError("VNEXT_STAGE_POLICY_INVALID")
        if sum(values[3:-1]) <= 0:
            raise ValueError("VNEXT_STAGE_HAS_NO_OBJECTIVE")


@dataclass(frozen=True)
class VNextFit8PolicyV1:
    seed: int = 8162026
    geometry_samples_per_subject: int = 4096
    mechanical_samples_per_subject: int = 2048
    appearance_samples_per_subject: int = 2048
    relation_pairs_per_subject: int = 1024
    extraction_l_dev_weight: float = 0.01
    extraction_flexible_weight_l1: float = 1.0e-4
    weight_decay: float = 0.01
    scene_learning_rate: float = 3e-5
    evidence_learning_rate: float = 1e-4
    structure_learning_rate: float = 5e-5
    checkpoint_interval: int = 100
    eval_interval: int = 100
    expected_subjects: tuple[str, ...] = SUBJECT_IDS
    stages: tuple[VNextStageV1, ...] = (
        VNextStageV1("G1_GEOMETRY", 1600, field_weight=1.0),
        VNextStageV1(
            "G2_FLEXICUBES_SURFACE",
            1200,
            scene_lr_scale=0.75,
            evidence_lr_scale=0.5,
            extraction_weight=1.0,
        ),
        VNextStageV1(
            "M1_MECHANICAL_DEMAND",
            1600,
            scene_lr_scale=0.5,
            evidence_lr_scale=1.0,
            field_weight=0.25,
            mechanical_weight=1.0,
        ),
        VNextStageV1(
            "A1_APPEARANCE_EVIDENCE",
            1200,
            scene_lr_scale=0.5,
            evidence_lr_scale=1.0,
            appearance_weight=1.0,
        ),
        VNextStageV1(
            "R1_ROLE_FREE_RELATIONS",
            1600,
            scene_lr_scale=0.5,
            evidence_lr_scale=1.0,
            relation_weight=1.0,
        ),
        VNextStageV1(
            "T1_MECHANICAL_STRUCTURE",
            2400,
            scene_lr_scale=0.35,
            evidence_lr_scale=0.5,
            structure_lr_scale=1.0,
            topology_weight=1.0,
            mechanical_weight=0.25,
        ),
        VNextStageV1(
            "J1_JOINT_FINETUNE",
            2400,
            scene_lr_scale=0.25,
            evidence_lr_scale=0.5,
            structure_lr_scale=0.5,
            field_weight=0.25,
            extraction_weight=0.25,
            mechanical_weight=0.5,
            appearance_weight=0.25,
            relation_weight=0.5,
            topology_weight=0.75,
        ),
    )

    def validate(self):
        counts = (
            self.geometry_samples_per_subject,
            self.mechanical_samples_per_subject,
            self.appearance_samples_per_subject,
            self.relation_pairs_per_subject,
            self.checkpoint_interval,
            self.eval_interval,
        )
        if tuple(self.expected_subjects) != SUBJECT_IDS or min(counts) <= 0:
            raise ValueError("VNEXT_FIT8_POLICY_INVALID")
        if min(self.extraction_l_dev_weight, self.extraction_flexible_weight_l1) < 0:
            raise ValueError("VNEXT_FIT8_EXTRACTION_REGULARIZER_INVALID")
        for stage in self.stages:
            stage.validate()

    @property
    def total_steps(self):
        return sum(stage.steps for stage in self.stages)


_REQUIRED_SOURCE = ("image_tokens", "token_xy", "camera_features")
_REQUIRED_GEOMETRY = ("geometry_points", "geometry_targets")
_REQUIRED_MECH = (
    "mechanical_points",
    "mechanical_normals",
    "mechanical_stretch_major",
    "mechanical_stretch_minor",
    "mechanical_tangent_direction_2theta",
    "mechanical_bend_demand",
    "mechanical_rigidity_probability",
)
_REQUIRED_EXTRACTION = ("teacher_surface_points_iris",)
_REQUIRED_APPEARANCE = (
    "appearance_points",
    "appearance_rgba",
    "appearance_alpha_boundary",
    "appearance_boundary",
)
_REQUIRED_RELATION = (
    "relation_point_a",
    "relation_point_b",
    "relation_same_drawable_affinity",
    "relation_cut_probability",
    "relation_attachment_affinity",
    "relation_alpha_boundary_probability",
    "relation_appearance_boundary_probability",
    "relation_front_probability",
    "relation_occlusion_probability",
    "relation_visibility_probability_a",
    "relation_visibility_probability_b",
)
_REQUIRED_TOPOLOGY = ("topology_input_ids", "topology_labels")
_REQUIRED_ALL = (
    _REQUIRED_SOURCE
    + _REQUIRED_GEOMETRY
    + _REQUIRED_MECH
    + _REQUIRED_EXTRACTION
    + _REQUIRED_APPEARANCE
    + _REQUIRED_RELATION
    + _REQUIRED_TOPOLOGY
)


class VNextFit8CohortV1:
    """Eight immutable shards resident as one cohort; no subject-serial model forward."""

    def __init__(self, data_root, manifest, device):
        self.root = Path(data_root).resolve()
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        rows = tuple(self.manifest.get("subjects") or ())
        ids = tuple(str(row["subject_id"]) for row in rows)
        if ids != SUBJECT_IDS:
            raise ValueError("VNEXT_FIT8_SUBJECT_ORDER_OR_ID_DRIFT")
        if self.manifest.get("teacher_inputs_in_inference_forward") is not False:
            raise ValueError("VNEXT_FIT8_TEACHER_FORWARD_FORBIDDEN")

        shards = []
        for row in rows:
            path = (self.root / str(row["shard_file"])).resolve()
            if not path.is_relative_to(self.root) or _sha(path) != str(row["shard_sha256"]):
                raise ValueError("VNEXT_FIT8_SHARD_HASH_OR_PATH_DRIFT")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if str(value.get("subject_id")) != str(row["subject_id"]):
                raise ValueError("VNEXT_FIT8_SHARD_SUBJECT_DRIFT")
            if value.get("teacher_inputs_in_inference_forward") is not False:
                raise ValueError("VNEXT_FIT8_SHARD_TEACHER_FORWARD_FORBIDDEN")
            if "cube_extraction_targets" in value or "topology_witness_xyz" in value:
                raise ValueError("VNEXT_FIT8_FORBIDDEN_TEACHER_INTERNAL_TARGET")
            for key in _REQUIRED_ALL:
                if key not in value or not isinstance(value[key], torch.Tensor):
                    raise ValueError("VNEXT_FIT8_SHARD_FIELD_MISSING:" + key)
            shards.append(value)

        def stack(key):
            values = []
            for value in shards:
                tensor = value[key]
                if key in _REQUIRED_SOURCE and tensor.ndim > 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                values.append(tensor)
            if len({tuple(value.shape) for value in values}) != 1:
                raise ValueError("VNEXT_FIT8_BATCH_SHAPE_DRIFT:" + key)
            out = torch.stack(values, dim=0).contiguous()
            if self.device.type == "cuda":
                out = out.pin_memory().to(self.device, non_blocking=True)
            else:
                out = out.to(self.device)
            return out

        self.tensors = {key: stack(key) for key in _REQUIRED_ALL}
        self._validate_shapes()

    def _validate_shapes(self):
        t = self.tensors
        if t["image_tokens"].shape[0] != 8:
            raise AssertionError("VNEXT_FIT8_NOT_BATCH8")
        if t["geometry_targets"].shape != t["geometry_points"].shape[:2]:
            raise ValueError("VNEXT_FIT8_GEOMETRY_TARGET_SHAPE")
        if t["teacher_surface_points_iris"].ndim != 3 or t["teacher_surface_points_iris"].shape[-1] != 3:
            raise ValueError("VNEXT_FIT8_TEACHER_SURFACE_SHAPE")
        if float(t["teacher_surface_points_iris"].abs().max()) > 1.000001:
            raise ValueError("VNEXT_FIT8_TEACHER_SURFACE_DOMAIN_DRIFT")
        if t["mechanical_normals"].shape != t["mechanical_points"].shape:
            raise ValueError("VNEXT_FIT8_MECHANICAL_NORMAL_SHAPE")
        mshape = t["mechanical_points"].shape[:2]
        for key in (
            "mechanical_stretch_major",
            "mechanical_stretch_minor",
            "mechanical_bend_demand",
            "mechanical_rigidity_probability",
        ):
            if t[key].shape != mshape:
                raise ValueError("VNEXT_FIT8_MECHANICAL_TARGET_SHAPE:" + key)
        if t["mechanical_tangent_direction_2theta"].shape != mshape + (2,):
            raise ValueError("VNEXT_FIT8_MECHANICAL_DIRECTION_SHAPE")
        if t["appearance_points"].ndim != 4 or t["appearance_points"].shape[1] != 8:
            raise ValueError("VNEXT_FIT8_APPEARANCE_VIEW_SHAPE")
        ashape = t["appearance_points"].shape[:3]
        if t["appearance_rgba"].shape != ashape + (4,):
            raise ValueError("VNEXT_FIT8_APPEARANCE_RGBA_SHAPE")
        for key in ("appearance_alpha_boundary", "appearance_boundary"):
            if t[key].shape != ashape:
                raise ValueError("VNEXT_FIT8_APPEARANCE_TARGET_SHAPE:" + key)
        if t["relation_point_a"].shape != t["relation_point_b"].shape:
            raise ValueError("VNEXT_FIT8_RELATION_POINT_SHAPE")
        if t["relation_point_a"].ndim != 4 or t["relation_point_a"].shape[1] != 8:
            raise ValueError("VNEXT_FIT8_RELATION_VIEW_SHAPE")
        rshape = t["relation_point_a"].shape[:3]
        for key in _REQUIRED_RELATION[2:]:
            if t[key].shape != rshape:
                raise ValueError("VNEXT_FIT8_RELATION_TARGET_SHAPE:" + key)
        if t["topology_input_ids"].shape != t["topology_labels"].shape:
            raise ValueError("VNEXT_FIT8_TOPOLOGY_TOKEN_SHAPE")

    @staticmethod
    def _indices(bank_count, sample_count, seed, step, device):
        if bank_count < sample_count:
            raise ValueError("VNEXT_FIT8_BANK_TOO_SMALL")
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(step) * 1000003)
        index = torch.stack(
            [torch.randperm(bank_count, generator=generator)[:sample_count] for _ in range(8)],
            dim=0,
        )
        return index.to(device)

    @staticmethod
    def _gather(bank, index):
        tail = bank.shape[2:]
        view = index.reshape(index.shape + (1,) * len(tail)).expand(index.shape + tail)
        return torch.gather(bank, 1, view)

    @classmethod
    def _gather_view(cls, bank, view_index, index):
        return cls._gather(bank[:, int(view_index)], index)

    def geometry_batch(self, policy, step):
        index = self._indices(
            self.tensors["geometry_points"].shape[1],
            policy.geometry_samples_per_subject,
            policy.seed + 11,
            step,
            self.device,
        )
        return (
            self._gather(self.tensors["geometry_points"], index),
            self._gather(self.tensors["geometry_targets"], index),
        )

    def mechanical_batch(self, policy, step):
        index = self._indices(
            self.tensors["mechanical_points"].shape[1],
            policy.mechanical_samples_per_subject,
            policy.seed + 23,
            step,
            self.device,
        )
        return {key: self._gather(self.tensors[key], index) for key in _REQUIRED_MECH}

    def appearance_batch(self, policy, step, view_index):
        bank = self.tensors["appearance_points"].shape[2]
        index = self._indices(
            bank, policy.appearance_samples_per_subject, policy.seed + 41, step, self.device
        )
        return {
            key: self._gather_view(self.tensors[key], view_index, index)
            for key in _REQUIRED_APPEARANCE
        }

    def relation_batch(self, policy, step, view_index):
        bank = self.tensors["relation_point_a"].shape[2]
        index = self._indices(
            bank, policy.relation_pairs_per_subject, policy.seed + 53, step, self.device
        )
        return {
            key: self._gather_view(self.tensors[key], view_index, index)
            for key in _REQUIRED_RELATION
        }


class VNextCheckpointStoreV1:
    def __init__(self, root, binding_hash):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding_hash = binding_hash

    def save(self, payload):
        step = int(payload["step"])
        name = f"step_{step:08d}_{uuid.uuid4().hex}.pt"
        tmp = self.root / (name + ".staged")
        torch.save(payload, tmp)
        digest = _sha(tmp)
        torch.load(tmp, map_location="cpu", weights_only=True)
        target = self.root / name
        os.replace(tmp, target)
        pointer = {
            "schema": "RealSaS.IRISVNextFit8CheckpointPointer.v2",
            "file": name,
            "sha256": digest,
            "step": step,
            "binding_sha256": self.binding_hash,
        }
        staged = self.root / "LATEST.json.staged"
        staged.write_text(json.dumps(pointer, sort_keys=True, indent=2))
        os.replace(staged, self.root / "LATEST.json")
        return pointer

    def load(self):
        pointer_path = self.root / "LATEST.json"
        if not pointer_path.exists():
            return None
        row = json.loads(pointer_path.read_text())
        if row["binding_sha256"] != self.binding_hash:
            raise ValueError("VNEXT_RESUME_BINDING_DRIFT")
        path = self.root / Path(row["file"]).name
        if _sha(path) != row["sha256"]:
            raise ValueError("VNEXT_RESUME_PAYLOAD_HASH_DRIFT")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["binding_sha256"] != self.binding_hash or int(value["step"]) != int(row["step"]):
            raise ValueError("VNEXT_RESUME_PAYLOAD_BINDING_DRIFT")
        return value


class IRISVNextFit8TrainerV1:
    """One optimizer step has one B=8 scene encode and no subject-serial model loop."""

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
        self.cohort = VNextFit8CohortV1(data_root, manifest, self.device)
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
            "schema": "RealSaS.IRISVNextFit8Binding.v3",
            "source": _source_identity(self.repo_root),
            "manifest_sha256": _vh(manifest),
            "policy": asdict(policy),
            "model_config": asdict(model_config),
            "flexicubes": flexi_identity,
            "optimizer_semantics": "ONE_BATCH8_SCENE_ENCODE__NO_SUBJECT_SERIAL_MODEL_FORWARD__ONE_BACKWARD__ONE_STEP",
            "surface_witness_semantics": "MODEL_FIELD_TO_PINNED_FLEXICUBES_TO_TESSA_CANONICAL_XYZ",
            "curriculum_semantics": "G1__G2_FLEXICUBES__M1__A1__R1__T1__J1",
            "teacher_inputs_in_inference_forward": False,
            "direct_flexicubes_parameter_teacher_targets": False,
            "teacher_surface_as_topology_witness": False,
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
        optimizer_policy = IRISVNextOptimizationPolicyV1(
            scene_learning_rate=policy.scene_learning_rate,
            evidence_learning_rate=policy.evidence_learning_rate,
            structure_learning_rate=policy.structure_learning_rate,
            weight_decay=policy.weight_decay,
            expected_cohort_size=8,
        )
        self.optimizer = build_vnext_optimizer_v1(self.model, optimizer_policy)
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
            "schema": "RealSaS.IRISVNextFit8Checkpoint.v3",
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
                    extraction_losses = extracted_surface_loss_v1(
                        extracted,
                        source["teacher_surface_points_iris"],
                        l_dev_weight=self.policy.extraction_l_dev_weight,
                        flexible_weight_l1=self.policy.extraction_flexible_weight_l1,
                    )
                    losses["extraction"] = extraction_losses["total"]
                    losses["extraction_chamfer"] = extraction_losses["surface_chamfer"]
                    losses["extraction_l_dev"] = extraction_losses["l_dev"]

            if stage.mechanical_weight > 0:
                mechanical_batch = self.cohort.mechanical_batch(self.policy, next_step)
                predicted = self.model.query_evidence(
                    planes, mechanical_batch["mechanical_points"]
                )["mechanical_demand"]
                losses["mechanical"] = mechanical_demand_loss_v1(
                    predicted, self._mechanical_truth(mechanical_batch)
                )

            if stage.appearance_weight > 0:
                appearance = self.cohort.appearance_batch(
                    self.policy, next_step, view_index
                )
                local = self.model.query_evidence(planes, appearance["appearance_points"])
                predicted = self.model.query_appearance(
                    local["H"], source["camera_features"][:, view_index]
                )
                rgba = F.smooth_l1_loss(
                    predicted["rgba_candidate"].float(), appearance["appearance_rgba"].float()
                )
                alpha_boundary = F.binary_cross_entropy(
                    predicted["alpha_boundary_probability"].clamp(1e-6, 1 - 1e-6),
                    appearance["appearance_alpha_boundary"],
                )
                appearance_boundary = F.binary_cross_entropy(
                    predicted["appearance_boundary_probability"].clamp(1e-6, 1 - 1e-6),
                    appearance["appearance_boundary"],
                )
                losses["appearance"] = rgba + 0.25 * (
                    alpha_boundary + appearance_boundary
                )

            if stage.relation_weight > 0:
                relation = self.cohort.relation_batch(self.policy, next_step, view_index)
                feature_a = self.model.query_evidence(planes, relation["relation_point_a"])["H"]
                feature_b = self.model.query_evidence(planes, relation["relation_point_b"])["H"]
                static = self.model.relations.static_relations(feature_a, feature_b)
                cameras = source["camera_features"][:, view_index, None, :].expand(
                    -1, feature_a.shape[1], -1
                )
                viewed = self.model.relations.view_relations(feature_a, feature_b, cameras)
                predicted = {**static, **viewed}
                truth = {
                    key.removeprefix("relation_"): relation[key]
                    for key in _REQUIRED_RELATION[2:]
                }
                losses["relations"] = graphics_relation_loss_v1(predicted, truth)

            if stage.topology_weight > 0:
                if extracted is None:
                    raise RuntimeError("VNEXT_FIT8_MODEL_SURFACE_WITNESS_MISSING")
                witness = self.model.query_evidence(planes, extracted.witness_xyz_iris)
                output = self.model.decode_topology(
                    scene_planes=planes,
                    witness_xyz=extracted.witness_xyz_tessa,
                    witness_normal=extracted.witness_normal,
                    witness_h=witness["H"],
                    witness_log_uncertainty=witness["log_uncertainty"],
                    witness_mechanical=witness["mechanical_demand"],
                    input_ids=source["topology_input_ids"].long(),
                    labels=source["topology_labels"].long(),
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
        if extracted is not None:
            result["extracted_vertex_counts"] = extracted.extracted_vertex_counts
            result["extracted_face_counts"] = extracted.extracted_face_counts
        return result
