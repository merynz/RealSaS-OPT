"""One image-only IRIS, jointly optimized over a hash-bound variable-size cohort.

Production semantics intentionally mirror the post-SDF-head Knight C lineage:
1) cold direct-F* stage,
2) fresh-optimizer C (direct F* + source silhouette) stage,
3) fresh-optimizer low-LR C continuation.

A "step" in this module is a GLOBAL optimizer update. Every global update receives
one equally weighted gradient contribution from every cohort subject before the
single optimizer.step(). Subject exposures are therefore N_subjects * global_steps.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

from models.iris.v4.scene_first_signed_v4 import (
    SceneFirstSignedGeometryV4,
    SceneFirstSignedGeometryConfigV4,
)
from models.iris.v5.source_signed_field_v5 import (
    source_signed_field_torch_v5,
    SourceSignedFieldPolicyV5,
)
from models.iris.v5.direct_fstar_fit_v5 import (
    direct_fstar_metrics_v5,
    direct_fstar_loss_v5,
    DirectFStarFitPolicyV5,
)
from models.iris.training.apparatus_v1 import (
    training_queries,
    training_queries_from_rng,
    training_rays,
)
from models.iris.v5.objective_tail_repair_v5 import (
    near_zero_wrong_sign_loss_v5,
    source_silhouette_wrong_sign_loss_v5,
)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".staged." + uuid.uuid4().hex)
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _validate_stage_schedule(policy, reference):
    expected = reference.get("shared_stage_schedule")
    actual = policy.get("training_stages")
    if expected is None:
        return
    if actual != expected:
        raise ValueError("KNIGHT_RECIPE_STAGE_SCHEDULE_DRIFT")
    if sum(int(row["steps"]) for row in actual) != int(policy["steps"]):
        raise ValueError("KNIGHT_RECIPE_TOTAL_STEP_DRIFT")


def validate_knight_recipe(policy, source):
    """Reject silent core/config/recipe drift before creating optimizer state."""
    if "knight_recipe_reference" not in policy:
        return
    root = Path(__file__).resolve().parents[3]
    reference = json.loads((root / policy["knight_recipe_reference"]).read_text())
    for path, digest in reference["core_source_sha256"].items():
        if file_hash(root / path) != digest or (source and source.get(path) != digest):
            raise ValueError("KNIGHT_RECIPE_CORE_SOURCE_DRIFT:" + path)
    for name, expected in reference["frozen_controls"].items():
        if policy.get(name) != expected:
            raise ValueError("KNIGHT_RECIPE_POLICY_DRIFT:" + name)
    if policy["model_config"] != reference["model_config"]:
        raise ValueError("KNIGHT_RECIPE_MODEL_CONFIG_DRIFT")
    if "engineering_eval_query_limit" in policy and not policy.get("scope", "").startswith(
        "RUNTIME_PREFLIGHT"
    ):
        raise ValueError("PRODUCTION_HELDOUT_SUBSET_FORBIDDEN")
    if not policy.get("scope", "").startswith("RUNTIME_PREFLIGHT"):
        _validate_stage_schedule(policy, reference)
        if policy.get("optimizer_update_semantics") != (
            "MEAN_GRADIENT_OVER_ALL_SUBJECTS_PER_GLOBAL_STEP"
        ):
            raise ValueError("KNIGHT_RECIPE_OPTIMIZER_UPDATE_SEMANTICS_DRIFT")
        if int(policy.get("subject_exposures_per_global_step", 0)) != int(
            policy["subject_count"]
        ):
            raise ValueError("KNIGHT_RECIPE_SUBJECT_EXPOSURE_SEMANTICS_DRIFT")


def source_identity(root):
    root = Path(root)
    paths = [
        "models/iris/training/cohort_v1.py",
        "models/iris/v4/scene_first_signed_v4.py",
        "models/iris/v2/dinov2_foundation_v2.py",
        "models/iris/v5/source_signed_field_v5.py",
        "models/iris/v5/direct_fstar_fit_v5.py",
        "models/iris/v5/objective_tail_repair_v5.py",
        "models/iris/training/apparatus_v1.py",
        "canonical/FIT8_IRIS_KNIGHT_RECIPE_REFERENCE_V1.json",
    ]
    return {p: file_hash(root / p) for p in paths}


class AtomicCheckpointStore:
    """Immutable payload first, verified pointer last. Orphans cannot become latest."""

    def __init__(self, root, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = binding

    def save(self, payload):
        step = int(payload["step"])
        name = f"step_{step:08d}_{uuid.uuid4().hex}.pt"
        staged = self.root / (name + ".staged")
        local_root = (
            Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        )
        with tempfile.NamedTemporaryFile(
            dir=local_root, prefix="realsas_cp_", suffix=".pt"
        ) as local:
            torch.save(payload, local)
            local.flush()
            os.fsync(local.fileno())
            digest = file_hash(local.name)
            torch.load(local.name, map_location="cpu", weights_only=True)
            with open(local.name, "rb") as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if file_hash(staged) != digest:
                raise RuntimeError("CHECKPOINT_COPY_CORRUPTED")
        target = self.root / name
        os.replace(staged, target)
        if file_hash(target) != digest:
            raise RuntimeError("CHECKPOINT_COPY_CORRUPTED")
        pointer = {
            "schema": "RealSaS.SharedFitCheckpointPointer.v1",
            "file": name,
            "sha256": digest,
            "step": step,
            "binding_sha256": self.binding,
        }
        latest = self.root / "LATEST.json"
        if latest.exists():
            atomic_json(self.root / "PREVIOUS.json", json.loads(latest.read_text()))
        atomic_json(latest, pointer)
        keep = {name}
        previous = self.root / "PREVIOUS.json"
        if previous.exists():
            keep.add(json.loads(previous.read_text())["file"])
        for old in self.root.glob("step_*.pt"):
            if old.name not in keep:
                old.unlink()
        return pointer

    def load(self):
        latest = self.root / "LATEST.json"
        if not latest.exists():
            return None
        pointer = json.loads(latest.read_text())
        if pointer["binding_sha256"] != self.binding:
            raise ValueError("RESUME_CODE_INPUT_POLICY_DRIFT")
        name = pointer["file"]
        if Path(name).name != name:
            raise ValueError("RESUME_PATH_ESCAPE")
        path = self.root / name
        if file_hash(path) != pointer["sha256"]:
            raise ValueError("RESUME_PAYLOAD_HASH_DRIFT")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["binding_sha256"] != self.binding
            or int(payload["step"]) != int(pointer["step"])
        ):
            raise ValueError("RESUME_PAYLOAD_BINDING_DRIFT")
        return payload


class SubjectCache:
    def __init__(self, root, manifest, device, maximum=2):
        self.root, self.manifest, self.device = Path(root), manifest, device
        self.maximum, self.cache = maximum, OrderedDict()

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.manifest["subjects"][index]
        path = (self.root / row["shard_file"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or file_hash(path) != row["shard_sha256"]:
            raise ValueError("SUBJECT_SHARD_HASH_OR_PATH_DRIFT")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (
            data["subject_id"] != row["subject_id"]
            or data["observation_binding"] != row["observation_binding"]
        ):
            raise ValueError("SUBJECT_OBSERVATION_BINDING_DRIFT")
        data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        self.cache[index] = data
        while len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return data


def field_targets(subject, points):
    return source_signed_field_torch_v5(
        points,
        signed_2d=subject["signed_2d"],
        center_xyz=subject["center_xyz"],
        normalization_half_extent=subject["normalization_half_extent"],
        camera_origins=subject["camera_origins"],
        camera_right=subject["camera_right"],
        camera_screen_up=subject["camera_screen_up"],
        camera_half_extent=subject["camera_half_extent"],
        policy=SourceSignedFieldPolicyV5(required_resolution=subject["resolution"]),
    )


def rng_state(generator):
    ns = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
        "queries": generator.get_state(),
    }


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state(
        (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
    )
    generator.set_state(state["queries"])


def _cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _stage_total_steps(policy):
    return sum(int(row["steps"]) for row in policy["training_stages"])


def _stage_prefix(policy, stage_index):
    return sum(int(row["steps"]) for row in policy["training_stages"][:stage_index])


class SharedIrisTrainer:
    """Joint Fit8 trainer with one equally weighted cohort update per global step."""

    def __init__(
        self,
        data_root,
        manifest,
        policy,
        source,
        checkpoint_root,
        output_root,
        device,
        persistent_result_root=None,
    ):
        self.device = torch.device(device)
        self.manifest, self.policy = manifest, policy
        validate_knight_recipe(policy, source)
        ids = [r["subject_id"] for r in manifest["subjects"]]
        if len(ids) != policy["subject_count"] or len(set(ids)) != len(ids) or not ids:
            raise ValueError("COHORT_CARDINALITY_OR_DUPLICATE")
        if manifest.get("teacher_inputs_in_forward") is not False:
            raise ValueError("COHORT_FORWARD_CONTRACT")
        if _stage_total_steps(policy) != int(policy["steps"]):
            raise ValueError("TRAINING_STAGE_TOTAL_STEP_DRIFT")
        self.binding = {
            "schema": "RealSaS.SharedIrisFitBinding.v1",
            "source": source,
            "manifest_sha256": value_hash(manifest),
            "policy_sha256": value_hash(policy),
            "subject_ids": ids,
            "device_class": self.device.type,
            "optimizer_update_semantics": policy["optimizer_update_semantics"],
        }
        self.binding_hash = value_hash(self.binding)
        seed = policy["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.model = SceneFirstSignedGeometryV4(
            SceneFirstSignedGeometryConfigV4(**policy["model_config"])
        ).to(self.device)
        if "initial_sdf_bias" in policy:
            with torch.no_grad():
                self.model.surface_field.sdf_head.bias.fill_(policy["initial_sdf_bias"])
        self.cache = SubjectCache(data_root, manifest, self.device)
        self.store = AtomicCheckpointStore(checkpoint_root, self.binding_hash)
        self.output = Path(output_root)
        self.output.mkdir(parents=True, exist_ok=True)
        self.persistent_results = (
            Path(persistent_result_root) if persistent_result_root else None
        )

        self.step = 0  # executed global optimizer updates
        self.counts = [0] * len(ids)  # one exposure per subject per global update
        self.trace = []
        self.evaluations = []
        self.completed_stages = []
        self.stage_index = 0
        self.stage_step = 0
        self.lineage_start_effective_step = 0
        self.stage_best_score = None
        self.stage_best_state = None
        self.final_best_state = None
        self.elapsed_before = 0.0
        self.started = time.monotonic()
        self.log_window = []
        self.optimizer = None
        self.scheduler = None

        # Direct stage reproduces the historical stateful train RNG shape, separately
        # for each new subject. Later C stages use the historical stateless schedules.
        self.direct_rngs = [
            np.random.default_rng(seed + 1001 + i * 1_000_000) for i in range(len(ids))
        ]
        self.replay_banks = self._rank_replay_banks()
        self._configure_optimizer()

        previous = self.store.load()
        if previous:
            self._restore(previous)
            print(
                f"[RESUME] verified_global_step={self.step} stage={self.stage['id']} "
                f"stage_step={self.stage_step} shared_models=1 active_shared_optimizers=1",
                flush=True,
            )
        else:
            self.save()

    @property
    def stage(self):
        return self.policy["training_stages"][self.stage_index]

    def _configure_optimizer(self):
        stage = self.stage
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(stage["learning_rate"]),
            weight_decay=float(self.policy["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(stage["steps"]),
            eta_min=float(stage["minimum_learning_rate"]),
        )

    @torch.no_grad()
    def _rank_replay_banks(self):
        banks = []
        self.model.eval()
        for index in range(len(self.counts)):
            s = self.cache.get(index)
            if (
                s.get("ray_court", {}).get("status") != "PASS"
                or s["ray_court"]["depth_samples"]
                != self.policy["silhouette_depth_samples"]
            ):
                raise ValueError("TRAINING_SOURCE_RAY_APPARATUS_UNCERTIFIED")
            if self.policy.get("knight_recipe_reference") and not self.policy.get(
                "scope", ""
            ).startswith("RUNTIME_PREFLIGHT"):
                census = s.get("source_exterior_census", {})
                if (
                    census.get("resolution") != 512
                    or census.get("source_only_census") is not True
                    or census.get("certified_exterior_count", 0) <= 0
                ):
                    raise ValueError("PREFIT_R512_SOURCE_CENSUS_REQUIRED")
            planes = self.model.encode_scene(
                s["image_tokens"], s["token_xy"], s["camera_features"]
            )
            pred = torch.cat(
                [
                    self.model.query(planes.float(), q[None])["sdf"].reshape(-1)
                    for q in s["replay_points"].split(2048)
                ]
            )
            err = (pred - s["replay_targets"]).abs()
            ids = torch.argsort(err, descending=True, stable=True)[
                : min(self.policy["hard_replay_bank_size"], len(err))
            ].cpu()
            banks.append(ids)
        return banks

    def _direct_rng_states(self):
        return [rng.bit_generator.state for rng in self.direct_rngs]

    def _restore_direct_rng_states(self, states):
        self.direct_rngs = []
        for state in states:
            rng = np.random.default_rng()
            rng.bit_generator.state = state
            self.direct_rngs.append(rng)

    def payload(self):
        return {
            "schema": "RealSaS.SharedIrisTrainingCheckpoint.v1",
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng_state(self.generator),
            "direct_rng_states": self._direct_rng_states(),
            "step": self.step,
            "counts": self.counts,
            "trace": self.trace,
            "evaluations": self.evaluations,
            "replay_banks": self.replay_banks,
            "stage_index": self.stage_index,
            "stage_step": self.stage_step,
            "lineage_start_effective_step": self.lineage_start_effective_step,
            "stage_best_score": self.stage_best_score,
            "stage_best_state": self.stage_best_state,
            "final_best_state": self.final_best_state,
            "completed_stages": self.completed_stages,
            "log_window": self.log_window,
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
        }

    def _restore(self, previous):
        self.step = int(previous["step"])
        self.counts = list(previous["counts"])
        self.trace = list(previous["trace"])
        self.evaluations = list(previous["evaluations"])
        self.replay_banks = previous["replay_banks"]
        self.stage_index = int(previous["stage_index"])
        self.stage_step = int(previous["stage_step"])
        self.lineage_start_effective_step = int(
            previous["lineage_start_effective_step"]
        )
        self.stage_best_score = previous["stage_best_score"]
        self.stage_best_state = previous["stage_best_state"]
        self.final_best_state = previous.get("final_best_state")
        self.completed_stages = list(previous["completed_stages"])
        self.log_window = list(previous.get("log_window", []))
        self.elapsed_before = float(previous["elapsed_seconds"])
        self.model.load_state_dict(previous["model"])
        self._configure_optimizer()
        self.optimizer.load_state_dict(previous["optimizer"])
        self.scheduler.load_state_dict(previous["scheduler"])
        restore_rng(previous["rng"], self.generator)
        self._restore_direct_rng_states(previous["direct_rng_states"])

    def save(self):
        pointer = self.store.save(self.payload())
        print(
            f"[CHECKPOINT] global_step={pointer['step']} stage={self.stage['id']} "
            f"stage_step={self.stage_step} sha256={pointer['sha256'][:12]}",
            flush=True,
        )

    def _queries_for_subject(self, index, subject, next_stage_step):
        mode = self.stage["query_schedule"]
        if mode == "STATEFUL_DIRECT_V5":
            return training_queries_from_rng(
                subject,
                self.policy,
                self.direct_rngs[index],
                self.replay_banks[index],
                self.device,
            )
        if mode == "STATELESS_C_V5":
            return training_queries(
                subject,
                self.policy,
                int(self.policy["query_schedule_seed"]) + index * 1_000_000,
                int(next_stage_step),
                self.replay_banks[index],
                self.device,
            )
        raise ValueError("UNKNOWN_QUERY_SCHEDULE:" + str(mode))

    def _subject_loss(self, index, next_stage_step):
        subject = self.cache.get(index)
        points = self._queries_for_subject(index, subject, next_stage_step)
        with torch.no_grad():
            target = field_targets(subject, points)
        with torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            planes = self.model.encode_scene(
                subject["image_tokens"], subject["token_xy"], subject["camera_features"]
            )
            prediction = (
                self.model.query(planes, points[None])["sdf"]
                .float()
                .reshape(-1)
            )
        huber = direct_fstar_loss_v5(
            prediction,
            target,
            policy=DirectFStarFitPolicyV5(huber_beta=self.policy["huber_beta"]),
        )["total"]
        sign_report = near_zero_wrong_sign_loss_v5(
            prediction, target, band=self.policy["near_zero_band"]
        )
        sign = sign_report["total"]
        silhouette = huber * 0.0
        silhouette_report = {
            "wrong_ray_count": 0,
            "wrong_ray_fraction": torch.zeros((), device=prediction.device),
        }
        silhouette_weight = float(self.stage["silhouette_loss_weight"])
        if silhouette_weight > 0.0:
            ray_count = self.policy["silhouette_rays_per_step"]
            ray_q, foreground = training_rays(
                subject,
                self.policy,
                int(self.policy["ray_schedule_seed"]) + index * 1_000_000,
                int(next_stage_step),
                self.device,
            )
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                ray_f = self.model.query(
                    planes, ray_q.reshape(1, -1, 3)
                )["sdf"].reshape(ray_count, -1)
            silhouette_report = source_silhouette_wrong_sign_loss_v5(
                ray_f, foreground, tau=self.policy["silhouette_softmin_tau"]
            )
            silhouette = silhouette_report["total"]
        loss = (
            huber
            + float(self.policy["sign_loss_weight"]) * sign
            + silhouette_weight * silhouette
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("TRAINING_NONFINITE_LOSS")
        row = {
            "subject_id": subject["subject_id"],
            "loss": float(loss.detach()),
            "huber": float(huber.detach()),
            "sign": float(sign.detach()),
            "sign_wrong_count": int(sign_report["wrong_sign_count"]),
            "sign_wrong_fraction": float(torch.as_tensor(sign_report["wrong_sign_fraction"]).detach().cpu()),
            "source_silhouette": float(silhouette.detach()),
            "silhouette_wrong_count": int(silhouette_report["wrong_ray_count"]),
            "silhouette_wrong_fraction": float(torch.as_tensor(silhouette_report["wrong_ray_fraction"]).detach().cpu()),
            "mae": float((prediction.detach() - target).abs().mean()),
            "view_index": "ALL_EIGHT_BALANCED"
            if silhouette_weight > 0.0
            else "ALL_EIGHT_SCENE_INPUT__NO_SILHOUETTE_OBJECTIVE",
            "query_mix": [
                len(points) // 2,
                len(points) // 4,
                len(points) // 8,
                len(points) // 8,
            ],
        }
        return loss, row

    def one_step(self):
        """One global optimizer update after all subjects contribute equally."""
        next_stage_step = self.stage_step + 1
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        rows = []
        n_subjects = len(self.counts)
        for index in range(n_subjects):
            loss, row = self._subject_loss(index, next_stage_step)
            (loss / n_subjects).backward()
            rows.append(row)
        grad = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.policy["gradient_clip"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self.stage_step += 1
        self.counts = [x + 1 for x in self.counts]
        if len(set(self.counts)) != 1 or self.counts[0] != self.step:
            raise RuntimeError("COHORT_JOINT_EXPOSURE_DRIFT")
        aggregate = {
            "mean_loss": float(np.mean([r["loss"] for r in rows])),
            "max_loss": float(max(r["loss"] for r in rows)),
            "mean_huber": float(np.mean([r["huber"] for r in rows])),
            "max_huber": float(max(r["huber"] for r in rows)),
            "mean_source_silhouette": float(
                np.mean([r["source_silhouette"] for r in rows])
            ),
            "max_source_silhouette": float(
                max(r["source_silhouette"] for r in rows)
            ),
            "mean_mae": float(np.mean([r["mae"] for r in rows])),
            "max_mae": float(max(r["mae"] for r in rows)),
            "max_sign_wrong_fraction": float(max(r["sign_wrong_fraction"] for r in rows)),
            "max_silhouette_wrong_fraction": float(max(r["silhouette_wrong_fraction"] for r in rows)),
            "gradient_norm": float(torch.as_tensor(grad).detach().cpu()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        row = {
            "global_step": self.step,
            "stage_id": self.stage["id"],
            "stage_step": self.stage_step,
            "subjects": rows,
            "aggregate": aggregate,
        }
        self.log_window.append(row)
        return row

    @staticmethod
    def _selection_score(rows, effective_step):
        dirty = any(int(r["sign_disagreement_count"]) != 0 for r in rows)
        return [
            1.0 if dirty else 0.0,
            max(float(r["sign_disagreement_fraction"]) for r in rows),
            max(float(r["mae"]) for r in rows),
            max(float(r["near_zero_mae"]) for r in rows),
            max(float(r["p95_abs"]) for r in rows),
            max(float(r["rmse"]) for r in rows),
            float(effective_step),
        ]

    @staticmethod
    def _worst_owner(rows, key):
        row = max(rows, key=lambda r: float(r[key]))
        return {"subject_id": row["subject_id"], "value": float(row[key])}

    @torch.no_grad()
    def evaluate(self, *, candidate=True, origin="scheduled"):
        self.model.eval()
        rows = []
        for index in range(len(self.counts)):
            s = self.cache.get(index)
            planes = self.model.encode_scene(
                s["image_tokens"], s["token_xy"], s["camera_features"]
            )
            limit = self.policy.get(
                "engineering_eval_query_limit", len(s["heldout_points"])
            )
            pred = torch.cat(
                [
                    self.model.query(planes.float(), q[None])["sdf"].reshape(-1)
                    for q in s["heldout_points"][:limit].split(2048)
                ]
            )
            m = direct_fstar_metrics_v5(
                pred,
                s["heldout_targets"][:limit],
                near_zero_band=self.policy["near_zero_band"],
            )
            rows.append({"subject_id": s["subject_id"], **m})
            print(
                f"[EVAL_SUBJECT] global={self.step} stage={self.stage['id']} "
                f"stage_step={self.stage_step} subject={s['subject_id']} "
                f"wrong_sign={m['sign_disagreement_count']}/{m['count']} "
                f"frac={m['sign_disagreement_fraction']:.6f} mae={m['mae']:.6f} "
                f"near_mae={m['near_zero_mae']:.6f} p95={m['p95_abs']:.6f} "
                f"rmse={m['rmse']:.6f}",
                flush=True,
            )
        effective_step = self.lineage_start_effective_step + self.stage_step
        score = self._selection_score(rows, effective_step)
        shared = {
            "dirty_subject_count": int(
                sum(int(r["sign_disagreement_count"]) != 0 for r in rows)
            ),
            "mean_sign_disagreement_fraction": float(np.mean([r["sign_disagreement_fraction"] for r in rows])),
            "mean_mae": float(np.mean([r["mae"] for r in rows])),
            "mean_near_zero_mae": float(np.mean([r["near_zero_mae"] for r in rows])),
            "mean_p95_abs": float(np.mean([r["p95_abs"] for r in rows])),
            "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
            "worst_sign_fraction": self._worst_owner(
                rows, "sign_disagreement_fraction"
            ),
            "worst_mae": self._worst_owner(rows, "mae"),
            "worst_near_zero_mae": self._worst_owner(rows, "near_zero_mae"),
            "worst_p95_abs": self._worst_owner(rows, "p95_abs"),
            "worst_rmse": self._worst_owner(rows, "rmse"),
        }
        evaluation = {
            "step": self.step,
            "stage_id": self.stage["id"],
            "stage_step": self.stage_step,
            "lineage_effective_step": effective_step,
            "origin": origin,
            "subjects": rows,
            "shared": shared,
            "selection_score_worst_first": score,
        }
        self.evaluations.append(evaluation)
        if candidate and (
            self.stage_best_score is None
            or tuple(score) < tuple(self.stage_best_score)
        ):
            self.stage_best_score = score
            self.stage_best_state = {
                "step": self.step,
                "stage_id": self.stage["id"],
                "stage_step": self.stage_step,
                "lineage_effective_step": effective_step,
                "state_dict": _cpu_state_dict(self.model),
                "model_config": self.policy["model_config"],
                "evaluation": evaluation,
                "binding": self.binding,
                "binding_sha256": self.binding_hash,
            }
        print(
            f"[EVAL_SHARED] global={self.step} stage={self.stage['id']} "
            f"stage_step={self.stage_step} effective={effective_step} "
            f"dirty_subjects={shared['dirty_subject_count']} "
            f"mean_sign={shared['mean_sign_disagreement_fraction']:.6f} "
            f"mean_mae={shared['mean_mae']:.6f} "
            f"worst_sign={shared['worst_sign_fraction']['value']:.6f}"
            f"@{shared['worst_sign_fraction']['subject_id']} "
            f"worst_mae={shared['worst_mae']['value']:.6f}"
            f"@{shared['worst_mae']['subject_id']} "
            f"selected_effective={None if self.stage_best_state is None else self.stage_best_state['lineage_effective_step']}",
            flush=True,
        )
        return evaluation

    def _refresh_direct_replay(self):
        if self.stage["replay_mode"] != "REFRESH_EVERY_EVAL":
            return
        self.replay_banks = self._rank_replay_banks()
        print(
            f"[REPLAY_REFRESH] stage={self.stage['id']} global={self.step} "
            f"banks={','.join(map(str,[len(x) for x in self.replay_banks]))}",
            flush=True,
        )

    def _stage_start_candidate(self):
        if self.stage.get("start_checkpoint_is_candidate", False):
            self.evaluate(candidate=True, origin="stage_start")

    def _complete_stage(self):
        if self.stage_best_state is None:
            raise RuntimeError("STAGE_NO_SELECTED_CHECKPOINT")
        selected = self.stage_best_state
        receipt = {
            "stage_id": self.stage["id"],
            "executed_global_start": _stage_prefix(self.policy, self.stage_index),
            "executed_global_end": self.step,
            "executed_steps": int(self.stage["steps"]),
            "selected_global_step": int(selected["step"]),
            "selected_stage_step": int(selected["stage_step"]),
            "selected_lineage_effective_step": int(selected["lineage_effective_step"]),
            "selection_score_worst_first": list(self.stage_best_score),
            "selected_evaluation": selected["evaluation"],
            "optimizer": "AdamW",
            "learning_rate": float(self.stage["learning_rate"]),
            "minimum_learning_rate": float(self.stage["minimum_learning_rate"]),
            "objective": self.stage["objective"],
            "replay_mode": self.stage["replay_mode"],
        }
        self.completed_stages.append(receipt)
        print(
            f"[STAGE_SELECTED] {self.stage['id']} selected_effective="
            f"{receipt['selected_lineage_effective_step']} executed_global={self.step}",
            flush=True,
        )
        self.model.load_state_dict(selected["state_dict"], strict=True)

        if self.stage_index == len(self.policy["training_stages"]) - 1:
            self.final_best_state = selected
            self.save()
            return

        self.lineage_start_effective_step = int(selected["lineage_effective_step"])
        self.stage_index += 1
        self.stage_step = 0
        self.stage_best_score = None
        self.stage_best_state = None

        # Historical C computes one fixed residual bank from its selected direct-F*
        # start and keeps it frozen through C + the low-LR continuation.
        if self.stage["replay_mode"] == "FIXED_FROM_STAGE_START":
            if self.stage["id"] == "C_OBJECTIVE":
                self.replay_banks = self._rank_replay_banks()
                print(
                    f"[REPLAY_FREEZE] stage={self.stage['id']} source=selected_direct "
                    f"bank_size={self.policy['hard_replay_bank_size']}",
                    flush=True,
                )
        self._configure_optimizer()
        self._stage_start_candidate()
        self.save()

    def _emit_log_window(self):
        if not self.log_window:
            return
        rows = self.log_window
        subject_ids = self.binding["subject_ids"]
        per_subject = {}
        for sid in subject_ids:
            sr = [
                s
                for row in rows
                for s in row["subjects"]
                if s["subject_id"] == sid
            ]
            per_subject[sid] = {
                "mean_loss": float(np.mean([x["loss"] for x in sr])),
                "mean_mae": float(np.mean([x["mae"] for x in sr])),
                "mean_silhouette": float(
                    np.mean([x["source_silhouette"] for x in sr])
                ),
            }
        summary = {
            "schema": "RealSaS.SharedIrisTrainingLogWindow.v1",
            "stage_id": self.stage["id"],
            "global_step": self.step,
            "stage_step": self.stage_step,
            "window_steps": len(rows),
            "mean_loss": float(
                np.mean([x["aggregate"]["mean_loss"] for x in rows])
            ),
            "worst_subject_window_mean_loss": max(
                x["mean_loss"] for x in per_subject.values()
            ),
            "mean_mae": float(np.mean([x["aggregate"]["mean_mae"] for x in rows])),
            "worst_subject_window_mean_mae": max(
                x["mean_mae"] for x in per_subject.values()
            ),
            "mean_silhouette": float(
                np.mean([x["aggregate"]["mean_source_silhouette"] for x in rows])
            ),
            "max_sign_wrong_fraction": float(max(x["aggregate"]["max_sign_wrong_fraction"] for x in rows)),
            "max_silhouette_wrong_fraction": float(max(x["aggregate"]["max_silhouette_wrong_fraction"] for x in rows)),
            "mean_gradient_norm": float(
                np.mean([x["aggregate"]["gradient_norm"] for x in rows])
            ),
            "learning_rate": float(rows[-1]["aggregate"]["learning_rate"]),
            "per_subject": per_subject,
        }
        self.trace.append(summary)
        elapsed = self.elapsed_before + time.monotonic() - self.started
        done = max(self.step, 1)
        eta = elapsed / done * (self.policy["steps"] - self.step)
        memory = (
            torch.cuda.max_memory_allocated() / 2**30
            if self.device.type == "cuda"
            else 0.0
        )
        print(
            f"[TRAIN100] global={self.step}/{self.policy['steps']} "
            f"stage={self.stage['id']} stage_step={self.stage_step}/{self.stage['steps']} "
            f"mean_loss={summary['mean_loss']:.6f} "
            f"worst_subject_loss={summary['worst_subject_window_mean_loss']:.6f} "
            f"mean_mae={summary['mean_mae']:.6f} "
            f"worst_subject_mae={summary['worst_subject_window_mean_mae']:.6f} "
            f"silhouette={summary['mean_silhouette']:.6f} "
            f"max_train_sign_wrong={summary['max_sign_wrong_fraction']:.6f} "
            f"max_train_ray_wrong={summary['max_silhouette_wrong_fraction']:.6f} "
            f"lr={summary['learning_rate']:.2g} elapsed_s={elapsed:.1f} "
            f"eta_s={eta:.0f} gpu_GiB={memory:.2f}",
            flush=True,
        )
        self.log_window = []

    def export(self, complete):
        selected = self.final_best_state if complete else self.stage_best_state
        if selected:
            path = self.output / "IRIS_SHARED_SELECTED.pt"
            temp = path.with_suffix(".staged")
            torch.save(selected, temp)
            os.replace(temp, path)
        result = {
            "schema": "RealSaS.SharedIrisFitExecutionPayload.v1",
            "status": (
                "TRAINING_COMPLETE__COMPILER_QUALIFICATION_PENDING"
                if complete
                else "RESUMABLE_PREFIX"
            ),
            "binding": self.binding,
            "binding_sha256": self.binding_hash,
            "actual_optimizer_steps": self.step,
            "shared_model_count": 1,
            "shared_optimizer_count": 1,
            "optimizer_stage_count": len(self.policy["training_stages"]),
            "optimizer_restarts": max(0, len(self.policy["training_stages"]) - 1),
            "subject_exposures": dict(zip(self.binding["subject_ids"], self.counts)),
            "total_subject_exposures": int(sum(self.counts)),
            "per_subject_evaluations": self.evaluations,
            "training_stage_receipts": self.completed_stages,
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            "selected_checkpoint_sha256": (
                file_hash(self.output / "IRIS_SHARED_SELECTED.pt") if selected else None
            ),
            "selected_lineage_effective_step": (
                int(selected["lineage_effective_step"]) if selected else None
            ),
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
            "unseen_claimed": False,
            "go_qualification_required": True,
        }
        atomic_json(self.output / "IRIS_SHARED_FIT_RESULT.json", result)
        atomic_json(self.output / "TRAINING_TRACE.json", self.trace)
        if self.persistent_results:
            self.persistent_results.mkdir(parents=True, exist_ok=True)
            for path in self.output.iterdir():
                if path.suffix not in (".json", ".pt"):
                    continue
                staged = self.persistent_results / (path.name + ".staged")
                shutil.copyfile(path, staged)
                if file_hash(staged) != file_hash(path):
                    raise RuntimeError("PERSISTENT_RESULT_COPY_HASH")
                os.replace(staged, self.persistent_results / path.name)
        return result

    def run(self, stop_after=None):
        end = min(int(self.policy["steps"]), stop_after or int(self.policy["steps"]))
        print(
            f"[TRAIN] device={self.device} parameters={sum(p.numel() for p in self.model.parameters())} "
            f"subjects={len(self.counts)} shared_models=1 update=mean_gradient_all_subjects "
            f"next_global_step={self.step+1} end={end}",
            flush=True,
        )
        try:
            while self.step < end:
                row = self.one_step()
                stage = self.stage

                if (
                    self.step % int(self.policy["log_interval"]) == 0
                    or self.stage_step == int(stage["steps"])
                ):
                    self._emit_log_window()

                if (
                    self.stage_step % int(stage["evaluation_interval"]) == 0
                    or self.stage_step == int(stage["steps"])
                ):
                    self.evaluate(candidate=True, origin="scheduled")
                    self._refresh_direct_replay()

                stage_finished = self.stage_step == int(stage["steps"])
                if stage_finished:
                    self._complete_stage()
                elif (
                    self.stage_step % int(stage["checkpoint_interval"]) == 0
                    or self.step == end
                ):
                    self.save()
                    self.export(False)

                if self.step == end:
                    # If end lands exactly on a stage transition, _complete_stage already
                    # committed the new stage-start checkpoint. Otherwise persist prefix.
                    if self.step < int(self.policy["steps"]):
                        self.save()
                        self.export(False)
                    break
        except BaseException:
            pointer = json.loads((self.store.root / "LATEST.json").read_text())
            print(
                f"[INTERRUPTED] observed_global_step={self.step} "
                f"committed_resume_step={pointer['step']}",
                flush=True,
            )
            raise
        return self.export(self.step == int(self.policy["steps"]))
