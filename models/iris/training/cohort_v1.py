"""One image-only IRIS and optimizer for a hash-bound variable-size cohort."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid
import numpy as np
import torch
from models.iris.v4.scene_first_signed_v4 import SceneFirstSignedGeometryV4, SceneFirstSignedGeometryConfigV4
from models.iris.v5.source_signed_field_v5 import source_signed_field_torch_v5, SourceSignedFieldPolicyV5
from models.iris.v5.direct_fstar_fit_v5 import direct_fstar_metrics_v5
from models.iris.v5.objective_tail_repair_v5 import (near_zero_wrong_sign_loss_v5,
    orthographic_source_rays_normalized_v5, stratified_ray_samples_v5, source_silhouette_wrong_sign_loss_v5)

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".staged." + uuid.uuid4().hex)
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def source_identity(root):
    root = Path(root)
    paths = ["models/iris/training/cohort_v1.py",
             "models/iris/v4/scene_first_signed_v4.py",
             "models/iris/v5/source_signed_field_v5.py",
             "models/iris/v5/direct_fstar_fit_v5.py",
             "models/iris/v5/objective_tail_repair_v5.py"]
    return {p: file_hash(root/p) for p in paths}

class AtomicCheckpointStore:
    """Immutable payload first, verified pointer last. Orphans cannot become latest."""
    def __init__(self, root, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = binding

    def save(self, payload):
        step = payload["step"]
        name = f"step_{step:08d}_{uuid.uuid4().hex}.pt"
        staged = self.root/(name+".staged")
        with staged.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        digest = file_hash(staged)
        torch.load(staged, map_location="cpu", weights_only=True)
        target = self.root/name
        os.replace(staged, target)
        if file_hash(target) != digest:
            raise RuntimeError("CHECKPOINT_COPY_CORRUPTED")
        pointer = {"schema": "RealSaS.SharedFitCheckpointPointer.v1", "file": name,
                   "sha256": digest, "step": step, "binding_sha256": self.binding}
        latest = self.root/"LATEST.json"
        if latest.exists():
            atomic_json(self.root/"PREVIOUS.json", json.loads(latest.read_text()))
        atomic_json(latest, pointer)
        keep = {name}
        previous = self.root/"PREVIOUS.json"
        if previous.exists():
            keep.add(json.loads(previous.read_text())["file"])
        for old in self.root.glob("step_*.pt"):
            if old.name not in keep:
                old.unlink()
        return pointer

    def load(self):
        latest = self.root/"LATEST.json"
        if not latest.exists():
            return None
        pointer = json.loads(latest.read_text())
        if pointer["binding_sha256"] != self.binding:
            raise ValueError("RESUME_CODE_INPUT_POLICY_DRIFT")
        name = pointer["file"]
        if Path(name).name != name:
            raise ValueError("RESUME_PATH_ESCAPE")
        path = self.root/name
        if file_hash(path) != pointer["sha256"]:
            raise ValueError("RESUME_PAYLOAD_HASH_DRIFT")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["binding_sha256"] != self.binding or payload["step"] != pointer["step"]:
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
        path = (self.root/row["shard_file"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or file_hash(path) != row["shard_sha256"]:
            raise ValueError("SUBJECT_SHARD_HASH_OR_PATH_DRIFT")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["subject_id"] != row["subject_id"] or data["observation_binding"] != row["observation_binding"]:
            raise ValueError("SUBJECT_OBSERVATION_BINDING_DRIFT")
        data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        self.cache[index] = data
        while len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return data

def field_targets(subject, points):
    return source_signed_field_torch_v5(points, signed_2d=subject["signed_2d"],
        center_xyz=subject["center_xyz"], normalization_half_extent=subject["normalization_half_extent"],
        camera_origins=subject["camera_origins"], camera_right=subject["camera_right"],
        camera_screen_up=subject["camera_screen_up"], camera_half_extent=subject["camera_half_extent"],
        policy=SourceSignedFieldPolicyV5(required_resolution=subject["resolution"]))

def rng_state(generator):
    ns = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            "queries": generator.get_state()}

def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    generator.set_state(state["queries"])

class SharedIrisTrainer:
    def __init__(self, data_root, manifest, policy, source, checkpoint_root, output_root, device,
                 persistent_result_root=None):
        self.device = torch.device(device)
        self.manifest, self.policy = manifest, policy
        ids = [r["subject_id"] for r in manifest["subjects"]]
        if len(ids) != policy["subject_count"] or len(set(ids)) != len(ids) or not ids:
            raise ValueError("COHORT_CARDINALITY_OR_DUPLICATE")
        if manifest.get("teacher_inputs_in_forward") is not False:
            raise ValueError("COHORT_FORWARD_CONTRACT")
        self.binding = {"schema": "RealSaS.SharedIrisFitBinding.v1", "source": source,
                        "manifest_sha256": value_hash(manifest), "policy_sha256": value_hash(policy),
                        "subject_ids": ids, "device_class": self.device.type}
        self.binding_hash = value_hash(self.binding)
        seed = policy["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.generator = torch.Generator().manual_seed(seed+1)
        self.model = SceneFirstSignedGeometryV4(SceneFirstSignedGeometryConfigV4(**policy["model_config"])).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=policy["learning_rate"],
                                           weight_decay=policy["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, policy["steps"],
                                                                   eta_min=policy["minimum_learning_rate"])
        # BF16 has the exponent range of FP32; retain AMP state even without scaling.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        self.cache = SubjectCache(data_root, manifest, self.device)
        self.store = AtomicCheckpointStore(checkpoint_root, self.binding_hash)
        self.output = Path(output_root); self.output.mkdir(parents=True, exist_ok=True)
        self.persistent_results = Path(persistent_result_root) if persistent_result_root else None
        self.step, self.counts, self.trace, self.evaluations = 0, [0]*len(ids), [], []
        self.best_score, self.best_state, self.elapsed_before = None, None, 0.
        self.started = time.monotonic()
        previous = self.store.load()
        if previous:
            self.model.load_state_dict(previous["model"]); self.optimizer.load_state_dict(previous["optimizer"])
            self.scheduler.load_state_dict(previous["scheduler"]); self.scaler.load_state_dict(previous["amp"])
            self.step, self.counts = previous["step"], previous["counts"]
            self.trace, self.evaluations = previous["trace"], previous["evaluations"]
            self.best_score, self.best_state = previous["best_score"], previous["best_state"]
            self.elapsed_before = previous["elapsed_seconds"]
            restore_rng(previous["rng"], self.generator)
            print(f"[RESUME] verified_step={self.step} shared_models=1 optimizer_states=1", flush=True)
        else:
            self.save()

    def payload(self):
        return {"schema": "RealSaS.SharedIrisTrainingCheckpoint.v1", "binding": self.binding,
                "binding_sha256": self.binding_hash, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "amp": self.scaler.state_dict(), "rng": rng_state(self.generator),
                "step": self.step, "counts": self.counts, "trace": self.trace, "evaluations": self.evaluations,
                "best_score": self.best_score, "best_state": self.best_state,
                "elapsed_seconds": self.elapsed_before + time.monotonic()-self.started}

    def save(self):
        pointer = self.store.save(self.payload())
        print(f"[CHECKPOINT] committed_step={pointer['step']} sha256={pointer['sha256'][:12]}", flush=True)

    def one_step(self):
        index = self.step % len(self.counts)
        subject = self.cache.get(index)
        count = self.policy["queries_per_step"]
        uniform = torch.rand((count//2, 3), generator=self.generator)*2.-1.
        picks = torch.randint(len(subject["near_points"]), (count-count//2,), generator=self.generator)
        points = torch.cat([uniform.to(self.device), subject["near_points"][picks.to(self.device)]])
        with torch.no_grad():
            target = field_targets(subject, points)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type=="cuda"):
            planes = self.model.encode_scene(subject["image_tokens"], subject["token_xy"], subject["camera_features"])
            # FP32 sampling preserves grid_sample semantics on both CPU and CUDA.
        prediction = self.model.query(planes.float(), points[None])["sdf"].float().reshape(-1)
        huber = torch.nn.functional.smooth_l1_loss(prediction, target, beta=self.policy["huber_beta"])
        sign = near_zero_wrong_sign_loss_v5(prediction, target, band=self.policy["near_zero_band"])["total"]
        silhouette = huber*0.
        view = (self.step//len(self.counts)) % 8
        if self.policy.get("silhouette_loss_weight", 0.) > 0.:
            pixels = subject["boundary_pixels"][view]
            ray_count = self.policy["silhouette_rays_per_step"]
            selected = torch.randint(len(pixels), (ray_count,), generator=self.generator)
            xy = pixels[selected].to(self.device)
            vi = torch.full((ray_count,),view,device=self.device,dtype=torch.long)
            base,direction,enter,exit = orthographic_source_rays_normalized_v5(xy,vi,
                resolution=subject["resolution"],center_xyz=subject["center_xyz"],
                normalization_half_extent=subject["normalization_half_extent"],
                camera_origins=subject["camera_origins"],camera_right=subject["camera_right"],
                camera_screen_up=subject["camera_screen_up"],camera_forward=subject["camera_forward"],
                camera_half_extent=subject["camera_half_extent"])
            phase = torch.rand(ray_count,generator=self.generator).to(self.device)
            ray_q = stratified_ray_samples_v5(base,direction,enter,exit,phase,
                       depth_samples=self.policy["silhouette_depth_samples"])
            ray_f = self.model.query(planes.float(),ray_q.reshape(1,-1,3))["sdf"].reshape(ray_count,-1)
            foreground = subject["signed_2d"][view,xy[:,1].long(),xy[:,0].long()] < 0
            silhouette = source_silhouette_wrong_sign_loss_v5(ray_f,foreground,
                                tau=self.policy["silhouette_softmin_tau"])["total"]
        loss = huber + self.policy["sign_loss_weight"]*sign + self.policy.get("silhouette_loss_weight",0.)*silhouette
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("TRAINING_NONFINITE_LOSS")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.policy["gradient_clip"],
                                             error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step()
        self.step += 1; self.counts[index] += 1
        if max(self.counts)-min(self.counts) > 1:
            raise RuntimeError("COHORT_EXPOSURE_DRIFT")
        row = {"step": self.step, "subject_id": subject["subject_id"], "loss": float(loss.detach()),
               "huber": float(huber.detach()), "sign": float(sign.detach()),
               "source_silhouette": float(silhouette.detach()), "view_index": view,
               "mae": float((prediction.detach()-target).abs().mean()),
               "gradient_norm": float(grad), "learning_rate": self.optimizer.param_groups[0]["lr"]}
        self.trace.append(row)
        return row

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        rows = []
        for index in range(len(self.counts)):
            s = self.cache.get(index)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type=="cuda"):
                planes = self.model.encode_scene(s["image_tokens"], s["token_xy"], s["camera_features"])
            pred = torch.cat([self.model.query(planes.float(), q[None])["sdf"].reshape(-1)
                              for q in s["heldout_points"].split(2048)])
            m = direct_fstar_metrics_v5(pred, s["heldout_targets"], near_zero_band=self.policy["near_zero_band"])
            rows.append({"subject_id": s["subject_id"], **m})
            print(f"[EVAL] step={self.step} {s['subject_id']} mae={m['mae']:.5f} "
                  f"wrong_sign={m['sign_disagreement_fraction']:.5f} near_mae={m['near_zero_mae']:.5f}",flush=True)
        score = [max(r["sign_disagreement_fraction"] for r in rows),
                 max(r["mae"] for r in rows), sum(r["sign_disagreement_count"] for r in rows)]
        evaluation = {"step": self.step, "subjects": rows, "selection_score_worst_first": score}
        self.evaluations.append(evaluation)
        if self.best_score is None or tuple(score) < tuple(self.best_score):
            self.best_score = score
            self.best_state = {"step": self.step, "state_dict": {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                               "model_config": self.policy["model_config"], "evaluation": evaluation,
                               "binding": self.binding, "binding_sha256": self.binding_hash}
        return evaluation

    def export(self, complete):
        if self.best_state:
            path = self.output/"IRIS_SHARED_SELECTED.pt"
            temp = path.with_suffix(".staged")
            torch.save(self.best_state, temp); os.replace(temp,path)
        result = {"schema": "RealSaS.SharedIrisFitExecutionPayload.v1",
                  "status": "TRAINING_COMPLETE__COMPILER_QUALIFICATION_PENDING" if complete else "RESUMABLE_PREFIX",
                  "binding": self.binding, "binding_sha256": self.binding_hash, "actual_optimizer_steps": self.step,
                  "shared_model_count": 1, "shared_optimizer_count": 1,
                  "subject_exposures": dict(zip(self.binding["subject_ids"], self.counts)),
                  "per_subject_evaluations": self.evaluations,
                  "elapsed_seconds": self.elapsed_before+time.monotonic()-self.started,
                  "selected_checkpoint_sha256": file_hash(self.output/"IRIS_SHARED_SELECTED.pt") if self.best_state else None,
                  "fit8_pass_claimed": False, "product_pass_claimed": False, "unseen_claimed": False,
                  "go_qualification_required": True}
        atomic_json(self.output/"IRIS_SHARED_FIT_RESULT.json", result)
        atomic_json(self.output/"TRAINING_TRACE.json", self.trace)
        if self.persistent_results:
            self.persistent_results.mkdir(parents=True, exist_ok=True)
            for path in self.output.iterdir():
                if path.suffix not in (".json", ".pt"):
                    continue
                staged = self.persistent_results/(path.name+".staged")
                shutil.copyfile(path,staged)
                if file_hash(staged) != file_hash(path):
                    raise RuntimeError("PERSISTENT_RESULT_COPY_HASH")
                os.replace(staged,self.persistent_results/path.name)
        return result

    def run(self, stop_after=None):
        end = min(self.policy["steps"], stop_after or self.policy["steps"])
        print(f"[TRAIN] device={self.device} parameters={sum(p.numel() for p in self.model.parameters())} "
              f"subjects={len(self.counts)} shared_models=1 next_step={self.step+1} end={end}",flush=True)
        try:
            while self.step < end:
                row = self.one_step()
                if self.step <= len(self.counts) or self.step % self.policy["log_interval"] == 0:
                    elapsed = self.elapsed_before+time.monotonic()-self.started
                    eta = elapsed/max(self.step,1)*(self.policy["steps"]-self.step)
                    memory = torch.cuda.max_memory_allocated()/2**30 if self.device.type=="cuda" else 0.
                    print(f"[STEP] {self.step}/{self.policy['steps']} {row['subject_id']} loss={row['loss']:.6f} "
                          f"mae={row['mae']:.6f} elapsed_s={elapsed:.1f} eta_s={eta:.0f} gpu_GiB={memory:.2f}",flush=True)
                if self.step % self.policy["evaluation_interval"] == 0 or self.step == end:
                    self.evaluate()
                if self.step % self.policy["checkpoint_interval"] == 0 or self.step == end:
                    self.save(); self.export(self.step==self.policy["steps"])
        except BaseException:
            # A cancellation inside optimizer.step may leave partial in-memory
            # state. Resume the last committed boundary; never seal that state.
            pointer = json.loads((self.store.root/"LATEST.json").read_text())
            print(f"[INTERRUPTED] observed_step={self.step} committed_resume_step={pointer['step']}",flush=True)
            raise
        return self.export(self.step==self.policy["steps"])
