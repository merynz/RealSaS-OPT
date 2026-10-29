"""Teacher-blind free-running TESSA unseen evaluator for Truth139.

Teacher topology is opened only for post-generation scoring.  Generation sees only the sealed
student conditioning cache and the fixed final TESSA checkpoint.  Every successfully decoded
sequence is packaged as ShapeProposal.v2 for the existing Platform `product.mesh` static consequence
court.  Generation/decode failures are recorded per subject rather than disappearing from averages.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from models.iris.training.cohort_v1 import file_hash
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import build_teacher_asset_sequence_v1, build_truncated_teacher_windows_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.objective_v2 import TESSAObjectiveWeightsV2
from .dual_stream_v31 import TESSADualStreamStructuralizerV31
from .topology_generation_v1 import generate_topology_v1
from .truth139_shape_proposal_v1 import package_truth139_shape_proposal_v1


CONDITIONING_SCHEMA = "RealSaS.Truth139TESSAConditioningIndex.v1"
TEACHER_SCHEMA = "RealSaS.Truth139TESSATeacherUnseenManifest.v1"
TRAIN_RESULT_SCHEMA = "RealSaS.Truth139TESSATrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.Truth139TESSACheckpoint.v1"
RECIPE_SCHEMA = "RealSaS.Truth139TESSATrainingRecipe.v1"
RESULT_SCHEMA = "RealSaS.Truth139TESSAUnseenEvaluation.v1"


def _require(value, code: str) -> None:
    if not value:
        raise ValueError(code)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _codec_config(cfg: TESSAStructuralizerConfigV2) -> TESSAConfigV1:
    return TESSAConfigV1(
        coordinate_bins=cfg.coordinate_bins,
        max_faces=cfg.max_faces,
        max_vertices=cfg.max_vertices,
        local_attention_window=cfg.local_attention_window,
        query_chunk_size=cfg.query_chunk_size,
        surface_latent_count=cfg.surface_memory_tokens,
    )


def _load_teacher_mesh(row: dict) -> tuple[np.ndarray, np.ndarray]:
    ref = dict(row.get("teacher_mesh") or {})
    path = Path(str(ref.get("path") or "")).resolve()
    _require(path.is_file() and file_hash(path) == str(ref.get("sha256") or ""),
             "TRUTH139_TESSA_UNSEEN_TEACHER_MESH_DRIFT:" + str(row.get("subject_id")))
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["vertices_normalized"], dtype=np.float32), np.asarray(z["faces"], dtype=np.int64)


def _load_conditioning(row: dict) -> dict:
    ref = dict(row.get("conditioning_cache") or {})
    path = Path(str(ref.get("path") or "")).resolve()
    _require(path.is_file() and file_hash(path) == str(ref.get("sha256") or ""),
             "TRUTH139_TESSA_UNSEEN_CONDITIONING_DRIFT:" + str(row.get("subject_id")))
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(value.get("schema") == "RealSaS.Truth139TESSAConditioningCache.v1"
             and value.get("teacher_topology_used") is False,
             "TRUTH139_TESSA_UNSEEN_CONDITIONING_AUTHORITY")
    return value


class Truth139TESSAUnseenEvaluatorV1:
    def __init__(self, *, conditioning_index: dict, teacher_manifest: dict, training_result: dict,
                 recipe: dict, checkpoint_path, output_root, device: str = "cuda",
                 maximum_generation_tokens: int = 131072) -> None:
        _require(conditioning_index.get("schema") == CONDITIONING_SCHEMA and conditioning_index.get("scope") == "UNSEEN",
                 "TRUTH139_TESSA_UNSEEN_CONDITIONING_SCHEMA")
        _require(teacher_manifest.get("schema") == TEACHER_SCHEMA, "TRUTH139_TESSA_UNSEEN_TEACHER_SCHEMA")
        _require(training_result.get("schema") == TRAIN_RESULT_SCHEMA, "TRUTH139_TESSA_UNSEEN_TRAIN_RESULT_SCHEMA")
        _require(recipe.get("schema") == RECIPE_SCHEMA, "TRUTH139_TESSA_UNSEEN_RECIPE_SCHEMA")
        split = str(recipe.get("split_identity_sha256") or "")
        _require(split and split == conditioning_index.get("split_identity_sha256") == teacher_manifest.get("split_identity_sha256") == training_result.get("split_identity_sha256"),
                 "TRUTH139_TESSA_UNSEEN_SPLIT_DRIFT")
        _require(conditioning_index.get("teacher_topology_used") is False, "TRUTH139_TESSA_UNSEEN_CONDITIONING_TEACHER_LEAKAGE")
        _require(teacher_manifest.get("optimizer_updates_allowed") is False and teacher_manifest.get("checkpoint_selection_allowed") is False,
                 "TRUTH139_TESSA_UNSEEN_TEACHER_FIREWALL")
        _require(training_result.get("unseen_opened") is False and training_result.get("unseen_gradient_updates") is False,
                 "TRUTH139_TESSA_UNSEEN_TRAIN_FIREWALL")
        _require(training_result.get("selection_rule") == "FIXED_FINAL_ENDPOINT__NO_CHECKPOINT_SHOPPING",
                 "TRUTH139_TESSA_UNSEEN_SELECTION_RULE")
        _require(int(maximum_generation_tokens) >= 8, "TRUTH139_TESSA_UNSEEN_GENERATION_BUDGET")

        conditioning_rows = {str(row["subject_id"]): row for row in conditioning_index.get("subjects") or ()}
        teacher_rows = {str(row["subject_id"]): row for row in teacher_manifest.get("subjects") or ()}
        _require(conditioning_rows and set(conditioning_rows) == set(teacher_rows), "TRUTH139_TESSA_UNSEEN_ROSTER_DRIFT")
        self.ids = sorted(conditioning_rows)
        self.conditioning_rows = conditioning_rows
        self.teacher_rows = teacher_rows

        checkpoint = Path(checkpoint_path).resolve()
        _require(checkpoint.is_file() and file_hash(checkpoint) == str(training_result.get("selected_checkpoint_sha256") or ""),
                 "TRUTH139_TESSA_UNSEEN_CHECKPOINT_DRIFT")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _require(payload.get("schema") == CHECKPOINT_SCHEMA, "TRUTH139_TESSA_UNSEEN_CHECKPOINT_SCHEMA")
        _require(payload.get("architecture_id") == TESSADualStreamStructuralizerV31.architecture_id,
                 "TRUTH139_TESSA_UNSEEN_CHECKPOINT_ARCHITECTURE")
        _require(payload.get("attention_semantics") == TESSADualStreamStructuralizerV31.attention_semantics,
                 "TRUTH139_TESSA_UNSEEN_ATTENTION_SEMANTICS")
        cfg = TESSAStructuralizerConfigV2(**dict(payload["config"])); cfg.validate()
        self.config = cfg
        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "TRUTH139_TESSA_UNSEEN_CUDA_REQUIRED")
        runtime = recipe["runtime"]
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False
        self.model = TESSADualStreamStructuralizerV31(
            scene_input_dim=int(payload["scene_input_dim"]),
            surface_input_dim=int(payload["surface_input_dim"]),
            config=cfg,
            backend=str(runtime["attention_backend"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        self.weights = TESSAObjectiveWeightsV2(**dict(recipe["objective"])); self.weights.validate()
        self.recipe = recipe
        self.training_result = training_result
        self.conditioning_index = conditioning_index
        self.teacher_manifest = teacher_manifest
        self.maximum_generation_tokens = int(maximum_generation_tokens)
        self.output = Path(output_root).resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.shape_root = self.output / "shape_proposals"; self.shape_root.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def _teacher_forced(self, scene: torch.Tensor, surface: torch.Tensor,
                        vertices: np.ndarray, faces: np.ndarray) -> dict:
        sequence = build_teacher_asset_sequence_v1(vertices, faces, cfg=_codec_config(self.config))
        windows = build_truncated_teacher_windows_v1(
            sequence, cfg=_codec_config(self.config),
            target_tokens=int(self.recipe["teacher_windows"]["target_tokens_per_window"]),
        )
        rows = []
        for window in windows:
            input_ids = torch.tensor([window.input_ids], device=self.device, dtype=torch.long)
            labels = torch.tensor([window.labels], device=self.device, dtype=torch.long)
            output = self.model(
                scene_source=scene, surface_source=surface,
                input_ids=input_ids, labels=labels,
                sequence_position_offset=int(window.sequence_position_offset),
                objective_weights=self.weights,
            )
            objective = output.objective
            _require(objective is not None, "TRUTH139_TESSA_UNSEEN_OBJECTIVE_MISSING")
            rows.append({
                "target_token_count": int(window.target_token_count),
                "total": float(objective.total.cpu()),
                "token_cross_entropy": float(objective.token_cross_entropy.cpu()),
                "coordinate_metric": float(objective.coordinate_metric.cpu()),
                "structural_special_token": float(objective.structural_special_token.cpu()),
                "coordinate_expected_abs_error_bins": float(objective.coordinate_expected_abs_error_bins.cpu()),
            })
        total_targets = sum(row["target_token_count"] for row in rows)
        def weighted(key: str) -> float:
            return float(sum(row[key] * row["target_token_count"] for row in rows) / total_targets)
        return {
            "window_count": len(rows), "target_token_count": int(total_targets),
            "weighted_total": weighted("total"),
            "weighted_token_cross_entropy": weighted("token_cross_entropy"),
            "weighted_coordinate_metric": weighted("coordinate_metric"),
            "weighted_structural_special_token": weighted("structural_special_token"),
            "weighted_coordinate_expected_abs_error_bins": weighted("coordinate_expected_abs_error_bins"),
        }

    def evaluate(self) -> dict:
        per_subject = []
        successful_shapes = []
        for subject_id in self.ids:
            conditioning = _load_conditioning(self.conditioning_rows[subject_id])
            vertices, faces = _load_teacher_mesh(self.teacher_rows[subject_id])
            scene = conditioning["scene_source"].to(self.device).float()
            surface = conditioning["surface_source"].to(self.device).float()
            teacher_forced = self._teacher_forced(scene, surface, vertices, faces)
            row = {
                "subject_id": subject_id,
                "teacher_forced": teacher_forced,
                "teacher_vertex_count": int(len(vertices)),
                "teacher_face_count": int(len(faces)),
            }
            try:
                generated = generate_topology_v1(
                    model=self.model, scene_source=scene, surface_source=surface,
                    maximum_tokens=self.maximum_generation_tokens,
                )
                bridge = package_truth139_shape_proposal_v1(
                    generated=generated,
                    conditioning_cache=conditioning,
                    iris_checkpoint_sha256=str(self.training_result["iris_checkpoint_sha256"]),
                    tessa_checkpoint_sha256=str(self.training_result["selected_checkpoint_sha256"]),
                )
                shape_path = self.shape_root / f"{subject_id}.shape_v2.npz"
                staged = shape_path.with_suffix(shape_path.suffix + ".partial")
                staged.write_bytes(bridge.payload); staged.replace(shape_path)
                shape_sha = file_hash(shape_path)
                row.update({
                    "free_running_status": "PASS_CANONICAL_DECODE__PENDING_COMPILER_PRODUCT_MESH_COURT",
                    "generated_token_count": int(len(generated.token_ids)),
                    "generated_vertex_count": int(len(generated.vertices_normalized)),
                    "generated_face_count": int(len(generated.faces)),
                    "generated_component_count": int(generated.component_count),
                    "generated_restart_count": int(generated.restart_count),
                    "shape_proposal": {"path": str(shape_path), "sha256": shape_sha},
                    "shape_bridge": bridge.diagnostics,
                })
                successful_shapes.append({"subject_id": subject_id, "path": str(shape_path), "sha256": shape_sha})
            except Exception as exc:
                row.update({
                    "free_running_status": "FAIL_FREE_RUNNING_GENERATION_OR_CANONICAL_DECODE",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "shape_proposal": None,
                })
            per_subject.append(row)
            del scene, surface
            torch.cuda.empty_cache()

        success = sum(row["free_running_status"].startswith("PASS_") for row in per_subject)
        teacher_forced_keys = (
            "weighted_total", "weighted_token_cross_entropy", "weighted_coordinate_metric",
            "weighted_structural_special_token", "weighted_coordinate_expected_abs_error_bins",
        )
        aggregate = {
            "free_running_success_count": int(success),
            "free_running_failure_count": int(len(per_subject) - success),
            "free_running_success_fraction": float(success / len(per_subject)),
        }
        for key in teacher_forced_keys:
            values = np.asarray([row["teacher_forced"][key] for row in per_subject], dtype=np.float64)
            aggregate["macro_" + key] = float(values.mean())
            aggregate["worst_" + key] = float(values.max())

        shape_index = {
            "schema": "RealSaS.Truth139TESSAShapeProposalIndex.v1",
            "scope": "UNSEEN_STATIC_MESH_CONSEQUENCE",
            "split_identity_sha256": self.recipe["split_identity_sha256"],
            "tessa_checkpoint_sha256": self.training_result["selected_checkpoint_sha256"],
            "iris_checkpoint_sha256": self.training_result["iris_checkpoint_sha256"],
            "subject_count": len(self.ids),
            "successful_shape_count": len(successful_shapes),
            "subjects": successful_shapes,
            "compiler_next_role": "product.mesh",
            "mechanics_promotion_allowed": False,
            "skin_or_motion_promotion_allowed": False,
            "canonical_mesh_authority_claimed": False,
        }
        _atomic_json(self.output / "SHAPE_PROPOSAL_INDEX.json", shape_index)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "UNSEEN_FREE_RUNNING_EVALUATED__COMPILER_MESH_CONSEQUENCE_PENDING",
            "split_identity_sha256": self.recipe["split_identity_sha256"],
            "conditioning_index_identity_sha256": self.conditioning_index["index_identity_sha256"],
            "teacher_manifest_identity_sha256": self.teacher_manifest["manifest_identity_sha256"],
            "tessa_training_binding_sha256": self.training_result["binding_sha256"],
            "tessa_checkpoint_sha256": self.training_result["selected_checkpoint_sha256"],
            "iris_checkpoint_sha256": self.training_result["iris_checkpoint_sha256"],
            "maximum_generation_tokens": self.maximum_generation_tokens,
            "per_subject": per_subject,
            "aggregate": aggregate,
            "shape_proposal_index": str(self.output / "SHAPE_PROPOSAL_INDEX.json"),
            "optimizer_created": False,
            "checkpoint_selection_performed": False,
            "compiler_mesh_court_executed": False,
            "tessa_pass_claimed": False,
            "compiler_m_pass_claimed": False,
            "product_pass_claimed": False,
        }
        _atomic_json(self.output / "TESSA_UNSEEN_EVALUATION.json", result)
        return result
