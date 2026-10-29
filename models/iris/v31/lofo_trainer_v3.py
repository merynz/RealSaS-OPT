"""IRIS V3.1 LOFO trainer V3: fixed validation with typed family abstention.

V3 keeps V2's held-out firewall and mirrored-pair partitioning.  The only new
scientific rule is fail-closed handling of a completely unobservable Stage48
pair family: training may proceed for geometry/mechanics/appearance only when
both the projection qualification and recipe explicitly disable pair gradients
and pair validation.  Partial pair evidence remains forbidden.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .fstar_teacher_v31 import exact_fstar_targets_v31
from .lofo_trainer_v1 import IrisV31LofoTrainer, _indices, _require, presentation_partition_v1
from .lofo_trainer_v2 import IrisV31LofoTrainerV2, pair_group_partition_v2


TRAINER_REVISION_V3 = "IRIS_V31_LOFO_TRAINER_V3_FIXED_VALIDATION_PAIR_ABSTAIN"


def _pair_recipe_contract(recipe: dict, pair_authorized: bool) -> None:
    _require(recipe.get("trainer_revision") == TRAINER_REVISION_V3, "IRIS_V31_LOFO_V3_RECIPE_BINDING_REQUIRED")
    schedule = list(recipe.get("schedule") or ())
    _require(schedule, "IRIS_V31_LOFO_V3_SCHEDULE_REQUIRED")
    pair_budget = int((recipe.get("sampling") or {}).get("pair_queries_per_subject_step", -1))
    pair_weights = [float((row.get("loss_weights") or {}).get("occlusion_attenuation", 0.0)) for row in schedule]
    pair_families = ["OCCLUSION_ATTENUATION" in set(row.get("families") or ()) for row in schedule]
    authority = str((recipe.get("truth_authorities") or {}).get("occlusion_attenuation") or "")
    if pair_authorized:
        _require(pair_budget > 0 and any(weight > 0 for weight in pair_weights), "IRIS_V31_LOFO_PAIR_RECIPE_DISABLED_WITH_AUTHORITY")
    else:
        _require(
            pair_budget == 0
            and all(weight == 0.0 for weight in pair_weights)
            and not any(pair_families)
            and authority == "ABSTAIN__NO_OBSERVATION_CONDITIONED_PAIR_EVIDENCE_IN_STAGE48_FIT8",
            "IRIS_V31_LOFO_PAIR_ABSTAIN_RECIPE_REQUIRED",
        )


class IrisV31LofoTrainerV3(IrisV31LofoTrainerV2):
    def __init__(self, *, manifest: dict, recipe: dict, fold_id: str, checkpoint_root, output_root, device="cuda"):
        qualification = dict(manifest.get("student_projection_qualification") or {})
        _require(manifest.get("student_training_authorized") is True, "IRIS_V31_LOFO_STUDENT_TRAINING_NOT_AUTHORIZED")
        _require(
            manifest.get("status") == "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT",
            "IRIS_V31_LOFO_INPUT_AUTHORIZATION_STATUS",
        )
        _require(
            qualification.get("schema") == "RealSaS.IRISV31StudentProjectionQualification.v1"
            and qualification.get("status") == "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED"
            and qualification.get("teacher_inputs_in_forward") is False
            and qualification.get("privileged_ids_in_target_packet") is False
            and qualification.get("semantic_part_taxonomy_used") is False
            and qualification.get("geometric_front_order_student_objective") is False
            and qualification.get("compiler_oracle_retains_draw_order_authority") is True
            and qualification.get("missing_pair_truth_becomes_negative") is False,
            "IRIS_V31_LOFO_STUDENT_PROJECTION_QUALIFICATION",
        )

        pair_authorized = qualification.get("pair_training_authorized")
        pair_subjects = int(qualification.get("pair_evidence_subject_count", -1))
        pair_rows = int(qualification.get("pair_evidence_valid_rows", -1))
        _require(pair_authorized in {True, False}, "IRIS_V31_LOFO_PAIR_AUTHORITY_MISSING")
        if pair_authorized:
            _require(pair_subjects >= 2 and pair_rows > 0, "IRIS_V31_LOFO_PAIR_EVIDENCE_AUTHORIZATION")
        else:
            _require(pair_subjects == 0 and pair_rows == 0, "IRIS_V31_LOFO_PAIR_ABSTAIN_EVIDENCE_DRIFT")
        _pair_recipe_contract(recipe, bool(pair_authorized))

        # Deliberately bypass V2.__init__ only because V2 hard-required positive
        # pair evidence.  V1 owns the unchanged model/optimizer/held-out firewall;
        # below we reapply V2's mirrored pair partition court exactly.
        IrisV31LofoTrainer.__init__(
            self,
            manifest=manifest,
            recipe=recipe,
            fold_id=fold_id,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            device=device,
        )
        fraction = float(recipe["sampling"]["presentation_train_fraction"])
        partition_seed = int(recipe["sampling"]["seed"])
        for subject in self.train_subjects.values():
            valid = np.flatnonzero(np.asarray(subject.presentation["pair_valid"], dtype=np.bool_))
            subject.partitions["pair"] = pair_group_partition_v2(
                subject.subject_id, valid, partition_seed, fraction
            )

    def _geometry_batch(self, subject, step: int):
        """V1 query sampling with V3.1's isolated exact-F* teacher authority."""
        total = int(self.recipe["sampling"]["geometry_queries_per_subject_step"])
        counts = (total // 2, total // 4, total - total // 2 - total // 4)
        rows = (("replay_points", "replay_targets"), ("near_points", None), ("straddle_points", None))
        q_parts, t_parts = [], []
        subject_seed = int(hashlib.sha256(subject.subject_id.encode()).hexdigest()[:8], 16)
        for offset, ((point_key, target_key), count) in enumerate(zip(rows, counts)):
            points = subject.geometry[point_key]
            ids = _indices(
                len(points),
                count,
                self.seed + step * 1009 + offset * 97 + subject_seed,
            )
            q = points[torch.from_numpy(ids)].to(self.device).float()
            q_parts.append(q)
            if target_key:
                target = subject.geometry[target_key][torch.from_numpy(ids)].to(self.device).float()
            else:
                target_np = exact_fstar_targets_v31(
                    subject.field_teacher(),
                    q.detach().cpu().numpy(),
                    device=self.device.type,
                )
                target = torch.from_numpy(target_np).to(self.device).float()
            t_parts.append(target)
        return torch.cat(q_parts)[None], torch.cat(t_parts)[None]


def run_fold_v3(*, input_manifest_path, recipe_path, fold_id, checkpoint_root, output_root, device="cuda"):
    import json
    from pathlib import Path

    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31LofoTrainerV3(
        manifest=manifest,
        recipe=recipe,
        fold_id=fold_id,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
