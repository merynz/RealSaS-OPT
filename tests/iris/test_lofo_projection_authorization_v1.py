from __future__ import annotations

import pytest

from models.iris.v31.lofo_trainer_v2 import IrisV31LofoTrainerV2


def test_lofo_refuses_stage48_oracle_candidate_without_student_projection_authority():
    manifest = {
        "schema": "RealSaS.IRISV31LofoInputManifest.v1",
        "heldout_selection_blind": True,
        "train_subjects_per_fold": 7,
        "student_training_authorized": False,
        "status": "LOFO_INPUTS_SEALED__TRAINING_NOT_STARTED",
    }
    with pytest.raises(ValueError, match="IRIS_V31_LOFO_STUDENT_TRAINING_NOT_AUTHORIZED"):
        IrisV31LofoTrainerV2(
            manifest=manifest,
            recipe={},
            fold_id="LOFO_00_fit8_snake_v1",
            checkpoint_root="/tmp/never",
            output_root="/tmp/never",
            device="cuda",
        )


def test_lofo_refuses_projection_without_cross_fold_pair_evidence():
    manifest = {
        "schema": "RealSaS.IRISV31LofoInputManifest.v1",
        "heldout_selection_blind": True,
        "train_subjects_per_fold": 7,
        "student_training_authorized": True,
        "status": "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT",
        "student_projection_qualification": {
            "schema": "RealSaS.IRISV31StudentProjectionQualification.v1",
            "status": "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED",
            "teacher_inputs_in_forward": False,
            "privileged_ids_in_target_packet": False,
            "semantic_part_taxonomy_used": False,
            "geometric_front_order_student_objective": False,
            "compiler_oracle_retains_draw_order_authority": True,
            "pair_evidence_subject_count": 1,
            "pair_evidence_valid_rows": 12,
        },
    }
    with pytest.raises(ValueError, match="IRIS_V31_LOFO_PAIR_EVIDENCE_AUTHORIZATION"):
        IrisV31LofoTrainerV2(
            manifest=manifest,
            recipe={},
            fold_id="LOFO_00_fit8_snake_v1",
            checkpoint_root="/tmp/never",
            output_root="/tmp/never",
            device="cuda",
        )
