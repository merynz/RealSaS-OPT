#!/usr/bin/env python3
from __future__ import annotations

"""Audit repaired FIT8 for Motion truth that is actually available now.

This court is expected to PASS by proving a *negative authority boundary*: the
current static kinematic calibration contains useful diagnostic response bytes,
but it does not contain authored physical DOF, physical impulse, temporal or
secondary-motion truth.  Those absences must remain unavailable rather than
becoming model labels by reinterpretation.
"""

import argparse
import json
from pathlib import Path
import zipfile

from models.iris.truth.motion_truth_availability_v1 import (
    evaluate_motion_truth_availability_v1,
)

SCHEMA = "RealSaS.MotionTruthAvailabilityCourt.v1"
EXPECTED_CAMPAIGN = "ee6e4e66-a7c8-42f3-ac53-87a96c9a2393"
EXPECTED_SUBJECT_COUNT = 8
EXPECTED_ACTION_COUNT = 4392


def _load(archive: zipfile.ZipFile, name: str):
    return json.loads(archive.read(name))


def audit(path: Path):
    with zipfile.ZipFile(path) as archive:
        preparation = _load(archive, "PREPARATION.json")
        observation = _load(archive, "evidence/OBSERVATION.json")
        if preparation.get("campaign_id") != EXPECTED_CAMPAIGN:
            raise RuntimeError("MOTION_TRUTH_CAMPAIGN_DRIFT")
        if observation.get("campaign_id") != EXPECTED_CAMPAIGN:
            raise RuntimeError("MOTION_TRUTH_OBSERVATION_CAMPAIGN_DRIFT")
        if observation.get("all_measurements_committed_and_exported") is not True:
            raise RuntimeError("MOTION_TRUTH_MEASUREMENTS_NOT_CLOSED")
        if observation.get("training_submitted") is not False:
            raise RuntimeError("MOTION_TRUTH_UNEXPECTED_TRAINING")
        if observation.get("truth_qualification_inferred") is not False:
            raise RuntimeError("MOTION_TRUTH_IMPLICIT_QUALIFICATION_FORBIDDEN")

        names = sorted(n for n in archive.namelist() if n.endswith("/TEACHER_SCENE.json"))
        if len(names) != EXPECTED_SUBJECT_COUNT:
            raise RuntimeError("MOTION_TRUTH_SUBJECT_COUNT_DRIFT")

        rows = []
        total_actions = 0
        for name in names:
            scene = _load(archive, name)
            report = evaluate_motion_truth_availability_v1(scene)
            action_count = len(scene.get("calibration", {}).get("actions", ()))
            total_actions += action_count
            row = report.to_dict()
            row["action_count"] = action_count
            rows.append(row)

        if total_actions != EXPECTED_ACTION_COUNT:
            raise RuntimeError("MOTION_TRUTH_ACTION_COUNT_DRIFT")

        for row in rows:
            diagnostic = row["kinematic_response_diagnostic"]
            if diagnostic["status"] != "MEASURED_DIAGNOSTIC_ONLY":
                raise RuntimeError("MOTION_TRUTH_KINEMATIC_DIAGNOSTIC_MISSING")
            for key in (
                "admissible_motion_subspace",
                "physical_impulse_response",
                "temporal_response",
                "secondary_motion_consumer",
            ):
                family = row[key]
                if family["status"] != "UNAVAILABLE":
                    raise RuntimeError("MOTION_TRUTH_UNEXPECTED_FAMILY_AUTHORITY:" + key)
                if family["student_target_allowed"] or family["compiler_consequence_allowed"]:
                    raise RuntimeError("MOTION_TRUTH_UNAVAILABLE_FAMILY_PROMOTED:" + key)

    return {
        "schema": SCHEMA,
        "campaign_id": EXPECTED_CAMPAIGN,
        "subject_count": len(rows),
        "isolated_kinematic_action_count": total_actions,
        "kinematic_response_diagnostic": "MEASURED_DIAGNOSTIC_ONLY",
        "admissible_motion_subspace": "UNAVAILABLE",
        "physical_impulse_response": "UNAVAILABLE",
        "temporal_response": "UNAVAILABLE",
        "secondary_motion_consumer": "UNAVAILABLE",
        "training_allowed": False,
        "qualification_minted": False,
        "court_status": "PASS_FAIL_CLOSED_AVAILABILITY_BOUNDARY",
        "next_required_truth_producer": (
            "AUTHORED_DOF_CONSTRAINT_OR_NATIVE_TEMPORAL_RESPONSE_ADAPTER"
        ),
        "subjects": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.artifact_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
