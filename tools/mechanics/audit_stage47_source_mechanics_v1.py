from __future__ import annotations

"""Read-only Stage47 source-native mechanical consequence audit.

This is an independent array/manifest verification of measured *source*
interventions. It is NOT a projection to a learned model, NOT CanonicalSkin
training truth, and NOT a comparison to the Mechanics runtime until a separate
source-to-canonical correspondence is qualified.

Inputs are exact files materialized from authorized Drive source evidence.
The tool cannot fetch from Drive, modify truth receipts, or certify unseen.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "RealSaS.MechanicsStage47SourceAudit.v1"


class SourceAuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def audit_stage47_source_mechanics_v1(
    teacher_scene: Path,
    geometry_response: Path,
    *,
    expected_teacher_sha256: str | None = None,
    expected_geometry_sha256: str | None = None,
) -> dict:
    teacher_sha = _sha256(teacher_scene)
    npz_sha = _sha256(geometry_response)
    if expected_teacher_sha256 and teacher_sha != expected_teacher_sha256:
        raise SourceAuditError("STAGE47_TEACHER_SHA_DRIFT")
    if expected_geometry_sha256 and npz_sha != expected_geometry_sha256:
        raise SourceAuditError("STAGE47_GEOMETRY_SHA_DRIFT")
    scene = json.loads(teacher_scene.read_text(encoding="utf-8"))
    if scene.get("schema") != "RealSaS.TeacherSceneIR.v2":
        raise SourceAuditError("STAGE47_TEACHER_SCHEMA_DRIFT")
    if scene.get("student_input_allowed") is not False or scene.get("qualification_minted") is not False:
        raise SourceAuditError("STAGE47_RESEARCH_SOURCE_AUTHORITY_INVALID")
    sources = scene.get("geometry_payloads", [])
    if len(sources) != 1 or sources[0].get("sha256") != npz_sha:
        raise SourceAuditError("STAGE47_SOURCE_BYTES_NOT_BOUND")
    if scene.get("coordinate_contract", {}).get("frame") != "SOURCE_WORLD":
        raise SourceAuditError("STAGE47_SOURCE_WORLD_FRAME_REQUIRED")
    if scene.get("calibration", {}).get("action_measure") != "UNSEALED__NO_IMPLICIT_UNIFORM_LAW":
        raise SourceAuditError("STAGE47_ACTION_MEASURE_UNEXPECTED")
    if scene.get("constraint_capabilities", {}).get("authored_physical_dof_or_joint_limit") != "UNKNOWN":
        raise SourceAuditError("STAGE47_AUTHORED_DOF_SCOPE_UNEXPECTED")
    actions = scene.get("calibration", {}).get("actions", [])
    if len(actions) < 2 or len(actions) % 2:
        raise SourceAuditError("STAGE47_ACTION_PAIR_COUNT_INVALID")
    with np.load(geometry_response, allow_pickle=False) as z:
        required = (
            "rest_vertices_source_world", "pose_identity_baseline",
            "sham_before", "sham_after", "isolated_position",
            "root_control_position", "source_weight_offsets",
            "source_weight_values", "source_weight_bone_indices_privileged",
            "source_weight_coverage",
        )
        if any(key not in z.files for key in required):
            raise SourceAuditError("STAGE47_REQUIRED_ARRAY_MISSING")
        rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        identity = np.asarray(z["pose_identity_baseline"], dtype=np.float64)
        before = np.asarray(z["sham_before"], dtype=np.float64)
        after = np.asarray(z["sham_after"], dtype=np.float64)
        isolated = np.asarray(z["isolated_position"], dtype=np.float64)
        root = np.asarray(z["root_control_position"], dtype=np.float64)
        offsets = np.asarray(z["source_weight_offsets"])
        weights = np.asarray(z["source_weight_values"], dtype=np.float64)
        bone_ids = np.asarray(z["source_weight_bone_indices_privileged"])
        covered = np.asarray(z["source_weight_coverage"])
        if rest.ndim != 2 or rest.shape[1] != 3 or rest.shape[0] < 1:
            raise SourceAuditError("STAGE47_REST_GEOMETRY_INVALID")
        n = rest.shape[0]
        if (identity.shape != rest.shape or before.shape != rest.shape or
                after.shape != rest.shape or root.shape != rest.shape or
                isolated.shape != (len(actions), n, 3)):
            raise SourceAuditError("STAGE47_RESPONSE_SHAPE_INVALID")
        if not all(np.isfinite(x).all() for x in (rest, identity, before, after, root, isolated, weights)):
            raise SourceAuditError("STAGE47_NONFINITE_ARRAY")
        if (offsets.shape != (n + 1,) or offsets[0] != 0
                or np.any(np.diff(offsets) < 0)
                or offsets[-1] != len(weights)
                or len(weights) != len(bone_ids)
                or covered.shape != (n,)):
            raise SourceAuditError("STAGE47_SOURCE_CSR_INVALID")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(bone_ids < 0):
            raise SourceAuditError("STAGE47_SOURCE_CSR_VALUES_INVALID")
        row_sums = np.zeros(n, dtype=np.float64)
        for index in range(n):
            row_sums[index] = weights[offsets[index]:offsets[index + 1]].sum()
        covered_rows = covered.astype(bool)
        if (np.any(covered_rows & ((offsets[1:] - offsets[:-1]) == 0))
                or np.any(np.abs(row_sums[covered_rows] - 1.0) > 1e-5)):
            raise SourceAuditError("STAGE47_WEIGHT_COVERAGE_OR_SIMPLEX_DRIFT")

        rest_identity_error = float(np.max(np.abs(rest - identity)))
        sham_error = float(np.max(np.abs(after - before)))
        delta = np.asarray(
            scene["calibration"]["root_translation_source_units"], dtype=np.float64,
        )
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise SourceAuditError("STAGE47_ROOT_CALIBRATION_INVALID")
        root_error = float(np.max(np.abs((root - rest) - delta[None])))
        if max(rest_identity_error, sham_error, root_error) > 1e-6:
            raise SourceAuditError("STAGE47_BASELINE_OR_SHAM_DRIFT")

        active_actions = 0
        response_l2_max: list[float] = []
        response_even_l2: list[float] = []
        source_control_indices: set[int] = set()
        for index in range(0, len(actions), 2):
            negative, positive = actions[index:index+2]
            if not (
                negative.get("sign") == -1 and positive.get("sign") == 1
                and negative.get("control_index_privileged") == positive.get("control_index_privileged")
                and negative.get("opaque_privileged_handle") == positive.get("opaque_privileged_handle")
                and negative.get("amplitude") == positive.get("amplitude")
                and negative.get("amplitude_index") == positive.get("amplitude_index")
                and negative.get("units") == positive.get("units") == "RADIAN"
                and negative.get("kind") == positive.get("kind")
                == "ISOLATED_KINEMATIC_PARAMETER_PERTURBATION"
            ):
                raise SourceAuditError("STAGE47_ACTION_SIGN_PAIR_DRIFT")
            amp = float(negative["amplitude"])
            if not np.isfinite(amp) or amp <= 0.0:
                raise SourceAuditError("STAGE47_ACTION_AMPLITUDE_INVALID")
            source_control_indices.add(int(negative["control_index_privileged"]))
            pair = isolated[index:index+2] - rest[None]
            norms = np.linalg.norm(pair, axis=-1)
            active_actions += int(np.any(norms[0] > 1e-9)) + int(np.any(norms[1] > 1e-9))
            response_l2_max.extend(map(float, norms.max(axis=1)))
            even = 0.5 * (isolated[index] + isolated[index+1]) - rest
            response_even_l2.append(float(np.linalg.norm(even)))
        q = lambda x, p: float(np.percentile(np.asarray(x), p))
        return {
            "schema": SCHEMA,
            "status": "PASS_SOURCE_MEASUREMENT_INTEGRITY_ONLY",
            "source_subject_key": scene["subject_key"],
            "source_sha256": scene["source_sha256"],
            "teacher_scene_sha256": teacher_sha,
            "source_geometry_npz_sha256": npz_sha,
            "vertex_count": n,
            "action_count": len(actions),
            "sign_pair_count": len(actions) // 2,
            "privileged_source_control_count": len(source_control_indices),
            "source_weight_covered_vertex_count": int(covered_rows.sum()),
            "source_weight_max_simplex_error": float(np.max(np.abs(row_sums[covered_rows] - 1.0))) if np.any(covered_rows) else None,
            "rest_identity_max_abs": rest_identity_error,
            "sham_max_abs": sham_error,
            "root_translation_max_abs_error": root_error,
            "nonzero_source_consequence_actions": active_actions,
            "source_displacement_median_world": q(response_l2_max, 50),
            "source_displacement_p90_world": q(response_l2_max, 90),
            "source_displacement_max_world": float(max(response_l2_max)),
            "source_pair_even_residual_median_world": q(response_even_l2, 50),
            "source_pair_even_residual_p90_world": q(response_even_l2, 90),
            "teacher_only_control_identity": True,
            "source_weight_lbs_equivalence_qualified": False,
            "compiled_mechanics_correspondence_qualified": False,
            "physical_joint_limits_qualified": False,
            "research_exposed_subject": True,
            "student_training_authorized": False,
            "graphics_authority_claimed": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--expected-npz-sha256")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_stage47_source_mechanics_v1(
        args.teacher, args.npz, expected_teacher_sha256=args.expected_teacher_sha256,
        expected_geometry_sha256=args.expected_npz_sha256,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
