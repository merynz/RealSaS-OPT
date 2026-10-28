"""FIT8 PresentationResponseTeacherV1 Platform producer.

This additive stage consumes the already-qualified source/observation parents and
the observation-rest TeacherSceneIR.v2 from Stage47.  It never mutates Stage47.
It measures renderer consequences twice, independently replays the admitted raw
source material evaluator twice, derives semantics-free surface response and
pose-conditioned ray order, and emits an oracle-qualified *candidate* package.

Student target admission is deliberately downstream because hidden presentation
and authored mechanical response are not automatically identifiable from static
RGBA observations.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from compiler.realsas_compiler_core.types import QualificationError
from models.iris.truth.presentation_qualification_v1 import qualify_subject_presentation_v1
from models.iris.truth.presentation_response_v1 import (
    build_measured_compositing_arrays_v1,
    build_pose_conditioned_order_arrays_v1,
    build_surface_response_arrays_v1,
)
from tools.observation.pinned_evaluator_runtime_v1 import launch_options

from .adapter_io import load_file_ref, sha256_file, stage_output_payload, write_json
from .truth_foundry_v1 import _artifact, _require


STAGE_ID = "48_PRESENTATION_RESPONSE_TRUTH"
MODE = "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1"


def _stage_output_ref(ctx: dict, stage_id: str, schema: str) -> dict:
    row = next((row for row in ctx["ledger"]["stages"] if row["id"] == stage_id), None)
    _require(row is not None, "PRESENTATION_TRUTH_UPSTREAM_STAGE_MISSING")
    matches = [output for output in row.get("outputs", ()) if output.get("schema") == schema]
    _require(len(matches) == 1, "PRESENTATION_TRUTH_UPSTREAM_SCHEMA_CARDINALITY")
    path = Path(str(matches[0]["path"])).resolve()
    _require(path.is_file() and sha256_file(path) == matches[0].get("sha256"), "PRESENTATION_TRUTH_UPSTREAM_OUTPUT_DRIFT")
    return {"path": str(path), "sha256": str(matches[0]["sha256"])}


def _run_blender(
    executable: Path,
    program: Path,
    arguments: list[object],
    log_path: Path,
    maximum_seconds: float,
    subject: str,
    *,
    runtime: dict | None,
    blend: Path | None = None,
) -> None:
    prefix = [str(executable), "-b"]
    if blend is None:
        prefix.append("--factory-startup")
    else:
        prefix.append(str(blend))
    command = prefix + ["--python-exit-code", "2", "--python", str(program), "--", *map(str, arguments)]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **launch_options(executable, runtime))
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"[PRESENTATION_TRUTH_HEARTBEAT] subject={subject} elapsed_s={elapsed:.1f}", flush=True)
                    _require(elapsed < maximum_seconds, "PRESENTATION_TRUTH_TIME_BUDGET_EXCEEDED")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    _require(code == 0, "PRESENTATION_TRUTH_BLENDER_FAILED__INSPECT_BOUND_LOG:" + str(log_path))


def _payload_ref(scene: dict, name: str, field: str) -> dict:
    matches = [row for row in scene.get(field) or () if Path(str(row.get("path") or "")).name == name]
    _require(len(matches) == 1, "PRESENTATION_TRUTH_SCENE_PAYLOAD_CARDINALITY:" + name)
    ref = dict(matches[0])
    load_file_ref(ref, json_required=False)
    return {"path": str(Path(ref["path"]).resolve()), "sha256": str(ref["sha256"])}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            raise QualificationError("PRESENTATION_TRUTH_OBJECT_ARRAY_FORBIDDEN:" + name)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise QualificationError("PRESENTATION_TRUTH_NONFINITE_ARRAY:" + name)
    np.savez_compressed(path, **arrays)


def produce_presentation_truth_stage(ctx: dict) -> dict:
    _require(ctx["stage"]["id"] == STAGE_ID, "PRESENTATION_TRUTH_STAGE_ID")
    _require(ctx["stage"]["policy"].get("product_pass_authority") is False, "PRESENTATION_TRUTH_CANNOT_OWN_PRODUCT_PASS")
    truth_cfg = dict(ctx["run_manifest"].get("truth_foundry") or {})
    root = ctx["run_root"] / "artifacts" / STAGE_ID
    root.mkdir(parents=True, exist_ok=True)
    subject = str(ctx["run_manifest"].get("subject_id") or "")
    _require(subject, "PRESENTATION_TRUTH_SUBJECT_REQUIRED")

    if truth_cfg.get("mode") == "SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS":
        payload = {
            "schema": "RealSaS.PresentationTruthRegressionExclusion.v1",
            "subject_id": subject,
            "reason": "SEPARATE_FROZEN_REGRESSION_MEMBER__NOT_A_FIT8_TRAINING_SUBJECT",
            "new_truth_produced": False,
            "training_authorized": False,
            "product_authority_changed": False,
        }
        return {
            "status": "PASS",
            "outputs": [write_json(root / "regression_exclusion.json", payload, authority_class="RESEARCH_EXCLUSION", schema=payload["schema"])],
            "diagnostics": {"regression_excluded": True, "training_authorized": False},
        }

    cfg = dict(ctx["run_manifest"].get("presentation_truth") or {})
    _require(cfg.get("mode") == MODE, "PRESENTATION_TRUTH_MODE_REQUIRED")
    scene = stage_output_payload(ctx, "47_SOURCE_RESPONSE_CALIBRATION", "RealSaS.TeacherSceneIR.v2")
    scene_ref = _stage_output_ref(ctx, "47_SOURCE_RESPONSE_CALIBRATION", "RealSaS.TeacherSceneIR.v2")
    _require(scene.get("subject_key") == subject and scene.get("rest_contract") == "EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY", "PRESENTATION_TRUTH_TEACHER_BINDING")
    _require((scene.get("observation_rest_binding") or {}).get("status") == "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY", "PRESENTATION_TRUTH_REST_NOT_EXACT")

    geometry_ref = _payload_ref(scene, "geometry_and_response.npz", "geometry_payloads")
    appearance_ref = _payload_ref(scene, "appearance.json", "appearance_payloads")
    blend_ref = _payload_ref(scene, "source_scene.blend", "appearance_payloads")
    geometry_path = Path(geometry_ref["path"])
    appearance_path = Path(appearance_ref["path"])
    blend_path = Path(blend_ref["path"])

    audit = dict(ctx["run_manifest"].get("source_audit") or {})
    _require(audit.get("profile") == "SOURCE_SCENE_BIND_V2", "PRESENTATION_TRUTH_SOURCE_PROFILE")
    candidate_ref = dict(audit.get("source_rest_candidate") or {})
    candidate_path = load_file_ref(candidate_ref, expected_schema="RealSaS.Controlled8ViewSourceRestCandidate.v2")
    camera_ref = dict((ctx["run_manifest"].get("observation") or {}).get("camera_bundle") or {})
    camera = load_file_ref(camera_ref, expected_schema="RealSaS.CameraProjectionBundle.v1")
    camera_path = Path(camera_ref["path"]).resolve()

    source_rows = [row for row in ctx["run_manifest"].get("source_files") or () if row.get("source_kind") == "PRIMARY"]
    _require(len(source_rows) == 1, "PRESENTATION_TRUTH_PRIMARY_SOURCE_REQUIRED")
    source_path = Path(str(source_rows[0]["path"])).resolve()
    source_sha = str(source_rows[0]["expected_sha256"])
    _require(source_path.is_file() and sha256_file(source_path) == source_sha, "PRESENTATION_TRUTH_PRIMARY_SOURCE_DRIFT")

    executable_ref = dict(truth_cfg.get("evaluator_binary") or {})
    executable = load_file_ref(executable_ref, json_required=False)
    runtime = truth_cfg.get("evaluator_runtime")
    maximum_seconds = float(cfg.get("maximum_seconds_per_subject", truth_cfg.get("maximum_seconds_per_subject", 1200)))
    maximum_rays = int(cfg.get("measured_rays_per_subject", 8))
    maximum_hits = int(cfg.get("maximum_hits_per_ray", 4))
    cycles_samples = int(cfg.get("cycles_samples", 256))
    _require(maximum_rays >= 1 and maximum_hits >= 2 and cycles_samples >= 16, "PRESENTATION_TRUTH_RENDER_POLICY")

    measurement_paths = []
    reference_paths = []
    for replay in ("A", "B"):
        measurement = root / f"PACKED_{replay}.json"
        if not measurement.exists():
            _run_blender(
                executable,
                ctx["repo_root"] / "tools/truth_foundry/blender_presentation_ray_measure_v1.py",
                [
                    "--subject-id", subject,
                    "--teacher-scene", scene_ref["path"],
                    "--geometry-npz", geometry_path,
                    "--candidate", candidate_ref["path"],
                    "--camera-bundle", camera_path,
                    "--expected-blend-sha256", blend_ref["sha256"],
                    "--output", measurement,
                    "--maximum-rays", maximum_rays,
                    "--maximum-hits", maximum_hits,
                    "--cycles-samples", cycles_samples,
                    "--large-footprint-fraction", float(cfg.get("large_footprint_fraction", 0.0625)),
                    "--small-footprint-fraction", float(cfg.get("small_footprint_fraction", 0.03125)),
                    "--convergence-tolerance", float(cfg.get("convergence_tolerance", 0.005)),
                    "--representation-tolerance", float(cfg.get("representation_tolerance", 0.005)),
                    "--measurement-tolerance", float(cfg.get("measurement_tolerance", 0.0075)),
                ],
                root / f"packed_{replay}.blender.log",
                maximum_seconds,
                subject,
                runtime=runtime,
                blend=blend_path,
            )
        measurement_paths.append(measurement)

        reference = root / f"SOURCE_REFERENCE_{replay}.json"
        if not reference.exists():
            _run_blender(
                executable,
                ctx["repo_root"] / "tools/truth_foundry/blender_source_ray_appearance_reference_v1.py",
                [
                    "--source", source_path,
                    "--expected-source-sha256", source_sha,
                    "--subject-id", subject,
                    "--candidate", candidate_ref["path"],
                    "--camera-bundle", camera_path,
                    "--appearance", appearance_path,
                    "--geometry-npz", geometry_path,
                    "--packed-measurement", measurement,
                    "--output", reference,
                ],
                root / f"source_reference_{replay}.blender.log",
                maximum_seconds,
                subject,
                runtime=runtime,
                blend=None,
            )
        reference_paths.append(reference)

    packed_a, packed_b = (json.loads(path.read_text()) for path in measurement_paths)
    source_a, source_b = (json.loads(path.read_text()) for path in reference_paths)
    qualification = qualify_subject_presentation_v1(
        packed_a,
        packed_b,
        source_a,
        source_b,
        appearance_tolerance=float(cfg.get("source_equivalence_tolerance", 0.005)),
        convergence_tolerance=float(cfg.get("convergence_tolerance", 0.005)),
    )
    _require(qualification["status"] == "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE", "PRESENTATION_TRUTH_SOURCE_EQUIVALENCE_FAILED")

    with np.load(geometry_path, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "faces", "isolated_position"}
        _require(required.issubset(set(z.files)), "PRESENTATION_TRUTH_GEOMETRY_ARRAYS_MISSING")
        rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64).copy()
        faces = np.asarray(z["faces"], dtype=np.int64).copy()
        isolated = np.asarray(z["isolated_position"], dtype=np.float64).copy()
    actions = list((scene.get("calibration") or {}).get("actions") or ())
    _require(len(actions) == isolated.shape[0] > 0, "PRESENTATION_TRUTH_ACTION_BINDING")

    surface_arrays = build_surface_response_arrays_v1(rest, faces, isolated, actions)
    order_arrays = build_pose_conditioned_order_arrays_v1(
        rest,
        faces,
        isolated,
        camera.get("cameras") or (),
        maximum_actions=int(cfg.get("pose_conditioned_action_budget", 16)),
        maximum_rays_per_view=int(cfg.get("geometric_rays_per_view", 32)),
    )
    compositing_arrays = build_measured_compositing_arrays_v1(
        packed_a,
        contribution_floor=float(cfg.get("student_observable_contribution_floor", 1.0e-6)),
    )
    arrays = {**surface_arrays, **order_arrays, **compositing_arrays}
    arrays_path = root / "PRESENTATION_RESPONSE_ARRAYS_V1.npz"
    _save_npz(arrays_path, arrays)

    summary = {
        "schema": "RealSaS.PresentationResponseTruthCandidate.v1",
        "subject_id": subject,
        "teacher_scene": scene_ref,
        "geometry_parent": geometry_ref,
        "appearance_parent": appearance_ref,
        "camera_bundle_sha256": str(camera_ref["sha256"]),
        "source_sha256": source_sha,
        "policy": {
            "action_measure": "CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_PER_SEALED_SIGN_AMPLITUDE_ROW",
            "semantic_part_labels_used": False,
            "material_or_owner_ids_used_as_student_targets": False,
            "pose_conditioned_action_budget": int(cfg.get("pose_conditioned_action_budget", 16)),
            "geometric_rays_per_view": int(cfg.get("geometric_rays_per_view", 32)),
            "measured_rays_per_subject": maximum_rays,
            "maximum_hits_per_ray": maximum_hits,
            "student_observable_contribution_floor": float(cfg.get("student_observable_contribution_floor", 1.0e-6)),
        },
        "renderer_subject_qualification": qualification,
        "array_sha256": sha256_file(arrays_path),
        "counts": {
            "surface_face_count": int(len(faces)),
            "action_count": int(len(actions)),
            "response_pair_count": int(len(arrays["response_pair_face"])),
            "pose_order_ray_count": int(len(arrays["order_ray_pose_index"])),
            "pose_order_pair_count": int(len(arrays["order_pair_face_a"])),
            "measured_composite_hit_count": int(len(arrays["composite_hit_face"])),
            "measured_composite_pair_count": int(len(arrays["composite_pair_face_a"])),
            "student_observable_composite_pair_count": int(np.count_nonzero(arrays["composite_pair_student_observable"])),
        },
        "families": {
            "SURFACE_METRIC_STRAIN": "MEASURED__CONTROL_BALANCED_AUTHORED_KINEMATIC_RESPONSE",
            "RAY_COMPOSITING": "MEASURED__SOURCE_EQUIVALENT_BOUNDED_RENDERER_INTERVENTION_DOMAIN",
            "RASTER_EDIT_RESPONSE": "MEASURED__HIT_REMOVAL_COUNTERFACTUAL_ON_BOUNDED_DOMAIN",
            "BOUNDARY_RESPONSE": "MEASURED__ADJACENT_FACE_RESPONSE_DISCONTINUITY_ONLY",
            "SURFACE_TRAJECTORY": "MEASURED_RAW_IN_STAGE47__NOT_REEXPORTED_AS_TEMPORAL_LAW",
            "RELATIVE_AFFINE_TRANSPORT": "UNSUPPORTED_IN_THIS_PRODUCER",
            "ORIENTED_GAP_AND_VELOCITY": "UNSUPPORTED_IN_THIS_PRODUCER",
            "ADMISSIBLE_MOTION_SUBSPACE": "UNSUPPORTED_IN_THIS_PRODUCER",
            "PHYSICAL_IMPULSE_RESPONSE": "UNAVAILABLE_WITHOUT_AUTHORED_PHYSICAL_SOLVER",
            "TEMPORAL_RESPONSE": "UNAVAILABLE_ON_OBSERVATION_REST_V2_WITHOUT_MEASURED_NATIVE_REST_BRIDGE",
        },
        "oracle_eligible_on_declared_renderer_domain": True,
        "student_supervision_authorized": False,
        "training_authorized": False,
        "canonical_mesh_binding_minted": False,
        "product_authority_changed": False,
        "authority": "UNQUALIFIED_PRESENTATION_RESPONSE_TRUTH_CANDIDATE",
    }
    summary_path = root / "PRESENTATION_RESPONSE_TRUTH_V1.json"
    outputs = [
        write_json(summary_path, summary, authority_class="UNQUALIFIED_PRESENTATION_RESPONSE_TRUTH", schema=summary["schema"]),
        _artifact(arrays_path, "RealSaS.PresentationResponseArrays.v1"),
    ]
    for path, schema in (
        (measurement_paths[0], "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"),
        (measurement_paths[1], "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"),
        (reference_paths[0], "RealSaS.SourceRayAppearanceReference.v1"),
        (reference_paths[1], "RealSaS.SourceRayAppearanceReference.v1"),
    ):
        outputs.append(_artifact(path, schema))
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "renderer_oracle_candidate": True,
            "source_equivalent": qualification["source_equivalent"],
            "student_supervision_authorized": False,
            **summary["counts"],
        },
    }
