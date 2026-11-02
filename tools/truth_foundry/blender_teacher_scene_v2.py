"""Observation-bind-rest teacher revision; historical native teachers stay intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import bpy
from mathutils import Matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.observation import blender_controlled_8view_v2 as observation
from tools.observation.evaluator_runtime_fingerprint_v1 import fingerprint
from tools.observation.pinned_evaluator_runtime_v1 import LIBRARY_SHA
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry_transport
from tools.truth_foundry.blender_teacher_scene_v1 import bound_ref, measure_imported_scene, sha256
from tools.truth_foundry.observation_rest_contract_v2 import validate_rest_binding


NUMERIC_REBASE_REL_TOL = 5.0e-7


def _canonicalize_exact_rest_response(output: Path, policy: dict, rest: dict) -> None:
    """Re-express tiny evaluator identity/gauge residuals on exact admitted REST.

    The raw POSE-mode evaluator remains the source of every isolated-control
    differential.  This function only accepts residuals inside a strict numeric
    envelope, subtracts the measured POSE identity offset from those responses,
    and writes them in the already-admitted exact REST frame.  The root control
    is a declared coordinate gauge, not a physical response target, so after its
    measured common-translation property is verified it is written exactly as
    REST + the preregistered translation.
    """
    output = Path(output)
    geometry_path = output / 'geometry_and_response.npz'
    scene_path = output / 'TEACHER_SCENE.json'
    if not geometry_path.is_file() or not scene_path.is_file():
        raise ValueError('TEACHER_SCENE_V2_REBASE_OUTPUT_MISSING')

    with np.load(geometry_path, allow_pickle=False) as z:
        arrays = {name: np.array(z[name], copy=True) for name in z.files}

    rest_vertices = np.asarray(arrays['rest_vertices_source_world'], dtype=np.float64)
    expected_rest = np.asarray(rest['vertices'], dtype=np.float64)
    pose_identity = np.asarray(arrays['pose_identity_baseline'], dtype=np.float64)
    if not np.array_equal(rest_vertices, expected_rest):
        raise ValueError('TEACHER_SCENE_V2_REBASE_REST_DRIFT')
    if pose_identity.shape != rest_vertices.shape:
        raise ValueError('TEACHER_SCENE_V2_REBASE_IDENTITY_LAYOUT')

    translation = np.asarray(policy['root_translation_source_units'], dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('TEACHER_SCENE_V2_REBASE_ROOT_POLICY')

    scale = float(np.linalg.norm(rest_vertices.max(axis=0) - rest_vertices.min(axis=0)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('TEACHER_SCENE_V2_REBASE_SCALE')
    tolerance = max(scale, float(np.linalg.norm(translation)), 1.0) * NUMERIC_REBASE_REL_TOL

    identity_offset = pose_identity - rest_vertices
    identity_max_abs = float(np.max(np.abs(identity_offset), initial=0.0))
    if not np.isfinite(identity_max_abs) or identity_max_abs > tolerance:
        raise ValueError(
            f'TEACHER_SCENE_OBSERVATION_REST_TO_IDENTITY_DRIFT:{identity_max_abs:.17g}>{tolerance:.17g}'
        )

    root_raw = np.asarray(arrays['root_control_position'], dtype=np.float64)
    if root_raw.shape != rest_vertices.shape:
        raise ValueError('TEACHER_SCENE_V2_REBASE_ROOT_LAYOUT')
    measured_root_delta = root_raw - pose_identity
    root_residual = float(np.max(np.linalg.norm(
        measured_root_delta - translation[None, :], axis=1
    ), initial=0.0))
    if not np.isfinite(root_residual) or root_residual > tolerance:
        raise ValueError(
            f'TEACHER_SCENE_ROOT_GAUGE_NOT_COMMON_TRANSLATION:{root_residual:.17g}>{tolerance:.17g}'
        )

    def rebase_pose(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != rest_vertices.shape:
            raise ValueError('TEACHER_SCENE_V2_REBASE_POSE_LAYOUT')
        return rest_vertices + (value - pose_identity)

    arrays['pose_identity_baseline'] = rest_vertices.copy()
    arrays['sham_before'] = rebase_pose(arrays['sham_before'])
    arrays['sham_after'] = rebase_pose(arrays['sham_after'])

    isolated = np.asarray(arrays['isolated_position'], dtype=np.float64)
    if isolated.ndim != 3 or isolated.shape[1:] != rest_vertices.shape:
        raise ValueError('TEACHER_SCENE_V2_REBASE_ISOLATED_LAYOUT')
    arrays['isolated_position'] = rest_vertices[None, :, :] + (
        isolated - pose_identity[None, :, :]
    )

    # Gauge authority is exact after measured common-translation verification.
    arrays['root_control_position'] = rest_vertices + translation[None, :]

    temp = geometry_path.with_name(geometry_path.name + '.partial.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(geometry_path)

    scene = json.loads(scene_path.read_text())
    payloads = list(scene.get('geometry_payloads') or ())
    matches = [row for row in payloads if Path(str(row.get('path') or '')).name == geometry_path.name]
    if len(matches) != 1:
        raise ValueError('TEACHER_SCENE_V2_REBASE_GEOMETRY_BINDING')
    matches[0]['sha256'] = sha256(geometry_path)
    matches[0]['size_bytes'] = geometry_path.stat().st_size

    calibration = dict(scene.get('calibration') or {})
    calibration['source_rest_to_pose_identity_max_abs'] = identity_max_abs
    calibration['common_root_translation_max_abs'] = root_residual
    calibration['exact_rest_response_rebase'] = {
        'status': 'PASS_NUMERIC_ONLY_REBASE_TO_EXACT_ADMITTED_REST',
        'relative_tolerance': NUMERIC_REBASE_REL_TOL,
        'absolute_tolerance_source_units': tolerance,
        'raw_pose_identity_max_abs': identity_max_abs,
        'raw_root_common_translation_residual': root_residual,
        'isolated_response_rule': 'EXACT_REST_PLUS_RAW_ACTION_MINUS_RAW_POSE_IDENTITY',
        'root_gauge_rule': 'MEASURE_COMMON_TRANSLATION_THEN_CANONICALIZE_TO_EXACT_REST_PLUS_POLICY_TRANSLATION',
        'physical_response_truth_changed': False,
    }
    scene['calibration'] = calibration
    scene['producer_revision'] = 3
    scene_path.write_text(json.dumps(scene, sort_keys=True, indent=2, allow_nan=False) + '\n')


def build(job, output):
    if job.get('schema') != 'RealSaS.TeacherSceneCalibrationJob.v2' or job.get('training_allowed') is not False:
        raise ValueError('TEACHER_SCENE_V2_JOB_SCOPE')
    runtime_fingerprint = None
    if job.get('evaluator_runtime') is not None:
        runtime_fingerprint = fingerprint()
        actual = runtime_fingerprint['libraries']
        if (runtime_fingerprint['binary_sha256'] != job['evaluator_binary_sha256']
                or runtime_fingerprint['libc'] != ['glibc', '2.39']
                or any(actual[name]['sha256'] != LIBRARY_SHA[key] for key, name in
                       [('loader', 'ld-linux-x86-64.so.2'), ('libc', 'libc.so.6'), ('libm', 'libm.so.6')])):
            raise ValueError('TEACHER_SCENE_PINNED_NUMERIC_RUNTIME_DRIFT')
    source = bound_ref(job['source'])
    values = {key: json.loads(bound_ref(job[key]).read_text()) for key in (
        'observation_admission', 'source_rest_candidate', 'source_scene_snapshot', 'policy', 'observation_import_geometry')}
    policy = values['policy']
    if (policy.get('schema') != 'RealSaS.TeacherKinematicCalibrationPolicy.v2'
            or policy.get('rest_contract') != 'EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY'):
        raise ValueError('TEACHER_SCENE_V2_POLICY')
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix='observation-rest-', dir=output.parent) as work:
        receipt = observation.import_rest(source, Path(work))
        meshes = source_transport._mesh_objects()
        rest = geometry_transport._extract_snapshot(meshes, freeze_triangulation=True)
        sealed = values['source_scene_snapshot']
        current = observation.scene_snapshot(job['source']['sha256'], Matrix(sealed['source_world_to_canonical']))
        proof = validate_rest_binding(job, values['observation_admission'], values['source_rest_candidate'],
            sealed, current, values['observation_import_geometry'], receipt, rest)

        def reset_scene():
            for obj in bpy.context.scene.objects:
                if any(observation.animation_counts(obj).values()):
                    raise ValueError('TEACHER_SCENE_V2_LIVE_ANIMATION_FORBIDDEN')
                if obj.type == 'ARMATURE':
                    obj.data.pose_position = 'REST'
            bpy.context.view_layer.update()

        try:
            # Measure raw POSE-mode differentials first.  Exact admitted REST is
            # re-established only after the raw identity and root gauge have
            # independently proved that their mismatch is numeric-only.
            measure_imported_scene(job, output, policy, sealed, meshes, rest,
                reset_scene=reset_scene, scene_schema='RealSaS.TeacherSceneIR.v2', native_clip_refs=[],
                equivalence_sha=job['observation_import_geometry']['sha256'], exact_identity=False,
                extra={'producer_revision': 2, 'rest_contract': 'EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY',
                       'observation_rest_binding': proof, 'native_temporal_rest_compatible': False,
                       'observation_runtime_binding': job['observation_runtime_binding'],
                       'evaluator_runtime_fingerprint': runtime_fingerprint})
            _canonicalize_exact_rest_response(output, policy, rest)
        except Exception:
            # A failed candidate may never be reused as a completed TeacherScene.
            if output.exists():
                shutil.rmtree(output)
            raise


if __name__ == '__main__':
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(raw)
    build(json.loads(Path(args.job).read_text()), args.output)
