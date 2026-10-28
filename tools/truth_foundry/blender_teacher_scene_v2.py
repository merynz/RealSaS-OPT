"""Observation-bind-rest teacher revision; historical native teachers stay intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.observation import blender_controlled_8view_v2 as observation
from tools.observation.evaluator_runtime_fingerprint_v1 import fingerprint
from tools.observation.pinned_evaluator_runtime_v1 import LIBRARY_SHA
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry_transport
from tools.truth_foundry.blender_teacher_scene_v1 import bound_ref, measure_imported_scene
from tools.truth_foundry.observation_rest_contract_v2 import validate_rest_binding


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

        measure_imported_scene(job, output, policy, sealed, meshes, rest,
            reset_scene=reset_scene, scene_schema='RealSaS.TeacherSceneIR.v2', native_clip_refs=[],
            equivalence_sha=job['observation_import_geometry']['sha256'], exact_identity=True,
            extra={'producer_revision': 2, 'rest_contract': 'EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY',
                   'observation_rest_binding': proof, 'native_temporal_rest_compatible': False,
                   'observation_runtime_binding': job['observation_runtime_binding'],
                   'evaluator_runtime_fingerprint': runtime_fingerprint})


if __name__ == '__main__':
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(raw)
    build(json.loads(Path(args.job).read_text()), args.output)
