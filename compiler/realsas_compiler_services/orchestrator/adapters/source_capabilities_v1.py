"""Account for calibration failure without erasing independent source geometry.

PASS is an accounting state. Only an actual TeacherScene output can supply
kinematic supervision. Original errors and hash-bound logs remain explicit.
"""
from pathlib import Path
from compiler.realsas_compiler_core.types import QualificationError
from .adapter_io import stage_output_payload,sha256_file
from .measurement_io_v1 import artifact as _artifact,require as _require,write,ref
from .truth_foundry_v2 import measure_teacher_scene_stage


def measure_source_capabilities_stage(ctx):
    _require(ctx['stage']['policy'].get('product_pass_authority') is False,'SOURCE_CAPABILITY_PRODUCT_AUTHORITY_FORBIDDEN')
    surface=stage_output_payload(ctx,'50_SOURCE_SURFACE_ACQUIRED','RealSaS.AdmittedSourceRestGeometry.v1')
    manifest=ctx['run_manifest'];root=ctx['run_root']/'artifacts'/ctx['stage']['id'];root.mkdir(parents=True,exist_ok=True)
    outputs=[];failure=None;teacher=None
    try:
        measured=measure_teacher_scene_stage(ctx);outputs=measured['outputs']
        rows=[r for r in outputs if r['schema']=='RealSaS.TeacherSceneIR.v2']
        _require(len(rows)==1,'SOURCE_CAPABILITY_TEACHER_CARDINALITY')
        teacher=rows[0]
    except QualificationError as exc:
        # Resource exceptions, interrupts and unexpected Python failures propagate;
        # this envelope preserves only typed measurement failures as evidence.
        logs=[ref(p) for p in sorted(root.glob('*.log')) if p.is_file()]
        failure={'error_type':type(exc).__name__,'reason':str(exc),'logs':logs}
    report={'schema':'RealSaS.SourceCapabilityMeasurement.v1','status':'SOURCE_CAPABILITIES_ACCOUNTED__NOT_TARGET_QUALIFICATION',
        'subject_id':manifest['subject_id'],'source_sha256':surface['source_sha256'],
        'teacher_scene':teacher,'measurement_failure':failure,'target_arrays_issued':teacher is not None,
        'authority':{'accounting_only':True,'response_family_qualified':False,'training_authorized':False}}
    path=write(root/'SOURCE_CAPABILITIES.json',report)
    return {'status':'PASS','outputs':outputs+[_artifact(path,report['schema'])],
            'diagnostics':{'accounting_complete':True,'calibration_produced':teacher is not None,
                           'measurement_failure':failure,'training_authorized':False}}
