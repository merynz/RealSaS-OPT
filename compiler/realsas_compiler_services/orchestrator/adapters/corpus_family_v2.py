"""Family accounting with independent source surface and optional calibration."""
import json
from pathlib import Path
import numpy as np
from .adapter_io import stage_output_payload,load_file_ref
from .measurement_io_v1 import artifact,require,ref,write
from .corpus_family_v1 import measure_subject_families
from tools.corpus.validate_source_capability_absence_v1 import validate as absence_court


def stage_ref(ctx,stage_id,schema):
    row=next(r for r in ctx['ledger']['stages'] if r['id']==stage_id)
    found=[r for r in row['outputs'] if r['schema']==schema]
    require(len(found)==1,'CORPUS_FAMILY_V2_PARENT_CARDINALITY:'+schema)
    return load_file_ref(found[0],json_required=False)


def produce_subject_family_stage(ctx):
    require(ctx['stage']['policy'].get('product_pass_authority') is False,'CORPUS_FAMILY_V2_PRODUCT_AUTHORITY_FORBIDDEN')
    manifest=ctx['run_manifest'];sid=manifest['subject_id']
    surface=stage_output_payload(ctx,'50_SOURCE_SURFACE_ACQUIRED','RealSaS.AdmittedSourceRestGeometry.v1')
    capabilities=stage_output_payload(ctx,'51_SOURCE_CAPABILITIES_ACCOUNTED','RealSaS.SourceCapabilityMeasurement.v1')
    require(surface['subject_id']==capabilities['subject_id']==sid and surface['source_sha256']==capabilities['source_sha256'],'CORPUS_FAMILY_V2_SOURCE_BINDING')
    surface_path=load_file_ref(surface['output'],json_required=False)
    root=ctx['run_root']/'artifacts'/ctx['stage']['id'];root.mkdir(parents=True,exist_ok=True)
    scene_path=None;geometry_path=surface_path;missing_path=None
    if capabilities['target_arrays_issued']:
        require(capabilities['measurement_failure'] is None,'CORPUS_FAMILY_V2_SUCCESS_WITH_FAILURE')
        scene_path=load_file_ref(capabilities['teacher_scene'],json_required=False);scene=json.loads(scene_path.read_text())
        require(scene['schema']=='RealSaS.TeacherSceneIR.v2' and scene['subject_key']==sid and scene['source_sha256']==surface['source_sha256'],'CORPUS_FAMILY_V2_TEACHER_BINDING')
        rows=[r for r in scene['geometry_payloads'] if Path(r['path']).name=='geometry_and_response.npz']
        require(len(rows)==1,'CORPUS_FAMILY_V2_CALIBRATION_PAYLOAD')
        geometry_path=load_file_ref(rows[0],json_required=False)
        with np.load(surface_path,allow_pickle=False) as a,np.load(geometry_path,allow_pickle=False) as b:
            for name in ('rest_vertices_source_world','faces','face_owner_privileged'):
                require(np.array_equal(a[name],b[name]),'CORPUS_FAMILY_V2_SURFACE_CALIBRATION_DRIFT:'+name)
    else:
        cap_path=stage_ref(ctx,'51_SOURCE_CAPABILITIES_ACCOUNTED','RealSaS.SourceCapabilityMeasurement.v1')
        missing_path=write(root/'MISSING_CALIBRATION_COURT.json',absence_court(cap_path,subject_id=sid,source_sha256=surface['source_sha256']))
    replay_path=load_file_ref(manifest['source_surface']['admitted_import_geometry'],json_required=False)
    snapshot_path=load_file_ref(manifest['source_audit']['source_scene_snapshot'],json_required=False)
    camera_path=load_file_ref(manifest['observation']['camera_bundle'],json_required=False)
    surface_index_path=stage_ref(ctx,'50_SOURCE_SURFACE_ACQUIRED','RealSaS.Corpus370FamilyEvidenceIndex.v1')
    surface_index=json.loads(surface_index_path.read_text());surface_row=surface_index['subjects'][0]
    require(surface_index['subject_count']==1 and surface_row['subject_id']==sid
            and surface_row['source_sha256']==surface['source_sha256'],'CORPUS_FAMILY_V2_SURFACE_INDEX_BINDING')
    row=measure_subject_families(sid=sid,source_sha=surface['source_sha256'],scene_path=scene_path,replay_path=replay_path,
        geometry_path=geometry_path,surface_geometry_path=surface_path,snapshot_path=snapshot_path,camera_path=camera_path,
        missing_target_court_path=missing_path,surface_family_claims=surface_row['families'],root=root)
    index={'schema':'RealSaS.Corpus370FamilyEvidenceIndex.v1','status':'SUBJECT_FAMILY_MEASUREMENT_ACCOUNTED__FINAL_ADMISSION_SEPARATE',
           'subject_count':1,'subjects':[row],'parents':{'source_rest_surface':ref(surface_path),
           'source_capabilities':ref(stage_ref(ctx,'51_SOURCE_CAPABILITIES_ACCOUNTED','RealSaS.SourceCapabilityMeasurement.v1'))},
           'component_sign_promoted_to_global':False,'training_authorized':False}
    write(root/'SUBJECT_FAMILY_EVIDENCE.json',index)
    outputs=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.npz'):
            schema=json.loads(path.read_text())['schema'] if path.suffix=='.json' else 'RealSaS.TruthFamilyNumericPayload.v1'
            outputs.append(artifact(path,schema))
    return {'status':'PASS','outputs':outputs,'diagnostics':{'accounting_complete':True,
            'family_status':{f:c['status'] for f,c in row['families'].items()},'training_authorized':False}}
