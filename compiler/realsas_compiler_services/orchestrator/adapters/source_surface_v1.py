"""Released source-only acquisition; no Blender animation/material dependency."""
import json,traceback
from pathlib import Path
from .adapter_io import load_file_ref,stage_output_payload
from .measurement_io_v1 import artifact as _artifact,require as _require,write,ref
from tools.corpus import source_rest_geometry_v1 as producer
from tools.corpus.validate_source_rest_geometry_v1 import validate
from tools.corpus import build_component_geometry_authority_v1 as geometry
from tools.corpus import validate_component_geometry_authority_v1 as geometry_court


def produce_source_surface_stage(ctx):
    _require(ctx['stage']['policy'].get('product_pass_authority') is False,'SOURCE_SURFACE_PRODUCT_AUTHORITY_FORBIDDEN')
    manifest=ctx['run_manifest'];sid=manifest['subject_id'];audit=manifest['source_audit']
    source=stage_output_payload(ctx,'01_SOURCE_BYTES_SEALED','RealSaS.SourceByteSeal.v1')
    stage_output_payload(ctx,'08_NORMALIZATION_DOMAIN_QUALIFIED','RealSaS.NormalizationDomainIR.v1')
    primary=[r for r in manifest['source_files'] if r.get('source_kind')=='PRIMARY']
    _require(len(primary)==1,'SOURCE_SURFACE_PRIMARY_CARDINALITY')
    source_sha=primary[0]['expected_sha256']
    _require(source_sha in {r['sha256'] for r in source['files']},'SOURCE_SURFACE_SOURCE_NOT_SEALED')
    cfg=manifest['source_surface']
    _require(cfg.get('mode')=='EXACT_ADMITTED_IMPORT_SURFACE_ONLY','SOURCE_SURFACE_MODE')
    imported=load_file_ref(cfg['admitted_import_geometry'],json_required=False)
    admission=load_file_ref(audit['iris_observation_admission'],json_required=False)
    candidate=load_file_ref(audit['source_rest_candidate'],json_required=False)
    snapshot=load_file_ref(audit['source_scene_snapshot'],json_required=False)
    root=ctx['run_root']/'artifacts'/ctx['stage']['id']
    producer.build(sid,source_sha,imported,admission,candidate,snapshot,root,matrix_tolerance=audit['matrix_tolerance'])
    path=root/'SOURCE_REST_GEOMETRY_MANIFEST_V1.json';court=validate(path);court_path=write(root/'COURT.json',court)
    _require(court['authority']['independent_source_rest_reconstruction_passed'],'SOURCE_SURFACE_COURT_REQUIRED')
    # Geometry courts live in this independent stage. A later calibration crash
    # (including resource failure) cannot erase these already emitted families.
    geo=root/'geometry';proof_path=geo/'COURT.json'
    try:
        camera=load_file_ref(manifest['observation']['camera_bundle'],json_required=False)
        gm=geometry.build(sid,root/'SOURCE_REST_GEOMETRY_V1.npz',snapshot,camera,geo)
        gp=geo/'COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json'
        result=geometry_court.validate(gm);write(proof_path,result)
        families={}
        for name,key in [('SURFACE_GEOMETRY_UDF','SURFACE_GEOMETRY'),('SURFACE_NORMALS','ORIENTED_NORMALS')]:
            status='MASKED' if result['family_candidate_outcomes'][key]=='UNAVAILABLE_CANDIDATE' else 'PASS'
            families[name]={'status':status,'reason_codes':[] if status=='PASS' else ['COURTED_NO_VALID_TARGET_ROWS'],
                'evidence_refs':[ref(gp,gm['schema'])],
                'court':{**ref(proof_path,result['schema']),'status':'PASS'}}
    except MemoryError:
        raise
    except Exception as exc:
        result={'schema':'RealSaS.CorpusFamilyExecutionFailure.v1','status':'FAIL','subject_id':sid,'source_sha256':source_sha,
            'reason':str(exc),'error_type':type(exc).__name__,'traceback':traceback.format_exc(),'training_authorized':False}
        write(proof_path,result)
        claim={'status':'FAIL','reason_codes':[str(exc).split(':')[0] or type(exc).__name__],'evidence_refs':[ref(proof_path,result['schema'])],
               'court':{**ref(proof_path,result['schema']),'status':'FAIL'}}
        families={name:claim for name in ('SURFACE_GEOMETRY_UDF','SURFACE_NORMALS')}
    write(root/'SURFACE_FAMILY_EVIDENCE.json',{'schema':'RealSaS.Corpus370FamilyEvidenceIndex.v1',
        'status':'SUBJECT_FAMILY_MEASUREMENT_ACCOUNTED__FINAL_ADMISSION_SEPARATE','subject_count':1,
        'subjects':[{'subject_id':sid,'source_sha256':source_sha,'families':families}],
        'component_sign_promoted_to_global':False,'training_authorized':False})
    outputs=[]
    for file in sorted(root.rglob('*')):
        if file.is_file() and file.suffix in ('.json','.npz'):
            schema=json.loads(file.read_text())['schema'] if file.suffix=='.json' else 'RealSaS.TruthFamilyNumericPayload.v1'
            outputs.append(_artifact(file,schema))
    return {'status':'PASS','outputs':outputs,
        'diagnostics':{'source_surface_acquired':True,'response_measured':False,'training_authorized':False}}
