"""Court for the absence of consumable targets, never a response PASS."""
import hashlib,json
from pathlib import Path


def validate(path,*,subject_id,source_sha256):
    m=json.loads(Path(path).read_text())
    if m.get('schema')!='RealSaS.SourceCapabilityMeasurement.v1' or m.get('status')!='SOURCE_CAPABILITIES_ACCOUNTED__NOT_TARGET_QUALIFICATION' or m.get('subject_id')!=subject_id or m.get('source_sha256')!=source_sha256:
        raise ValueError('SOURCE_ABSENCE_PARENT')
    if m.get('teacher_scene') is not None or m.get('target_arrays_issued') is not False:
        raise ValueError('SOURCE_ABSENCE_CANNOT_MASK_PRESENT_TARGETS')
    failure=m.get('measurement_failure') or {}
    if not failure.get('reason') or failure.get('error_type')!='QualificationError':raise ValueError('SOURCE_ABSENCE_FAILURE_REQUIRED')
    for row in failure.get('logs') or []:
        with Path(row['path']).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError('SOURCE_ABSENCE_LOG_DRIFT')
    if m.get('authority',{}).get('accounting_only') is not True or m['authority'].get('training_authorized') is not False:
        raise ValueError('SOURCE_ABSENCE_AUTHORITY')
    return {'schema':'RealSaS.SourceCapabilityAbsenceCourt.v1','status':'PASS_MISSING_TARGET_ACCOUNTING__NO_TARGET_AUTHORITY',
        'subject_id':subject_id,'source_sha256':source_sha256,'reason':'CALIBRATION_TARGET_NOT_PRODUCED',
        'recorded_measurement_failure':failure['reason'],'family_candidate_outcomes':{
            'SURFACE_TOPOLOGY':'UNAVAILABLE_CANDIDATE','CAUSAL_RESPONSE':'UNAVAILABLE_CANDIDATE','TESSA_TEACHER':'UNAVAILABLE_CANDIDATE'},
        'authority':{'independent_missing_target_check_passed':True,'qualified_target_count':0,'training_authorized':False}}
