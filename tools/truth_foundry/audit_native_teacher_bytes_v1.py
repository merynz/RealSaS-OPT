#!/usr/bin/env python3
"""Read-only byte/schema/capability audit of existing native-motion teachers.

This does not run a new scientific build or infer causal capability from clips.
Source equivalence, timebase, rendering and isolated intervention qualification
are distinct from readable, hash-matching NPZ bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED_ARRAYS = {'rest_vertices_source_world','faces','face_owner','face_material','posed_vertices_source_world'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _bound_path(root, relative):
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or Path(relative).is_absolute():
        raise ValueError('NATIVE_AUDIT_PATH_ESCAPE')
    return path


def audit_teacher(row, *, teachers, source_row, stage01_root):
    result={'subject_id':row['subject_id'],'teacher_sha256':row['teacher_sha256'],
            'status':'MISSING','byte_schema_verified':False,'source_equivalence_qualified':False,
            'isolated_intervention_available':False,'source_timebase_qualified':False,
            'source_appearance_sampler_available':False,'complete_ray_truth_available':False,
            'physics_solver_available':False,'qualification_minted':False}
    payload=_bound_path(teachers,row['teacher_file'])
    receipt_path=_bound_path(teachers,row['receipt_file'])
    if not receipt_path.is_file():
        result['reason']='RECEIPT_BYTES_MISSING';return result
    if sha256(receipt_path) != row['receipt_sha256']:
        raise ValueError('NATIVE_AUDIT_RECEIPT_HASH_DRIFT:'+row['subject_id'])
    receipt=json.loads(receipt_path.read_text())
    raw=source_row['raw_source'];admission_ref=source_row['source_scene_evidence']['iris_observation_admission']
    source=_bound_path(stage01_root,raw['path']);admission_path=_bound_path(stage01_root,admission_ref['path'])
    if sha256(source)!=raw['sha256'] or sha256(admission_path)!=admission_ref['sha256']:
        raise ValueError('NATIVE_AUDIT_SOURCE_BYTES_DRIFT')
    admission=json.loads(admission_path.read_text())
    if (receipt.get('schema')!='RealSaS.IRISVNextAuthoredTeacherExport.v1'
            or receipt.get('subject_id')!=row['legacy_subject_id']
            or receipt.get('source',{}).get('sha256')!=raw['sha256']
            or receipt.get('observation_admission_sha256')!=admission_ref['sha256']
            or receipt.get('coordinate_domain')!='SOURCE_WORLD__NOT_REALSAS_CANONICAL'
            or receipt.get('transport_revision')!=row['teacher_transport_revision']
            or receipt.get('teacher_only') is not True
            or receipt.get('teacher_rig_skin_motion_allowed_as_inference_input') is not False
            or receipt.get('canonical_M_G_W_minted') is not False):
        raise ValueError('NATIVE_AUDIT_SOURCE_RECEIPT_BINDING_DRIFT:'+row['subject_id'])
    clips=admission['native_motion_inventory']
    if (receipt['native_motion_inventory_expected']!=clips or receipt['native_motion_inventory_exported']!=clips
            or len(clips)!=row['native_motion_count'] or len(set(clips))!=len(clips)):
        raise ValueError('NATIVE_AUDIT_NATIVE_ACTION_BINDING_DRIFT')
    result['receipt_bytes_verified']=True
    result['native_clip_count']=len(clips)
    if not payload.is_file():
        result['reason']='PAYLOAD_BYTES_MISSING';return result
    if (sha256(payload)!=row['teacher_sha256'] or receipt['output']['sha256']!=row['teacher_sha256']
            or payload.stat().st_size!=receipt['output']['size_bytes']):
        raise ValueError('NATIVE_AUDIT_PAYLOAD_HASH_DRIFT:'+row['subject_id'])
    with np.load(payload,allow_pickle=False) as arrays:
        if set(arrays.files)!=EXPECTED_ARRAYS:
            raise ValueError('NATIVE_AUDIT_UNREVIEWED_TRANSPORT_LAYOUT')
        rest,faces,posed=(arrays[k] for k in ('rest_vertices_source_world','faces','posed_vertices_source_world'))
        v,t,f=row['rest_vertex_count'],row['sampled_frame_count'],receipt['face_count']
        if (rest.shape!=(v,3) or posed.shape!=(t,v,3) or faces.shape!=(f,3)
                or rest.dtype.kind!='f' or posed.dtype.kind!='f' or faces.dtype.kind not in 'iu'
                or not np.isfinite(rest).all() or not np.isfinite(posed).all()
                or f<=0 or faces.min()<0 or faces.max()>=v):
            raise ValueError('NATIVE_AUDIT_GEOMETRY_ARRAY_DRIFT')
        for key in ('face_owner','face_material'):
            if arrays[key].shape!=(f,) or arrays[key].dtype.kind not in 'iu':
                raise ValueError('NATIVE_AUDIT_PRIVILEGED_INDEX_LAYOUT_DRIFT')
        frames=receipt['sampled_frames']
        if (len(frames)!=t or receipt['sampled_frame_count']!=t
                or receipt['rest_vertex_count']!=v or {r['action'] for r in frames}!=set(clips)):
            raise ValueError('NATIVE_AUDIT_FRAME_BINDING_DRIFT')
        for clip in clips:
            times=[r['frame'] for r in frames if r['action']==clip]
            if not times or any(type(x) is not int for x in times) or any(a>=b for a,b in zip(times,times[1:])):
                raise ValueError('NATIVE_AUDIT_FRAME_ORDER_DRIFT')
        result.update(status='NATIVE_GEOMETRY_BYTES_VERIFIED__CAUSAL_TRUTH_PENDING',byte_schema_verified=True,
                      vertices=v,faces=f,frames=t,payload_size_bytes=payload.stat().st_size,
                      numeric_array_bytes=sum(arrays[k].nbytes for k in arrays.files),
                      supplied_arrays=sorted(arrays.files),
                      available_measurement_scope='SOURCE_WORLD_REST_AND_NATIVE_CLIP_VERTEX_POSITIONS_ONLY',
                      reason='NO_ISOLATED_ACTION_BANK_SOURCE_CLOCK_MATERIAL_SAMPLER_OR_RAY_PAYLOAD')
    return result


def audit(*,batch_path,expected_batch_sha256,teachers,stage01_root):
    if sha256(batch_path)!=expected_batch_sha256:
        raise ValueError('NATIVE_AUDIT_BATCH_HASH_DRIFT')
    batch=json.loads(Path(batch_path).read_text())
    if batch.get('schema')!='RealSaS.IRISVNextFit8TeacherBatch.v1':
        raise ValueError('NATIVE_AUDIT_BATCH_SCHEMA')
    manifest=json.loads((Path(stage01_root)/'BOOTSTRAP_MANIFEST.json').read_text())
    sources={r['subject_id']:r for r in manifest['subjects']}
    rows=batch['subjects']
    if len(rows)!=len(sources) or {r['legacy_subject_id'] for r in rows}!=set(sources):
        raise ValueError('NATIVE_AUDIT_COHORT_SOURCE_DRIFT')
    results=[audit_teacher(r,teachers=teachers,source_row=sources[r['legacy_subject_id']],stage01_root=stage01_root) for r in rows]
    return {'schema':'RealSaS.NativeTeacherByteCapabilityAudit.v1','batch_sha256':expected_batch_sha256,
            'subject_count':len(results),'verified_payload_count':sum(r['byte_schema_verified'] for r in results),
            'subjects':results,'blender_version_recorded':batch['blender_version'],
            'scope':'READ_ONLY_EXISTING_SEALED_BYTES__NOT_TRUTH_QUALIFICATION',
            'training_ready':False,'qualification_minted':False,'scientific_pass_minted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','expected-batch-sha256','teachers','stage01-root','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    result=audit(batch_path=a.batch,expected_batch_sha256=a.expected_batch_sha256,teachers=a.teachers,stage01_root=a.stage01_root)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('subject_count','verified_payload_count','training_ready')},sort_keys=True))


if __name__=='__main__':
    main()
