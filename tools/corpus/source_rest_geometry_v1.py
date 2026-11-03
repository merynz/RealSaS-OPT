"""Acquire source surface geometry independently of skin, actions and textures.

The exact observation-admitted import is an input, not a new TeacherScene. No
motion arrays, zero interventions, skin or appearance stand-ins are manufactured.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot

SCHEMA='RealSaS.AdmittedSourceRestGeometry.v1'
STATUS='PASS_ADMITTED_SOURCE_REST_CANDIDATE__FAMILY_COURTS_REQUIRED'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ref(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p)}


def require(value,code):
    if not value:raise ValueError(code)


def decode_import(imported):
    v=np.asarray(imported['vertices'],np.float64);f=np.asarray(imported['faces'])
    require(v.ndim==2 and v.shape[1:]==(3,) and len(v)>0 and np.isfinite(v).all(),'SOURCE_REST_VERTICES')
    require(f.ndim==2 and f.shape[1:]==(3,) and len(f)>0 and f.dtype.kind in 'iu' and np.all(f>=0) and np.all(f<len(v)),'SOURCE_REST_FACES')
    objects=imported['objects'];names=[x['name'] for x in objects]
    require(len(objects)>0 and len(set(names))==len(objects),'SOURCE_REST_OBJECT_IDENTITY')
    blocks={};offset=0;covered=np.zeros(len(f),bool)
    for row in objects:
        count=row['vertex_count'];require(type(count) is int and count>=0 and offset+count<=len(v),'SOURCE_REST_VERTEX_BLOCK')
        selected=(f[:,0]>=offset)&(f[:,0]<offset+count)
        local=f[selected]-offset
        require(not len(local) or bool(np.all(local>=0) and np.all(local<count)),'SOURCE_REST_CROSS_OBJECT_FACE')
        require(not np.any(covered&selected),'SOURCE_REST_DUPLICATE_FACE_BLOCK')
        blocks[row['name']]=(v[offset:offset+count],local)
        covered|=selected;offset+=count
    require(offset==len(v) and np.all(covered),'SOURCE_REST_OBJECT_COVERAGE')
    vertices=[];faces=[];owners=[];index=0
    for owner,name in enumerate(sorted(names)):
        bv,bf=blocks[name];vertices.append(bv);faces.append(bf+index)
        owners.extend([owner]*len(bf));index+=len(bv)
    return {'rest_vertices_source_world':np.concatenate(vertices),'faces':np.concatenate(faces).astype(np.int64),
            'face_owner_privileged':np.asarray(owners,np.int32)},sorted(names)


def build(subject_id,source_sha256,import_path,admission_path,candidate_path,snapshot_path,output_dir,*,matrix_tolerance=1e-6):
    ip,ap,cp,sp=map(lambda p:Path(p).resolve(),(import_path,admission_path,candidate_path,snapshot_path))
    imported,admission,candidate,snapshot=[json.loads(p.read_text()) for p in (ip,ap,cp,sp)]
    require(admission.get('schema')=='RealSaS.IrisBindObservationAdmission.v1' and admission.get('status')=='PASS_IRIS_BIND_OBSERVATIONS','SOURCE_REST_ADMISSION')
    require(candidate.get('schema')=='RealSaS.Controlled8ViewSourceRestCandidate.v2','SOURCE_REST_CANDIDATE')
    require(all(x.get('subject_id')==subject_id and x.get('source_sha256')==source_sha256 for x in (admission,candidate)),'SOURCE_REST_SUBJECT_SOURCE')
    require(sha(ip)==admission['imported_geometry_sha256'],'SOURCE_REST_EXACT_IMPORT_BYTES_DRIFT')
    require(sha(cp)==admission['candidate_sha256'] and sha(sp)==candidate['source_scene_snapshot']['sha256'],'SOURCE_REST_LINEAGE_DRIFT')
    require(snapshot['source_sha256']==source_sha256,'SOURCE_REST_SNAPSHOT_SOURCE')
    validate_imported_scene_snapshot(snapshot,tolerance=matrix_tolerance)
    recipe=imported['import'];expected=candidate['import']
    for key in ('source_sha256','operation','importer_options','pose_policy','raw_source_mutated','excluded_import_controls'):
        require(recipe.get(key)==expected.get(key),'SOURCE_REST_IMPORT_RECIPE:'+key)
    require(recipe['source_sha256']==source_sha256 and recipe['pose_policy']=='SOURCE_BIND_REST__NO_ANIMATION' and recipe['raw_source_mutated'] is False,'SOURCE_REST_IMPORT_SCOPE')
    require(len(imported['vertices'])==admission['imported_vertex_count'] and imported['bone_count']==admission['bone_count']==len(snapshot['bones']),'SOURCE_REST_ADMITTED_CARDINALITY')
    nodes={r['source_import_name']:r for r in snapshot['nodes'] if r['type']=='MESH'}
    for obj in imported['objects']:
        require(obj['name'] in nodes,'SOURCE_REST_SNAPSHOT_OBJECT_SET')
        matrix=np.asarray(obj['world'],np.float64);reference=np.asarray(nodes[obj['name']]['node_to_source_world'],np.float64)
        require(matrix.shape==reference.shape==(4,4) and np.isfinite(matrix).all() and np.allclose(matrix,reference,atol=matrix_tolerance,rtol=0),'SOURCE_REST_OBJECT_FRAME')
    arrays,names=decode_import(imported)
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    output=root/'SOURCE_REST_GEOMETRY_V1.npz';tmp=output.with_suffix('.partial.npz');np.savez_compressed(tmp,**arrays);tmp.replace(output)
    result={'schema':SCHEMA,'status':STATUS,'subject_id':subject_id,'source_sha256':source_sha256,
            'parents':{'admitted_import':ref(ip),'observation_admission':ref(ap),'source_rest_candidate':ref(cp),'source_snapshot':ref(sp)},
            'output':ref(output),'owner_order':names,'counts':{'vertices':len(arrays['rest_vertices_source_world']),'faces':len(arrays['faces']),'owners':len(names)},
            'policy':{'owner_order':'SOURCE_IMPORT_NAME_ASCENDING','matrix_tolerance':float(matrix_tolerance)},
            'authority':{'source_rest_candidate_only':True,'kinematic_response_measured':False,'skin_authority_minted':False,
                         'appearance_authority_minted':False,'family_authority_minted':False,'training_authorized':False}}
    path=root/'SOURCE_REST_GEOMETRY_MANIFEST_V1.json';path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result
