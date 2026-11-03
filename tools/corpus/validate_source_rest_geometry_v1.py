"""Independent source-rest index reconstruction; no producer import."""
import hashlib,json
from pathlib import Path
import numpy as np
from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot


def bound(row):
    p=Path(row['path'])
    with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=row['sha256']:raise ValueError('SOURCE_REST_COURT_BYTES_DRIFT')
    return p


def validate(manifest_path):
    p=Path(manifest_path);m=json.loads(p.read_text())
    if m.get('schema')!='RealSaS.AdmittedSourceRestGeometry.v1' or m.get('status')!='PASS_ADMITTED_SOURCE_REST_CANDIDATE__FAMILY_COURTS_REQUIRED':raise ValueError('SOURCE_REST_COURT_PARENT')
    parents=m['parents'];a=json.loads(bound(parents['observation_admission']).read_text());c=json.loads(bound(parents['source_rest_candidate']).read_text())
    s=json.loads(bound(parents['source_snapshot']).read_text());ip=bound(parents['admitted_import']);raw=json.loads(ip.read_text())
    validate_imported_scene_snapshot(s,tolerance=m['policy']['matrix_tolerance'])
    if a.get('status')!='PASS_IRIS_BIND_OBSERVATIONS' or any(x['source_sha256']!=m['source_sha256'] for x in (a,c,s)) or a['subject_id']!=m['subject_id'] or c['subject_id']!=m['subject_id']:raise ValueError('SOURCE_REST_COURT_SOURCE')
    if a['imported_geometry_sha256']!=parents['admitted_import']['sha256'] or a['candidate_sha256']!=parents['source_rest_candidate']['sha256'] or c['source_scene_snapshot']['sha256']!=parents['source_snapshot']['sha256']:raise ValueError('SOURCE_REST_COURT_LINEAGE')
    if raw['import']!=c['import']:
        # Observation recipes can include diagnostics in addition to the sealed
        # fields. Compare the actual scope fields independently.
        for k in ('source_sha256','operation','importer_options','pose_policy','raw_source_mutated','excluded_import_controls'):
            if raw['import'].get(k)!=c['import'].get(k):raise ValueError('SOURCE_REST_COURT_RECIPE')
    if raw['import'].get('pose_policy')!='SOURCE_BIND_REST__NO_ANIMATION' or raw['import'].get('raw_source_mutated') is not False or raw['import'].get('source_sha256')!=m['source_sha256']:raise ValueError('SOURCE_REST_COURT_IMPORT_SCOPE')
    with np.load(bound(m['output']),allow_pickle=False) as z:
        if set(z.files)!={'rest_vertices_source_world','faces','face_owner_privileged'}:raise ValueError('SOURCE_REST_COURT_EXTRA_CAPABILITY_ARRAY')
        v=z['rest_vertices_source_world'];f=z['faces'];owner=z['face_owner_privileged']
    raw_v=np.asarray(raw['vertices'],np.float64);raw_f=np.asarray(raw['faces']);offset=0;blocks={};seen=set()
    if raw_v.ndim!=2 or raw_v.shape[1:]!=(3,) or not np.isfinite(raw_v).all() or raw_f.ndim!=2 or raw_f.shape[1:]!=(3,) or raw_f.dtype.kind not in 'iu' or np.any(raw_f<0) or np.any(raw_f>=len(raw_v)):raise ValueError('SOURCE_REST_COURT_RAW_ARRAYS')
    for row in raw['objects']:
        name=row['name'];count=row['vertex_count']
        if name in blocks or type(count) is not int or count<0 or offset+count>len(raw_v):raise ValueError('SOURCE_REST_COURT_BLOCK')
        ids=np.flatnonzero((raw_f[:,0]>=offset)&(raw_f[:,0]<offset+count));local=raw_f[ids]-offset
        if len(local) and (local.min()<0 or local.max()>=count):raise ValueError('SOURCE_REST_COURT_CROSS_OBJECT')
        seen.update(map(int,ids));blocks[name]=(raw_v[offset:offset+count],local);offset+=count
    if offset!=len(raw_v) or len(seen)!=len(raw_f):raise ValueError('SOURCE_REST_COURT_COVERAGE')
    if len(raw_v)!=a['imported_vertex_count'] or raw['bone_count']!=a['bone_count'] or raw['bone_count']!=len(s['bones']):raise ValueError('SOURCE_REST_COURT_CARDINALITY')
    ordered=sorted(blocks);expected_v=[];expected_f=[];expected_owner=[];offset=0
    nodes={r['source_import_name']:r for r in s['nodes'] if r['type']=='MESH'}
    for row in raw['objects']:
        if row['name'] not in nodes or not np.allclose(np.asarray(row['world']),np.asarray(nodes[row['name']]['node_to_source_world']),atol=m['policy']['matrix_tolerance'],rtol=0):raise ValueError('SOURCE_REST_COURT_FRAME')
    for oi,name in enumerate(ordered):
        bv,bf=blocks[name];expected_v.extend(bv);expected_f.extend(bf+offset);expected_owner.extend([oi]*len(bf));offset+=len(bv)
    if not np.array_equal(v,np.asarray(expected_v)) or not np.array_equal(f,np.asarray(expected_f)) or not np.array_equal(owner,np.asarray(expected_owner)):raise ValueError('SOURCE_REST_COURT_GEOMETRY')
    if m['owner_order']!=ordered or m['authority'].get('kinematic_response_measured') is not False or m['authority'].get('training_authorized') is not False:raise ValueError('SOURCE_REST_COURT_AUTHORITY')
    return {'schema':'RealSaS.AdmittedSourceRestGeometryCourt.v1','status':'PASS_SOURCE_REST_ACQUISITION_COURT__FAMILY_AUTHORITY_SEPARATE',
            'subject_id':m['subject_id'],'source_sha256':m['source_sha256'],'counts':{'vertices':len(v),'faces':len(f),'owners':len(ordered)},
            'authority':{'independent_source_rest_reconstruction_passed':True,'response_authority_minted':False,'family_authority_minted':False,'training_authorized':False}}
