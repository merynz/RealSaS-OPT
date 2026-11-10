#!/usr/bin/env python3
"""Full-universe, source-key-scoped, resumable physical evidence worker.

This worker is not a training-admission engine. It never turns metadata, a
container parse, a normalized derivative, or a license claim into a PASS for
character, rights, orientation, front/back art, lineage or model supervision.

Inputs:
  * V24 16,397-source queue (.csv/.csv.gz)
  * Source location index (.csv), fields: source_key,scope,artifact_path,
    optional structure_path,expected_sha256,expected_structure_sha256.
Scope is one of ORIGINAL_BYTES, NORMALIZED_DERIVATIVE, PROCESSED_DERIVATIVE.
Physical bytes must be mounted in a trusted local staging area before running.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,pathlib,sqlite3,struct,time,zipfile,io
from collections import Counter

ALLOWED_SCOPES={'ORIGINAL_BYTES','NORMALIZED_DERIVATIVE','PROCESSED_DERIVATIVE'}
COLUMNS=['source_key','scope','artifact_path','structure_path','expected_sha256','expected_structure_sha256']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def open_csv(path):
    return gzip.open(path,'rt',encoding='utf8',newline='') if str(path).endswith('.gz') else open(path,encoding='utf8',newline='')

def initialize(db):
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=10000')
    db.execute('''CREATE TABLE IF NOT EXISTS source_court (
        source_key TEXT PRIMARY KEY, provider TEXT NOT NULL, scope TEXT NOT NULL,
        status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
        artifact_path TEXT NOT NULL DEFAULT '', file_sha256 TEXT NOT NULL DEFAULT '',
        structure_sha256 TEXT NOT NULL DEFAULT '', bytes INTEGER NOT NULL DEFAULT 0,
        evidence_json TEXT NOT NULL DEFAULT '{}', verified_at TEXT NOT NULL DEFAULT '',
        training_authorized INTEGER NOT NULL DEFAULT 0 CHECK(training_authorized=0),
        original_rights_admitted INTEGER NOT NULL DEFAULT 0 CHECK(original_rights_admitted=0),
        character_verified INTEGER NOT NULL DEFAULT 0 CHECK(character_verified=0),
        front_back_verified INTEGER NOT NULL DEFAULT 0 CHECK(front_back_verified=0)
    )''')
    db.commit()

def read_universe(path):
    with open_csv(path) as f: roster=list(csv.DictReader(f))
    keys=[r['source_key'] for r in roster]
    if len(set(keys))!=len(keys):raise ValueError('UNIVERSE_DUPLICATE_SOURCE_KEY')
    if len(roster)!=16397:raise ValueError('UNIVERSE_CARDINALITY_NOT_16397')
    if any(r.get('training_authorized')!='False' or r.get('rights_training_admission')!='False' for r in roster):
        raise ValueError('UNIVERSE_AUTHORITY_BREACH')
    return {r['source_key']:r for r in roster}

def read_index(path,universe):
    with open_csv(path) as f: rows=list(csv.DictReader(f))
    out={}
    for r in rows:
        key=r['source_key'];scope=r['scope'];root=r.get('artifact_path','')
        if key not in universe:raise ValueError('INDEX_UNREGISTERED_SOURCE:'+key)
        if scope not in ALLOWED_SCOPES:raise ValueError('INDEX_INVALID_SCOPE:'+scope)
        if key in out:raise ValueError('INDEX_DUPLICATE_SOURCE:'+key)
        if not root:raise ValueError('INDEX_NO_ARTIFACT_PATH:'+key)
        if scope=='NORMALIZED_DERIVATIVE' and universe[key]['allowlisted_master726_normalized']!='True':
            raise ValueError('NON_ALLOWLISTED_NPZ_COURT:'+key)
        if universe[key]['protected_holdout']=='True':
            raise ValueError('HOLDOUT_FORBIDDEN:'+key)
        out[key]=r
    return out

def glb_header(p):
    with p.open('rb') as f:
        if f.read(4)!=b'glTF':raise ValueError('INVALID_GLB_MAGIC')
        version,total=struct.unpack('<II',f.read(8))
        if version!=2 or total!=p.stat().st_size:raise ValueError('INVALID_GLB_LENGTH_OR_VERSION')
        clen,ctype=struct.unpack('<II',f.read(8))
        if ctype!=0x4e4f534a or clen<2 or clen>total:raise ValueError('GLB_JSON_CHUNK_INVALID')
        raw=f.read(clen).rstrip(b' \t\r\n\x00')
    return json.loads(raw)

def scene_container_observation(p):
    ext=p.suffix.lower(); n=p.stat().st_size
    if ext=='.glb':
        d=glb_header(p)
        return ('RAW_SOURCE_CONTAINER_STRUCTURE_ONLY',{
            'format':'glb','mesh_declarations':len(d.get('meshes',[])),
            'primitive_declarations':sum(len(m.get('primitives',[])) for m in d.get('meshes',[])),
            'skin_declarations':len(d.get('skins',[])),
            'node_declarations':len(d.get('nodes',[])),
            'material_declarations':len(d.get('materials',[])),
            'image_declarations':len(d.get('images',[])),
            'animation_declarations':len(d.get('animations',[])),
            'decoded_accessor_geometry_courted':False})
    if ext=='.gltf':
        d=json.loads(p.read_text(encoding='utf8'))
        if d.get('asset',{}).get('version')!='2.0':raise ValueError('UNSUPPORTED_GLTF_VERSION')
        return ('RAW_SOURCE_CONTAINER_STRUCTURE_ONLY',{
            'format':'gltf','mesh_declarations':len(d.get('meshes',[])),
            'primitive_declarations':sum(len(m.get('primitives',[])) for m in d.get('meshes',[])),
            'skin_declarations':len(d.get('skins',[])),'animation_declarations':len(d.get('animations',[])),
            'external_dependencies':[x.get('uri') for kind in ('buffers','images') for x in d.get(kind,[]) if x.get('uri') and not x['uri'].startswith('data:')],
            'external_dependencies_complete':False,'decoded_accessor_geometry_courted':False})
    if ext=='.obj':
        verts=faces=0;unknown_refs=0
        with p.open('r',encoding='utf8',errors='replace') as f:
            for line in f:
                if line.startswith('v '):verts+=1
                elif line.startswith('f '):
                    tok=line.split()[1:]
                    if len(tok)<3:raise ValueError('OBJ_FACE_LT3')
                    faces+=1
                    for ref in tok:
                        i=int(ref.split('/')[0]);
                        if i==0 or (i>0 and i>verts) or (i<0 and -i>verts): unknown_refs+=1
        if unknown_refs:raise ValueError('OBJ_FACE_REFERENCE_UNRESOLVED')
        return ('RAW_SOURCE_OBJ_VERTEX_FACE_RECORDS_ONLY',{
            'format':'obj','vertex_records':verts,'polygon_records':faces,
            'triangulation_courted':False,'mesh_component_ownership_courted':False})
    if ext=='.fbx':
        with p.open('rb') as f:magic=f.read(27)
        return ('RAW_SOURCE_BYTES_REHASHED_IMPORT_REQUIRED',{'format':'fbx','binary_header':magic.startswith(b'Kaydara FBX Binary'),'requires_authoritative_import':True})
    if ext=='.blend':
        with p.open('rb') as f:magic=f.read(7)
        if magic!=b'BLENDER':raise ValueError('INVALID_BLEND_MAGIC')
        return ('RAW_SOURCE_BYTES_REHASHED_IMPORT_REQUIRED',{'format':'blend','requires_authoritative_import':True})
    if ext=='.dae':
        with p.open('rb') as f:magic=f.read(4096)
        if b'COLLADA' not in magic:raise ValueError('INVALID_COLLADA_HEADER')
        return ('RAW_SOURCE_BYTES_REHASHED_IMPORT_REQUIRED',{'format':'dae','requires_authoritative_import':True})
    if ext=='.npz':
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('NPZ_CRC_FAILURE')
            info=[{'name':i.filename,'uncompressed_bytes':i.file_size} for i in z.infolist()]
        return ('RAW_NPZ_CONTAINER_ONLY',{'format':'npz','entries':info,'array_shape_and_geometry_not_yet_courted':True})
    if ext=='.zip':
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('ZIP_CRC_FAILURE')
            info=[{'name':i.filename,'uncompressed_bytes':i.file_size} for i in z.infolist()]
        return ('RAW_ARCHIVE_CRC_ONLY',{'format':'zip','entries':info,'content_semantics_not_courted':True})
    return ('RAW_SOURCE_BYTES_REHASHED_UNPARSED',{'format':ext,'bytes':n})

def inspect(record):
    p=pathlib.Path(record['artifact_path'])
    if not p.is_file():raise FileNotFoundError('SOURCE_NOT_MOUNTED:'+str(p))
    if not p.stat().st_size:raise ValueError('EMPTY_SOURCE_BYTES')
    result={'artifact_sha256':digest(p),'bytes':p.stat().st_size}
    expect=(record.get('expected_sha256') or '').strip().lower()
    if expect and result['artifact_sha256']!=expect:raise ValueError('SOURCE_SHA256_MISMATCH')
    scope=record['scope']
    status,obs=scene_container_observation(p)
    result.update(obs)
    result['scope']=scope
    result['original_byte_parity']=scope=='ORIGINAL_BYTES'
    result['rights_training_admission']=False
    result['character_verified']=False
    result['anatomical_front_verified']=False
    result['front_back_verified']=False
    result['lineage_dedup_verified']=False
    result['skin_influence_semantics_verified']=False
    structure_sha=''
    structure=record.get('structure_path') or ''
    if scope=='NORMALIZED_DERIVATIVE':
        if p.suffix.lower()!='.npz':raise ValueError('NORMALIZED_SCOPE_REQUIRES_NPZ')
        if not structure:raise ValueError('NORMALIZED_SCOPE_REQUIRES_STRUCTURE')
        sp=pathlib.Path(structure)
        if not sp.is_file():raise FileNotFoundError('MISSING_SOURCE_STRUCTURE:'+structure)
        structure_sha=digest(sp);json.loads(sp.read_text(encoding='utf8'))
        expected=(record.get('expected_structure_sha256') or '').lower()
        if expected and structure_sha!=expected:raise ValueError('STRUCTURE_SHA256_MISMATCH')
        # Stronger numerical evidence is only granted by the separate V20-V24 typed numeric court.
        status='NORMALIZED_DERIVATIVE_BYTES_AND_CRC_OBSERVED_NUMERIC_COURT_PENDING'
    return status,result,structure_sha

def run(args):
    universe=read_universe(args.universe);index=read_index(args.index,universe)
    db=sqlite3.connect(args.db);initialize(db)
    existing={k:s for k,s in db.execute('SELECT source_key,status FROM source_court')}
    # Write 16,397 explicit rows, even for unlocated sources. No registry candidate disappears.
    for key,r in universe.items():
        if key not in existing:
            # Existing immutable V20-V25 source receipts are counted as prior scoped evidence,
            # not falsely tagged unreviewed. Indexed files still receive a new local readback.
            prior=(r.get('normalized_numeric_source_court')=='True' and key not in index)
            prior_state='PRIOR_NORMALIZED_NUMERIC_RECEIPT_RECORDED_NOT_LOCALLY_REHASHED' if prior else 'NOT_MOUNTED_UNREVIEWED'
            db.execute('INSERT INTO source_court (source_key,provider,scope,status) VALUES (?,?,?,?)',
                       (key,r['provider'],'',prior_state))
    db.commit()
    ok=fail=skip=0
    for key,r in sorted(index.items()):
        old=existing.get(key)
        if old and old not in ('NOT_MOUNTED_UNREVIEWED','COURT_FAILED') and not args.recheck:
            skip+=1;continue
        try:
            status,obs,structure_sha=inspect(r)
            artifact_sha=obs['artifact_sha256'];bytes_=obs['bytes'];error=''
            ok+=1
        except Exception as e:
            status='COURT_FAILED';obs={};structure_sha=artifact_sha='';bytes_=0;error=f'{type(e).__name__}: {e}'[:500];fail+=1
        db.execute('''UPDATE source_court SET scope=?,status=?,error=?,artifact_path=?,file_sha256=?,
                     structure_sha256=?,bytes=?,evidence_json=?,verified_at=? WHERE source_key=?''',
                   (r['scope'],status,error,r['artifact_path'],artifact_sha,structure_sha,bytes_,
                    json.dumps(obs,sort_keys=True,ensure_ascii=False),time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),key))
        db.commit()  # durable resume after each candidate
        if args.max_sources and ok+fail>=args.max_sources:break
    csvp=pathlib.Path(args.report);csvp.parent.mkdir(parents=True,exist_ok=True)
    counters=Counter()
    with gzip.open(csvp,'wt',newline='',encoding='utf8') as f:
        fields=['source_key','provider','scope','status','error','file_sha256','structure_sha256','bytes',
                'verified_at','training_authorized','original_rights_admitted','character_verified','front_back_verified']
        w=csv.writer(f);w.writerow(fields)
        for row in db.execute('SELECT '+','.join(fields)+' FROM source_court ORDER BY source_key'):
            counters[row[3]]+=1;w.writerow(row)
    assert sum(counters.values())==16397
    report={'source_universe':16397,'newly_inspected':ok,'newly_failed':fail,'cached_skipped':skip,
            'physical_statuses':dict(counters),'training_admitted':0,'original_rights_admitted':0,
            'verified_character':0,'verified_two_view_front_back':0}
    pathlib.Path(args.summary).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    if fail:raise SystemExit(2)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--universe',type=pathlib.Path,required=True)
    ap.add_argument('--index',type=pathlib.Path,required=True)
    ap.add_argument('--db',type=pathlib.Path,required=True)
    ap.add_argument('--report',type=pathlib.Path,required=True)
    ap.add_argument('--summary',type=pathlib.Path,required=True)
    ap.add_argument('--max-sources',type=int,default=0)
    ap.add_argument('--recheck',action='store_true')
    run(ap.parse_args())