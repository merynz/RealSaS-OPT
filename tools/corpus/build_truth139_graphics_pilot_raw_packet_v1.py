#!/usr/bin/env python3
"""Build a byte-exact light raw-source packet for the frozen Truth139 Graphics pilot."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

WORK_SCHEMA='RealSaS.Truth139MaterializationWorkOrders.v1'
PILOT_SCHEMA='RealSaS.Truth139GraphicsMaterializationPilot.v1'
APPARATUS_SCHEMA='RealSaS.Truth139ApparatusManifest.v1'
CLOSURE_SCHEMA='RealSaS.Truth139StructuralCorpusClosureReceipt.v1'
OUTPUT_SCHEMA='RealSaS.Truth139GraphicsPilotRawPacket.v1'
STATUS='PASS_TRUTH139_GRAPHICS_PILOT_RAW_BYTES_PACKED__RENDER_AND_TRUTH_NOT_PERFORMED'

def sha(path: Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def dh(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load(path, code):
    path=Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(code+':'+str(path))
    v=json.loads(path.read_text())
    if not isinstance(v,dict): raise ValueError(code+'_NOT_OBJECT')
    return path,v

def build(*,source_root,work_orders,pilot,apparatus,closure_receipt,output_root):
    source_root=Path(source_root).resolve(); output_root=Path(output_root).resolve()
    wp,work=load(work_orders,'RAW_PACKET_WORK'); pp,p=load(pilot,'RAW_PACKET_PILOT'); ap,a=load(apparatus,'RAW_PACKET_APPARATUS'); cp,c=load(closure_receipt,'RAW_PACKET_CLOSURE')
    if work.get('schema')!=WORK_SCHEMA: raise ValueError('RAW_PACKET_WORK_SCHEMA')
    if c.get('schema')!=CLOSURE_SCHEMA or c.get('status')!='PASS_TRUTH139_STRUCTURAL_CORPUS_CLOSED__RICH_TRUTH_MATERIALIZATION_READY': raise ValueError('RAW_PACKET_CLOSURE_SCHEMA_STATUS')
    expected_work_file_sha=str(((c.get('artifacts') or {}).get('TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json') or {}).get('sha256') or '')
    if expected_work_file_sha!=sha(wp): raise ValueError('RAW_PACKET_CLOSURE_WORK_ORDER_FILE_DRIFT')
    if p.get('schema')!=PILOT_SCHEMA or p.get('status')!='READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY': raise ValueError('RAW_PACKET_PILOT_SCHEMA_STATUS')
    if a.get('schema')!=APPARATUS_SCHEMA or a.get('status')!='READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED': raise ValueError('RAW_PACKET_APPARATUS_SCHEMA_STATUS')
    work_id=str(work.get('work_order_identity_sha256') or '')
    if p.get('parent_work_order_identity_sha256')!=work_id or a.get('parent_work_order_identity_sha256')!=work_id: raise ValueError('RAW_PACKET_WORK_IDENTITY_DRIFT')
    if a.get('parent_pilot_identity_sha256')!=p.get('pilot_identity_sha256'): raise ValueError('RAW_PACKET_PILOT_IDENTITY_DRIFT')
    rows={str(r.get('subject_id') or ''):r for r in work.get('subjects') or ()}
    app={str(r.get('subject_key') or ''):r for r in a.get('selected') or ()}
    frozen=list(p.get('selected') or ())
    if len(frozen)!=int(p.get('pilot_size',-1)) or len(app)!=len(frozen): raise ValueError('RAW_PACKET_CARDINALITY')
    if output_root.exists():
        if any(output_root.iterdir()): raise ValueError('RAW_PACKET_OUTPUT_NOT_EMPTY')
    output_root.mkdir(parents=True,exist_ok=True)
    files_root=output_root/'source_root'; files_root.mkdir()
    packed=[]
    for fr in frozen:
        sid=str(fr.get('subject_id') or '')
        row=rows.get(sid); ar=app.get(sid)
        if row is None or ar is None: raise ValueError('RAW_PACKET_SUBJECT_BINDING:'+sid)
        src=dict(row.get('source') or {})
        if dict(ar.get('source') or {})!=src: raise ValueError('RAW_PACKET_APPARATUS_SOURCE_DRIFT:'+sid)
        if src.get('sha256')!=fr.get('source_sha256') or src.get('corpus_relative_path')!=fr.get('corpus_relative_path'): raise ValueError('RAW_PACKET_PILOT_SOURCE_DRIFT:'+sid)
        deps=list(src.get('dependencies') or ())
        if int(src.get('dependency_count',len(deps)))!=len(deps): raise ValueError('RAW_PACKET_DEPENDENCY_COUNT:'+sid)
        # Current pilot was deliberately chosen from self-contained GLTFs. Fail closed rather than
        # guessing dependency URI semantics if a future pilot changes this property.
        if deps: raise ValueError('RAW_PACKET_EXTERNAL_DEPENDENCIES_REQUIRE_EXPLICIT_PACKER_V2:'+sid)
        rel=Path(str(src.get('corpus_relative_path') or ''))
        if rel.is_absolute() or '..' in rel.parts or not rel.parts: raise ValueError('RAW_PACKET_SOURCE_PATH:'+sid)
        original=(source_root/rel).resolve()
        if not original.is_relative_to(source_root) or not original.is_file(): raise FileNotFoundError('RAW_PACKET_SOURCE_MISSING:'+sid)
        actual=sha(original); expected=str(src.get('sha256') or '')
        if actual!=expected: raise ValueError('RAW_PACKET_SOURCE_SHA:'+sid+':'+actual+'!='+expected)
        target=(files_root/rel).resolve()
        if not target.is_relative_to(files_root): raise ValueError('RAW_PACKET_TARGET_ESCAPE:'+sid)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(original,target)
        if sha(target)!=expected: raise ValueError('RAW_PACKET_COPY_SHA:'+sid)
        packed.append({'pilot_index':int(fr.get('pilot_index',-1)),'subject_id':sid,'corpus_relative_path':rel.as_posix(),'sha256':expected,'byte_size':target.stat().st_size,'dependency_count':0})
    meta=output_root/'metadata'; meta.mkdir()
    bindings=[]
    for name,path in [('TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json',wp),('TRUTH139_GRAPHICS_MATERIALIZATION_PILOT_V1.json',pp),('TRUTH139_GRAPHICS_APPARATUS_MANIFEST_V1.json',ap),('STRUCTURAL_CORPUS_CLOSURE_RECEIPT.json',cp)]:
        dst=meta/name; shutil.copyfile(path,dst); bindings.append({'path':'metadata/'+name,'sha256':sha(dst)})
    out={'schema':OUTPUT_SCHEMA,'date':'2026-10-29','status':STATUS,'work_order_identity_sha256':work_id,'pilot_identity_sha256':p['pilot_identity_sha256'],'apparatus_identity_sha256':a['apparatus_identity_sha256'],'subject_count':len(packed),'subjects':packed,'metadata_files':bindings,'structural_closure':{'github_run_id':str(c.get('github_run_id') or ''),'git_commit_sha':str(c.get('git_commit_sha') or ''),'receipt_identity_sha256':str(c.get('receipt_identity_sha256') or ''),'work_order_file_sha256':expected_work_file_sha},'policy':{'exact_source_bytes_only':True,'external_dependencies_supported':False,'render_performed':False,'derived_truth_materialized':False,'training_performed':False,'split_created':False,'qualification_minted':False},'training_performed':False,'qualification_minted':False}
    out['packet_identity_sha256']=dh(out)
    manifest=output_root/'RAW_PACKET_MANIFEST.json'; manifest.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    return out

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--source-root',required=True,type=Path); pa.add_argument('--work-orders',required=True,type=Path); pa.add_argument('--pilot',required=True,type=Path); pa.add_argument('--apparatus',required=True,type=Path); pa.add_argument('--closure-receipt',required=True,type=Path); pa.add_argument('--output-root',required=True,type=Path); args=pa.parse_args()
    r=build(source_root=args.source_root,work_orders=args.work_orders,pilot=args.pilot,apparatus=args.apparatus,closure_receipt=args.closure_receipt,output_root=args.output_root)
    print(json.dumps({'status':r['status'],'subject_count':r['subject_count'],'packet_identity_sha256':r['packet_identity_sha256']},sort_keys=True))
if __name__=='__main__': main()
