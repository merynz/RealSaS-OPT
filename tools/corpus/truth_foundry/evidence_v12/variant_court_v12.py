#!/usr/bin/env python3
"""Read-only legacy variant evidence: exact asset linkage and zero authority promotion.

Reads four historical JSON receipts per candidate, never normalized.npz or 3D assets.
No source byte or ML rights claims. Input references are SHA locked by the Action.
"""
import argparse, csv, gzip, hashlib, json, math, pathlib, collections

FAMILIES=('source_structure.json','technical_audit.json','admission.json','normalization_meta.json')
STATUS='OBSERVED_LEGACY_VARIANT_NOT_INDEPENDENTLY_COURTED'
FACTS={
    'source_vertex_face_count':'mesh_geometry',
    'skin_weight_normalization':'skin_statistics',
    'zero_weight_vertices_and_unbound_mesh':'skin_statistics',
    'canonical_frame_transform':'canonical_transform',
    'source_name_only_teacher_hint_isolation':'teacher_name_hints',
    'original_asset_provider_locator':'provider_locator_claim',
    'attribution_obligations':'historical_attribution_obligation',
}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def jdump(o):return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def as_nonneg_int(x):
    if isinstance(x,bool):return None
    try:
        v=int(x)
        if v<0 or float(x)!=v:return None
        return v
    except(TypeError,ValueError,OverflowError):return None
def finite(x):
    try:
        f=float(x)
        return f if math.isfinite(f) else None
    except(TypeError,ValueError,OverflowError):return None
def jsonread(p):
    if p.stat().st_size>8*1024*1024:raise ValueError('UNBOUNDED_JSON_GT_8M')
    return json.loads(p.read_text(encoding='utf-8'))
def parse_subject(row,folder):
    key=row['source_key'];cid=row['candidate_id'];orig_sha=row.get('original_sha_claim') or ''
    source={'source_key':key,'candidate_id':cid,'variant_path_expected':str(folder.name),
            'court_scope':'HISTORICAL_VARIANT_RECEIPT_NOT_FULL_SOURCE_SCENE','training_authorized':False,
            'source_original_bytes_rehashed':False,'current_rights_court':False}
    facts={};meshes=[];bones=[];issues=[];files={};docs={}
    for f in FAMILIES:
        path=folder/f
        if not path.is_file():
            issues.append('MISSING_'+f.upper().replace('.','_'));continue
        try:
            files[f]=sha(path)
            o=jsonread(path)
            if not isinstance(o,dict):raise ValueError('JSON_ROOT_NOT_OBJECT')
            docs[f]=o
        except(Exception) as e:
            issues.append('INVALID_'+f.upper().replace('.','_')+':'+type(e).__name__)
    source['receipt_sha256']=files
    ad=docs.get('admission.json') or {}
    si=ad.get('source_identity') or {}
    if ad:
        if ad.get('candidate_id')!=cid: issues.append('ADMISSION_CANDIDATE_ID_DRIFT')
        if ad.get('canonical_asset_id')!=key.split(':',1)[-1]:issues.append('ADMISSION_ASSET_ID_DRIFT')
        if row['provider']!=cid.split(':',1)[0]:issues.append('PROVIDER_CANDIDATE_ID_DRIFT')
        claim=str(si.get('source_sha256') or '')
        if orig_sha and claim and orig_sha!=claim:issues.append('HISTORICAL_SHA_CLAIM_DRIFT')
        source.update({'historical_admission_decision':ad.get('decision'),
                       'historical_technical_status':ad.get('technical'),
                       'historical_legal_decision':(ad.get('legal') or {}).get('decision'),
                       'historical_object_license_metadata':(ad.get('legal') or {}).get('object_license'),
                       'historical_source_sha_claim':claim,
                       'historical_source_size_claim':si.get('source_size_bytes'),
                       'historical_provider_revision':si.get('revision'),
                       'historical_source_locator':si.get('source_url'),
                       'historical_split_claim':ad.get('split_state'),
                       'historical_attribution_required':(ad.get('legal') or {}).get('attribution_required'),
                       'historical_rights_court_is_current':False})
        if si.get('source_url'):facts['original_asset_provider_locator']={'url':si['source_url'],'source_sha_claim':claim}
        if isinstance((ad.get('legal') or {}).get('attribution_required'),bool):
            facts['attribution_obligations']={'historical_attribution_required':ad['legal']['attribution_required'],
                                             'historical_legal_decision_not_current':(ad.get('legal') or {}).get('decision')}
    st=docs.get('source_structure.json') or {}
    if st:
        objects=st.get('mesh_objects') or []
        count=as_nonneg_int(st.get('bone_count'))
        names=st.get('bone_names') or []
        if not isinstance(objects,list) or not isinstance(names,list):issues.append('INVALID_SCENE_LAYOUT');objects=[];names=[]
        if count is not None and len(names)!=count:issues.append('BONE_COUNT_NAME_DRIFT')
        if len(names)!=len(set(str(n) for n in names)):issues.append('DUPLICATE_BONE_NAMES')
        span_end=0;seen_idx=set();verts=0;faces=0
        for i,m in enumerate(objects):
            if not isinstance(m,dict):issues.append('INVALID_MESH_OBJECT');continue
            start=as_nonneg_int(m.get('vertex_start'))
            v=as_nonneg_int(m.get('vertex_count'))
            f=as_nonneg_int(m.get('face_count'))
            if start is None or v is None or f is None:issues.append('INVALID_MESH_COUNTS');continue
            if v==0 or f==0:issues.append('EMPTY_MESH_OBJECT')
            if start<span_end:issues.append('OVERLAPPING_VERTEX_SPANS')
            span_end=max(span_end,start+v)
            verts+=v;faces+=f
            meshes.append({'source_key':key,'mesh_index':i,'name_teacher_only':str(m.get('name') or ''),
                           'vertex_start':start,'vertex_count':v,'face_count':f,'source_doc_sha256':files.get('source_structure.json','')})
        bones=[{'source_key':key,'bone_index':i,'name_teacher_only':str(n),'source_doc_sha256':files.get('source_structure.json','')} for i,n in enumerate(names)]
        source.update({'historical_variant_mesh_count':len(objects),'historical_variant_vertex_count':verts,
                       'historical_variant_face_count':faces,'historical_variant_bone_count':count,
                       'historical_variant_armature_name_teacher_only':st.get('armature'),
                       'historical_variant_original_source_path_claim':st.get('source')})
        if objects and len(meshes)==len(objects):facts['source_vertex_face_count']={'mesh_objects':len(objects),'vertices':verts,'faces':faces}
        if names and count==len(names):facts['source_name_only_teacher_hint_isolation']={'mesh_names':len(objects),'bone_name_count':len(names),'student_input_eligible':False}
    ta=docs.get('technical_audit.json') or {}
    if ta:
        sm=ta.get('skin_metrics') or {}
        norm=finite(sm.get('row_sum_mean'));err=finite(sm.get('row_sum_p99_abs_err'));zero=finite(sm.get('zero_row_fraction'))
        source.update({'historical_skin_pass':ta.get('skin_pass'),
                       'historical_geometry_pass':ta.get('geometry_pass'),
                       'historical_rig_pass':ta.get('rig_pass'),
                       'historical_skin_row_sum_mean':norm,
                       'historical_skin_row_p99_abs_err':err,
                       'historical_skin_zero_row_fraction':zero,
                       'historical_nondegenerate_face_fraction':finite((ta.get('geometry_metrics') or {}).get('nondegenerate_fraction'))})
        if norm is not None and err is not None and norm>=0 and err>=0:
            facts['skin_weight_normalization']={'historical_row_sum_mean':norm,'historical_row_sum_p99_abs_err':err,'skin_pass_claim':ta.get('skin_pass')}
        if zero is not None and 0<=zero<=1:
            facts['zero_weight_vertices_and_unbound_mesh']={'historical_zero_row_fraction':zero}
        bc=as_nonneg_int((ta.get('rig_metrics') or {}).get('bone_count'))
        if st and bc is not None and bc!=source.get('historical_variant_bone_count'):
            issues.append('TECHNICAL_VS_STRUCTURE_BONE_COUNT_DRIFT')
    nm=docs.get('normalization_meta.json') or {}
    if nm:
        center=nm.get('center');scale=finite(nm.get('scale'))
        inv=finite(nm.get('inverse_transform_max_abs_err'))
        if isinstance(center,list) and len(center)==3 and all(finite(x) is not None for x in center) and scale is not None and scale>0:
            facts['canonical_frame_transform']={'historical_center':[float(x) for x in center],'historical_scale':scale,
                                                'historical_inverse_error':inv}
            source.update({'historical_canonical_center':center,'historical_canonical_scale':scale,
                           'historical_inverse_transform_max_abs_err':inv})
        else:issues.append('INVALID_CANONICAL_TRANSFORM')
    if len(files)!=len(FAMILIES):issues.append('INCOMPLETE_VARIANT_RECEIPT_BUNDLE')
    fatal=[s for s in issues if any(z in s for z in ('DRIFT','INVALID_','OVERLAPPING_','DUPLICATE_'))]
    if fatal:
        # Record the contradictory raw documents for review, not even aggregate observed facts.
        facts={}
    source['inspection_state']='CONFLICT_QUARANTINED' if fatal else ('FOUR_DOCUMENTS_OBSERVED' if len(files)==4 else ('NO_VARIANT_RECEIPT_FOUND' if len(files)==0 else 'PARTIAL_RECEIPT_OBSERVED'))
    source['diagnostic_issues']=issues
    source['facet_candidates']=facts
    source['record_sha256']=hashlib.sha256(jdump({'files':files,'facts':facts,'issues':issues,'source_key':key}).encode()).hexdigest()
    return source,meshes,bones

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--queue',required=True);ap.add_argument('--baseline-inventory',required=True)
    ap.add_argument('--baseline-states',required=True);ap.add_argument('--baseline-values',required=True)
    ap.add_argument('--variant-dir',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    with open(args.queue,newline='',encoding='utf-8') as f: queue=list(csv.DictReader(f))
    assert len(queue)==1527 and len({x['source_key'] for x in queue})==1527
    assert len({x['candidate_id'] for x in queue})==1527
    base=pathlib.Path(args.variant_dir)
    inventory={}
    with gzip.open(args.baseline_inventory,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f);infields=r.fieldnames
        for row in r:
            if row['source_key'] in {x['source_key'] for x in queue}:
                inventory[row['source_key']]=row
    assert len(inventory)==1527
    hist=collections.Counter();kfacts=collections.Counter();records={}
    mesh_count=0;bone_count=0
    with gzip.open(dest/'LEGACY_VARIANT_SOURCE_RECEIPTS_1527.jsonl.gz','wt',encoding='utf-8') as jout,\
         gzip.open(dest/'LEGACY_VARIANT_MESH_DETAIL.csv.gz','wt',encoding='utf-8',newline='') as mout,\
         gzip.open(dest/'LEGACY_VARIANT_BONE_NAMES_TEACHER_ONLY.csv.gz','wt',encoding='utf-8',newline='') as bout:
        mw=csv.DictWriter(mout,fieldnames=['source_key','mesh_index','name_teacher_only','vertex_start','vertex_count','face_count','source_doc_sha256']);mw.writeheader()
        bw=csv.DictWriter(bout,fieldnames=['source_key','bone_index','name_teacher_only','source_doc_sha256']);bw.writeheader()
        for row in queue:
            key=row['source_key']; state=inventory[key].get('master_v11_source_recovery_status','')
            if state=='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED':
                rec={'source_key':key,'candidate_id':row['candidate_id'],'inspection_state':'SKIPPED_PROTECTED_HOLDOUT','facet_candidates':{},'diagnostic_issues':[],'source_original_bytes_rehashed':False,'training_authorized':False}
                meshes=[];bones=[]
            else:
                folder=base/row['candidate_id'].replace(':','_',1)
                rec,meshes,bones=parse_subject(row,folder)
            records[key]=rec
            for m in meshes:mw.writerow(m)
            for b in bones:bw.writerow(b)
            mesh_count+=len(meshes);bone_count+=len(bones)
            hist[rec['inspection_state']]+=1
            for k in rec['facet_candidates']:kfacts[k]+=1
            jout.write(jdump(rec)+'\n')
    deltas=[];observed_old=0;changes=collections.Counter()
    with gzip.open(args.baseline_states,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f); fields=r.fieldnames; assert len(fields)==265 and fields[0]=='source_key'
        with gzip.open(args.baseline_values,'rt',encoding='utf-8',newline='') as vf:
            vr=csv.DictReader(vf);assert vr.fieldnames==fields
            with gzip.open(dest/'FACET_STATES_EFFECTIVE_V12_16397x264.csv.gz','wt',encoding='utf-8',newline='') as sf,\
                 gzip.open(dest/'FACET_VALUES_EFFECTIVE_V12_16397x264.csv.gz','wt',encoding='utf-8',newline='') as of:
                sw=csv.DictWriter(sf,fieldnames=fields);sw.writeheader();vw=csv.DictWriter(of,fieldnames=fields);vw.writeheader()
                total=0
                for st,val in zip(r,vr):
                    total+=1;assert st['source_key']==val['source_key']
                    rec=records.get(st['source_key'])
                    if rec:
                        for facet,datum in rec['facet_candidates'].items():
                            assert facet in fields
                            if st[facet]!='UNKNOWN_UNREVIEWED':
                                observed_old+=1;continue
                            assert not val[facet],('unknown facet has nonempty value',st['source_key'],facet)
                            st[facet]=STATUS
                            val[facet]=jdump({'observation':datum,'lineage':'LEGACY_VARIANT_JSON_RECEIPT',
                                             'source_record_sha256':rec['record_sha256'],
                                             'authority':'HISTORICAL_SOURCE_DERIVATIVE_NOT_CURRENT_COURT'})
                            deltas.append({'source_key':st['source_key'],'facet':facet,'status':STATUS,'value':val[facet],
                                           'source_record_sha256':rec['record_sha256']})
                            changes[facet]+=1
                    sw.writerow(st);vw.writerow(val)
    assert total==16397
    with gzip.open(dest/'FACET_OBSERVATION_DELTA_V12.csv.gz','wt',encoding='utf-8',newline='') as f:
        fieldnames=['source_key','facet','status','value','source_record_sha256'];w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(deltas)
    with gzip.open(args.baseline_inventory,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f);oldfields=r.fieldnames
        newfields=['variant_v12_inspection_state','variant_v12_legacy_mesh_objects','variant_v12_legacy_vertices','variant_v12_legacy_faces','variant_v12_legacy_bones','variant_v12_skin_row_mean','variant_v12_skin_row_p99_err','variant_v12_skin_zero_row_fraction','variant_v12_canonical_scale','variant_v12_prior_legal_decision','variant_v12_prior_admission','variant_v12_receipt_sha256','variant_v12_not_original_bytes_rehash','variant_v12_diagnostic_issues_json']
        assert not (set(oldfields)&set(newfields))
        with gzip.open(dest/'SOURCE_INVENTORY_DEEP_16397_V12.csv.gz','wt',encoding='utf-8',newline='') as o:
            w=csv.DictWriter(o,fieldnames=oldfields+newfields);w.writeheader();n=0
            for row in r:
                n+=1;rec=records.get(row['source_key'])
                if rec:
                    row.update({
                        'variant_v12_inspection_state':rec['inspection_state'],
                        'variant_v12_legacy_mesh_objects':rec.get('historical_variant_mesh_count',''),
                        'variant_v12_legacy_vertices':rec.get('historical_variant_vertex_count',''),
                        'variant_v12_legacy_faces':rec.get('historical_variant_face_count',''),
                        'variant_v12_legacy_bones':rec.get('historical_variant_bone_count',''),
                        'variant_v12_skin_row_mean':rec.get('historical_skin_row_sum_mean',''),
                        'variant_v12_skin_row_p99_err':rec.get('historical_skin_row_p99_abs_err',''),
                        'variant_v12_skin_zero_row_fraction':rec.get('historical_skin_zero_row_fraction',''),
                        'variant_v12_canonical_scale':rec.get('historical_canonical_scale',''),
                        'variant_v12_prior_legal_decision':rec.get('historical_legal_decision',''),
                        'variant_v12_prior_admission':rec.get('historical_admission_decision',''),
                        'variant_v12_receipt_sha256':rec.get('record_sha256',''),
                        'variant_v12_not_original_bytes_rehash':'TRUE',
                        'variant_v12_diagnostic_issues_json':jdump(rec.get('diagnostic_issues',[]))})
                w.writerow(row)
    assert n==16397
    outputs={f.name:sha(f) for f in sorted(dest.iterdir()) if f.is_file()}
    summary={'schema':'RealSaS.TruthFoundryLegacyVariantsV12','source_rows':16397,'facet_columns':264,
             'missing_full_import_receipts_examined':1527,'legacy_variant_state_counts':dict(hist),
             'source_ledger_mesh_detail_rows':mesh_count,'bone_name_teacher_only_rows':bone_count,
             'new_observations':len(deltas),'new_observations_by_facet':dict(changes),'already_present_cells_skipped':observed_old,
             'full_scene_original_byte_rehash_count':0,'current_rights_court_pass':0,'new_training_admissions':0,
             'holdout_source_variants_read':0,'source_recovery_status_not_changed':True,
             'baseline_input_sha256':{k:sha(getattr(args,k)) for k in ('queue','baseline_inventory','baseline_states','baseline_values')},
             'output_sha256':outputs}
    (dest/'SUMMARY_VARIANTS_V12.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('baseline_input_sha256','output_sha256')},indent=2))
if __name__=='__main__':main()