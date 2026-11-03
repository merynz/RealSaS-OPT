"""Export a bounded, read-only slice of sealed inputs and native failure bytes.

This does not execute Blender, retry subjects, call Platform mutation APIs, or mint
truth. It lets a small scientific diagnostic use exact inputs without transferring
the entire1.6GB observation bundle through a provider with a256MB file limit.
"""
import argparse,hashlib,json,re
from pathlib import Path,PurePosixPath
import zipfile


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('EVIDENCE_SLICE_MEMBER_ESCAPE')
    return str(p)


def export(bundle,expected_sha,subjects,output,*,preparation_root=None,maximum_bytes=128<<20):
    bundle=Path(bundle);output=Path(output);subjects=sorted(set(subjects))
    if not subjects or len(subjects)>12:raise ValueError('EVIDENCE_SLICE_SUBJECT_LIMIT')
    if any(not re.fullmatch(r'(corpus|truth139)_[0-9a-f]{20}',s) for s in subjects):raise ValueError('EVIDENCE_SLICE_SUBJECT_ID')
    if sha(bundle)!=expected_sha:raise ValueError('EVIDENCE_SLICE_BUNDLE_HASH')
    output.mkdir(parents=True,exist_ok=True);used=0;entries={};native_rows=[]
    with zipfile.ZipFile(bundle) as source:
        manifest_raw=source.read('BOOTSTRAP_MANIFEST.json');m=json.loads(manifest_raw)
        roster={r['subject_id']:r for r in m['subjects']}
        if m.get('candidate_subject_count')!=370 or len(roster)!=370 or any(s not in roster for s in subjects):
            raise ValueError('EVIDENCE_SLICE_EXACT_SCOPE')
        selected=[roster[s] for s in subjects]
        def file_refs(value):
            if isinstance(value,list):
                for item in value:yield from file_refs(item)
            elif isinstance(value,dict):
                if 'path' in value and 'sha256' in value and 'size_bytes' in value:yield value
                else:
                    for child in value.values():yield from file_refs(child)
        refs={r['path']:r for row in selected for r in file_refs(row)}
        staged=output/'EVIDENCE_SLICE.partial.zip';final=output/'EVIDENCE_SLICE.zip'
        with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as target:
            def add(name,raw,scope):
                nonlocal used
                name=safe_name(name)
                if name in entries:return
                if used+len(raw)>maximum_bytes:raise ValueError('EVIDENCE_SLICE_BYTE_LIMIT')
                target.writestr(name,raw);entries[name]={'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'scope':scope};used+=len(raw)
            add('BOOTSTRAP_MANIFEST.json',manifest_raw,'ORIGINAL_SEALED_MANIFEST')
            for name,row in refs.items():
                if row['size_bytes']>maximum_bytes-used:raise ValueError('EVIDENCE_SLICE_BYTE_LIMIT')
                raw=source.read(safe_name(name))
                if len(raw)!=row['size_bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('EVIDENCE_SLICE_MEMBER_DRIFT')
                add(name,raw,'ORIGINAL_SEALED_INPUT')
            if preparation_root is not None:
                root=Path(preparation_root).resolve()
                # Read only the two known preparation indexes. Binding identities
                # locate exact selected subject directories, never a broad scan.
                for relative in ('observations_v2/CORPUS370_PREPARED_OBSERVATION_PACKETS_V2.json',
                                 'observations_rescue_v3/CORPUS370_PREPARED_OBSERVATION_PACKETS_V3.json'):
                    index_path=root/relative
                    if not index_path.is_file():continue
                    index=json.loads(index_path.read_text());binding=index.get('binding')
                    explicit={r['subject_id']:r for r in index.get('subjects',[])}
                    for sid in subjects:
                        locations=[]
                        packet=explicit.get(sid,{}).get('packet_path')
                        if packet:locations.append(Path(packet).parent.parent)
                        if binding:
                            payload={'binding':binding,'subject_id':sid,'source_sha256':roster[sid]['raw_source']['sha256']}
                            identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                            locations.append(index_path.parent/'objects'/identity)
                        for location in dict.fromkeys(locations):
                            if not location.resolve().is_relative_to(root):raise ValueError('EVIDENCE_SLICE_NATIVE_ESCAPE')
                            imported=location/'imported'/sid/'imported.json'
                            if imported.is_file():
                                if not imported.resolve().is_relative_to(root):raise ValueError('EVIDENCE_SLICE_NATIVE_SYMLINK_ESCAPE')
                                if imported.stat().st_size>maximum_bytes-used:raise ValueError('EVIDENCE_SLICE_BYTE_LIMIT')
                                name='native/'+sid+'/'+location.name+'/imported.json';raw=imported.read_bytes()
                                add(name,raw,'UNQUALIFIED_NATIVE_WORKING_BYTES')
                                native_rows.append({'subject_id':sid,'source_path':str(imported),'slice_path':name,'sha256':hashlib.sha256(raw).hexdigest()})
                            observations=location/'observations'/sid
                            for file in sorted(observations.glob('*')) if observations.is_dir() else []:
                                if not file.is_file() or file.suffix not in ('.json','.png','.bin'):continue
                                if not file.resolve().is_relative_to(root):raise ValueError('EVIDENCE_SLICE_NATIVE_SYMLINK_ESCAPE')
                                if file.stat().st_size>maximum_bytes-used:raise ValueError('EVIDENCE_SLICE_BYTE_LIMIT')
                                add('native/'+sid+'/'+location.name+'/observations/'+file.name,file.read_bytes(),'UNQUALIFIED_NATIVE_WORKING_BYTES')
            result={'schema':'RealSaS.Corpus370EvidenceSlice.v1','source_bundle_sha256':expected_sha,
                    'subjects':selected,'requested_subject_ids':subjects,'files':list(entries.values()),
                    'native_imports':native_rows,'uncompressed_bytes':used,'qualification_minted':False,
                    'training_authorized':False,'source_data_mutated':False}
            target.writestr('EVIDENCE_SLICE.json',json.dumps(result,sort_keys=True,indent=2)+'\n')
        staged.replace(final)
    receipt={**result,'archive':{'path':str(final),'sha256':sha(final),'size_bytes':final.stat().st_size}}
    (output/'EVIDENCE_SLICE_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'exported_subject_count':len(subjects),'archive_bytes':final.stat().st_size,'uncompressed_bytes':used,'native_import_count':len(native_rows),'training_authorized':False}),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--subjects',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preparation-root',type=Path)
    a=p.parse_args();export(a.bundle,a.expected_sha256,a.subjects.split(','),a.output,preparation_root=a.preparation_root)
