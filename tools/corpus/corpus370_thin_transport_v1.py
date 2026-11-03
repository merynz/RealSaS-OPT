"""Transport existing sealed input bytes using the already-published source bundle.

The original capsule seal/SubjectInputs remain unchanged. This is not a court.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def require(value,reason):
    if not value:raise ValueError(reason)


def safe_member(name):
    p=PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name,'THIN_MEMBER_ESCAPE')
    return p


def fresh_runtime(original_executable,cache):
    from tools.observation.install_pinned_evaluator_runtime_v1 import install
    original=Path(original_executable);root=cache/'provider'/'blender-4.0.2-linux-x64'/'numeric-runtime-239'
    root.mkdir(parents=True,exist_ok=True)
    # Reuse authenticated download caches, never an already-executed installation.
    for source,destination in (
        (original.parent.parent.parent/'blender-4.0.2-linux-x64.tar.xz',root.parent.parent/'blender-4.0.2-linux-x64.tar.xz'),
        (original.parent/'libc6_2.39-0ubuntu8.6_amd64.deb',root/'libc6_2.39-0ubuntu8.6_amd64.deb')):
        if source.is_file() and not destination.exists():os.link(source,destination)
    install(root/'blender-glibc239')
    return root


def build(config,operator,output):
    old=config['existing_capsule'];source=Path(old['directory'])
    archive=source/'CORPUS370_COMPUTE_CAPSULE_V1.zip';seal_path=source/'COMPUTE_CAPSULE.json'
    require(sha(archive)==old['archive_sha256'] and archive.stat().st_size==old['archive_size_bytes'],'THIN_PARENT_ARCHIVE_DRIFT')
    require(sha(seal_path)==old['seal_sha256'],'THIN_PARENT_SEAL_DRIFT')
    raw=seal_path.read_bytes();seal=json.loads(raw)
    require(seal['code_sha']==old['code_sha'] and seal['preparation_sha256']==config['preparation']['sha256'],'THIN_PARENT_SCOPE_DRIFT')
    require(seal['training_authorized'] is False and seal['qualification_minted'] is False,'THIN_AUTHORITY_DRIFT')
    base=operator['cohort_bundle'];prefix='/home/monster/realsas_authority/truth_cohort_bundles_v2/'+base['sha256']+'/'
    base_members={r['sha256']:r['path'][len(prefix):] for r in seal['files'] if r['path'].startswith(prefix)}
    require(base_members,'THIN_BASE_MAPPING_EMPTY')
    runtime_prefix=str(Path(operator['evaluator_executable']).parent)+'/'
    output.mkdir(parents=True,exist_ok=True)
    recreated=fresh_runtime(operator['evaluator_executable'],output)
    providers={};overlay={}
    for row in seal['files']:
        digest=row['sha256'];path=row['path']
        if digest in providers:continue
        if digest in base_members:
            member=base_members[digest];safe_member(member)
            providers[digest]={'provider':'base','member':member}
        elif path.startswith(runtime_prefix):
            member=path[len(runtime_prefix):];safe_member(member);candidate=recreated/member
            if candidate.is_file() and sha(candidate)==digest:
                providers[digest]={'provider':'runtime','member':member}
            else:
                providers[digest]={'provider':'overlay','member':row['object']};overlay[digest]=row
        else:
            providers[digest]={'provider':'overlay','member':row['object']};overlay[digest]=row
    transport={'schema':'RealSaS.Corpus370ThinTransport.v1','original_seal_sha256':old['seal_sha256'],
        'original_archive_sha256':old['archive_sha256'],'code_sha':seal['code_sha'],'base_bundle':base,
        'runtime_prefix':runtime_prefix,'providers':providers,'qualification_minted':False,'training_authorized':False}
    target=output/'CORPUS370_COMPUTE_THIN_V1.zip';staged=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(archive) as original,zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        out.writestr('COMPUTE_CAPSULE.json',raw)
        out.writestr('THIN_TRANSPORT.json',json.dumps(transport,sort_keys=True,indent=2)+'\n')
        for row in overlay.values():
            raw=original.read(row['object']);require(hashlib.sha256(raw).hexdigest()==row['sha256'],'THIN_OVERLAY_BYTES_DRIFT')
            out.writestr(row['object'],raw)
    staged.replace(target)
    receipt={'schema':'RealSaS.Corpus370ThinTransportReceipt.v1','thin_sha256':sha(target),'thin_size_bytes':target.stat().st_size,
        'original_seal_sha256':old['seal_sha256'],'original_archive_sha256':old['archive_sha256'],
        'code_sha':seal['code_sha'],'base_bundle':base,'provider_counts':{k:sum(p['provider']==k for p in providers.values()) for k in ('base','runtime','overlay')},
        'overlay_uncompressed_bytes':sum(r['size_bytes'] for r in overlay.values()),
        'qualification_minted':False,'training_authorized':False}
    (output/'THIN_TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


def hydrate(thin_path,expected_thin_sha,expected_seal_sha,base_path,cache):
    require(sha(thin_path)==expected_thin_sha,'THIN_PACKAGE_HASH_DRIFT')
    with zipfile.ZipFile(thin_path) as thin:
        names=thin.namelist();require(len(names)==len(set(names)),'THIN_DUPLICATE_MEMBER')
        for name in names:safe_member(name)
        raw=thin.read('COMPUTE_CAPSULE.json');require(hashlib.sha256(raw).hexdigest()==expected_seal_sha,'THIN_SEAL_HASH_DRIFT')
        seal=json.loads(raw);transport=json.loads(thin.read('THIN_TRANSPORT.json'))
        require(transport['original_seal_sha256']==expected_seal_sha and transport['code_sha']==seal['code_sha'],'THIN_SCOPE_DRIFT')
        require(transport['training_authorized'] is False and transport['qualification_minted'] is False
                and seal['training_authorized'] is False and seal['qualification_minted'] is False,'THIN_AUTHORITY_DRIFT')
        providers=transport['providers'];require(set(providers)=={r['sha256'] for r in seal['files']},'THIN_PROVIDER_COVERAGE')
        require(sha(base_path)==transport['base_bundle']['sha256'],'THIN_BASE_BUNDLE_DRIFT')
        from tools.observation.install_pinned_evaluator_runtime_v1 import install
        runtime=cache/'provider'/'blender-4.0.2-linux-x64'/'numeric-runtime-239'
        install(runtime/'blender-glibc239')
        roots=[Path(p) for p in seal['allowed_original_roots']]
        require(all(p.is_absolute() and str(p).startswith('/home/monster/') and len(p.parts)>3 for p in roots),'THIN_ROOT_SCOPE')
        objects=cache/'objects';objects.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(base_path) as base:
            for index,row in enumerate(seal['files'],1):
                digest=row['sha256'];provider=providers[digest];member=provider['member'];safe_member(member)
                require(row['object']=='objects/'+digest,'THIN_OBJECT_ADDRESS_DRIFT')
                obj=objects/digest
                if not obj.is_file() or sha(obj)!=digest:
                    partial=obj.with_suffix('.partial')
                    if provider['provider']=='base':
                        with base.open(member) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,length=8<<20)
                    elif provider['provider']=='runtime':
                        require((runtime/member).resolve().is_relative_to(runtime.parent.resolve()),'THIN_RUNTIME_ESCAPE')
                        shutil.copyfile(runtime/member,partial)
                    elif provider['provider']=='overlay':
                        with thin.open(member) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,length=8<<20)
                    else:raise ValueError('THIN_UNKNOWN_PROVIDER')
                    require(partial.stat().st_size==row['size_bytes'] and sha(partial)==digest,'THIN_OBJECT_BYTES_DRIFT:'+row['path'])
                    partial.replace(obj)
                destination=Path(row['path'])
                require(destination.is_absolute() and '..' not in destination.parts
                    and any(destination.is_relative_to(root) and destination.resolve().is_relative_to(root.resolve()) for root in roots),'THIN_DESTINATION_ESCAPE')
                if destination.exists():require(destination.is_file() and sha(destination)==digest,'THIN_EXISTING_BYTES_DRIFT:'+str(destination))
                else:destination.parent.mkdir(parents=True,exist_ok=True);os.link(obj,destination)
                if row['executable']:destination.chmod(destination.stat().st_mode|0o111)
                if index%250==0:print(f'INPUT_VERIFY {index}/{len(seal["files"])}',flush=True)
    (cache/'COMPUTE_CAPSULE.json').write_bytes(raw)
    require(sha(cache/'COMPUTE_CAPSULE.json')==expected_seal_sha,'THIN_FINAL_SEAL_DRIFT')
    return seal


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--operator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    build(json.loads(args.config.read_text()),json.loads(args.operator.read_text()),args.output)
