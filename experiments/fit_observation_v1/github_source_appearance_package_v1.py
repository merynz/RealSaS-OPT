from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote, unquote, urlparse

import requests

IMAGE_EXTS={'.png','.jpg','.jpeg','.bmp','.tga','.tif','.tiff','.webp','.dds','.exr','.hdr'}
MATERIAL_EXTS={'.mat','.mtl'}
UNITY_META_EXTS={'.meta'}
MAX_DEP_FILES=256
MAX_DEP_BYTES=512*1024*1024
MAX_DEP_DEPTH=2


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def github_parts(blob_url: str):
    u=urlparse(blob_url)
    if u.netloc.lower()!='github.com':
        raise RuntimeError('GitHub-linked source required for this materializer: '+blob_url)
    seg=[unquote(x) for x in u.path.strip('/').split('/')]
    if len(seg)<5 or seg[2]!='blob':
        raise RuntimeError('Unsupported GitHub blob URL: '+blob_url)
    return seg[0],seg[1],seg[3],'/'.join(seg[4:])


def raw_url(org: str, repo: str, commit: str, rel: str) -> str:
    return f'https://raw.githubusercontent.com/{quote(org,safe="")}/{quote(repo,safe="")}/{quote(commit,safe="")}/{quote(rel,safe="/")}'


def http_download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(dst)+'.part'); tmp.unlink(missing_ok=True)
    with requests.get(url,stream=True,timeout=(30,600),allow_redirects=True,
                      headers={'User-Agent':'RealSaS-FitObservation/1.0'}) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for ch in r.iter_content(4<<20):
                if ch: f.write(ch)
    os.replace(tmp,dst)
    return dst


def github_directory_files(org: str, repo: str, commit: str, parent: str):
    out=[]; total=0; queue=[(parent,0)]
    while queue:
        path,depth=queue.pop(0)
        api=f'https://api.github.com/repos/{quote(org,safe="")}/{quote(repo,safe="")}/contents/{quote(path,safe="/")}'
        rr=requests.get(api,params={'ref':commit},timeout=(30,180),headers={'User-Agent':'RealSaS-FitObservation/1.0'})
        rr.raise_for_status(); items=rr.json()
        if isinstance(items,dict): items=[items]
        for e in items:
            typ=e.get('type'); rp=e.get('path','')
            if typ=='file':
                ext=Path(rp).suffix.lower()
                if ext in IMAGE_EXTS|MATERIAL_EXTS|UNITY_META_EXTS:
                    size=int(e.get('size') or 0); out.append((rp,size)); total+=size
                    if len(out)>MAX_DEP_FILES or total>MAX_DEP_BYTES:
                        raise RuntimeError('appearance dependency bound exceeded')
            elif typ=='dir' and depth<MAX_DEP_DEPTH:
                queue.append((rp,depth+1))
    return out


def materialize(admission: dict, output_dir: Path) -> dict:
    si=admission['source_identity']
    fid=str(si.get('file_identifier') or si.get('source_url') or '')
    exp_sha=str(si['source_sha256']).lower(); exp_size=int(si['source_size_bytes'])
    org,repo,commit,rel=github_parts(fid); parent=Path(rel).parent.as_posix()
    shutil.rmtree(output_dir,ignore_errors=True); output_dir.mkdir(parents=True,exist_ok=True)
    src=output_dir/Path(rel).name
    http_download(raw_url(org,repo,commit,rel),src)
    got_sha=sha256_file(src); got_size=src.stat().st_size
    if got_sha!=exp_sha or got_size!=exp_size:
        raise RuntimeError(f'byte authority mismatch sha={got_sha}/{exp_sha} bytes={got_size}/{exp_size}')
    entries=github_directory_files(org,repo,commit,parent); recovered=[]; failures=[]
    for rp,_ in entries:
        rel_from_parent=Path(rp).relative_to(Path(parent)) if parent not in ('','.') else Path(rp)
        try:
            http_download(raw_url(org,repo,commit,rp),output_dir/rel_from_parent); recovered.append(str(rel_from_parent))
        except Exception as e:
            failures.append({'path':rp,'error':repr(e)})
    return {
        'schema':'RealSaS.SourceAppearancePackage.v1','candidate_id':admission.get('candidate_id'),
        'canonical_asset_id':admission.get('canonical_asset_id'),'source_file':str(src),
        'source_sha256':got_sha,'source_size_bytes':got_size,'github_org':org,'github_repo':repo,
        'github_commit':commit,'github_relpath':rel,'dependency_candidates':len(entries),
        'dependency_recovered':recovered,'dependency_failures':failures,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--admission',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--manifest',required=True)
    args=ap.parse_args(); admission=json.loads(Path(args.admission).read_text(encoding='utf-8'))
    result=materialize(admission,Path(args.output_dir)); Path(args.manifest).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__': main()
