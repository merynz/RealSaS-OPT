#!/usr/bin/env python3
"""Read-only recovery of exact requested artifact bytes and Platform capability.

This does not execute a teacher, create an Attempt, import artifacts or qualify
data. Filenames locate candidates; only expected SHA256+size admit a recovery.
No branch-name or latest-file selection is used. Search roots are explicit data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(request, *, authority_root, platform_root, output, fetch_json=None):
    if request.get('schema') != 'RealSaS.TruthInputInspectionRequest.v1':
        raise ValueError('TRUTH_INSPECT_REQUEST_SCHEMA')
    if request.get('read_only') is not True:
        raise ValueError('TRUTH_INSPECT_READ_ONLY_REQUIRED')
    authority, platform, output = map(lambda p:Path(p).resolve(), (authority_root, platform_root, output))
    output.mkdir(parents=True, exist_ok=True)
    roots = []
    for value in request['search_roots']:
        base = {'authority':authority, 'platform':platform}[value['root']]
        relative = Path(value['relative'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('TRUTH_INSPECT_ROOT_ESCAPE')
        path = (base/relative).resolve()
        path.relative_to(base)
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('TRUTH_INSPECT_OUTPUT_INSIDE_SEARCH')
        roots.append(path)
    expected = request['artifacts']
    if len({r['sha256'] for r in expected}) != len(expected):
        raise ValueError('TRUTH_INSPECT_DUPLICATE_REQUEST')
    for row in expected:
        if (len(row['sha256']) != 64 or any(x not in '0123456789abcdef' for x in row['sha256'])
                or not isinstance(row['size_bytes'], int) or row['size_bytes'] < 0
                or Path(row['file_name']).name != row['file_name']):
            raise ValueError('TRUTH_INSPECT_IDENTITY_INVALID')
    names = {r['file_name'] for r in expected} | {r['sha256'] for r in expected}
    by_sha = {r['sha256']:r for r in expected}
    found, mismatches = {}, []
    deadline = time.monotonic()+request.get('max_search_seconds', 60)
    visited, truncated = 0, False
    for root in roots:
        if not root.is_dir():
            continue
        for folder, directories, files in os.walk(root, followlinks=False):
            directories[:] = [n for n in directories if n not in {'.git', 'node_modules', '.venv'} and not (Path(folder)/n).is_symlink()]
            visited += 1
            if time.monotonic() > deadline or visited > request.get('max_directories', 100000):
                truncated = True
                break
            for name in names.intersection(files):
                path = Path(folder)/name
                if path.is_symlink():
                    continue
                candidates = [r for r in expected if name in {r['file_name'], r['sha256']} and path.stat().st_size == r['size_bytes']]
                if not candidates:
                    continue
                digest = sha256(path)
                if digest not in by_sha:
                    mismatches.append({'file_name':name, 'sha256':digest})
                    continue
                row = by_sha[digest]
                target = output/'payloads'/(digest+'.bin')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if sha256(target) != digest or target.stat().st_size != row['size_bytes']:
                    raise ValueError('TRUTH_INSPECT_RECOVERY_COPY_DRIFT')
                found[digest] = {'file_name':row['file_name'], 'sha256':digest, 'size_bytes':row['size_bytes'],
                                 'recovered_relative_path':str(target.relative_to(output))}
            if len(found) == len(expected):
                break
        if truncated or len(found) == len(expected):
            break
    platform_state = {}
    if fetch_json:
        for route in ('/v1/ops/status', '/v1/stages', '/v1/model-channels', '/v1/experiments'):
            try:
                platform_state[route] = fetch_json(route)
            except Exception as exc:
                platform_state[route] = {'read_error':type(exc).__name__, 'message':str(exc)}
    result = {'schema':'RealSaS.TruthInputInspection.v1', 'status':'RECOVERED' if len(found)==len(expected) else 'INCOMPLETE',
              'recovered':list(found.values()), 'missing':[r for r in expected if r['sha256'] not in found],
              'mismatched_candidates':mismatches, 'search_truncated':truncated, 'directories_visited':visited,
              'platform_state':platform_state, 'platform_mutation_performed':False, 'teacher_executed':False,
              'qualification_minted':False, 'training_executed':False}
    (output/'INSPECTION.json').write_text(json.dumps(result, sort_keys=True, indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','authority-root','platform-root','output','expected-main-sha'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--api-root', default='http://127.0.0.1:8080')
    a=p.parse_args()
    deployment=json.loads((Path(a.platform_root)/'deployment.json').read_text())
    if deployment['code_sha'] != a.expected_main_sha:
        raise ValueError('TRUTH_INSPECT_DEPLOYED_MAIN_DRIFT')
    def get(route):
        with urllib.request.urlopen(a.api_root.rstrip('/')+route, timeout=15) as response:
            return json.load(response)
    result=inspect(json.loads(Path(a.request).read_text()), authority_root=a.authority_root,
                   platform_root=a.platform_root, output=a.output, fetch_json=get)
    print(json.dumps({k:result[k] for k in ('status','search_truncated','directories_visited','missing')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
