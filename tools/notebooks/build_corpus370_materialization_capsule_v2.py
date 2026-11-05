#!/usr/bin/env python3
"""Corpus370 Run-All materialization V2 using exact Drive CODE capsule.

The private repo cannot be cloned anonymously from Colab. The capsule is
generated from a pinned Git commit, and the notebook verifies BOTH capsule
and corpus bundle before importing code or copying data. Readiness still
requires the existing PASS checkpoint, split and family court bundle.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path,PurePosixPath
from models.materialization_contract_v1 import require
from tools.notebooks.build_corpus370_materialization_v1 import build as original

def relative(name):
    p=PurePosixPath(name)
    require(bool(name) and not p.is_absolute()
            and '..' not in p.parts and '\\' not in name
            and name not in ('.',''),'CAPSULE_NOTEBOOK_DRIVE_RELATIVE_PATH')

def build(checkpoint_path,seal_path,*,zip_sha256,drive_bundle_path,
          code_sha,code_archive_sha256,drive_code_path):
    notebook=original(checkpoint_path,seal_path,zip_sha256=zip_sha256,
                      drive_bundle_path=drive_bundle_path,code_sha=code_sha)
    require(re.fullmatch('[0-9a-f]{64}',code_archive_sha256) is not None,
            'CAPSULE_NOTEBOOK_EXACT_CODE_ARCHIVE_SHA_REQUIRED')
    relative(drive_code_path)
    cfg=notebook['metadata']['realsas']
    cfg['code_archive_sha256']=code_archive_sha256
    cfg['drive_code_path']=drive_code_path
    source=r'''import hashlib,json,shutil,sys,time,zipfile,subprocess
from pathlib import Path,PurePosixPath
from google.colab import drive
START=time.monotonic()
CONFIG=__CONFIG__
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive')
ARCHIVE=DRIVE/CONFIG['drive_bundle_path']
CODE_ARCHIVE=DRIVE/CONFIG['drive_code_path']
assert ARCHIVE.is_file(), 'CORPUS370_BUNDLE_MISSING'
assert CODE_ARCHIVE.is_file(), 'CORPUS370_CODE_CAPSULE_MISSING'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
assert digest(CODE_ARCHIVE)==CONFIG['code_archive_sha256'], 'CORPUS370_CODE_CAPSULE_SHA_MISMATCH'
CODE=Path('/content/realsas-corpus370-code')/CONFIG['code_archive_sha256']
CODE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(CODE_ARCHIVE) as z:
    names=z.namelist()
    assert len(names)==len(set(names)), 'CODE_CAPSULE_DUPLICATE_ENTRY'
    assert 'CODE_CAPSULE_MANIFEST.json' in names,'CODE_CAPSULE_MANIFEST_ABSENT'
    manifest=json.loads(z.read('CODE_CAPSULE_MANIFEST.json'))
    assert manifest['schema']=='RealSaS.Corpus370MaterializationCodeCapsule.v1'
    assert manifest['status']=='SEALED_MINIMAL_EXECUTABLE_CODE_ONLY__NOT_MATERIALIZATION_OR_TRAINING'
    assert manifest['source_git_commit']==CONFIG['code_sha']
    assert manifest['source_code_from_immutable_git_objects'] is True
    assert manifest['git_credentials_embedded'] is False
    assert manifest['source_truth_files_embedded'] is False
    assert manifest['checkpoint_requirement_preserved'] is True
    assert manifest['training_authorized'] is False
    required={
      'models/materialization_contract_v1.py',
      'models/iris/v32/training_input_contract_v1.py',
      'models/tessa/v32/training_input_contract_v1.py',
      'tools/corpus/materialize_corpus370_sealed_bundle_v1.py',
    }
    listed={x['path']:x for x in manifest['content_files']}
    assert required.issubset(listed),'CODE_CAPSULE_IMPORT_CLOSURE_MISSING'
    assert set(names)==set(listed)|{'CODE_CAPSULE_MANIFEST.json'},'CODE_CAPSULE_UNSEALED_EXTRA'
    assert len(listed)<=24,'CODE_CAPSULE_FILE_BUDGET'
    assert sum(x['size_bytes'] for x in listed.values())<4_000_000,'CODE_CAPSULE_ZIP_BOMB_BUDGET'
    for name,row in listed.items():
        p=PurePosixPath(name)
        assert name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
        info=z.getinfo(name)
        assert not info.is_dir() and (info.external_attr>>16)&0o170000 != 0o120000
        assert info.file_size==row['size_bytes'] and info.file_size<1_000_000
        data=z.read(name)
        assert hashlib.sha256(data).hexdigest()==row['sha256'],'CODE_CAPSULE_FILE_SHA_DRIFT'
        destination=(CODE/name).resolve()
        assert destination.is_relative_to(CODE.resolve()),'CODE_CAPSULE_PATH_ESCAPE'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            assert digest(destination)==row['sha256'],'CODE_CAPSULE_RESUME_DRIFT'
        else:
            tmp=destination.with_suffix(destination.suffix+'.partial')
            tmp.write_bytes(data)
            tmp.replace(destination)
subprocess.run([sys.executable,'-m','pip','install','--quiet','numpy==2.2.6'],check=True)
sys.path.insert(0,str(CODE))
from models.materialization_contract_v1 import sha256
assert sha256(CODE_ARCHIVE)==CONFIG['code_archive_sha256'],'CODE_CAPSULE_SECOND_HASH_DRIFT'
print('CODE_CAPSULE_PASS',CONFIG['code_sha'],round(time.monotonic()-START,1),'s')
'''
    source=source.replace('__CONFIG__',repr(cfg))
    notebook['cells'][1]['source']=source.splitlines(keepends=True)
    notebook['metadata']['colab']['name']='Corpus370_Materialize_Sealed_Code_Run_All_v2.ipynb'
    return notebook

def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','seal','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('zip-sha256','drive-bundle-path','code-sha',
                 'code-archive-sha256','drive-code-path'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    result=build(a.checkpoint,a.seal,zip_sha256=a.zip_sha256,
        drive_bundle_path=a.drive_bundle_path,code_sha=a.code_sha,
        code_archive_sha256=a.code_archive_sha256,drive_code_path=a.drive_code_path)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':
    main()
