#!/usr/bin/env python3
"""Generate Run All only after the exact materialization checkpoint is ready."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

from models.materialization_contract_v1 import require, sha256


def cell(kind, source):
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': value.update(execution_count=None, outputs=[])
    return value


def build(checkpoint_path, seal_path, *, zip_sha256, drive_bundle_path, code_sha):
    checkpoint = json.loads(checkpoint_path.read_text())
    seal = json.loads(seal_path.read_text())
    require(checkpoint.get('schema') == 'RealSaS.Corpus370MaterializationCheckpoint.v1'
            and checkpoint.get('status') == 'PASS_PRE_MATERIALIZATION_CHECKPOINT'
            and checkpoint.get('counts_are_final') is True and checkpoint.get('materialization_authorized') is True
            and checkpoint.get('training_authorized') is False, 'NOTEBOOK_CHECKPOINT_NOT_READY')
    require(seal.get('schema') == 'RealSaS.Corpus370MaterializationBundle.v1'
            and seal.get('status') == 'PASS_SEALED_STATIC_CORPUS_BUNDLE'
            and seal.get('training_authorized') is False and seal['checkpoint']['sha256'] == sha256(checkpoint_path),
            'NOTEBOOK_BUNDLE_CHECKPOINT_BINDING')
    require(re.fullmatch('[0-9a-f]{40}', code_sha) is not None and re.fullmatch('[0-9a-f]{64}', zip_sha256) is not None,
            'NOTEBOOK_EXACT_IDENTITIES_REQUIRED')
    path = PurePosixPath(drive_bundle_path)
    require(bool(drive_bundle_path) and not path.is_absolute() and '..' not in path.parts and '\\' not in drive_bundle_path,
            'NOTEBOOK_DRIVE_RELATIVE_PATH')
    cfg = {'code_sha': code_sha, 'archive_sha256': zip_sha256, 'seal_sha256': sha256(seal_path),
           'drive_bundle_path': drive_bundle_path, 'seal_relative_path': 'CORPUS370_MATERIALIZATION_BUNDLE_V1.json'}
    summary = {model: {k: row[k] for k in ('eligible_subject_count', 'partial_supervision_subject_count', 'split_counts')}
               for model, row in checkpoint['models'].items()}
    intro = '# Corpus370 — statik corpus materyalizasyonu\n\n'
    intro += 'Run All: Drive bağlama → sabit kaynak sürümü → paket hash kontrolü → kesintiden devam → model consumer doğrulaması.\n\n'
    intro += 'Bu notebook subject kabulü veya split üretmez. Sign kanıtı bulunmayan family için hedef üretmez. Eğitim başlatmaz.\n\n'
    intro += 'Sabit checkpoint:\n\n```json\n' + json.dumps(summary, indent=2, ensure_ascii=False) + '\n```\n'
    setup = 'import json, subprocess, sys, time\nfrom pathlib import Path\nfrom google.colab import drive\n'
    setup += 'START = time.monotonic()\nCONFIG = ' + repr(cfg) + '\n'
    setup += '''drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive')
ARCHIVE = DRIVE / CONFIG['drive_bundle_path']
assert ARCHIVE.is_file(), f'Sabit corpus paketi bulunamadı: {ARCHIVE}'
CODE = Path('/content/realsas-corpus370-materialization-code')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/merynz/RealSaS-OPT.git', str(CODE)], check=True)
subprocess.run(['git', '-C', str(CODE), 'fetch', 'origin', CONFIG['code_sha']], check=True)
subprocess.run(['git', '-C', str(CODE), 'checkout', '--detach', CONFIG['code_sha']], check=True)
assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == CONFIG['code_sha']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy==2.2.6'], check=True)
sys.path.insert(0, str(CODE))
print('SETUP_PASS', round(time.monotonic() - START, 1), 'seconds')
'''
    extract = '''import hashlib, shutil, zipfile
from tools.corpus.materialize_corpus370_sealed_bundle_v1 import inside
from models.materialization_contract_v1 import sha256
print('Arşiv doğrulanıyor:', ARCHIVE.stat().st_size, 'bytes', flush=True)
assert sha256(ARCHIVE) == CONFIG['archive_sha256'], 'CORPUS_ARCHIVE_HASH_DRIFT'
BUNDLE = Path('/content/corpus370-sealed-inputs') / CONFIG['archive_sha256']
BUNDLE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), 'DUPLICATE_ZIP_MEMBER'
    for member in archive.infolist():
        assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP_SYMLINK_FORBIDDEN'
        target = inside(BUNDLE, member.filename.rstrip('/'))
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_suffix(target.suffix + '.partial')
        with archive.open(member) as source, staged.open('wb') as destination:
            shutil.copyfileobj(source, destination, length=8 << 20)
        staged.replace(target)
SEAL = BUNDLE / CONFIG['seal_relative_path']
assert sha256(SEAL) == CONFIG['seal_sha256'], 'CORPUS_SEAL_HASH_DRIFT'
print('ARCHIVE_PASS', len(names), 'members', round(time.monotonic() - START, 1), 'seconds')
'''
    run = '''from tools.corpus.materialize_corpus370_sealed_bundle_v1 import materialize
OUTPUT = DRIVE / 'corpus370_materialized' / CONFIG['seal_sha256']
receipt = materialize(SEAL, CONFIG['seal_sha256'], BUNDLE, OUTPUT)
print(json.dumps({k: receipt[k] for k in ('status', 'verified_file_count', 'verified_payload_bytes', 'elapsed_seconds', 'training_authorized')}, indent=2))
print('Doğrulama raporu:', OUTPUT / 'CORPUS370_MATERIALIZATION_RECEIPT_V1.json')
print('Toplam süre:', round(time.monotonic() - START, 1), 'seconds')
'''
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': 'Corpus370_Materialize_Run_All_v1.ipynb'},
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}, 'realsas': cfg},
        'cells': [cell('markdown', intro), cell('code', setup), cell('code', extract), cell('code', run)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seal', type=Path, required=True)
    p.add_argument('--zip-sha256', required=True)
    p.add_argument('--drive-bundle-path', required=True)
    p.add_argument('--code-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    notebook = build(a.checkpoint, a.seal, zip_sha256=a.zip_sha256, drive_bundle_path=a.drive_bundle_path, code_sha=a.code_sha)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__': main()
