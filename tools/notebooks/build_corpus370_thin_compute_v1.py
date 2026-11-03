"""Generate a compute worker only after the small exact transport is available."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
from tools.corpus import corpus370_thin_transport_v1 as thin
from tools.notebooks.build_corpus370_compute_worker_v1 import build as original_notebook


def build(seal_path,receipt_path,thin_receipt_path,drive_folder):
    receipt=json.loads(thin_receipt_path.read_text())
    if receipt['training_authorized'] is not False or receipt['qualification_minted'] is not False:raise ValueError('THIN_NOTEBOOK_SCOPE')
    result=original_notebook(seal_path,receipt_path,drive_folder)
    cfg=result['metadata']['realsas']
    if receipt['original_seal_sha256']!=cfg['seal_sha256'] or receipt['code_sha']!=cfg['code_sha']:raise ValueError('THIN_NOTEBOOK_SEAL_DRIFT')
    cfg.update(thin_sha256=receipt['thin_sha256'],thin_size_bytes=receipt['thin_size_bytes'],
        base_sha256=receipt['base_bundle']['sha256'],
        base_drive_path='corpus370_preparation_v1/'+receipt['base_bundle']['name'])
    result['metadata']['colab']['name']='Corpus370_Compute_Run_All_FIXED_v3.ipynb'
    result['cells'][0]['source']=['''# Corpus370 — mevcut Drive girdileriyle Run All

**Yüksek RAM → Run All.** Önceki notebooktaki yayın bekleme hücresini durdurun.
Bu sürümün küçük girdi paketi teslimden önce Drive’da doğrulanır; saatlerce yayın beklemez.
Eksik veya hash’i değişmiş bir dosya varsa açık hata ile durur.

Drive’daki mevcut1.64GB source bundle bir defa yerel diske kopyalanıp hash ile doğrulanır.
Blender/glibc aynı sabitlenmiş public dağıtımlardan kurulur. Yerel runtime cache ve manifest
farkları küçük paketten alınır.18,286 girdi adresinin her biri orijinal seal’deki SHA ve boyutla
doğrulanmadan hesaplama başlamaz. Hiçbir source/Attempt/subject kimliği değiştirilmez.

294 subject sırayla ölçülür;76 önceki hazırlama hatası korunur. Biten subject çıktısı Drive’a
ayrı arşivlenir; kesintiden sonra Run All ile doğrulanmış biten subject’ler atlanır.
Bu hesaplama/doğrulama worker’ıdır. Final eligibility, lineage split, materialization veya
training izni vermez; optimizer başlatmaz.
''']
    setup=''.join(result['cells'][1]['source'])
    start=setup.index('CONFIG = ');end=setup.index('\n',start)
    setup='CONFIG = '+repr(cfg)+setup[end:]
    setup=setup.replace("ARCHIVE=DRIVE/'CORPUS370_COMPUTE_CAPSULE_V1.zip'\nassert ARCHIVE.is_file(), f'Compute girdisi bulunamadı: {ARCHIVE}'\n",'')
    result['cells'][1]['source']=setup.splitlines(keepends=True)
    helpers='\n\n'.join(inspect.getsource(getattr(thin,name)) for name in ('sha','require','safe_member','place_object','hydrate'))
    extraction='''THIN=DRIVE/'CORPUS370_COMPUTE_THIN_V1.zip'
BASE=Path('/content/drive/MyDrive')/CONFIG['base_drive_path']
assert THIN.is_file(),f'Küçük girdi paketi bulunamadı: {THIN}. Eski yayın bekleme hücresini kullanmayın.'
assert BASE.is_file(),f'Mevcut source bundle bulunamadı: {BASE}'
CACHE=Path('/content/corpus370-worker-inputs')/CONFIG['thin_sha256']
CACHE.mkdir(parents=True,exist_ok=True)
script="""import hashlib,json,os,shutil,sys,zipfile
from pathlib import Path,PurePosixPath
"""+'''+repr(helpers)+'''+"""
thin=Path(sys.argv[1]);base=Path(sys.argv[2]);cache=Path(sys.argv[3]);cfg=json.loads(sys.argv[4])
local_base=cache/'BASE_BUNDLE.zip'
if not local_base.is_file() or sha(local_base)!=cfg['base_sha256']:
    partial=local_base.with_suffix('.partial');copied=0;last=0;total=base.stat().st_size
    print('DRIVE_SOURCE_COPY_START',total,'bytes',flush=True)
    with base.open('rb') as src,partial.open('wb') as dst:
        while block:=src.read(8<<20):
            dst.write(block);copied+=len(block)
            if copied-last>=64<<20:
                print(f'DRIVE_SOURCE_COPY {copied}/{total} bytes ({100*copied/total:.1f}%)',flush=True);last=copied
    require(sha(partial)==cfg['base_sha256'],'THIN_BASE_BUNDLE_DRIFT');partial.replace(local_base)
print('SOURCE_BUNDLE_HASH_PASS; PINNED_RUNTIME_AND_ALL_INPUTS_NEXT',flush=True)
seal=hydrate(thin,cfg['thin_sha256'],cfg['seal_sha256'],local_base,cache)
print('ALL_INPUTS_VERIFIED',len(seal['files']),'files',seal['worker_subject_count'],'subjects',flush=True)
"""
subprocess.run([PYTHON,'-u','-c',script,str(THIN),str(BASE),str(CACHE),json.dumps(CONFIG)],check=True)
print('INPUTS_PASS',round(time.monotonic()-START,1),'seconds')
'''
    result['cells'][2]['source']=extraction.splitlines(keepends=True)
    for index,cell in enumerate(result['cells']):cell['id']='corpus370-fixed3-'+str(index)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('seal','receipt','thin-receipt','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--drive-folder',required=True);args=parser.parse_args()
    notebook=build(args.seal,args.receipt,args.thin_receipt,args.drive_folder)
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
