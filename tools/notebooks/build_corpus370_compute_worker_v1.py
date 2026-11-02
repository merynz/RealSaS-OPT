#!/usr/bin/env python3
"""Generate the separate heavy-compute notebook from a real sealed input export."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path, PurePosixPath
import re
from tools.notebooks.build_corpus370_materialization_v1 import cell
from models.materialization_contract_v1 import require,sha256


def effective_memory_gib(meminfo_text=None,cgroup_limits=None):
    """Respect the container limit as well as the host's visible memory."""
    if meminfo_text is None:meminfo_text=Path('/proc/meminfo').read_text()
    host_kib=int(next(line.split()[1] for line in meminfo_text.splitlines() if line.startswith('MemTotal:')))
    limits=[host_kib*1024]
    if cgroup_limits is None:
        cgroup_limits=[]
        for name in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            try:cgroup_limits.append(Path(name).read_text().strip())
            except OSError:pass
    for value in cgroup_limits:
        try:value=int(value)
        except (ValueError,TypeError):continue
        if value>0:limits.append(value)
    return min(limits)/(1024**3)


def build(seal_path,receipt_path,drive_folder):
    seal=json.loads(seal_path.read_text());receipt=json.loads(receipt_path.read_text())
    require(seal.get('schema')=='RealSaS.Corpus370ComputeCapsule.v1'
            and seal.get('qualification_minted') is False and seal.get('training_authorized') is False,
            'WORKER_NOTEBOOK_COMPUTE_SCOPE')
    require(receipt['seal_sha256']==sha256(seal_path) and receipt['code_sha']==seal['code_sha'],
            'WORKER_NOTEBOOK_SEAL_BINDING')
    require(re.fullmatch('[a-f0-9]{40}',seal['code_sha']) is not None
            and re.fullmatch('[a-f0-9]{64}',receipt['archive_sha256']) is not None,'WORKER_NOTEBOOK_EXACT_IDENTITIES')
    p=PurePosixPath(drive_folder)
    require(bool(drive_folder) and not p.is_absolute() and '..' not in p.parts and '\\' not in drive_folder,'WORKER_NOTEBOOK_DRIVE_PATH')
    cfg={'code_sha':seal['code_sha'],'archive_sha256':receipt['archive_sha256'],
         'seal_sha256':receipt['seal_sha256'],'drive_folder':drive_folder}
    intro='''# Corpus370 — ağır hesaplama worker’ı

**Çalışma zamanı → Çalışma zamanı türünü değiştir → Yüksek RAM** seçin, ardından **Run All**.
Bu iş ağırlıkla CPU ve RAM kullanır; GPU seçmek tek başına belleği artırmaz.
Önerilen RAM: en az 24 GiB. Tek subject, tek process; iç hesaplar bellek için parçalara ayrılır.

370 adayın 294’ü bu ölçüm hattına girer; önceki 76 observation hazırlama hatası açıkça korunur.
Paket exact Platform Campaign/Attempt/SubjectInput kimliklerine ve kaynak hash’lerine bağlıdır.
Kaynak adresleri korunur; sign veya court kriteri değiştirilmez.

Her subject bittiğinde tüm çıktıları Drive’a yazılır. Oturum kesilirse aynı notebook’u Run All
ile tekrar çalıştırın; tamamlanan arşivler hash ile kontrol edilerek atlanır.
İşlem sırasında subject/stage, süre ve bellek raporlanır; hata varsa ilgili log açıkça gösterilir.

Bu notebook yeni ölçüm ve bağımsız court çıktıları üretir. Platform’a import/doğrulama,
nihai family matrix, global sign/graphics, lineage-safe split ve materialization checkpoint
ayrı kapılardır. Bu dosya materialization veya training notebook’u değildir; optimizer çalıştırmaz.
'''
    setup='CONFIG = '+repr(cfg)+'\n'+'''import json, os, subprocess, sys, time
from pathlib import Path
from google.colab import drive
'''+inspect.getsource(effective_memory_gib)+'''
START=time.monotonic()
ram_gib=effective_memory_gib()
print(f'Kullanılabilir çalışma zamanı RAM sınırı: {ram_gib:.1f} GiB')
assert ram_gib>=22, 'Yüksek RAM çalışma zamanı seçin (en az 24 GiB önerilir), sonra Run All.'
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','libxi6','libxrender1','libxfixes3','libxrandr2',
                'libxxf86vm1','libsm6','libice6','libgl1','libxkbcommon0','libxinerama1'],check=True)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive')/CONFIG['drive_folder']
ARCHIVE=DRIVE/'CORPUS370_COMPUTE_CAPSULE_V1.zip'
assert ARCHIVE.is_file(), f'Compute girdisi bulunamadı: {ARCHIVE}'
CODE=Path('/content/realsas-corpus370-worker-code')
if not CODE.exists():
    subprocess.run(['git','clone','--no-checkout','https://github.com/merynz/RealSaS-OPT.git',str(CODE)],check=True)
subprocess.run(['git','-C',str(CODE),'fetch','origin',CONFIG['code_sha']],check=True)
subprocess.run(['git','-C',str(CODE),'checkout','--detach',CONFIG['code_sha']],check=True)
assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==CONFIG['code_sha']
VENV=Path('/content/realsas-corpus370-worker-venv')
if not (VENV/'bin/python').is_file():subprocess.run([sys.executable,'-m','venv','--without-pip',str(VENV)],check=True)
PYTHON=str(VENV/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',PYTHON,'install','--quiet','-r',str(CODE/'requirements/mainline-ci.txt'),
                '-r',str(CODE/'requirements/truth-family-courts.txt')],check=True)
os.environ['PYTHONPATH']=str(CODE)
os.environ['REALSAS_AUTHORITY_ROOT']='/home/monster/realsas_authority'
os.chdir(CODE)
print('SETUP_PASS',round(time.monotonic()-START,1),'seconds')
'''
    extract='''CACHE=Path('/content/corpus370-worker-inputs')/CONFIG['archive_sha256']
script="""import json,sys
from pathlib import Path
from tools.corpus.run_corpus370_compute_worker_v1 import hydrate
from models.materialization_contract_v1 import sha256
seal=hydrate(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
assert sha256(Path(sys.argv[3])/'COMPUTE_CAPSULE.json')==sys.argv[4], 'CAPSULE_SEAL_HASH_DRIFT'
print('INPUTS_VERIFIED',seal['worker_subject_count'],'subjects; all',seal['candidate_subject_count'],'candidates accounted')
"""
subprocess.run([PYTHON,'-u','-c',script,str(ARCHIVE),CONFIG['archive_sha256'],str(CACHE),CONFIG['seal_sha256']],check=True)
print('INPUTS_PASS',round(time.monotonic()-START,1),'seconds')
'''
    run='''OUTPUT=DRIVE/'results'/CONFIG['seal_sha256']
script="""import json,sys
from pathlib import Path
from tools.corpus.run_corpus370_compute_worker_v1 import run
seal=json.loads((Path(sys.argv[1])/'COMPUTE_CAPSULE.json').read_text())
report=run(seal,Path(sys.argv[2]),Path(sys.argv[1]))
print(json.dumps({k:report[k] for k in ('accounted_subject_count','candidate_subject_count','elapsed_seconds','qualification_minted','training_authorized')},indent=2))
"""
subprocess.run([PYTHON,'-u','-c',script,str(CACHE),str(OUTPUT)],check=True)
print('Hesaplama çıktıları:',OUTPUT)
print('Özet:',OUTPUT/'COMPUTE_PROGRESS.json')
print('Toplam süre:',round(time.monotonic()-START,1),'seconds')
print('Sonraki adım: Platform import/bağımsız doğrulama → final eligibility/split → materialization notebook.')
'''
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Corpus370_Compute_Worker_Run_All_v1.ipynb'},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'realsas':cfg},
        'cells':[dict(cell(kind,source),id='corpus370-worker-'+str(i)) for i,(kind,source) in enumerate(
            [('markdown',intro),('code',setup),('code',extract),('code',run)])]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--drive-folder',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    notebook=build(a.seal,a.receipt,a.drive_folder)
    a.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
