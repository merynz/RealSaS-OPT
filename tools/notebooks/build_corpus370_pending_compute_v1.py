"""Bind a compute notebook before a large immutable input upload finishes.

The original per-SubjectInput section digests are the out-of-band input binding.
The publisher's archive/seal hashes additionally protect transport integrity.
No eligibility, split, qualification or training permission is produced here.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
from tools.notebooks.build_corpus370_materialization_v1 import cell
from tools.notebooks.build_corpus370_compute_worker_v1 import effective_memory_gib,preserve_interrupted_work


def verify_scope(archive_path,scope):
    import hashlib,json,zipfile
    def check(value,reason):
        if not value:raise ValueError(reason)
    def digest(value):
        return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        seal=json.loads(archive.read('COMPUTE_CAPSULE.json'))
        for key,expected in scope['header'].items():check(seal.get(key)==expected,'PENDING_CAPSULE_SCOPE:'+key)
        ref=seal['plan']
        check(ref['sha256']==scope['plan_sha256'] and hashlib.sha256(archive.read(ref['object'])).hexdigest()==scope['plan_sha256'],
              'PENDING_EXECUTION_PLAN_DRIFT')
        check(digest(seal['excluded_subjects'])==scope['excluded_subjects_sha256'],'PENDING_EXCLUSION_DRIFT')
        rows=seal['subjects'];expected=scope['subjects']
        check(len(rows)==len(expected) and {r['subject_id'] for r in rows}==set(expected),'PENDING_SUBJECT_SET_DRIFT')
        for row in rows:
            parent=expected[row['subject_id']]
            for key in ('platform_subject_id','subject_input_id','parent_attempt_id'):
                check(row[key]==parent[key],'PENDING_SUBJECT_BINDING:'+key)
            ref=row['manifest'];raw=archive.read(ref['object'])
            check(hashlib.sha256(raw).hexdigest()==ref['sha256'],'PENDING_MANIFEST_HASH')
            manifest=json.loads(raw)
            check(manifest['subject_id']==row['subject_id'] and manifest['run_id']==parent['run_id']
                  and manifest['execution_class']=='WITNESS','PENDING_MANIFEST_IDENTITY')
            check(set(manifest)==set(scope['section_keys'])|{'run_id','subject_id','execution_class'},'PENDING_MANIFEST_KEYS')
            sections={k:manifest[k] for k in scope['section_keys']}
            check(digest(sections)==parent['sections_sha256'],'PENDING_SUBJECT_INPUT_BYTES_DRIFT')
    return seal


def scope_from_preparation(preparation,bootstrap,repairs,*,preparation_sha,code_sha,section_keys,plan_sha):
    def digest(v):return hashlib.sha256((json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
    by_input={r['subject_input_id']:(sid,r) for sid,r in repairs['subjects'].items()}
    attempts={r['SubjectID']:r['AttemptID'] for r in preparation['campaign_run']['receipts']}
    subjects={}
    for row in bootstrap['run_bindings']:
        sid,data=by_input[row['subject_input_id']]
        subjects[sid]={'platform_subject_id':row['subject_id'],'subject_input_id':row['subject_input_id'],
            'parent_attempt_id':attempts[row['subject_id']],'sections_sha256':data['sections_sha256'],
            'run_id':Path(row['run_manifest_path']).parent.name}
    if len(subjects)!=294 or len(attempts)!=294:raise ValueError('PENDING_EXACT294_SCOPE')
    return {'header':{'schema':'RealSaS.Corpus370ComputeCapsule.v1','code_sha':code_sha,
        'parent_campaign_id':preparation['campaign_id'],'parent_engine_release_id':preparation['engine_release_id'],
        'preparation_sha256':preparation_sha,'target_stage_id':'49_SUBJECT_TRUTH_FAMILIES',
        'candidate_subject_count':370,'worker_subject_count':294,'excluded_subject_count':76,
        'qualification_minted':False,'training_authorized':False},
        'section_keys':sorted(section_keys),'subjects':subjects,'excluded_subjects_sha256':digest(bootstrap['excluded_subjects']),
        'plan_sha256':plan_sha}


def build(scope,drive_folder):
    config={'code_sha':scope['header']['code_sha'],'drive_folder':drive_folder,
            'publisher_run_id':41526186943,'training_authorized':False}
    intro='''# Corpus370 — yüksek RAM hesaplama ve doğrulama worker’ı

**Colab → Yüksek RAM → Tümünü çalıştır (Run All).** En az24GiB RAM önerilir.
Bu notebook henüz tamamlanmakta olan girdi yüklemesini otomatik bekler; hazır olunca
paket hash’lerini ve önceden sabitlenmiş294 SubjectInput / Attempt kimliğini doğrular.
370 adayın76 önceki observation hazırlama hatası korunur.

Bu ölçüm aşamasıdır. Eğitim/optimizer başlatmaz; final eligibility, global sign,
lineage split veya training izni üretmez. Üretilen kanıtların Platform’a bağlanması
ve model/truth incelemesindeki açıkların kapatılması ayrıca gereklidir.

Subject’ler tek process ile sırayla çalışır. Biten çıktılar Drive’a ayrı arşivlenir.
Kesilirse Run All ile yeniden açın; biten arşivler doğrulanır ve atlanır. Aynı oturumda
kalan yarım çalışma ayrıca korunur, sonra yalnız tamamlanmamış subject yeniden hesaplanır.
'''
    setup='CONFIG='+repr(config)+'\n'+'''import json,os,subprocess,sys,time
from pathlib import Path
from google.colab import drive
'''+inspect.getsource(effective_memory_gib)+'''
START=time.monotonic()
ram_gib=effective_memory_gib()
print(f'Gerçek RAM sınırı: {ram_gib:.1f} GiB',flush=True)
assert ram_gib>=22,'Yüksek RAM seçin (en az24GiB önerilir), sonra Run All.'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive')/CONFIG['drive_folder']
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','libxi6','libxrender1','libxfixes3','libxrandr2','libxxf86vm1','libsm6','libice6','libgl1','libxkbcommon0','libxinerama1'],check=True)
CODE=Path('/content/realsas-corpus370-worker-code')
if not CODE.exists():
    CODE.mkdir(parents=True)
    subprocess.run(['git','init',str(CODE)],check=True)
    subprocess.run(['git','-C',str(CODE),'remote','add','origin','https://github.com/merynz/RealSaS-OPT.git'],check=True)
subprocess.run(['git','-C',str(CODE),'fetch','--depth','1','origin',CONFIG['code_sha']],check=True)
subprocess.run(['git','-C',str(CODE),'checkout','--detach',CONFIG['code_sha']],check=True)
assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==CONFIG['code_sha']
VENV=Path('/content/realsas-corpus370-worker-venv')
if not (VENV/'bin/python').is_file():subprocess.run([sys.executable,'-m','venv','--without-pip',str(VENV)],check=True)
PYTHON=str(VENV/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',PYTHON,'install','--quiet','-r',str(CODE/'requirements/mainline-ci.txt'),'-r',str(CODE/'requirements/truth-family-courts.txt')],check=True)
os.environ['PYTHONPATH']=str(CODE)
os.environ['REALSAS_AUTHORITY_ROOT']='/home/monster/realsas_authority'
os.chdir(CODE)
print('SETUP_PASS',round(time.monotonic()-START,1),'seconds')
'''
    verify='EXPECTED_SCOPE='+repr(scope)+'\n'+'''RECEIPT=DRIVE/'COMPUTE_CAPSULE_RECEIPT.json'
deadline=time.monotonic()+3*60*60
while not RECEIPT.is_file():
    assert time.monotonic()<deadline,'Girdi yayını henüz tamamlanmadı. Aynı notebook daha sonra Run All ile devam eder.'
    print('Girdi paketi Drive’a yükleniyor; doğrulanmış yayın bekleniyor...',round(time.monotonic()-START),'seconds',flush=True)
    time.sleep(30)
published=json.loads(RECEIPT.read_text())
assert published['code_sha']==CONFIG['code_sha'] and published['training_authorized'] is False,'PUBLISHER_SCOPE_DRIFT'
ARCHIVE=DRIVE/'CORPUS370_COMPUTE_CAPSULE_V1.zip'
CACHE=Path('/content/corpus370-worker-inputs')/published['archive_sha256']
CACHE.mkdir(parents=True,exist_ok=True)
(CACHE/'EXPECTED_SCOPE.json').write_text(json.dumps(EXPECTED_SCOPE))
script="""import json,sys
from pathlib import Path
from models.materialization_contract_v1 import sha256
from tools.corpus.run_corpus370_compute_worker_v1 import hydrate
"""+'''+repr(inspect.getsource(verify_scope))+'''+"""
archive=Path(sys.argv[1]);receipt=json.loads(Path(sys.argv[2]).read_text());cache=Path(sys.argv[3])
assert sha256(archive)==receipt['archive_sha256'],'PUBLISHER_ARCHIVE_HASH_DRIFT'
scope=json.loads((cache/'EXPECTED_SCOPE.json').read_text())
verify_scope(archive,scope)
seal=hydrate(archive,receipt['archive_sha256'],cache)
assert sha256(cache/'COMPUTE_CAPSULE.json')==receipt['seal_sha256'],'PUBLISHER_SEAL_HASH_DRIFT'
print('EXACT_INPUT_SCOPE_VERIFIED',seal['worker_subject_count'],'subjects',flush=True)
"""
subprocess.run([PYTHON,'-u','-c',script,str(ARCHIVE),str(RECEIPT),str(CACHE)],check=True)
print('INPUTS_PASS',round(time.monotonic()-START,1),'seconds')
'''
    run='''OUTPUT=DRIVE/'results'/published['seal_sha256']
script="""import json,sys
from pathlib import Path
from tools.corpus.run_corpus370_compute_worker_v1 import run
from compiler.realsas_compiler_core.hashing import content_sha256
"""+'''+repr(inspect.getsource(preserve_interrupted_work))+'''+"""
cache=Path(sys.argv[1]);out=Path(sys.argv[2]);seal=json.loads((cache/'COMPUTE_CAPSULE.json').read_text())
recovered=preserve_interrupted_work(seal,content_sha256(seal),out,Path('/home/monster/realsas_authority'))
if recovered:print('INTERRUPTED_COMPUTE_PRESERVED',len(recovered),flush=True)
result=run(seal,out,cache)
print(json.dumps({k:result[k] for k in ('accounted_subject_count','candidate_subject_count','elapsed_seconds','qualification_minted','training_authorized')},indent=2))
"""
subprocess.run([PYTHON,'-u','-c',script,str(CACHE),str(OUTPUT)],check=True)
print('Sonuçlar:',OUTPUT)
print('Özet:',OUTPUT/'COMPUTE_PROGRESS.json')
print('Toplam süre:',round(time.monotonic()-START,1),'seconds')
'''
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Corpus370_High_RAM_Compute_Run_All_v1.ipynb'},
        'kernelspec':{'display_name':'Python3','name':'python3','language':'python'},'language_info':{'name':'python'},
        'realsas':config},'cells':[dict(cell(kind,src),id='corpus370-pending-'+str(i)) for i,(kind,src) in enumerate(
            [('markdown',intro),('code',setup),('code',verify),('code',run)])]}


def main():
    import argparse
    from tools.platform_truth_cohort_bootstrap_v1 import plan_manifest_keys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','bootstrap','repairs','plan','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('preparation-sha256','code-sha','drive-folder'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.preparation.read_bytes()).hexdigest()!=args.preparation_sha256:raise ValueError('PENDING_PREPARATION_HASH_DRIFT')
    preparation=json.loads(args.preparation.read_text())
    for path,key in ((args.bootstrap,'bootstrap_receipt'),(args.repairs,'source_dependency_repair_index')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=preparation[key]['sha256']:raise ValueError('PENDING_PARENT_REFERENCE_DRIFT:'+key)
    scope=scope_from_preparation(preparation,json.loads(args.bootstrap.read_text()),json.loads(args.repairs.read_text()),
        preparation_sha=args.preparation_sha256,code_sha=args.code_sha,
        section_keys=plan_manifest_keys(json.loads(args.plan.read_text())),plan_sha=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(build(scope,args.drive_folder),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'notebook':str(args.output),'worker_subject_count':len(scope['subjects']),'training_authorized':False}))


if __name__=='__main__':main()
