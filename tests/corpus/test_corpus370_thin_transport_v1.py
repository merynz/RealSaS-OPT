"""Existing input bytes remain exact when their transport containers change."""
import hashlib
import ast
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
from tools.corpus import corpus370_thin_transport_v1 as thin


def digest(raw):return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def source(tmp_path,monkeypatch):
    base=tmp_path/'base.zip'
    with zipfile.ZipFile(base,'w') as out:out.writestr('asset.bin',b'base-asset')
    base_sha=thin.sha(base);prefix='/home/monster/realsas_authority/truth_cohort_bundles_v2/'+base_sha+'/'
    runtime='/home/monster/.cache/realsas/blender/official/blender-4.0.2-linux-x64/numeric-runtime-239/'
    inputs=[(prefix+'asset.bin',b'base-asset'),('/home/monster/realsas_authority/other/asset',b'base-asset'),
            (runtime+'lib/ordinary.bin',b'official-runtime'),(runtime+'cache.pyc',b'native-specific-cache'),
            ('/home/monster/realsas_authority/run/manifest.json',b'original-manifest')]
    rows=[{'path':p,'sha256':digest(raw),'size_bytes':len(raw),'object':'objects/'+digest(raw),'executable':False} for p,raw in inputs]
    seal={'schema':'RealSaS.Corpus370ComputeCapsule.v1','code_sha':'a'*40,'preparation_sha256':'b'*64,
          'files':rows,'allowed_original_roots':['/home/monster/realsas_authority',runtime.rstrip('/')],
          'qualification_minted':False,'training_authorized':False}
    original=tmp_path/'original';original.mkdir();seal_path=original/'COMPUTE_CAPSULE.json'
    seal_path.write_text(json.dumps(seal));archive=original/'CORPUS370_COMPUTE_CAPSULE_V1.zip'
    with zipfile.ZipFile(archive,'w') as out:
        for raw in {raw for _,raw in inputs}:out.writestr('objects/'+digest(raw),raw)
    config={'preparation':{'sha256':'b'*64},'existing_capsule':{'directory':str(original),'archive_sha256':thin.sha(archive),
        'archive_size_bytes':archive.stat().st_size,'seal_sha256':thin.sha(seal_path),'code_sha':'a'*40}}
    operator={'cohort_bundle':{'sha256':base_sha,'file_id':'fixture','name':'base.zip'},'evaluator_executable':runtime+'blender-glibc239'}
    recreated=tmp_path/'fresh';(recreated/'lib').mkdir(parents=True)
    (recreated/'lib/ordinary.bin').write_bytes(b'official-runtime')
    (recreated/'cache.pyc').write_bytes(b'new-cache-different-bytes')
    monkeypatch.setattr(thin,'fresh_runtime',lambda *args:recreated)
    return config,operator,base,tmp_path/'transport',seal_path


def test_dedup_uses_existing_bundle_and_keeps_unreproducible_cache_bytes(source):
    config,operator,base,out,seal=source
    receipt=thin.build(config,operator,out)
    assert receipt['provider_counts']=={'base':1,'runtime':1,'overlay':2}
    with zipfile.ZipFile(out/'CORPUS370_COMPUTE_THIN_V1.zip') as package:
        assert package.read('COMPUTE_CAPSULE.json')==seal.read_bytes()
        plan=json.loads(package.read('THIN_TRANSPORT.json'))
        cache_digest=digest(b'native-specific-cache')
        assert package.read(plan['providers'][cache_digest]['member'])==b'native-specific-cache'
    assert receipt['qualification_minted'] is False and receipt['training_authorized'] is False


def test_corrupted_parent_is_rejected_before_repack(source):
    config,operator,base,out,seal=source
    seal.write_text('{}')
    with pytest.raises(ValueError,match='THIN_PARENT_SEAL_DRIFT'):thin.build(config,operator,out)


def test_corrupted_transport_is_rejected_before_any_hydration(source):
    config,operator,base,out,seal=source;receipt=thin.build(config,operator,out)
    package=out/'CORPUS370_COMPUTE_THIN_V1.zip';package.write_bytes(package.read_bytes()+b'drift')
    with pytest.raises(ValueError,match='THIN_PACKAGE_HASH_DRIFT'):
        thin.hydrate(package,receipt['thin_sha256'],receipt['original_seal_sha256'],base,out/'cache')


def test_wrong_base_bytes_are_rejected_before_runtime_install(source):
    config,operator,base,out,seal=source;receipt=thin.build(config,operator,out)
    base.write_bytes(b'wrong-bundle')
    with pytest.raises(ValueError,match='THIN_BASE_BUNDLE_DRIFT'):
        thin.hydrate(out/'CORPUS370_COMPUTE_THIN_V1.zip',receipt['thin_sha256'],receipt['original_seal_sha256'],base,out/'cache')


def test_distinct_filesystems_keep_exact_bytes_with_atomic_copy(tmp_path,monkeypatch):
    obj=tmp_path/'object';obj.write_bytes(b'bytes');dest=tmp_path/'elsewhere'/'input'
    def separate_mount(*args):raise OSError(errno.EXDEV,'different filesystem')
    monkeypatch.setattr(thin.os,'link',separate_mount)
    thin.place_object(obj,dest,digest(b'bytes'),5)
    assert dest.read_bytes()==b'bytes' and not list(dest.parent.glob('*.thin-partial'))


def test_new_notebook_constructs_self_contained_hydration_without_a_publisher_wait(source):
    from tools.notebooks.build_corpus370_thin_compute_v1 import build
    config,operator,base,out,seal=source;receipt=thin.build(config,operator,out)
    original_receipt=out/'ORIGINAL_RECEIPT.json'
    original_receipt.write_text(json.dumps({'code_sha':'a'*40,'seal_sha256':thin.sha(seal),'archive_sha256':'c'*64}))
    notebook=build(seal,original_receipt,out/'THIN_TRANSPORT_RECEIPT.json','folder')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    assert 'ARCHIVE.is_file' not in ''.join(notebook['cells'][1]['source'])
    code=''.join(notebook['cells'][2]['source'])
    assert 'time.sleep' not in code and 'while not' not in code
    root=out/'drive';(root/'corpus370_preparation_v1').mkdir(parents=True)
    (root/'corpus370_preparation_v1/base.zip').write_bytes(base.read_bytes())
    code=code.replace("Path('/content/drive/MyDrive')",'Path('+repr(str(root))+')')
    code=code.replace("Path('/content/corpus370-worker-inputs')",'Path('+repr(str(out/'cache'))+')')
    calls=[]
    namespace={'Path':Path,'DRIVE':out,'CONFIG':notebook['metadata']['realsas'],'json':json,'PYTHON':'python',
        'subprocess':SimpleNamespace(run=lambda args,**kwargs:calls.append(args)),'time':SimpleNamespace(monotonic=lambda:0),'START':0}
    exec(compile(code,'thin-notebook-extract','exec'),namespace)
    script=calls[0][3];tree=ast.parse(script)
    definitions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert {'hydrate','place_object','sha','safe_member','require'}<=definitions
