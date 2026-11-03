"""Control-plane binding tests; these fixtures never mint scientific evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from tools.notebooks.build_corpus370_pending_compute_v1 import build, verify_scope


def digest(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


@pytest.fixture
def capsule():
    sections={'source_files':[{'path':'/source/asset','sha256':'f'*64}]}
    parent={'platform_subject_id':'platform','subject_input_id':'input','parent_attempt_id':'attempt',
            'run_id':'ORIGINAL_RUN','sections_sha256':digest(sections)}
    scope={'header':{'schema':'RealSaS.Corpus370ComputeCapsule.v1','code_sha':'a'*40,
                     'qualification_minted':False,'training_authorized':False},
           'section_keys':list(sections),'subjects':{'subject':parent},'excluded_subjects_sha256':digest([]),
           'plan_sha256':hashlib.sha256(b'plan').hexdigest()}
    manifest={**sections,'run_id':'ORIGINAL_RUN','subject_id':'subject','execution_class':'WITNESS'}
    seal={**scope['header'],'excluded_subjects':[],'plan':{'sha256':scope['plan_sha256'],'object':'objects/plan'},
          'subjects':[{'subject_id':'subject',**{k:parent[k] for k in ('platform_subject_id','subject_input_id','parent_attempt_id')}}]}
    return scope,manifest,seal


def archive(tmp_path,manifest,seal):
    raw=json.dumps(manifest).encode();seal=copy.deepcopy(seal)
    seal['subjects'][0]['manifest']={'object':'objects/manifest','sha256':hashlib.sha256(raw).hexdigest()}
    path=tmp_path/'input.zip'
    with zipfile.ZipFile(path,'w') as out:
        out.writestr('objects/plan',b'plan')
        out.writestr('objects/manifest',raw)
        out.writestr('COMPUTE_CAPSULE.json',json.dumps(seal))
    return path


def test_original_subject_input_matches_even_without_a_preknown_transport_hash(tmp_path,capsule):
    scope,manifest,seal=capsule
    assert verify_scope(archive(tmp_path,manifest,seal),scope)==seal | {'subjects':[seal['subjects'][0] | {
        'manifest':{'object':'objects/manifest','sha256':hashlib.sha256(json.dumps(manifest).encode()).hexdigest()}}]}


@pytest.mark.parametrize('drift',['section','input','attempt','exclusion','training','run','extra_section'])
def test_rehashed_transport_does_not_authorize_changed_scientific_inputs(tmp_path,capsule,drift):
    scope,manifest,seal=capsule
    if drift=='section':manifest['source_files'][0]['sha256']='b'*64
    elif drift=='input':seal['subjects'][0]['subject_input_id']='different'
    elif drift=='attempt':seal['subjects'][0]['parent_attempt_id']='different'
    elif drift=='exclusion':seal['excluded_subjects']=[{'subject_id':'different'}]
    elif drift=='training':seal['training_authorized']=True
    elif drift=='run':manifest['run_id']='different'
    else:manifest['extra']={}
    with pytest.raises(ValueError):verify_scope(archive(tmp_path,manifest,seal),scope)


def test_generated_cells_embed_their_worker_functions_without_builder_dependencies(tmp_path,capsule):
    scope,_,_=capsule
    notebook=build(scope,'corpus370_compute_worker_v1/'+'a'*40)
    calls=[]
    subprocess=SimpleNamespace(run=lambda args,**kwargs:calls.append(args))
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'COMPUTE_CAPSULE.json').write_text(json.dumps({'subjects':[]}))
    namespace={'Path':Path,'DRIVE':tmp_path,'published':{'seal_sha256':'b'*64},'CACHE':cache,
               'PYTHON':'python','subprocess':subprocess,'START':0,'time':SimpleNamespace(monotonic=lambda:1)}
    run_cell=''.join(notebook['cells'][3]['source'])
    exec(compile(run_cell,'notebook-run-cell','exec'),namespace)
    script=calls[0][3]
    tree=ast.parse(script)
    assert any(isinstance(node,ast.FunctionDef) and node.name=='preserve_interrupted_work' for node in tree.body)
    assert 'inspect.getsource' not in script
    verify_cell=''.join(notebook['cells'][2]['source'])
    # Reach script construction using an existing publication, without hydrating inputs.
    (tmp_path/'COMPUTE_CAPSULE_RECEIPT.json').write_text(json.dumps({'code_sha':'a'*40,'training_authorized':False,
        'archive_sha256':'c'*64,'seal_sha256':'b'*64}))
    namespace.update(CONFIG={'code_sha':'a'*40},json=json)
    # Only cache paths are relocated; the generated scientific subprocess remains unexecuted.
    verify_cell=verify_cell.replace("Path('/content/corpus370-worker-inputs')",'Path('+repr(str(tmp_path/'cache-inputs'))+')')
    exec(compile(verify_cell,'notebook-verify-cell','exec'),namespace)
    tree=ast.parse(calls[-1][3])
    assert any(isinstance(node,ast.FunctionDef) and node.name=='verify_scope' for node in tree.body)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
