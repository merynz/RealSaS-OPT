import ast
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from tools.corpus import export_corpus370_compute_capsule_v1 as exporter
from tools.corpus import run_corpus370_compute_worker_v1 as worker
from tools.notebooks.build_corpus370_compute_worker_v1 import build as notebook


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    return {'path':str(path),'sha256':exporter.sha256(path)}


def test_exact294_export_preserves370_scope_input_bytes_and_notebook_identity(tmp_path,monkeypatch):
    repo=tmp_path/'repo';authority=tmp_path/'authority';platform=tmp_path/'platform'
    put(repo/exporter.PLAN,{'schema':'Synthetic.Plan'})
    binary=authority/'runtime/blender';binary.parent.mkdir(parents=True);binary.write_bytes(b'pinned-executable')
    binary.chmod(0o755)
    resources=authority/'runtime-resources';resources.mkdir();(resources/'python.bin').write_bytes(b'embedded-python')
    (binary.parent/'resources').symlink_to(resources,target_is_directory=True)
    texture=authority/'source/texture.bin';texture.parent.mkdir();texture.write_bytes(b'exact-texture')
    rows=[];members=[];receipts=[]
    for i in range(294):
        sid=str(i);rid='CAMPAIGN_c_'+sid
        manifest=authority/'runs'/rid/'run_manifest.json'
        put(manifest,{'run_id':rid,'subject_id':sid,'source_files':[{'path':str(texture),'expected_sha256':exporter.sha256(texture)}],
            'truth_foundry':{'evaluator_binary':{'path':str(binary),'sha256':exporter.sha256(binary)}}})
        rows.append({'subject_id':sid,'subject_input_id':'input-'+sid,'run_manifest_path':str(manifest),'run_ledger_path':str(manifest.parent/'ACTIVE_RUN_V2.json')})
        members.append({'subject_id':sid,'attempt_id':'attempt-'+sid,'final_state':'OPEN'})
        receipts.append({'SubjectID':sid,'AttemptID':'attempt-'+sid})
    bootstrap=put(authority/'BOOTSTRAP.json',{'run_bindings':rows,'excluded_subjects':[{'subject_id':'excluded-'+str(i),'reason':'OBSERVATION_FAIL'} for i in range(76)]})
    prep=put(authority/'PREPARATION.json',{'bootstrap_receipt':bootstrap,'campaign_id':'c','engine_release_id':'r',
        'member_count':294,'candidate_subject_count':370,'excluded_subject_count':76,'source_code_sha':'1'*40,'campaign_run':{'receipts':receipts}})
    monkeypatch.setattr(exporter,'api',lambda *a,**k:{'members':members})
    monkeypatch.setattr(exporter.subprocess,'check_output',lambda *a,**k:'2'*40)
    out=tmp_path/'out'
    receipt=exporter.build(Path(prep['path']),prep['sha256'],authority_root=authority,platform_root=platform,repo=repo,output=out,api_root='test')
    seal=json.loads((out/'COMPUTE_CAPSULE.json').read_text())
    assert seal['worker_subject_count']==294 and len(seal['excluded_subjects'])==76
    assert seal['qualification_minted'] is False and seal['training_authorized'] is False
    assert any(r['path']==str(binary.parent/'resources/python.bin') for r in seal['files'])
    with zipfile.ZipFile(out/'CORPUS370_COMPUTE_CAPSULE_V1.zip') as archive:
        assert len(archive.namelist())==len(set(r['object'] for r in seal['files']))+1
        for row in seal['files']:assert hashlib.sha256(archive.read(row['object'])).hexdigest()==row['sha256']
    value=notebook(out/'COMPUTE_CAPSULE.json',out/'COMPUTE_CAPSULE_RECEIPT.json','worker/pinned')
    assert value['metadata']['realsas']['archive_sha256']==receipt['archive_sha256']
    for cell in value['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    altered=json.loads((out/'COMPUTE_CAPSULE_RECEIPT.json').read_text());altered['seal_sha256']='0'*64
    put(out/'bad.json',altered)
    with pytest.raises(ValueError,match='SEAL_BINDING'):notebook(out/'COMPUTE_CAPSULE.json',out/'bad.json','worker')


@pytest.mark.parametrize('name',['../escape','/absolute','objects/../../escape','objects\\escape'])
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError,match='PATH_ESCAPE'):worker.safe_member(name)


def test_resume_rejects_a_foreign_subject_scope_before_payload_access():
    seal={'identity':'capsule','code_sha':'2'*40}
    subject={'subject_id':'s','parent_attempt_id':'a'}
    result={'schema':'RealSaS.Corpus370ComputeSubjectReceipt.v1','capsule_identity':'foreign',
            'subject_id':'s','parent_attempt_id':'a','code_sha':'2'*40,'qualification_minted':False,'training_authorized':False,'files':[]}
    with pytest.raises(ValueError,match='RESUME_SCOPE_DRIFT'):worker.verify_result(result,seal,subject)


def test_memory_exhaustion_is_not_a_scientific_family_fail(monkeypatch,tmp_path):
    from compiler.realsas_compiler_services.orchestrator.adapters import corpus_family_v1 as family
    def oom(*a,**k):raise MemoryError('measured-capacity-failure')
    monkeypatch.setattr(family.geometry,'build',oom)
    with pytest.raises(MemoryError):family.measure_subject_families(sid='s',source_sha='1'*64,
        scene_path=tmp_path/'scene',replay_path=tmp_path/'replay',geometry_path=tmp_path/'geom',
        snapshot_path=tmp_path/'snapshot',camera_path=tmp_path/'camera',root=tmp_path/'out')
    assert not (tmp_path/'out/geometry/COURT.json').exists()
