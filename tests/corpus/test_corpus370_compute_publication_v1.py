"""Transport does not regenerate or requalify a scientific package."""
import hashlib
import json
import subprocess
import pytest
from tools.corpus import publish_corpus370_compute_capsule_v1 as publisher


def save(path,value):path.write_text(json.dumps(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def existing(tmp_path):
    source=tmp_path/'original';source.mkdir()
    archive=source/'CORPUS370_COMPUTE_CAPSULE_V1.zip';archive.write_bytes(b'fixture-input-not-scientific-evidence')
    seal=source/'COMPUTE_CAPSULE.json'
    save(seal,{'schema':'RealSaS.Corpus370ComputeCapsule.v1','code_sha':'a'*40,'preparation_sha256':'p'*64,
        'candidate_subject_count':370,'worker_subject_count':294,'excluded_subject_count':76,
        'training_authorized':False,'qualification_minted':False})
    receipt={'code_sha':'a'*40,'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,
        'seal_sha256':sha(seal),'candidate_subject_count':370,'worker_subject_count':294,'training_authorized':False}
    save(source/'COMPUTE_CAPSULE_RECEIPT.json',receipt)
    config={'drive_folder':'fixture','preparation':{'sha256':'p'*64},
        'existing_capsule':{'directory':str(source),**{k:receipt[k] for k in ('code_sha','archive_sha256','archive_size_bytes','seal_sha256')}}}
    return config,source,tmp_path/'publication'


def test_replication_keeps_original_bytes_and_code_binding(existing):
    config,source,out=existing
    before={p.name:p.read_bytes() for p in source.iterdir()}
    publisher.prepare(config,out)
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    assert (out/'CORPUS370_COMPUTE_CAPSULE_V1.zip').read_bytes()==before['CORPUS370_COMPUTE_CAPSULE_V1.zip']
    assert json.loads((out/'PUBLISH_PROGRESS.json').read_text())['measurement_repeated'] is False
    n=json.loads((out/'Corpus370_Compute_Worker_Run_All_v1.ipynb').read_text())
    assert n['metadata']['realsas']['code_sha']=='a'*40


def test_changed_local_bytes_cannot_be_republished(existing):
    config,source,out=existing
    (source/'CORPUS370_COMPUTE_CAPSULE_V1.zip').write_bytes(b'drift')
    with pytest.raises(ValueError,match='PUBLISH_ARCHIVE_DRIFT'):publisher.prepare(config,out)


def test_network_failure_preserves_package_and_never_publishes_readiness(existing,monkeypatch):
    config,source,out=existing;publisher.prepare(config,out);calls=[]
    def fail_archive(args,**kwargs):
        calls.append((args,kwargs))
        if 'copyto' in args and args[args.index('copyto')+1].endswith('.zip'):raise subprocess.TimeoutExpired(args,kwargs['timeout'])
    monkeypatch.setattr(publisher.subprocess,'run',fail_archive)
    with pytest.raises(subprocess.TimeoutExpired):publisher.publish(out)
    assert not any('copyto' in args and args[args.index('copyto')+1].endswith('RECEIPT.json') for args,_ in calls)
    assert (source/'CORPUS370_COMPUTE_CAPSULE_V1.zip').is_file()
    assert json.loads((out/'PUBLISH_PROGRESS.json').read_text())['status']=='PUBLICATION_FAILED__PACKAGE_PRESERVED'
    assert all('--timeout' in args and kwargs['timeout']<=1800 for args,kwargs in calls)


def test_readiness_is_uploaded_only_after_remote_archive_check(existing,monkeypatch):
    config,source,out=existing;publisher.prepare(config,out);calls=[]
    monkeypatch.setattr(publisher.subprocess,'run',lambda args,**kwargs:calls.append(args))
    result=publisher.publish(out)
    check=next(i for i,args in enumerate(calls) if 'check' in args)
    receipt=next(i for i,args in enumerate(calls) if 'copyto' in args and args[args.index('copyto')+1].endswith('RECEIPT.json'))
    assert check<receipt and result['training_authorized'] is False
