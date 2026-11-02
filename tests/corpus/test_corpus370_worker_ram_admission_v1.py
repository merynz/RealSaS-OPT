import pytest
import json
import zipfile
from tools.notebooks.build_corpus370_compute_worker_v1 import effective_memory_gib,preserve_interrupted_work


@pytest.mark.parametrize('host,limits,expected',[
    (256,['8589934592'],8),
    (256,['max','34359738368'],32),
    (24,['max','9223372036854771712'],24),
    (16,[],16),
    (32,['invalid','0','-1'],32),
])
def test_capacity_uses_the_smallest_actual_host_or_container_limit(host,limits,expected):
    assert effective_memory_gib(f'MemTotal: {host*1024**2} kB\n',limits)==expected


def test_only_interrupted_compute_is_archived_before_a_clean_retry(tmp_path):
    identity='a'*64;authority=tmp_path/'authority';out=tmp_path/'out';out.mkdir()
    subjects=[{'subject_id':s,'platform_subject_id':s,'parent_attempt_id':'attempt-'+s} for s in ('unfinished','completed')]
    for s in subjects:
        root=authority/'runs'/('COMPUTE_'+identity[:16]+'_'+s['platform_subject_id'])
        root.mkdir(parents=True);(root/'partial.npz').write_bytes(b'unsealed-bytes')
    (out/'completed.json').write_text('{}')
    unrelated=authority/'runs'/'ACTUAL_PLATFORM_ATTEMPT';unrelated.mkdir()
    result=preserve_interrupted_work({'subjects':subjects},identity,out,authority)
    assert len(result)==1 and result[0]['qualification_minted'] is False
    assert (authority/'runs'/('COMPUTE_'+identity[:16]+'_completed')).is_dir()
    assert unrelated.is_dir()
    assert not (authority/'runs'/('COMPUTE_'+identity[:16]+'_unfinished')).exists()
    archive=next((out/'recovery').glob('*.zip'))
    with zipfile.ZipFile(archive) as zipped:assert zipped.read('partial.npz')==b'unsealed-bytes'
    assert json.loads(archive.with_suffix('.json').read_text())['status'].startswith('UNSEALED_INTERRUPTED_COMPUTE')
