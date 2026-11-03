import hashlib,json,zipfile
import pytest
from tools.corpus.export_corpus370_evidence_slice_v1 import export


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def bundle(root,*,path='sources/asset.glb'):
    raw=b'original source bytes';digest=hashlib.sha256(raw).hexdigest();rows=[]
    for i in range(370):
        rows.append({'subject_id':f'corpus_{i:020x}','raw_source':{'path':path,'sha256':digest,'size_bytes':len(raw)}})
    p=root/'bundle.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('BOOTSTRAP_MANIFEST.json',json.dumps({'candidate_subject_count':370,'subjects':rows}));z.writestr(path,raw)
    return p,rows[0]['subject_id']


def test_slice_preserves_original_bytes_and_cannot_mint_authority(tmp_path):
    p,sid=bundle(tmp_path);r=export(p,sha(p),[sid],tmp_path/'out')
    assert r['training_authorized'] is False and r['qualification_minted'] is False and r['source_data_mutated'] is False
    with zipfile.ZipFile(r['archive']['path']) as z:
        assert z.read('sources/asset.glb')==b'original source bytes'
        assert set(z.namelist())=={'sources/asset.glb','BOOTSTRAP_MANIFEST.json','EVIDENCE_SLICE.json'}


def test_unknown_subject_and_wrong_bundle_hash_fail_closed(tmp_path):
    p,sid=bundle(tmp_path)
    with pytest.raises(ValueError,match='BUNDLE_HASH'):export(p,'0'*64,[sid],tmp_path/'out')
    with pytest.raises(ValueError,match='EXACT_SCOPE'):export(p,sha(p),['corpus_'+'f'*20],tmp_path/'out')


def test_selected_archive_path_escape_and_size_limit_are_rejected(tmp_path):
    p,sid=bundle(tmp_path,path='../outside')
    with pytest.raises(ValueError,match='MEMBER_ESCAPE'):export(p,sha(p),[sid],tmp_path/'out')
    p,sid=bundle(tmp_path)
    with pytest.raises(ValueError,match='BYTE_LIMIT'):export(p,sha(p),[sid],tmp_path/'out',maximum_bytes=10)


def test_native_packet_cannot_read_outside_preparation_root(tmp_path):
    p,sid=bundle(tmp_path);root=tmp_path/'prep';index=root/'observations_v2/CORPUS370_PREPARED_OBSERVATION_PACKETS_V2.json'
    index.parent.mkdir(parents=True);index.write_text(json.dumps({'subjects':[{'subject_id':sid,'packet_path':'/tmp/outside/packet/'+sid}]}))
    with pytest.raises(ValueError,match='NATIVE_ESCAPE'):export(p,sha(p),[sid],tmp_path/'out',preparation_root=root)
