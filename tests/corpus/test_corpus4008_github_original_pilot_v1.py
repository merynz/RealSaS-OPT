"""GitHub original reacquisition is constrained to immutable blobs and source SHA."""
import hashlib
import pytest

from tools.corpus.recover_corpus4008_github_original_pilot_v1 import (
    pinned_raw_url,choose_pilot,fetch_one,
)


def test_exact_immutable_github_blob_only():
    commit="a"*40
    url="https://github.com/good/repo/blob/"+commit+"/art/Bad%20Guy.FBX"
    result,ext=pinned_raw_url(url)
    assert result=="https://raw.githubusercontent.com/good/repo/"+commit+"/art/Bad%20Guy.FBX"
    assert ext==".fbx"
    for url in [
        "https://github.com/good/repo/blob/main/thing.fbx",
        "https://github.com/good/repo/blob/"+commit+"/../private.fbx",
        "https://evil.test/good/repo/blob/"+commit+"/thing.fbx",
        "https://github.com/good/repo/blob/"+commit+"/script.py",
        "https://github.com/good/repo/blob/"+commit+"/file.fbx?raw=1",
    ]:
        with pytest.raises(ValueError):
            pinned_raw_url(url)


def test_reacquisition_refuses_html_sha_mismatch_and_bogus_redirect(tmp_path):
    class FakeResponse:
        status_code=200
        headers={"Content-Type":"application/octet-stream"}
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def iter_content(self,chunk_size):yield self.data

    class FakeSession:
        def __init__(self,data):self.data=data
        def get(self,*args,**kwargs):
            assert kwargs["allow_redirects"] is False
            return FakeResponse(self.data)
    blob=b"exact synthetic original glb"
    h=hashlib.sha256(blob).hexdigest()
    row={
        "source_expected_sha256":h,
        "candidate_id":"objaverse_animated_originals:"+h,
        "canonical_asset_id":"test",
        "historical_split":"FIT",
        "source_url_private":"https://github.com/good/repo/blob/"+("f"*40)+"/foo.glb",
    }
    a=fetch_one(row,tmp_path,max_bytes=1024,session=FakeSession(blob))
    assert a["source_bytes_recovered"] is True
    assert (tmp_path/(h+".glb")).read_bytes()==blob
    (tmp_path/(h+".glb")).unlink()
    bad=fetch_one(row,tmp_path,max_bytes=1024,session=FakeSession(b"not the source"))
    assert bad["status"]=="SHA_MISMATCH__REACQUISITION_NOT_ACCEPTED"
    assert not (tmp_path/(h+".glb")).exists()


def test_pilot_preserves_holdout_and_source_formats():
    rows=[]
    for ext in (".fbx",".glb",".gltf"):
        for i in range(5):
            h=hashlib.sha256((ext+str(i)).encode()).hexdigest()
            rows.append({
                "candidate_id":"objaverse_animated_originals:"+h,
                "source_expected_sha256":h,
                "historical_split":"EXTERNAL_HOLDOUT" if i==0 else "FIT",
                "historical_legal_decision":"PASS",
                "source_candidate_metadata_count":1,
                "source_url_private":"https://github.com/test/repo/blob/"+("f"*40)+"/asset"+ext,
            })
    picked=choose_pilot(rows,2)
    assert len(picked)==6
    assert all(x["historical_split"]!="EXTERNAL_HOLDOUT" for x in picked)
