"""Private-repo auth-free code ZIP plus strict existing readiness notebook gates."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import pytest
from models.materialization_contract_v1 import sha256
from tools.corpus.build_corpus370_materialization_code_capsule_v1 import build as capsule,FILES
from tools.notebooks.build_corpus370_materialization_capsule_v2 import build as notebook

def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()

def test_code_capsule_comes_from_git_objects_not_modified_worktree(tmp_path):
    repo=tmp_path/"mini";repo.mkdir()
    git(repo,"init","-q");git(repo,"config","user.name","Test")
    git(repo,"config","user.email","test@example.org")
    for p in FILES:
        path=repo/p;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("CLAIMED_IMMUTABLE_SOURCE = True\n")
    git(repo,"add",".");git(repo,"commit","-qm","sealed")
    sha=git(repo,"rev-parse","HEAD")
    modified=repo/FILES[0];modified.write_text("CLAIMED_IMMUTABLE_SOURCE = False\n")
    target=tmp_path/"capsule.zip";receipt=capsule(repo,sha,target)
    assert receipt["source_git_commit"]==sha
    assert receipt["training_authorized"] is False
    assert receipt["packed_code_file_count"]==len(FILES)
    assert receipt["code_archive_sha256"]==sha256(target)
    with zipfile.ZipFile(target) as z:
        index=json.loads(z.read("CODE_CAPSULE_MANIFEST.json"))
        assert index["source_git_commit"]==sha
        assert set(FILES)==set(z.namelist())-{"CODE_CAPSULE_MANIFEST.json"}
        assert z.read(FILES[0])==b"CLAIMED_IMMUTABLE_SOURCE = True\n"
        assert index["source_truth_files_embedded"] is False
        assert index["git_credentials_embedded"] is False

def ready(tmp_path):
    cp=tmp_path/"checkpoint.json"
    cp.write_text(json.dumps({
        "schema":"RealSaS.Corpus370MaterializationCheckpoint.v1",
        "status":"PASS_PRE_MATERIALIZATION_CHECKPOINT",
        "counts_are_final":True,"materialization_authorized":True,
        "training_authorized":False,
        "models":{"IRIS":{"eligible_subject_count":1,
           "partial_supervision_subject_count":1,
           "split_counts":{"train":1,"val":0,"global_unseen":0}}}
    }))
    seal=tmp_path/"seal.json"
    seal.write_text(json.dumps({
      "schema":"RealSaS.Corpus370MaterializationBundle.v1",
      "status":"PASS_SEALED_STATIC_CORPUS_BUNDLE",
      "training_authorized":False,
      "checkpoint":{"sha256":sha256(cp)}
    }))
    return cp,seal

def invoke(cp,seal):
    return notebook(cp,seal,
       zip_sha256="a"*64,drive_bundle_path="private-corpus/bundle.zip",
       code_sha="b"*40,code_archive_sha256="c"*64,
       drive_code_path="private-corpus/code-capsule.zip")

def test_secure_run_all_compiles_and_uses_two_drive_archives_no_private_git_auth(tmp_path):
    n=invoke(*ready(tmp_path))
    assert len(n["cells"])==4
    assert n["metadata"]["realsas"]["code_archive_sha256"]=="c"*64
    assert n["metadata"]["realsas"]["code_sha"]=="b"*40
    assert n["metadata"]["colab"]["name"]=="Corpus370_Materialize_Sealed_Code_Run_All_v2.ipynb"
    for cell in n["cells"]:
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),"<code-cell>","exec")
    setup="".join(n["cells"][1]["source"])
    assert "git clone" not in setup and "git fetch" not in setup
    assert "CODE_CAPSULE_FILE_SHA_DRIFT" in setup
    assert "CODE_CAPSULE_UNSEALED_EXTRA" in setup
    assert "CODE_CAPSULE_ZIP_BOMB_BUDGET" in setup
    assert "CAPSULE_SECOND_HASH_DRIFT" in setup
    assert "git_credentials_embedded" in setup and "training_authorized" in setup
    assert "from models.materialization_contract_v1 import sha256" in setup
    assert "materialize(" in "".join(n["cells"][3]["source"])

@pytest.mark.parametrize("mutation",[
   "blocked_checkpoint","bad_seal","invalid_code_archive_hash",
   "path_escape","absolute_path","backslash_path"
])
def test_no_ready_checkpoint_no_unsafe_code_capsule_paths(tmp_path,mutation):
    cp,seal=ready(tmp_path)
    kwargs={"zip_sha256":"a"*64,"drive_bundle_path":"bundle.zip",
            "code_sha":"b"*40,"code_archive_sha256":"c"*64,
            "drive_code_path":"capsule.zip"}
    if mutation=="blocked_checkpoint":
        d=json.loads(cp.read_text());d["status"]="BLOCKED_PRE_MATERIALIZATION";cp.write_text(json.dumps(d))
    elif mutation=="bad_seal":
        d=json.loads(seal.read_text());d["checkpoint"]["sha256"]="0"*64;seal.write_text(json.dumps(d))
    elif mutation=="invalid_code_archive_hash":kwargs["code_archive_sha256"]="z"*64
    elif mutation=="path_escape":kwargs["drive_code_path"]="../../run.py"
    elif mutation=="absolute_path":kwargs["drive_code_path"]="/tmp/code.zip"
    elif mutation=="backslash_path":kwargs["drive_code_path"]="folder\\code.zip"
    with pytest.raises(ValueError):
        notebook(cp,seal,**kwargs)

def test_code_capsule_rejects_forged_git_sha_and_unreachable_commit(tmp_path):
    with pytest.raises(ValueError):
        capsule(tmp_path,"F"*40,tmp_path/"no.zip")
