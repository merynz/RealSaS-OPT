#!/usr/bin/env python3
"""Token-free, immutable minimal Corpus370 materialization CODE capsule.

The historical notebook cloned a PRIVATE GitHub repository without auth.
This capsule is built from GIT OBJECTS at an exact source commit, never from
modified working-tree bytes. It carries no private data, credentials, secrets,
or training/admission flags. It does not bypass the readiness checkpoint.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path

SCHEMA="RealSaS.Corpus370MaterializationCodeCapsule.v1"
FILES=(
  "models/materialization_contract_v1.py",
  "models/iris/v32/training_input_contract_v1.py",
  "models/tessa/v32/training_input_contract_v1.py",
  "tools/corpus/materialize_corpus370_sealed_bundle_v1.py",
)
OPTIONAL_INIT=(
  "models/__init__.py","models/iris/__init__.py",
  "models/iris/v32/__init__.py","models/tessa/__init__.py",
  "models/tessa/v32/__init__.py","tools/__init__.py",
  "tools/corpus/__init__.py",
)

def require(ok,code):
    if not ok:raise ValueError(code)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def run_git(repo,*args):
    p=subprocess.run(["git","-C",str(repo),*args],
                     check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    require(p.returncode==0,"CAPSULE_GIT_OBJECT_NOT_AVAILABLE:"+"/".join(args))
    return p.stdout

def build(repo,commit,output):
    repo=Path(repo).resolve()
    require(repo.is_dir() and len(commit)==40 and all(c in "0123456789abcdef" for c in commit),
            "CAPSULE_EXACT_SOURCE_COMMIT_REQUIRED")
    require(run_git(repo,"cat-file","-t",commit).strip()==b"commit",
            "CAPSULE_SOURCE_SHA_NOT_COMMIT")
    require(run_git(repo,"rev-parse",commit).decode().strip()==commit,
            "CAPSULE_GIT_REVISION_DRIFT")
    payload={}
    for name in FILES+OPTIONAL_INIT:
        proc=subprocess.run(["git","-C",str(repo),"show",f"{commit}:{name}"],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        if proc.returncode!=0:
            require(name in OPTIONAL_INIT,"CAPSULE_REQUIRED_CODE_MISSING:"+name)
            continue
        require(len(proc.stdout)<1_000_000,"CAPSULE_CODE_FILE_OVERSIZE:"+name)
        payload[name]=proc.stdout
    require(set(FILES).issubset(payload)
            and sum(len(x) for x in payload.values())<4_000_000,
            "CAPSULE_MINIMAL_SOURCE_CODE_CLOSURE")
    manifest={
       "schema":SCHEMA,
       "status":"SEALED_MINIMAL_EXECUTABLE_CODE_ONLY__NOT_MATERIALIZATION_OR_TRAINING",
       "source_git_commit":commit,
       "content_files":[{"path":k,"sha256":sha(v),"size_bytes":len(v)}
                        for k,v in sorted(payload.items())],
       "required_module_paths":list(FILES),
       "source_code_from_immutable_git_objects":True,
       "git_credentials_embedded":False,
       "source_truth_files_embedded":False,
       "optimizer_embedded":False,
       "checkpoint_requirement_preserved":True,
       "training_authorized":False,
    }
    target=Path(output).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):
            z.writestr(name,data)
        z.writestr("CODE_CAPSULE_MANIFEST.json",
                   json.dumps(manifest,sort_keys=True,indent=2).encode()+b"\n")
    return {"status":manifest["status"],"source_git_commit":commit,
            "code_archive_sha256":sha(target.read_bytes()),
            "code_archive_bytes":target.stat().st_size,
            "packed_code_file_count":len(payload),"training_authorized":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,default=Path("."))
    p.add_argument("--code-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.repo,a.code_sha,a.output),sort_keys=True),flush=True)
