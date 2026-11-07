#!/usr/bin/env python3
"""Bounded-memory, read-only Corpus370 V5 portability bill of materials.

The source V5 manifests intentionally contain sealed ABSOLUTE paths rooted in
the original WSL host. Hosted execution must reconstruct these SAME paths and
bytes rather than rewrite source JSON (which would invalidate signed evidence).
This producer inventories only; it does not expose the private manifest in a
public Actions artifact or mint any training truth.

The source extracted tree is included to retain glTF/texture/FBX dependencies
that are not listed as independent source roster entries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input

SCHEMA="RealSaS.Corpus370HostedPortabilityInventory.v1"
V5_SHA="8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA="925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
# The original REST index SHA is passed to the CLI and must equal this exact
# independent source-rest court receipt from the completed 352 sweep.
REST_PIN="384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"
SOURCE_ARCHIVE_SHA="5c92e77ae07b215fa4234d221faf27659bc0c47a54d2711d6e49db3d820b8fa9"
REQUIRED_BASE=Path("/home/monster/realsas_authority")


def require(ok,why):
    if not ok:raise ValueError(why)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as inp:
        for b in iter(lambda:inp.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def capture(v5_path,roster_path,rest_path,source_root,*,sha_all:bool=True):
    for p,expected in ((v5_path,V5_SHA),(roster_path,ROSTER_SHA),(rest_path,REST_PIN)):
        require(p.is_file() and digest(p)==expected,"HOSTED_INVENTORY_PINNED_INDEX_DRIFT:"+p.name)
    v5=json.loads(v5_path.read_text())
    roster=json.loads(roster_path.read_text())
    rest=json.loads(rest_path.read_text())
    require(v5.get("subject_count")==370 and
            (v5.get("prepared_subject_count"),v5.get("failure_count"))==(352,18)
            and roster.get("subject_count")==370
            and rest.get("summary",{}).get("independent_source_rest_pass")==352
            and all(x.get("training_authorized") is False for x in (v5,roster,rest)),
            "HOSTED_INVENTORY_COHORT_IDENTITY")
    by_source={x["subject_id"]:x for x in roster["subjects"]}
    by_rest=rest["subjects"]
    require(len(by_source)==370 and len(by_rest)==352,
            "HOSTED_INVENTORY_SOURCE_REST_SUBJECT_PARTITION")
    files={}
    categories=Counter()
    def add(file,kind,expected=None):
        p=Path(file)
        require(p.is_absolute() and p.is_relative_to(REQUIRED_BASE)
                and not p.is_symlink() and p.is_file(),
                "HOSTED_INVENTORY_UNSAFE_OR_MISSING_FILE:"+str(p))
        actual=str(p)
        require(p.stat().st_size>=0, "HOSTED_INVENTORY_INVALID_SIZE")
        if actual in files:
            if expected:
                require(files[actual]["sha256"]==expected,"HOSTED_INVENTORY_HASH_CONFLICT:"+actual)
        else:
            checksum=digest(p) if sha_all or expected is None else expected
            if expected:
                require(checksum==expected,"HOSTED_INVENTORY_PARENT_HASH_DRIFT:"+actual)
            files[actual]={
                "path":actual,"size_bytes":p.stat().st_size,"sha256":checksum,
            }
        categories[kind]+=1
    add(v5_path,"V5_INDEX",V5_SHA)
    add(roster_path,"EXACT_ROSTER",ROSTER_SHA)
    add(rest_path,"SOURCE_REST_INDEX",REST_PIN)
    for row in v5["subjects"]:
        sid=row["subject_id"]; src=by_source[sid]; target=by_rest[sid]
        require(src["source_sha256"]==row["source_sha256"]
                and target["status"]=="PASS_SOURCE_REST_INDEPENDENT_COURT"
                and target["input"]["source_sha256"]==row["source_sha256"],
                "HOSTED_INVENTORY_SUBJECT_IDENTITY:"+sid)
        for key,ref in row_input(row).items():
            add(Path(ref["path"]),key,ref["sha256"])
        manifest=target["manifest"]
        add(Path(manifest["path"]),"SOURCE_REST_MANIFEST",manifest["sha256"])
        content=json.loads(Path(manifest["path"]).read_text())
        require(content["subject_id"]==sid and
                content["source_sha256"]==row["source_sha256"]
                and content["authority"]["training_authorized"] is False,
                "HOSTED_INVENTORY_REST_MANIFEST_WRONG_SOURCE")
        out=content["output"]
        add(Path(out["path"]),"SOURCE_REST_NPZ",out["sha256"])
        original=(source_root/src["source_relative_path"])
        require(original.resolve().is_relative_to(source_root.resolve()),
                "HOSTED_INVENTORY_SOURCE_TRAVERSAL:"+sid)
        add(original,"SOURCE_ORIGINAL",src["source_sha256"])
    for row in v5["failures"]:
        sid=row["subject_id"];src=by_source[sid]
        require(src["source_sha256"]==row["source_sha256"],
                "HOSTED_INVENTORY_FAILED_SOURCE_IDENTITY")
        original=(source_root/src["source_relative_path"])
        require(original.resolve().is_relative_to(source_root.resolve()),
                "HOSTED_INVENTORY_FAILED_SOURCE_TRAVERSAL")
        add(original,"SOURCE_OBSERVATION_FAIL_PRESERVED",src["source_sha256"])
    # Include all external .gltf textures and .bin file dependencies. Source
    # file validity is measured; external file closure remains a transport job.
    for base, dirs, names in os.walk(source_root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
        for name in names:
            path=Path(base)/name
            if not path.is_symlink() and path.is_file():
                add(path,"SOURCE_TREE_DEPENDENCY")
    total=sum(v["size_bytes"] for v in files.values())
    return {
        "schema":SCHEMA,
        "status":"SHA_BOUND_INPUT_MANIFEST__NO_HOSTED_BYTE_TRANSPORT_YET",
        "training_authorized":False,
        "source_cohort":370,"observation_qualified":352,"observation_failed":18,
        "files":sorted(files.values(),key=lambda x:x["path"]),
        "summary":{
            "unique_file_count":len(files),
            "total_input_bytes":total,
            "total_input_gib":round(total/1024**3,3),
            "conservative_double_staging_gib":round(2*total/1024**3,3),
            "category_reference_counts":dict(sorted(categories.items())),
            "fits_14gib_when_double_staged":2*total<14*1024**3,
            "source_archive_dependency_count":categories["SOURCE_TREE_DEPENDENCY"],
            "preserves_original_source_absolute_paths":True,
            "private_inventory_must_not_upload_to_public_artifact":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--source-rest-index",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--summary-output",type=Path,required=True)
    a=p.parse_args()
    v=capture(a.v5_index,a.exact_source_roster,a.source_rest_index,a.source_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n")
    a.summary_output.parent.mkdir(parents=True,exist_ok=True)
    a.summary_output.write_text(json.dumps({
        "schema":v["schema"],"status":v["status"],"summary":v["summary"],
        "training_authorized":False,
    },sort_keys=True,indent=2)+"\n")
    print(json.dumps(v["summary"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
