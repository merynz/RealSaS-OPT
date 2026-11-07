#!/usr/bin/env python3
"""Build/hydrate private SHA-addressed V5 Corpus370 shards for hosted CPU workers.

This tool transports exact BYTES and original absolute paths. It does not
rewrite any V5 manifest, grant truth authority, or publish private files to
public GitHub artifacts. Upload shards to a PRIVATE Drive folder via rclone;
GitHub-hosted readers require a dedicated short-scoped Drive credential in an
Actions secret, never a public URL/PR artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile

SCHEMA="RealSaS.Corpus370HostedPrivateV5Shards.v1"
INV_SCHEMA="RealSaS.Corpus370HostedPortabilityInventory.v1"
ALLOWED_ROOT=Path("/home/monster/realsas_authority")
MAX_STORAGE_BYTES=4*1024**3
MIN_SHARD=64*1024**2
MAX_SHARD=3*1024**3


def require(value,code):
    if not value:
        raise ValueError(code)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda:src.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(path,root=None):
    root=Path(root) if root is not None else ALLOWED_ROOT
    p=Path(path)
    require(p.is_absolute() and p!=root and p.is_relative_to(root)
            and ".." not in p.parts,
            "HOSTED_SHARD_UNSAFE_ABSOLUTE_SOURCE_PATH")
    rel=p.relative_to(root)
    require(all(x not in (".","..","") for x in rel.parts)
            and str(rel)!="","HOSTED_SHARD_UNSAFE_RELATIVE_PATH")
    return rel


def _validate_inventory(doc):
    require(doc.get("schema")==INV_SCHEMA
            and doc.get("status")=="SHA_BOUND_INPUT_MANIFEST__NO_HOSTED_BYTE_TRANSPORT_YET"
            and doc.get("source_cohort")==370
            and doc.get("observation_qualified")==352
            and doc.get("observation_failed")==18
            and doc.get("training_authorized") is False,
            "HOSTED_SHARD_EXACT_INPUT_CENSUS_REQUIRED")
    files=doc.get("files") or []
    require(files and len(files)==doc["summary"]["unique_file_count"],
            "HOSTED_SHARD_INVENTORY_COMPLETENESS")
    seen=set()
    objects={}
    for f in files:
        p=str(f["path"]);rel=safe_relative(p)
        require(p not in seen and len(f["sha256"])==64 and
                all(c in "0123456789abcdef" for c in f["sha256"]) and
                type(f["size_bytes"]) is int and f["size_bytes"]>=0,
                "HOSTED_SHARD_INVALID_REF")
        seen.add(p)
        old=objects.get(f["sha256"])
        if old is None:
            objects[f["sha256"]]={
                "sha256":f["sha256"],"size_bytes":f["size_bytes"],"source_path":p,
            }
        else:
            require(old["size_bytes"]==f["size_bytes"],"HOSTED_SHARD_SHA_SIZE_COLLISION")
    require(sum(v["size_bytes"] for v in objects.values())<=MAX_STORAGE_BYTES,
            "HOSTED_SHARD_14GIB_DISK_BUDGET_EXCEEDED")
    return files,objects


def partition(objects,max_shard_bytes):
    require(MIN_SHARD<=max_shard_bytes<=MAX_SHARD,
            "HOSTED_SHARD_SIZE_LIMIT_INVALID")
    groups=[];current=[];used=0
    for digest_id,row in sorted(objects.items()):
        size=row["size_bytes"]
        require(size<=max_shard_bytes,
                "HOSTED_SHARD_ONE_OBJECT_EXCEEDS_CHUNK_LIMIT:"+digest_id)
        if current and used+size>max_shard_bytes:
            groups.append(current);current=[];used=0
        current.append(row);used+=size
    if current:groups.append(current)
    require(groups,"HOSTED_SHARD_EMPTY_PARTITION")
    return groups


def _write_zip(path,objects):
    stage=path.with_suffix(".zip.partial")
    with zipfile.ZipFile(stage,"w",compression=zipfile.ZIP_STORED,
                         allowZip64=True) as out:
        for obj in objects:
            src=Path(obj["source_path"])
            require(src.is_file() and not src.is_symlink()
                    and src.stat().st_size==obj["size_bytes"]
                    and digest(src)==obj["sha256"],
                    "HOSTED_SHARD_SOURCE_FILE_DRIFT:"+obj["sha256"])
            out.write(src,arcname="objects/"+obj["sha256"])
            require(digest(src)==obj["sha256"],
                    "HOSTED_SHARD_SOURCE_CHANGED_DURING_COPY")
    stage.replace(path)


def export(inventory_path,output_root,max_shard_bytes=768*1024**2):
    inventory_path=Path(inventory_path)
    output_root=Path(output_root)
    doc=json.loads(inventory_path.read_text())
    files,objects=_validate_inventory(doc)
    groups=partition(objects,max_shard_bytes)
    output_root.mkdir(parents=True,exist_ok=True)
    staged=[]
    for i,group in enumerate(groups,1):
        name=f"private-objects-{i:04d}.zip"
        path=output_root/name
        if not path.exists():
            _write_zip(path,group)
        require(path.is_file(),"HOSTED_SHARD_ARCHIVE_MISSING")
        # Verify every member digest even if a cached archive is being reused.
        with zipfile.ZipFile(path) as z:
            expected={"objects/"+o["sha256"]:o for o in group}
            require(set(z.namelist())==set(expected),"HOSTED_SHARD_ARCHIVE_MEMBER_DRIFT")
            for name_in,obj in expected.items():
                h=hashlib.sha256()
                with z.open(name_in) as stream:
                    for block in iter(lambda:stream.read(8<<20),b""):
                        h.update(block)
                require(h.hexdigest()==obj["sha256"],
                        "HOSTED_SHARD_CACHED_ARCHIVE_OBJECT_DRIFT")
        staged.append({
            "name":name,"sha256":digest(path),"size_bytes":path.stat().st_size,
            "object_count":len(group),
            "object_ids":[o["sha256"] for o in group],
        })
        print(json.dumps({"event":"HOSTED_PRIVATE_V5_SHARD_SEALED",
                          "index":i,"total":len(groups),
                          "archive_bytes":staged[-1]["size_bytes"],
                          "object_count":len(group)},sort_keys=True),flush=True)
    manifest={
        "schema":SCHEMA,"status":"SEALED_PRIVATE_TRANSPORT__NO_TRAINING_AUTHORITY",
        "parent_inventory_sha256":digest(inventory_path),
        "absolute_original_root":str(ALLOWED_ROOT),
        "subject_count":370,"observation_pass_count":352,"observation_fail_count":18,
        "file_count":len(files),
        "object_count":len(objects),
        "files":[{"relative_path":str(safe_relative(f["path"])),
                  "sha256":f["sha256"],"size_bytes":f["size_bytes"]}
                 for f in files],
        "shards":staged,
        "training_authorized":False,
        "family_authority_minted":False,
        "public_artifact_export_allowed":False,
    }
    path=output_root/"PRIVATE_V5_SHARD_MANIFEST.json"
    if path.is_file():
        require(json.loads(path.read_text())==manifest,
                "HOSTED_SHARD_EXISTING_MANIFEST_DRIFT")
    else:
        tmp=path.with_suffix(".partial")
        tmp.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
        tmp.replace(path)
    receipt={
        "schema":SCHEMA,"status":manifest["status"],
        "manifest_sha256":digest(path),
        "subject_count":370,"source_file_count":len(files),
        "unique_object_count":len(objects),
        "shard_count":len(staged),
        "shard_total_bytes":sum(x["size_bytes"] for x in staged),
        "estimated_download_and_extract_gib":round(
            (sum(x["size_bytes"] for x in staged)
             +sum(v["size_bytes"] for v in objects.values()))/1024**3,3),
        "training_authorized":False,
    }
    (output_root/"PUBLIC_PATH_FREE_RECEIPT.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


def _object_from_shard(z,zip_name,obj_dest,digest_id,expected_size):
    require(zip_name=="objects/"+digest_id,"HOSTED_SHARD_OBJECT_MEMBER_NOT_CANONICAL")
    temp=obj_dest.with_suffix(".partial")
    h=hashlib.sha256();total=0
    with z.open(zip_name) as src, temp.open("wb") as dst:
        for block in iter(lambda:src.read(8<<20),b""):
            dst.write(block);h.update(block);total+=len(block)
    require(total==expected_size and h.hexdigest()==digest_id,
            "HOSTED_SHARD_HYDRATE_OBJECT_SHA_MISMATCH")
    temp.replace(obj_dest)


def hydrate(manifest_path,private_dir,object_cache,*,expected_manifest_sha,
            target_root=ALLOWED_ROOT):
    manifest_path=Path(manifest_path)
    require(digest(manifest_path)==expected_manifest_sha,
            "HOSTED_SHARD_PINNED_MANIFEST_SHA_MISMATCH")
    doc=json.loads(manifest_path.read_text())
    require(doc.get("schema")==SCHEMA and doc.get("training_authorized") is False
            and doc.get("family_authority_minted") is False
            and doc.get("public_artifact_export_allowed") is False
            and (doc.get("subject_count"),doc.get("observation_pass_count"),
                 doc.get("observation_fail_count"))==(370,352,18)
            and doc.get("absolute_original_root")==str(ALLOWED_ROOT),
            "HOSTED_SHARD_REHYDRATION_SCOPE")
    private_dir=Path(private_dir);object_cache=Path(object_cache)
    object_cache.mkdir(parents=True,exist_ok=True)
    by_object={}
    for f in doc["files"]:
        rel=Path(f["relative_path"])
        require(not rel.is_absolute() and ".." not in rel.parts and
                f["relative_path"]==str(rel)
                and len(rel.parts)>=1,
                "HOSTED_SHARD_TARGET_PATH_ESCAPE")
        by_object.setdefault(f["sha256"],f["size_bytes"])
        require(by_object[f["sha256"]]==f["size_bytes"],
                "HOSTED_SHARD_OBJECT_SIZE_DRIFT")
    require(len(by_object)==doc["object_count"] and
            len(doc["files"])==doc["file_count"],
            "HOSTED_SHARD_HYDRATION_MANIFEST_COUNTS")
    encountered=set()
    for chunk in doc["shards"]:
        name=chunk["name"]
        require(name.startswith("private-objects-") and name.endswith(".zip") and
                "/" not in name and "\\" not in name,
                "HOSTED_SHARD_UNSAFE_ARCHIVE_NAME")
        path=private_dir/name
        require(path.is_file() and path.stat().st_size==chunk["size_bytes"] and
                digest(path)==chunk["sha256"],
                "HOSTED_SHARD_DOWNLOAD_ARCHIVE_SHA_DRIFT")
        with zipfile.ZipFile(path) as z:
            expected={"objects/"+x for x in chunk["object_ids"]}
            require(set(z.namelist())==expected and len(expected)==chunk["object_count"],
                    "HOSTED_SHARD_ARCHIVE_SET_DRIFT")
            for object_id in chunk["object_ids"]:
                require(object_id in by_object and object_id not in encountered,
                        "HOSTED_SHARD_UNEXPECTED_OR_DUPLICATE_OBJECT")
                encountered.add(object_id)
                dest=object_cache/object_id
                if not dest.is_file() or digest(dest)!=object_id:
                    _object_from_shard(z,"objects/"+object_id,dest,
                                       object_id,by_object[object_id])
    require(encountered==set(by_object),"HOSTED_SHARD_INCOMPLETE_CAS")
    root=Path(target_root)
    require(root.is_absolute() and root.resolve()==root,
            "HOSTED_SHARD_TARGET_ROOT_INVALID")
    for f in doc["files"]:
        rel=Path(f["relative_path"]);dest=root/rel
        require(dest.resolve().is_relative_to(root),
                "HOSTED_SHARD_DEST_SYMLINK_ESCAPE")
        src=object_cache/f["sha256"]
        if dest.exists():
            require(dest.is_file() and digest(dest)==f["sha256"],
                    "HOSTED_SHARD_EXISTING_TARGET_DRIFT")
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            try:
                os.link(src,dest)
            except OSError:
                tmp=dest.with_suffix(dest.suffix+".partial")
                shutil.copyfile(src,tmp)
                require(digest(tmp)==f["sha256"],"HOSTED_SHARD_CROSS_DEVICE_COPY_DRIFT")
                tmp.replace(dest)
    # Every original path must be reconstructed with identical SHA, enabling
    # unmodified V5 source rest and native action identity courts.
    require(all(digest(root/f["relative_path"])==f["sha256"]
                for f in doc["files"]),"HOSTED_SHARD_FINAL_SOURCE_DRIFT")
    return {"status":"EXACT_ABSOLUTE_V5_SOURCE_BYTES_REHYDRATED",
            "file_count":len(doc["files"]),
            "subject_count":370,"training_authorized":False}


def main():
    p=argparse.ArgumentParser()
    sp=p.add_subparsers(dest="mode",required=True)
    a=sp.add_parser("export")
    a.add_argument("--private-inventory",type=Path,required=True)
    a.add_argument("--output-dir",type=Path,required=True)
    a.add_argument("--max-shard-mib",type=int,default=768)
    b=sp.add_parser("hydrate")
    b.add_argument("--manifest",type=Path,required=True)
    b.add_argument("--expected-manifest-sha256",required=True)
    b.add_argument("--private-dir",type=Path,required=True)
    b.add_argument("--object-cache",type=Path,required=True)
    b.add_argument("--target-root",type=Path,default=ALLOWED_ROOT)
    args=p.parse_args()
    if args.mode=="export":
        result=export(args.private_inventory,args.output_dir,
                      args.max_shard_mib*1024**2)
    else:
        result=hydrate(args.manifest,args.private_dir,args.object_cache,
                       expected_manifest_sha=args.expected_manifest_sha256,
                       target_root=args.target_root)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
