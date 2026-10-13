from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.source import (
    seal_source_bytes,
    seal_source_license_provenance,
)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path:Path, *, shared:bool=True):
    tmp_path.mkdir(parents=True,exist_ok=True)
    files={
        "RAW_SOURCE":tmp_path/"subject.fbx",
        "LICENSE_BYTES":tmp_path/"License.txt",
        "SOURCE_PAGE_SNAPSHOT":tmp_path/"source_page.html",
        "CORPUS_SEAL":tmp_path/"corpus_seal.json",
    }
    files["RAW_SOURCE"].write_bytes(b"raw-source")
    files["LICENSE_BYTES"].write_bytes(b"CC0-1.0 exact license bytes")
    files["SOURCE_PAGE_SNAPSHOT"].write_bytes(b"<html>source page snapshot</html>")
    files["CORPUS_SEAL"].write_bytes(b'{"schema":"test-corpus-seal"}')
    source_files=[{"role":role,"path":str(path),"expected_sha256":sha(path),
                   "expected_size_bytes":path.stat().st_size} for role,path in files.items()]
    manifest={
        "subject_id":"fit8_fixture",
        "source_files":source_files,
        "source_license":{
            "profile":"EXACT_SOURCE_PROVENANCE_V2",
            "source_pack":"quaternius-test-pack",
            "license_name":"CC0-1.0",
            "license_ref":"https://creativecommons.org/publicdomain/zero/1.0/",
            "source_url_or_locator":"https://example.invalid/sealed-source-page",
            "primary_source_sha256":sha(files["RAW_SOURCE"]),
            "expected_source_page_sha256":sha(files["SOURCE_PAGE_SNAPSHOT"]),
            "license_bytes":{"path":str(files["LICENSE_BYTES"]),"sha256":sha(files["LICENSE_BYTES"])},
            "source_page_snapshot":{"path":str(files["SOURCE_PAGE_SNAPSHOT"]),"sha256":sha(files["SOURCE_PAGE_SNAPSHOT"])},
            "corpus_seal":{"path":str(files["CORPUS_SEAL"]),"sha256":sha(files["CORPUS_SEAL"])},
        },
        "iris_fit":{"training_profile":"SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"} if shared else {},
    }
    ctx={"run_id":"PROVENANCE_TEST","run_root":tmp_path/"run","run_manifest":manifest,
         "ledger":{"stages":[]},"stage":{"id":"01_SOURCE_BYTES_SEALED"}}
    return ctx,files


def install_stage1(ctx):
    result=seal_source_bytes(ctx)
    assert result["status"]=="PASS",result
    ctx["ledger"]["stages"].append({"id":"01_SOURCE_BYTES_SEALED","status":"PASS","outputs":result["outputs"]})
    ctx["stage"]={"id":"02_SOURCE_LICENSE_PROVENANCE"}
    return result


def test_shared_fit_requires_exact_source_provenance_profile(tmp_path):
    ctx,_=fixture(tmp_path)
    install_stage1(ctx)
    ctx["run_manifest"]["source_license"].pop("profile")
    result=seal_source_license_provenance(ctx)
    assert result["status"]=="BLOCKED"
    assert result["blockers"]==["SHARED_IRIS_EXACT_SOURCE_PROVENANCE_REQUIRED"]


def test_exact_source_license_page_and_corpus_bytes_are_stage1_bound(tmp_path):
    ctx,files=fixture(tmp_path)
    install_stage1(ctx)
    result=seal_source_license_provenance(ctx)
    assert result["status"]=="PASS",result
    assert result["diagnostics"]["exact_provenance_bound"] is True
    payload=Path(result["outputs"][0]["path"]).read_text()
    assert sha(files["RAW_SOURCE"]) in payload
    assert sha(files["LICENSE_BYTES"]) in payload
    assert sha(files["SOURCE_PAGE_SNAPSHOT"]) in payload
    assert sha(files["CORPUS_SEAL"]) in payload


def test_unsealed_or_drifted_provenance_fails_closed(tmp_path):
    ctx,files=fixture(tmp_path)
    install_stage1(ctx)
    ctx["run_manifest"]["source_license"]["source_page_snapshot"]={
        "path":str(files["SOURCE_PAGE_SNAPSHOT"]),"sha256":"f"*64}
    with pytest.raises(QualificationError,match="SOURCE_PROVENANCE_SOURCE_PAGE_SNAPSHOT_NOT_STAGE1_SEALED"):
        seal_source_license_provenance(ctx)

    ctx,_=fixture(tmp_path/"page")
    install_stage1(ctx)
    ctx["run_manifest"]["source_license"]["expected_source_page_sha256"]="e"*64
    with pytest.raises(QualificationError,match="SOURCE_PROVENANCE_SOURCE_PAGE_HASH_DRIFT"):
        seal_source_license_provenance(ctx)

    ctx,_=fixture(tmp_path/"license")
    install_stage1(ctx)
    ctx["run_manifest"]["source_license"]["license_name"]="UNKNOWN"
    with pytest.raises(QualificationError,match="SOURCE_PROVENANCE_LICENSE_POLICY_DRIFT"):
        seal_source_license_provenance(ctx)


def test_nonshared_legacy_provenance_remains_readable(tmp_path):
    ctx,_=fixture(tmp_path,shared=False)
    install_stage1(ctx)
    info=ctx["run_manifest"]["source_license"]
    for key in ("profile","primary_source_sha256","expected_source_page_sha256","license_bytes","source_page_snapshot","corpus_seal"):
        info.pop(key,None)
    result=seal_source_license_provenance(ctx)
    assert result["status"]=="PASS"
    assert result["diagnostics"]["exact_provenance_bound"] is False
