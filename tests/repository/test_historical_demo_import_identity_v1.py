from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator import mainline


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch):
    repo_root=tmp_path/"repo"
    authority_root=tmp_path/"authority"
    run_id="DEMO_IMPORT_TEST"
    stage_id="01_SOURCE_BYTES_SEALED"
    run_root=authority_root/"runs"/run_id
    output_dir=run_root/"artifacts"/stage_id
    import_dir=run_root/"imports"/"historical_stage08"
    canonical_dir=repo_root/"canonical"
    output_dir.mkdir(parents=True)
    import_dir.mkdir(parents=True)
    canonical_dir.mkdir(parents=True)

    monkeypatch.setattr(mainline,"ROOT",repo_root)
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT",str(authority_root))

    archive=import_dir/"evidence.zip"
    archive.write_bytes(b"sealed historical archive")
    output=output_dir/"source.json"
    output.write_text('{"schema":"Example"}\n',encoding="utf-8")

    authority={
        "schema":"RealSaS.DemoHistoricalStageImportAuthority.v1",
        "status":"APPROVED_DEMO_ONLY",
        "subject_id":"SUBJECT",
        "target_run_id":run_id,
        "source_evidence_archive":{"sha256":_sha(archive)},
        "imported_stage_ids":[stage_id],
        "exact_outputs":{
            stage_id:{
                "member":"01_SOURCE_BYTES_SEALED/source.json",
                "schema":"Example.Schema.v1",
                "authority_class":"SEALED_SOURCE_EVIDENCE",
                "sha256":_sha(output),
                "bytes":output.stat().st_size,
            }
        },
        "scope":{
            "execution_class":"DEMO_WITNESS",
            "product_authority_claimed":False,
            "scientific_product_pass_forbidden":True,
            "exact_byte_import_required":True,
            "recomputation_forbidden":True,
        },
    }
    authority_path=canonical_dir/"IMPORT.json"
    authority_path.write_text(
        json.dumps(authority,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    stage={
        "id":stage_id,
        "manifest_keys":["source_files"],
        "policy":{"fail_closed":True,"output_hash_required":True},
    }
    manifest={"source_files":[{"role":"SOURCE","sha256":"abc"}]}
    ledger={
        "run_id":run_id,
        "subject_id":"SUBJECT",
        "execution_class":"DEMO_WITNESS",
        "pipeline_plan_sha256":"p"*64,
    }
    historical_import={
        "schema":"RealSaS.HistoricalStageImportIdentity.v1",
        "stage_id":stage_id,
        "authority_path":"canonical/IMPORT.json",
        "authority_sha256":_sha(authority_path),
        "archive_path":str(archive),
        "archive_sha256":_sha(archive),
        "source_member":"01_SOURCE_BYTES_SEALED/source.json",
        "source_output_sha256":_sha(output),
        "manifest_subset_hash":mainline.content_sha256(
            mainline._manifest_subset(manifest,stage)
        ),
        "policy_hash":mainline.content_sha256(stage["policy"]),
    }
    fingerprint,policy_hash=mainline._historical_import_fingerprint(
        ledger=ledger,
        manifest=manifest,
        stage=stage,
        historical_import=historical_import,
    )
    row={
        "id":stage_id,
        "status":"PASS_DEMO_ONLY",
        "input_fingerprint":fingerprint,
        "implementation_hash":mainline.content_sha256(historical_import),
        "policy_hash":policy_hash,
        "outputs":[{
            "path":str(output),
            "sha256":_sha(output),
            "bytes":output.stat().st_size,
            "authority_class":"SEALED_SOURCE_EVIDENCE",
            "schema":"Example.Schema.v1",
        }],
        "historical_import":historical_import,
    }
    return ledger,manifest,stage,row,archive,output


def test_historical_demo_import_survives_unrelated_adapter_changes(
    tmp_path,monkeypatch
):
    ledger,manifest,stage,row,_archive,_output=_fixture(
        tmp_path,monkeypatch
    )
    assert mainline._verify_historical_imported_pass(
        plan={"stages":[stage]},
        ledger=ledger,
        manifest=manifest,
        stage=stage,
        row=row,
    )


def test_historical_demo_import_fails_closed_on_archive_manifest_or_policy_drift(
    tmp_path,monkeypatch
):
    ledger,manifest,stage,row,archive,_output=_fixture(
        tmp_path,monkeypatch
    )

    archive.write_bytes(b"tampered")
    assert not mainline._verify_historical_imported_pass(
        plan={"stages":[stage]},
        ledger=ledger,
        manifest=manifest,
        stage=stage,
        row=row,
    )

    ledger,manifest,stage,row,_archive,_output=_fixture(
        tmp_path/"second",monkeypatch
    )
    manifest["source_files"][0]["sha256"]="changed"
    assert not mainline._verify_historical_imported_pass(
        plan={"stages":[stage]},
        ledger=ledger,
        manifest=manifest,
        stage=stage,
        row=row,
    )

    ledger,manifest,stage,row,_archive,_output=_fixture(
        tmp_path/"third",monkeypatch
    )
    stage["policy"]["new_rule"]="changed"
    assert not mainline._verify_historical_imported_pass(
        plan={"stages":[stage]},
        ledger=ledger,
        manifest=manifest,
        stage=stage,
        row=row,
    )
