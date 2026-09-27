from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.orchestrator.status_semantics import (
    DEMO_ONLY_STATUS,
)


OBSERVATION_ROLES = {
    "SOURCE_SCENE_AUDIT",
    "SOURCE_CAMERA_BUNDLE",
    "SOURCE_NORMALIZATION",
    "SOURCE_RENDER_RECEIPT",
}


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _is_observation_role(role: str) -> bool:
    return (
        role in OBSERVATION_ROLES
        or role.startswith("SOURCE_RASTER_V")
        or role.startswith("SOURCE_FOREGROUND_MASK_V")
    )


def _safe_extract(zf: zipfile.ZipFile, root: Path) -> None:
    root=root.resolve()
    for info in zf.infolist():
        target=(root/info.filename).resolve()
        if root not in target.parents and target!=root:
            raise RuntimeError(
                f"HISTORICAL_IMPORT_ARCHIVE_PATH_ESCAPE:{info.filename}"
            )
    zf.extractall(root)


def _remap_observation_manifest(
    historical: dict,
    current: dict,
    *,
    observation_dir: Path,
) -> dict:
    patched=copy.deepcopy(current)

    current_source_by_role={
        str(row.get("role") or ""):row
        for row in current.get("source_files") or ()
    }
    historical_source=list(historical.get("source_files") or ())

    # Preserve the current run's exact source FBX/texture/license paths while
    # requiring byte identity with the historical Stage08 evidence.
    for role in ("SOURCE_FBX","SOURCE_TEXTURE","SOURCE_LICENSE"):
        old=next(
            (row for row in historical_source if row.get("role")==role),
            None,
        )
        now=current_source_by_role.get(role)
        if not old or not now:
            raise RuntimeError(
                f"HISTORICAL_IMPORT_SOURCE_ROLE_MISSING:{role}"
            )
        if (
            str(old.get("expected_sha256") or "")
            != str(now.get("expected_sha256") or "")
            or int(old.get("expected_size_bytes",-1))
            != int(now.get("expected_size_bytes",-2))
        ):
            raise RuntimeError(
                f"HISTORICAL_IMPORT_SOURCE_ROLE_IDENTITY_DRIFT:{role}"
            )
        path=Path(str(now.get("path") or "")).expanduser().resolve()
        if (
            not path.is_file()
            or path.stat().st_size!=int(now["expected_size_bytes"])
            or _sha(path)!=str(now["expected_sha256"])
        ):
            raise RuntimeError(
                f"HISTORICAL_IMPORT_CURRENT_SOURCE_BYTES_INVALID:{role}"
            )

    retained=[
        copy.deepcopy(row)
        for row in current.get("source_files") or ()
        if not _is_observation_role(str(row.get("role") or ""))
    ]
    imported_obs_rows=[]
    for row in historical_source:
        role=str(row.get("role") or "")
        if not _is_observation_role(role):
            continue
        source_name=Path(str(row.get("path") or "")).name
        path=(observation_dir/source_name).resolve()
        expected_sha=str(row.get("expected_sha256") or "")
        expected_size=int(row.get("expected_size_bytes",-1))
        if (
            not path.is_file()
            or expected_size<0
            or path.stat().st_size!=expected_size
            or _sha(path)!=expected_sha
        ):
            raise RuntimeError(
                f"HISTORICAL_IMPORT_OBSERVATION_BYTES_INVALID:{role}"
            )
        imported=copy.deepcopy(row)
        imported["path"]=str(path)
        imported_obs_rows.append(imported)
    patched["source_files"]=retained+imported_obs_rows

    patched["source_audit"]=copy.deepcopy(
        historical.get("source_audit") or {}
    )
    patched["admission"]=copy.deepcopy(
        historical.get("admission") or {}
    )
    patched["normalization"]=copy.deepcopy(
        historical.get("normalization") or {}
    )

    old_obs=copy.deepcopy(historical.get("observation") or {})
    camera=dict(old_obs.get("camera_bundle") or {})
    camera_path=(observation_dir/"camera_bundle.json").resolve()
    if (
        not camera_path.is_file()
        or _sha(camera_path)!=str(camera.get("sha256") or "")
    ):
        raise RuntimeError(
            "HISTORICAL_IMPORT_CAMERA_BUNDLE_BYTES_INVALID"
        )
    camera["path"]=str(camera_path)
    old_obs["camera_bundle"]=camera

    rasters=[]
    for row in old_obs.get("source_rasters") or ():
        vi=int(row["view_index"])
        image=dict(row["image"])
        path=(observation_dir/f"V{vi}.png").resolve()
        if not path.is_file() or _sha(path)!=str(image.get("sha256") or ""):
            raise RuntimeError(
                f"HISTORICAL_IMPORT_RASTER_BYTES_INVALID:V{vi}"
            )
        image["path"]=str(path)
        rasters.append({"view_index":vi,"image":image})
    old_obs["source_rasters"]=rasters

    masks=[]
    for row in old_obs.get("source_foreground_masks") or ():
        vi=int(row["view_index"])
        mask=dict(row["mask"])
        path=(observation_dir/f"V{vi}.mask.bin").resolve()
        if not path.is_file() or _sha(path)!=str(mask.get("sha256") or ""):
            raise RuntimeError(
                f"HISTORICAL_IMPORT_MASK_BYTES_INVALID:V{vi}"
            )
        mask["path"]=str(path)
        masks.append({"view_index":vi,"mask":mask})
    old_obs["source_foreground_masks"]=masks
    patched["observation"]=old_obs
    return patched


def import_historical_stage08(
    *,
    repo_root: Path,
    authority_root: Path,
    run_id: str,
    archive: Path,
    import_authority_path: Path,
    receipt_path: Path,
) -> dict:
    repo_root=repo_root.resolve()
    authority_root=authority_root.resolve()
    archive=archive.resolve()
    import_authority_path=import_authority_path.resolve()
    if not archive.is_file():
        raise RuntimeError("HISTORICAL_IMPORT_ARCHIVE_MISSING")
    if not import_authority_path.is_file():
        raise RuntimeError("HISTORICAL_IMPORT_AUTHORITY_MISSING")

    authority=_load(import_authority_path)
    if (
        authority.get("schema")
        != "RealSaS.DemoHistoricalStageImportAuthority.v1"
        or authority.get("status")!="APPROVED_DEMO_ONLY"
        or authority.get("target_run_id")!=run_id
    ):
        raise RuntimeError("HISTORICAL_IMPORT_AUTHORITY_INVALID")
    archive_expected=str(
        (authority.get("source_evidence_archive") or {}).get("sha256")
        or ""
    )
    if len(archive_expected)!=64 or _sha(archive)!=archive_expected:
        raise RuntimeError("HISTORICAL_IMPORT_ARCHIVE_SHA_MISMATCH")

    run_root=(authority_root/"runs"/run_id).resolve()
    ledger_path=run_root/"ACTIVE_RUN_V2.json"
    manifest_path=run_root/"run_manifest.json"
    if not ledger_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("HISTORICAL_IMPORT_TARGET_RUN_MISSING")
    ledger=_load(ledger_path)
    manifest=_load(manifest_path)
    if (
        ledger.get("execution_class")!="DEMO_WITNESS"
        or ledger.get("subject_id")!=authority.get("subject_id")
        or manifest.get("run_id")!=run_id
        or manifest.get("subject_id")!=authority.get("subject_id")
    ):
        raise RuntimeError("HISTORICAL_IMPORT_TARGET_IDENTITY_DRIFT")

    plan=mainline.load_json(repo_root/"canonical"/"MAINLINE_EXECUTION_PLAN_V2.json")
    mainline.validate_plan(plan)
    plan_by={str(row["id"]):row for row in plan["stages"]}
    ledger_by={str(row["id"]):row for row in ledger["stages"]}

    persistent_import_dir=run_root/"imports"/"historical_stage08"
    persistent_import_dir.mkdir(parents=True,exist_ok=True)
    persistent_archive=persistent_import_dir/archive.name
    if archive!=persistent_archive:
        shutil.copy2(archive,persistent_archive)
    if _sha(persistent_archive)!=archive_expected:
        raise RuntimeError("HISTORICAL_IMPORT_PERSISTED_ARCHIVE_SHA_DRIFT")

    with tempfile.TemporaryDirectory(
        prefix="realsas_historical_stage08_"
    ) as tmp:
        extracted=Path(tmp)
        with zipfile.ZipFile(persistent_archive,"r") as zf:
            _safe_extract(zf,extracted)

        historical_manifest=_load(extracted/"run_manifest.json")
        historical_ledger=_load(extracted/"ACTIVE_RUN_V1.json")
        if (
            historical_manifest.get("run_id")!=authority.get("source_run_id")
            or historical_manifest.get("subject_id")!=authority.get("subject_id")
            or historical_ledger.get("run_id")!=authority.get("source_run_id")
        ):
            raise RuntimeError("HISTORICAL_IMPORT_SOURCE_IDENTITY_DRIFT")

        # Restore the exact observation byte domain first. Downstream consumers
        # must not combine historical Stage07 hashes with newly rendered PNGs.
        observation_dir=run_root/"inputs"/"observations"
        if observation_dir.exists():
            shutil.rmtree(observation_dir)
        shutil.copytree(extracted/"observations",observation_dir)

        patched_manifest=_remap_observation_manifest(
            historical_manifest,
            manifest,
            observation_dir=observation_dir,
        )
        _write(manifest_path,patched_manifest)
        manifest=patched_manifest

        authority_sha=_sha(import_authority_path)
        expected_outputs=dict(authority.get("exact_outputs") or {})
        imported_stage_ids=tuple(
            map(str,authority.get("imported_stage_ids") or ())
        )
        if imported_stage_ids!=tuple(
            f"{ordinal:02d}_{plan['stages'][ordinal-1]['id'].split('_',1)[1]}"
            for ordinal in range(1,9)
        ):
            # The authority must name exactly the first eight canonical stages,
            # in order. Do not permit a historical import to skip into later
            # product stages.
            raise RuntimeError(
                "HISTORICAL_IMPORT_STAGE_SCOPE_NOT_EXACT_01_08"
            )

        historical_by={
            str(row["id"]):row
            for row in historical_ledger.get("stages") or ()
        }
        imported_rows=[]
        for stage_id in imported_stage_ids:
            expected=dict(expected_outputs.get(stage_id) or {})
            member=str(expected.get("member") or "")
            source_path=(extracted/member).resolve()
            if not source_path.is_file():
                raise RuntimeError(
                    f"HISTORICAL_IMPORT_MEMBER_MISSING:{stage_id}:{member}"
                )
            if (
                _sha(source_path)!=str(expected.get("sha256") or "")
                or source_path.stat().st_size!=int(expected.get("bytes",-1))
            ):
                raise RuntimeError(
                    f"HISTORICAL_IMPORT_MEMBER_IDENTITY_DRIFT:{stage_id}"
                )

            artifact_dir=run_root/"artifacts"/stage_id
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            artifact_dir.mkdir(parents=True)
            output_path=artifact_dir/source_path.name
            shutil.copy2(source_path,output_path)

            stage=plan_by[stage_id]
            manifest_subset_hash=mainline.content_sha256(
                mainline._manifest_subset(manifest,stage)
            )
            policy_hash=mainline.content_sha256(stage["policy"])
            identity={
                "schema":"RealSaS.HistoricalStageImportIdentity.v1",
                "stage_id":stage_id,
                "authority_path":str(
                    import_authority_path.relative_to(repo_root).as_posix()
                ),
                "authority_sha256":authority_sha,
                "archive_path":str(persistent_archive),
                "archive_sha256":archive_expected,
                "source_run_id":str(authority["source_run_id"]),
                "source_commit":str(authority["source_commit"]),
                "source_member":member,
                "source_output_sha256":str(expected["sha256"]),
                "manifest_subset_hash":manifest_subset_hash,
                "policy_hash":policy_hash,
            }
            fingerprint,_=mainline._historical_import_fingerprint(
                ledger=ledger,
                manifest=manifest,
                stage=stage,
                historical_import=identity,
            )
            historical_row=historical_by.get(stage_id) or {}
            row=ledger_by[stage_id]
            row.update(
                status=DEMO_ONLY_STATUS,
                attempts=max(
                    int(row.get("attempts",0)),
                    int(historical_row.get("attempts",0)),
                ),
                input_fingerprint=fingerprint,
                implementation_hash=mainline.content_sha256(identity),
                policy_hash=policy_hash,
                outputs=[{
                    "path":str(output_path.resolve()),
                    "sha256":str(expected["sha256"]),
                    "bytes":int(expected["bytes"]),
                    "authority_class":str(expected["authority_class"]),
                    "schema":str(expected["schema"]),
                }],
                diagnostics_hash=str(
                    historical_row.get("diagnostics_hash") or ""
                ),
                blockers=[],
                wall_seconds=float(
                    historical_row.get("wall_seconds") or 0.0
                ),
                performance=dict(
                    historical_row.get("performance") or {}
                ),
                historical_import=identity,
            )
            imported_rows.append({
                "stage_id":stage_id,
                "output_sha256":str(expected["sha256"]),
                "output_path":str(output_path.resolve()),
            })

    # Any downstream state was derived from the drifted observation lineage.
    # Reset it rather than pretending it remains valid.
    for stage in plan["stages"][8:]:
        row=ledger_by[str(stage["id"])]
        row.pop("historical_import",None)
        row.update(
            status="PENDING",
            input_fingerprint="",
            implementation_hash="",
            policy_hash="",
            outputs=[],
            diagnostics_hash="",
            blockers=[],
            wall_seconds=0.0,
            performance={},
        )

    ledger.setdefault("history",[]).append({
        "event":"DEMO_EXACT_HISTORICAL_STAGE08_IMPORTED",
        "authority_path":str(
            import_authority_path.relative_to(repo_root).as_posix()
        ),
        "authority_sha256":_sha(import_authority_path),
        "archive_path":str(persistent_archive),
        "archive_sha256":archive_expected,
        "imported_stage_ids":list(imported_stage_ids),
        "downstream_reset_from_stage":"09_IRIS_FIT_PREREGISTERED",
        "product_authority_claimed":False,
    })
    mainline._refresh(plan,ledger)
    _write(ledger_path,ledger)

    # Re-open and verify using the same resume verifier that future runs use.
    verify_ledger=_load(ledger_path)
    verify_manifest=_load(manifest_path)
    if mainline._verify_existing_passes(
        plan,verify_ledger,verify_manifest
    ):
        raise RuntimeError(
            "HISTORICAL_IMPORT_NOT_STABLE_UNDER_RESUME_VERIFIER"
        )
    mainline.validate_ledger(plan,verify_ledger)

    stage07=_load(
        Path(ledger_by["07_OBSERVATION_CONTRACT_QUALIFIED"]["outputs"][0]["path"])
    )
    stage08=_load(
        Path(ledger_by["08_NORMALIZATION_DOMAIN_QUALIFIED"]["outputs"][0]["path"])
    )
    semantic=dict(authority.get("semantic_bindings") or {})
    if (
        stage07.get("observation_set_hash")
        != semantic.get("observation_set_hash")
        or stage08.get("normalization_hash")
        != semantic.get("normalization_hash")
        or stage08.get("camera_set_binding_hash")
        != semantic.get("camera_set_hash")
    ):
        raise RuntimeError("HISTORICAL_IMPORT_SEMANTIC_BINDING_DRIFT")

    receipt={
        "schema":"RealSaS.KnightDemoHistoricalStage08ImportReceipt.v1",
        "status":"PASS_DEMO_ONLY",
        "run_id":run_id,
        "subject_id":ledger["subject_id"],
        "archive_sha256":archive_expected,
        "authority_sha256":_sha(import_authority_path),
        "camera_set_hash":semantic.get("camera_set_hash"),
        "observation_set_hash":semantic.get("observation_set_hash"),
        "normalization_hash":semantic.get("normalization_hash"),
        "imported_stages":imported_rows,
        "downstream_reset_from_stage":"09_IRIS_FIT_PREREGISTERED",
        "resume_verifier_passed":True,
        "product_authority_claimed":False,
    }
    _write(receipt_path,receipt)
    return receipt


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",default=".")
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--archive",required=True)
    p.add_argument("--import-authority",required=True)
    p.add_argument("--receipt",required=True)
    a=p.parse_args()
    receipt=import_historical_stage08(
        repo_root=Path(a.repo_root),
        authority_root=Path(a.authority_root),
        run_id=a.run_id,
        archive=Path(a.archive),
        import_authority_path=Path(a.import_authority),
        receipt_path=Path(a.receipt),
    )
    print(
        "HISTORICAL_STAGE08_IMPORT_PASS",
        json.dumps(receipt,sort_keys=True),
    )


if __name__=="__main__":
    main()
