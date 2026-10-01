from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    resolved_path,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_output_payload(run_root: Path, ledger: dict, stage_id: str) -> tuple[Path, dict]:
    by_id={str(row["id"]):row for row in ledger["stages"]}
    row=by_id[stage_id]
    outputs=list(row.get("outputs") or ())
    if not outputs:
        raise RuntimeError(f"IRIS_BINDING_DIAG_STAGE_OUTPUT_MISSING:{stage_id}")
    # These stages each carry one primary typed output. Prefer exact schema names
    # so diagnostics remain stable if an auxiliary output is added later.
    expected={
        "07_OBSERVATION_CONTRACT_QUALIFIED":"RealSaS.QualifiedObservationSetIR.v1",
        "08_NORMALIZATION_DOMAIN_QUALIFIED":"RealSaS.NormalizationDomainIR.v1",
    }[stage_id]
    matches=[row for row in outputs if str(row.get("schema") or "")==expected]
    if len(matches)!=1:
        raise RuntimeError(
            f"IRIS_BINDING_DIAG_OUTPUT_CARDINALITY:{stage_id}:{len(matches)}"
        )
    path=Path(str(matches[0]["path"])).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"IRIS_BINDING_DIAG_OUTPUT_FILE_MISSING:{path}")
    return path,_load_json(path)


def diagnose(*, authority_root: Path, run_id: str) -> dict:
    run_root=(authority_root/"runs"/run_id).resolve()
    ledger=_load_json(run_root/"ACTIVE_RUN_V2.json")
    manifest=_load_json(run_root/"run_manifest.json")

    stage07_path,stage07=_stage_output_payload(
        run_root,ledger,"07_OBSERVATION_CONTRACT_QUALIFIED"
    )
    stage08_path,stage08=_stage_output_payload(
        run_root,ledger,"08_NORMALIZATION_DOMAIN_QUALIFIED"
    )

    demo=dict(
        (dict(manifest.get("iris_fit") or {}).get("demo_frozen_import") or {})
    )
    prereg_ref=dict(demo.get("research_preregistration") or {})
    closure_ref=dict(demo.get("research_closure") or {})
    if not prereg_ref or not closure_ref:
        raise RuntimeError("IRIS_BINDING_DIAG_FROZEN_IMPORT_REFS_MISSING")

    prereg=load_file_ref(
        prereg_ref,
        expected_schema="RealSaS.IRIS.V5TP64LongHorizonLowLRPreregistration.v1",
    )
    closure=load_file_ref(
        closure_ref,
        expected_schema="RealSaS.IRIS.V5TP64LongHorizonLowLRClosure.v1",
    )
    frozen=dict(prereg.get("frozen_inputs") or {})

    current_observation=str(stage07.get("observation_set_hash") or "")
    current_normalization=str(stage08.get("normalization_hash") or "")
    frozen_observation=str(frozen.get("observation_set_binding_hash") or "")
    frozen_normalization=str(frozen.get("normalization_binding_hash") or "")

    relevant_history=[]
    for event in ledger.get("history") or ():
        blob=json.dumps(event,sort_keys=True)
        if (
            "09_IRIS_FIT_PREREGISTERED" in blob
            or "STALE_PASS_IDENTITY" in blob
            or "DEPENDENCY_SUBGRAPH_INVALIDATED" in blob
        ):
            relevant_history.append(event)

    stage09=next(
        row for row in ledger["stages"]
        if str(row["id"])=="09_IRIS_FIT_PREREGISTERED"
    )

    return {
        "schema":"RealSaS.KnightIrisBindingDriftDiagnostic.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "stage07_output_path":str(stage07_path),
        "stage08_output_path":str(stage08_path),
        "research_preregistration_path":str(
            resolved_path(str(prereg_ref.get("path") or ""))
        ),
        "research_closure_path":str(
            resolved_path(str(closure_ref.get("path") or ""))
        ),
        "current_observation_set_hash":current_observation,
        "frozen_observation_set_binding_hash":frozen_observation,
        "observation_binding_matches":current_observation==frozen_observation,
        "current_normalization_hash":current_normalization,
        "frozen_normalization_binding_hash":frozen_normalization,
        "normalization_binding_matches":current_normalization==frozen_normalization,
        "frozen_stage08_evidence_sha256":str(
            frozen.get("stage08_evidence_sha256") or ""
        ),
        "historical_iris_run_id":str(closure.get("run_id") or ""),
        "historical_selected_checkpoint_sha256":str(
            ((closure.get("arms") or {}).get("C") or {}).get(
                "selected_checkpoint_sha256"
            ) or ""
        ),
        "current_stage09_status":str(stage09.get("status") or ""),
        "current_stage09_attempts":int(stage09.get("attempts") or 0),
        "current_stage09_input_fingerprint":str(
            stage09.get("input_fingerprint") or ""
        ),
        "current_stage09_implementation_hash":str(
            stage09.get("implementation_hash") or ""
        ),
        "relevant_ledger_history":relevant_history[-20:],
        "diagnostic_only":True,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--authority-root",required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    report=diagnose(
        authority_root=Path(args.authority_root).expanduser().resolve(),
        run_id=args.run_id,
    )
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("KNIGHT_IRIS_BINDING_DRIFT_DIAGNOSTIC",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
