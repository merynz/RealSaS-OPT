from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    caa_compile_artifact_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _load_source_inputs,
)


def _ctx(repo_root: Path, authority_root: Path, run_id: str) -> dict:
    run_root=(authority_root/"runs"/run_id).resolve()
    return {
        "repo_root":repo_root.resolve(),
        "authority_root":authority_root.resolve(),
        "run_root":run_root,
        "run_id":run_id,
        "run_manifest_path":run_root/"run_manifest.json",
        "run_manifest":json.loads((run_root/"run_manifest.json").read_text()),
        "stage":{"id":"CAA_ABSTENTION_OWNER_DIAGNOSTIC"},
        "ledger":json.loads((run_root/"ACTIVE_RUN_V2.json").read_text()),
    }


def diagnose(*,repo_root:Path,authority_root:Path,run_id:str)->dict:
    ctx=_ctx(repo_root,authority_root,run_id)
    artifact=caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,"21_CAA_COMPILE","RealSaS.CAACompileArtifactIR.v2"
        )
    )
    arrays=_load_compile_arrays(artifact)
    observation=qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    _rgba,masks=_load_source_inputs(ctx,observation)

    direct=np.asarray(arrays["direct_valid"],dtype=bool)
    source_xy=np.asarray(arrays["source_xy"],dtype=np.float64)
    provenance=np.asarray(arrays["provenance"],dtype=np.uint8)
    n=direct.shape[1]
    source_class=np.full(direct.shape,-1,dtype=np.int8)
    for view in range(8):
        ids=np.flatnonzero(direct[view])
        if not len(ids):
            continue
        mask=np.asarray(masks[view],dtype=bool)
        xy=source_xy[view,ids]
        ix=np.rint(xy[:,0]).astype(np.int64)
        iy=np.rint(xy[:,1]).astype(np.int64)
        if np.any(
            (ix<0)|(ix>=mask.shape[1])|(iy<0)|(iy>=mask.shape[0])
        ):
            raise RuntimeError("CAA_ABSTENTION_SOURCE_CLASS_COORDINATE_DRIFT")
        source_class[view,ids]=mask[iy,ix].astype(np.int8)

    any_direct=np.any(direct,axis=0)
    any_fg=np.any(direct & (source_class==1),axis=0)
    any_bg=np.any(direct & (source_class==0),axis=0)
    global_unseen=~any_direct

    per_view=[]
    totals={"unsupported":0,"global_unseen":0,"background_only":0,"foreground_seen":0}
    for target in range(8):
        unsupported=provenance[target]==CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"]
        count=int(np.count_nonzero(unsupported))
        gu=int(np.count_nonzero(unsupported & global_unseen))
        bg=int(np.count_nonzero(unsupported & ~global_unseen & any_bg & ~any_fg))
        fg=int(np.count_nonzero(unsupported & any_fg))
        if gu+bg+fg!=count:
            raise RuntimeError(
                f"CAA_ABSTENTION_OWNER_ACCOUNTING_DRIFT:{target}:{count}:{gu}:{bg}:{fg}"
            )
        totals["unsupported"]+=count
        totals["global_unseen"]+=gu
        totals["background_only"]+=bg
        totals["foreground_seen"]+=fg
        per_view.append({
            "view_index":target,
            "unsupported_sample_count":count,
            "global_all_view_unseen_count":gu,
            "background_only_elsewhere_count":bg,
            "foreground_seen_elsewhere_count":fg,
            "global_all_view_unseen_fraction_of_unsupported":(
                float(gu)/count if count else 0.0
            ),
            "background_only_fraction_of_unsupported":(
                float(bg)/count if count else 0.0
            ),
            "foreground_seen_fraction_of_unsupported":(
                float(fg)/count if count else 0.0
            ),
        })

    total=totals["unsupported"]
    return {
        "schema":"RealSaS.CAAAbstentionOwnerDiagnostic.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "sample_count_per_direction":n,
        "globally_unseen_canonical_sample_count":int(np.count_nonzero(global_unseen)),
        "globally_unseen_canonical_sample_fraction":float(np.mean(global_unseen)),
        "canonical_sample_with_any_foreground_source_count":int(np.count_nonzero(any_fg)),
        "canonical_sample_with_background_only_source_count":int(np.count_nonzero(any_bg & ~any_fg)),
        "unsupported_directional_sample_count":total,
        "unsupported_global_all_view_unseen_count":totals["global_unseen"],
        "unsupported_background_only_elsewhere_count":totals["background_only"],
        "unsupported_foreground_seen_elsewhere_count":totals["foreground_seen"],
        "unsupported_global_all_view_unseen_fraction":(
            float(totals["global_unseen"])/total if total else 0.0
        ),
        "unsupported_background_only_elsewhere_fraction":(
            float(totals["background_only"])/total if total else 0.0
        ),
        "unsupported_foreground_seen_elsewhere_fraction":(
            float(totals["foreground_seen"])/total if total else 0.0
        ),
        "per_view":per_view,
        "owner_contract":{
            "global_all_view_unseen":"CAA_CANONICAL_SURFACE_COMPLETION_C_P",
            "background_only_elsewhere":"VISIBILITY_SILHOUETTE_OR_GEOMETRY_OWNER__DO_NOT_PAINT_WITH_COMPLETION",
            "foreground_seen_elsewhere":"DONOR_SELECTION_OR_LOCAL_COMPLETION_OWNER",
        },
        "thresholds_changed":False,
        "diagnostic_only":True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--authority-root",required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    report=diagnose(
        repo_root=Path(a.repo_root),
        authority_root=Path(a.authority_root),
        run_id=a.run_id,
    )
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("CAA_ABSTENTION_OWNER_DIAGNOSTIC",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
