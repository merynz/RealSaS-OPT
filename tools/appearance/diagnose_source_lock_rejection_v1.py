from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    caa_compile_artifact_from_dict,
    caa_preregistration_from_dict,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    erode_binary_mask,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.visibility_v2 import (
    rasterize_visible_owner,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _load_source_inputs,
)


def _ctx(authority_root: Path, run_id: str) -> dict:
    run_root=(authority_root/"runs"/run_id).resolve()
    return {
        "repo_root":Path(".").resolve(),
        "authority_root":authority_root.resolve(),
        "run_root":run_root,
        "run_id":run_id,
        "run_manifest_path":run_root/"run_manifest.json",
        "run_manifest":json.loads((run_root/"run_manifest.json").read_text()),
        "ledger":json.loads((run_root/"ACTIVE_RUN_V2.json").read_text()),
        "stage":{"id":"CAA_SOURCE_LOCK_REJECTION_DIAGNOSTIC"},
    }


def _fraction(n:int,d:int)->float:
    return float(n)/float(max(1,d))


def diagnose(*,authority_root:Path,run_id:str)->dict:
    ctx=_ctx(authority_root,run_id)
    prereg=caa_preregistration_from_dict(
        stage_output_payload(
            ctx,"20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    artifact=caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,"21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    arrays=_load_compile_arrays(
        artifact,
        required_names={
            "sample_positions","sample_face_index","source_xy",
            "direct_valid","face_support_by_view",
        },
    )
    candidate=canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    cameras=qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,"05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation=qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,"07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    source_rgba,source_masks=_load_source_inputs(ctx,observation)

    positions=np.asarray(arrays["sample_positions"],dtype=np.float64)
    sample_face=np.asarray(arrays["sample_face_index"],dtype=np.int32)
    source_xy=np.asarray(arrays["source_xy"],dtype=np.float64)
    direct_valid=np.asarray(arrays["direct_valid"],dtype=bool)
    face_support=np.asarray(arrays["face_support_by_view"],dtype=np.float64)
    if positions.shape!=(len(sample_face),3):
        raise RuntimeError("CAA_SOURCE_LOCK_SAMPLE_POSITION_SHAPE")
    if source_xy.shape!=(8,len(sample_face),2):
        raise RuntimeError("CAA_SOURCE_LOCK_SOURCE_XY_SHAPE")
    if direct_valid.shape!=(8,len(sample_face)):
        raise RuntimeError("CAA_SOURCE_LOCK_DIRECT_VALID_SHAPE")

    min_cos=float(prereg.source_lock_policy["min_abs_normal_camera_cos"])
    erosion=int(prereg.source_lock_policy["boundary_safe_erosion_px"])
    min_alpha=int(prereg.source_lock_policy["min_source_alpha_u8"])
    by_view={int(c.view_index):c for c in cameras.cameras}
    rows=[]

    for view in range(8):
        camera=by_view[view]
        image=np.asarray(source_rgba[view],dtype=np.uint8)
        mask=np.asarray(source_masks[view],dtype=bool)
        safe_fg=erode_binary_mask(mask,erosion)
        safe_bg=erode_binary_mask(~mask,erosion)
        projected=np.asarray(
            project_points_xyz_v3(positions,camera),dtype=np.float64
        )
        # Stage21 computes direct validity from float64 projection and only
        # then stores source_xy as float32 for downstream qualification. Rebuild
        # the admission decision from the original float64 semantics; separately
        # measure any storage-precision index drift.
        xy=projected[:,:2]-0.5
        ix=np.rint(xy[:,0]).astype(np.int64)
        iy=np.rint(xy[:,1]).astype(np.int64)
        stored_xy=source_xy[view]
        stored_ix=np.rint(stored_xy[:,0]).astype(np.int64)
        stored_iy=np.rint(stored_xy[:,1]).astype(np.int64)
        source_xy_index_drift_count=int(
            np.count_nonzero(
                (stored_ix!=ix)|(stored_iy!=iy)
            )
        )
        in_bounds=(
            (ix>=0)&(ix<image.shape[1])
            &(iy>=0)&(iy<image.shape[0])
            &np.isfinite(projected[:,2])&(projected[:,2]>0.0)
        )

        fg_safe=np.zeros(len(sample_face),dtype=bool)
        bg_safe=np.zeros(len(sample_face),dtype=bool)
        alpha_fg=np.zeros(len(sample_face),dtype=bool)
        alpha_bg=np.zeros(len(sample_face),dtype=bool)
        source_pixel_fg=np.zeros(len(sample_face),dtype=bool)
        ids=np.flatnonzero(in_bounds)
        if len(ids):
            x=ix[ids]; y=iy[ids]
            fg_safe[ids]=safe_fg[y,x]
            bg_safe[ids]=safe_bg[y,x]
            alpha_fg[ids]=image[y,x,3]>=min_alpha
            alpha_bg[ids]=image[y,x,3]==0
            source_pixel_fg[ids]=mask[y,x]

        visibility=rasterize_visible_owner(
            candidate,camera,width=image.shape[1],height=image.shape[0],
        )
        first_hit=np.zeros(len(sample_face),dtype=bool)
        if len(ids):
            x=ix[ids];y=iy[ids]
            first_hit[ids]=(
                visibility.owner_face_index[y,x]==sample_face[ids]
            )
        angle_safe=face_support[view,sample_face]>=min_cos
        appearance_support=(
            (fg_safe&alpha_fg)|(bg_safe&alpha_bg)
        )
        reconstructed=(
            in_bounds&first_hit&angle_safe&appearance_support
        )
        if not np.array_equal(reconstructed,direct_valid[view]):
            mismatch=int(np.count_nonzero(reconstructed^direct_valid[view]))
            raise RuntimeError(
                f"CAA_SOURCE_LOCK_RECONSTRUCTION_DRIFT:V{view}:{mismatch}"
            )

        # Exclusive rejection funnel. Ordering is deliberately physical:
        # projection -> first-hit geometry -> source-safe mask -> alpha -> angle.
        remaining=~direct_valid[view]
        reason=np.full(len(sample_face),"DIRECT",dtype=object)
        take=remaining&~in_bounds
        reason[take]="OUT_OF_BOUNDS_OR_BEHIND"; remaining&=~take
        take=remaining&~first_hit
        reason[take]="NOT_FIRST_HIT"; remaining&=~take
        safe_region=fg_safe|bg_safe
        take=remaining&~safe_region
        reason[take]="BOUNDARY_EROSION_UNSAFE"; remaining&=~take
        alpha_supported=(fg_safe&alpha_fg)|(bg_safe&alpha_bg)
        take=remaining&~alpha_supported
        reason[take]="ALPHA_POLICY_UNSUPPORTED"; remaining&=~take
        take=remaining&~angle_safe
        reason[take]="GRAZING_ANGLE_UNSAFE"; remaining&=~take
        if np.any(remaining):
            reason[remaining]="UNCLASSIFIED"

        direct_count=int(np.count_nonzero(direct_valid[view]))
        visible_count=int(np.count_nonzero(in_bounds&first_hit))
        source_fg_first_hit=in_bounds&first_hit&source_pixel_fg
        source_bg_first_hit=in_bounds&first_hit&~source_pixel_fg
        rejected=~direct_valid[view]
        reasons={}
        for name in (
            "OUT_OF_BOUNDS_OR_BEHIND","NOT_FIRST_HIT",
            "BOUNDARY_EROSION_UNSAFE","ALPHA_POLICY_UNSUPPORTED",
            "GRAZING_ANGLE_UNSAFE","UNCLASSIFIED",
        ):
            m=reason==name
            reasons[name]={
                "sample_count":int(np.count_nonzero(m)),
                "fraction_of_all_rejected":_fraction(
                    int(np.count_nonzero(m)),
                    int(np.count_nonzero(rejected)),
                ),
                "source_foreground_first_hit_count":int(
                    np.count_nonzero(m&source_fg_first_hit)
                ),
                "source_background_first_hit_count":int(
                    np.count_nonzero(m&source_bg_first_hit)
                ),
            }

        rows.append({
            "view_index":view,
            "sample_count":len(sample_face),
            "source_xy_float32_index_drift_count":(
                source_xy_index_drift_count
            ),
            "direct_sample_count":direct_count,
            "direct_fraction_all_samples":_fraction(
                direct_count,len(sample_face)
            ),
            "first_hit_in_bounds_sample_count":visible_count,
            "direct_fraction_of_first_hit_samples":_fraction(
                direct_count,visible_count
            ),
            "source_foreground_first_hit_sample_count":int(
                np.count_nonzero(source_fg_first_hit)
            ),
            "direct_source_foreground_first_hit_sample_count":int(
                np.count_nonzero(
                    direct_valid[view]&source_fg_first_hit
                )
            ),
            "direct_fraction_of_source_foreground_first_hit":_fraction(
                int(np.count_nonzero(
                    direct_valid[view]&source_fg_first_hit
                )),
                int(np.count_nonzero(source_fg_first_hit)),
            ),
            "source_background_first_hit_sample_count":int(
                np.count_nonzero(source_bg_first_hit)
            ),
            "direct_source_background_first_hit_sample_count":int(
                np.count_nonzero(
                    direct_valid[view]&source_bg_first_hit
                )
            ),
            "direct_fraction_of_source_background_first_hit":_fraction(
                int(np.count_nonzero(
                    direct_valid[view]&source_bg_first_hit
                )),
                int(np.count_nonzero(source_bg_first_hit)),
            ),
            "rejection_reasons":reasons,
        })

    aggregate={}
    reason_names=(
        "OUT_OF_BOUNDS_OR_BEHIND","NOT_FIRST_HIT",
        "BOUNDARY_EROSION_UNSAFE","ALPHA_POLICY_UNSUPPORTED",
        "GRAZING_ANGLE_UNSAFE","UNCLASSIFIED",
    )
    for name in reason_names:
        aggregate[name]={
            "sample_count":sum(
                row["rejection_reasons"][name]["sample_count"]
                for row in rows
            ),
            "source_foreground_first_hit_count":sum(
                row["rejection_reasons"][name][
                    "source_foreground_first_hit_count"
                ]
                for row in rows
            ),
            "source_background_first_hit_count":sum(
                row["rejection_reasons"][name][
                    "source_background_first_hit_count"
                ]
                for row in rows
            ),
        }
    fg_total=sum(
        row["source_foreground_first_hit_sample_count"] for row in rows
    )
    fg_direct=sum(
        row["direct_source_foreground_first_hit_sample_count"]
        for row in rows
    )
    aggregate["source_foreground_first_hit_total"]=fg_total
    aggregate["source_foreground_first_hit_direct"]=fg_direct
    aggregate["source_foreground_first_hit_direct_fraction"]=_fraction(
        fg_direct,fg_total
    )
    return {
        "schema":"RealSaS.CAASourceLockRejectionDiagnostic.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "source_lock_policy":dict(prereg.source_lock_policy),
        "views":rows,
        "aggregate":aggregate,
        "interpretation_contract":{
            "diagnostic_only":True,
            "thresholds_changed":False,
            "direct_valid_reconstructed_exactly":True,
            "rejection_reason_order":[
                "OUT_OF_BOUNDS_OR_BEHIND","NOT_FIRST_HIT",
                "BOUNDARY_EROSION_UNSAFE","ALPHA_POLICY_UNSUPPORTED",
                "GRAZING_ANGLE_UNSAFE",
            ],
            "product_authority_minted":False,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    report=diagnose(
        authority_root=Path(a.authority_root),
        run_id=a.run_id,
    )
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(
        "CAA_SOURCE_LOCK_REJECTION_DIAGNOSTIC",
        json.dumps(report["aggregate"],sort_keys=True),
    )


if __name__=="__main__":
    main()
