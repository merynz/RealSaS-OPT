from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    _surface_sample_geometry_adaptive,
    bilinear_rgba_u8,
    erode_binary_mask,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    surface_sample_neighbors,
)
from compiler.realsas_compiler_core.appearance_variational_completion_v1 import (
    solve_weighted_surface_dirichlet,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)


def _load_context(*, repo_root: Path, authority_root: Path, run_id: str):
    run_root=(authority_root/"runs"/run_id).resolve()
    ledger=json.loads((run_root/"ACTIVE_RUN_V2.json").read_text())
    manifest=json.loads((run_root/"run_manifest.json").read_text())
    return run_root,{
        "repo_root":repo_root.resolve(),
        "authority_root":authority_root.resolve(),
        "run_root":run_root,
        "run_id":run_id,
        "run_manifest_path":run_root/"run_manifest.json",
        "run_manifest":manifest,
        "stage":{"id":"CAA_VARIATIONAL_CHALLENGER_DIAGNOSTIC"},
        "ledger":ledger,
    }


def diagnose(*, repo_root: Path, authority_root: Path, run_id: str) -> dict:
    run_root,ctx=_load_context(
        repo_root=repo_root,
        authority_root=authority_root,
        run_id=run_id,
    )
    candidate=canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    cameras=qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation=qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    rgba_by_view,masks_by_view=_load_source_inputs(ctx,observation)
    policy=json.loads(
        (
            repo_root
            /"canonical"
            /"CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
        ).read_text()
    )
    source_policy=dict(policy["source_lock_policy"])
    erosion=int(source_policy["boundary_safe_erosion_px"])
    min_alpha=int(source_policy["min_source_alpha_u8"])
    min_cos=float(source_policy["min_abs_normal_camera_cos"])

    face_count=len(candidate.faces)
    resolutions=np.full(face_count,4,dtype=np.int32)
    (
        positions,
        sample_face,
        sample_component,
        component_ids,
        face_normals,
        offsets,
    )=_surface_sample_geometry_adaptive(candidate,resolutions)
    sample_count=len(positions)
    graph=surface_sample_neighbors(
        positions=positions,
        face_count=face_count,
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=tuple(
            tuple(map(str,face)) for face in candidate.faces
        ),
    )

    foreground_valid=np.zeros((8,sample_count),dtype=bool)
    foreground_rgba=np.zeros((8,sample_count,4),dtype=np.uint8)
    face_support=np.zeros((8,face_count),dtype=np.float64)
    camera_by_view={int(c.view_index):c for c in cameras.cameras}
    assert set(camera_by_view)==set(range(8))

    for view in range(8):
        camera=camera_by_view[view]
        image=np.asarray(rgba_by_view[view],dtype=np.uint8)
        mask=np.asarray(masks_by_view[view],dtype=bool)
        safe_fg=erode_binary_mask(mask,erosion)
        visibility=rasterize_visible_owner(
            candidate,
            camera,
            width=image.shape[1],
            height=image.shape[0],
        )
        projected=np.asarray(
            project_points_xyz_v3(positions,camera),
            dtype=np.float64,
        )
        xy=projected[:,:2]-0.5
        ix=np.rint(xy[:,0]).astype(np.int64)
        iy=np.rint(xy[:,1]).astype(np.int64)
        in_bounds=(
            (ix>=0)&(ix<image.shape[1])
            &(iy>=0)&(iy<image.shape[0])
            &np.isfinite(projected[:,2])
            &(projected[:,2]>0.0)
        )
        fg_safe=np.zeros(sample_count,dtype=bool)
        visible=np.zeros(sample_count,dtype=bool)
        alpha_safe=np.zeros(sample_count,dtype=bool)
        idx=np.flatnonzero(in_bounds)
        if len(idx):
            x=ix[idx]; y=iy[idx]
            fg_safe[idx]=safe_fg[y,x]
            visible[idx]=(
                visibility.owner_face_index[y,x]
                == sample_face[idx]
            )
            alpha_safe[idx]=image[y,x,3]>=min_alpha

        forward=np.asarray(camera.forward,dtype=np.float64)
        forward=forward/np.linalg.norm(forward)
        cos=np.abs(face_normals@forward)
        face_support[view]=cos
        valid=(
            in_bounds
            &visible
            &fg_safe
            &alpha_safe
            &(cos[sample_face]>=min_cos)
        )
        foreground_valid[view]=valid
        if np.any(valid):
            sampled=bilinear_rgba_u8(image,xy[valid])
            foreground_rgba[view,valid]=sampled

    best_view=np.full(sample_count,-1,dtype=np.int16)
    best_score=np.full(sample_count,-1.0,dtype=np.float64)
    for view in range(8):
        score=face_support[view,sample_face]
        improve=foreground_valid[view]&(score>best_score+1.0e-12)
        best_view[improve]=view
        best_score[improve]=score[improve]

    known=best_view>=0
    canonical_rgba=np.zeros((sample_count,4),dtype=np.uint8)
    known_idx=np.flatnonzero(known)
    donors=best_view[known_idx].astype(np.int64)
    canonical_rgba[known_idx]=foreground_rgba[donors,known_idx]
    pm=np.zeros((sample_count,4),dtype=np.float64)
    if len(known_idx):
        pm[known_idx]=straight_srgb_rgba_u8_to_premultiplied_linear(
            canonical_rgba[known_idx]
        )

    solved,stats=solve_weighted_surface_dirichlet(
        values=pm,
        known_mask=known,
        sample_component=np.asarray(sample_component,dtype=np.int32),
        positions=np.asarray(positions,dtype=np.float64),
        graph=graph,
        rtol=1.0e-9,
        atol=1.0e-11,
        maxiter=2048,
    )
    solved=np.clip(solved,0.0,1.0)
    solved[:,:3]=np.minimum(solved[:,:3],solved[:,3:4])
    encoded=premultiplied_linear_to_straight_srgb_u8(solved)
    if not np.array_equal(encoded[known_idx],canonical_rgba[known_idx]):
        # Transport quantization may differ by one unit after a PM roundtrip;
        # source authority itself remains the exact pre-solve PM array.
        max_source_transport_delta=int(
            np.max(
                np.abs(
                    encoded[known_idx].astype(np.int16)
                    -canonical_rgba[known_idx].astype(np.int16)
                ),
                initial=0,
            )
        )
    else:
        max_source_transport_delta=0

    unknown=~known
    unknown_alpha=encoded[unknown,3].astype(np.float64)
    known_alpha=canonical_rgba[known,3].astype(np.float64)
    component=np.asarray(sample_component,dtype=np.int32)
    component_rows=[]
    for cid,name in enumerate(component_ids):
        take=component==cid
        component_rows.append({
            "component_index":int(cid),
            "component_id":str(name),
            "sample_count":int(np.count_nonzero(take)),
            "known_foreground_anchor_count":int(
                np.count_nonzero(take&known)
            ),
            "completed_sample_count":int(
                np.count_nonzero(take&unknown)
            ),
        })

    report={
        "schema":"RealSaS.CAAVariationalCompletionChallenger.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "control_lattice_resolution":4,
        "face_count":int(face_count),
        "sample_count":int(sample_count),
        "known_foreground_anchor_count":int(np.count_nonzero(known)),
        "completed_sample_count":int(np.count_nonzero(unknown)),
        "completed_fraction":float(np.mean(unknown)),
        "component_count":len(component_ids),
        "components":component_rows,
        "solver":stats.to_dict(),
        "completed_alpha_u8":{
            "minimum":int(np.min(unknown_alpha,initial=255)),
            "p01":float(np.quantile(unknown_alpha,0.01)) if len(unknown_alpha) else 0.0,
            "p05":float(np.quantile(unknown_alpha,0.05)) if len(unknown_alpha) else 0.0,
            "median":float(np.median(unknown_alpha)) if len(unknown_alpha) else 0.0,
            "p95":float(np.quantile(unknown_alpha,0.95)) if len(unknown_alpha) else 0.0,
            "maximum":int(np.max(unknown_alpha,initial=0)),
            "zero_alpha_fraction":float(np.mean(unknown_alpha==0)) if len(unknown_alpha) else 0.0,
        },
        "source_anchor_alpha_u8":{
            "minimum":int(np.min(known_alpha,initial=255)),
            "median":float(np.median(known_alpha)) if len(known_alpha) else 0.0,
            "maximum":int(np.max(known_alpha,initial=0)),
        },
        "source_constraints":{
            "premultiplied_linear_exact":bool(
                np.array_equal(solved[known],pm[known])
            ),
            "maximum_roundtrip_rgba8_delta":max_source_transport_delta,
        },
        "authority":{
            "diagnostic_only":True,
            "shipping_policy_changed":False,
            "product_authority_claimed":False,
            "source_only_constraints":True,
            "cross_component_completion_forbidden":True,
            "learned_pixel_generation_used":False,
        },
    }
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",default=".")
    parser.add_argument("--authority-root",required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    report=diagnose(
        repo_root=Path(args.repo_root).resolve(),
        authority_root=Path(args.authority_root).resolve(),
        run_id=args.run_id,
    )
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("CAA_VARIATIONAL_CHALLENGER",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
