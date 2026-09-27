from __future__ import annotations

import argparse
import json
from collections import deque
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
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    rgba_l1_premultiplied,
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
        "stage":{"id":"CAA_VARIATIONAL_CHALLENGER_V2_DIAGNOSTIC"},
        "ledger":ledger,
    }


def _deterministic_holdout_regions(
    *,
    known: np.ndarray,
    graph,
    component: np.ndarray,
    sizes: tuple[int,...]=(16,64,256,1024),
    repeats_per_size: int=2,
):
    known=np.asarray(known,dtype=bool)
    component=np.asarray(component,dtype=np.int32)
    candidates=np.flatnonzero(known).astype(np.int64)
    if len(candidates)==0:
        raise RuntimeError("CAA_VARIATIONAL_HOLDOUT_NO_SOURCE")
    used=np.zeros(len(known),dtype=bool)
    regions=[]
    for size_index,size in enumerate(sizes):
        for repeat in range(repeats_per_size):
            fraction=(size_index*repeats_per_size+repeat+1)/float(
                len(sizes)*repeats_per_size+1
            )
            base=int(round(fraction*(len(candidates)-1)))
            chosen=None
            for delta in range(len(candidates)):
                for sign in (1,-1):
                    cursor=base+sign*delta
                    if not (0<=cursor<len(candidates)):
                        continue
                    seed=int(candidates[cursor])
                    if used[seed]:
                        continue
                    cid=int(component[seed])
                    q=deque([seed])
                    local_seen={seed}
                    region=[]
                    while q and len(region)<int(size):
                        node=int(q.popleft())
                        if (
                            known[node]
                            and not used[node]
                            and int(component[node])==cid
                        ):
                            region.append(node)
                        if len(region)>=int(size):
                            break
                        for raw in graph[node]:
                            nxt=int(raw)
                            if (
                                nxt not in local_seen
                                and known[nxt]
                                and not used[nxt]
                                and int(component[nxt])==cid
                            ):
                                local_seen.add(nxt)
                                q.append(nxt)
                    if len(region)==int(size):
                        chosen=tuple(region)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(
                    f"CAA_VARIATIONAL_HOLDOUT_REGION_UNAVAILABLE:{size}:{repeat}"
                )
            idx=np.asarray(chosen,dtype=np.int64)
            used[idx]=True
            boundary=set()
            for node in chosen:
                for raw in graph[int(node)]:
                    nxt=int(raw)
                    if (
                        known[nxt]
                        and not used[nxt]
                        and int(component[nxt])==int(component[node])
                    ):
                        boundary.add(nxt)
            if not boundary:
                raise RuntimeError(
                    f"CAA_VARIATIONAL_HOLDOUT_WITHOUT_BOUNDARY:{size}:{repeat}"
                )
            regions.append({
                "requested_size":int(size),
                "repeat":int(repeat),
                "component_index":int(component[seed]),
                "indices":chosen,
                "boundary_count":int(len(boundary)),
            })
    return regions


def _metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    error=rgba_l1_premultiplied(predicted,truth)
    return {
        "sample_count":int(len(error)),
        "mean_rgba_l1":float(np.mean(error)) if len(error) else 0.0,
        "p50_rgba_l1":float(np.quantile(error,0.50)) if len(error) else 0.0,
        "p95_rgba_l1":float(np.quantile(error,0.95)) if len(error) else 0.0,
        "p99_rgba_l1":float(np.quantile(error,0.99)) if len(error) else 0.0,
        "max_rgba_l1":float(np.max(error)) if len(error) else 0.0,
    }


def diagnose(*, repo_root: Path, authority_root: Path, run_id: str) -> dict:
    _run_root,ctx=_load_context(
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
    quality_policy=dict(policy["completion_quality_policy"])
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
    component=np.asarray(sample_component,dtype=np.int32)
    graph=surface_sample_neighbors(
        positions=positions,
        face_count=face_count,
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=tuple(
            tuple(map(str,face)) for face in candidate.faces
        ),
    )

    direct_valid=np.zeros((8,sample_count),dtype=bool)
    direct_rgba=np.zeros((8,sample_count,4),dtype=np.uint8)
    face_support=np.zeros((8,face_count),dtype=np.float64)
    camera_by_view={int(c.view_index):c for c in cameras.cameras}
    assert set(camera_by_view)==set(range(8))

    for view in range(8):
        camera=camera_by_view[view]
        image=np.asarray(rgba_by_view[view],dtype=np.uint8)
        mask=np.asarray(masks_by_view[view],dtype=bool)
        safe_fg=erode_binary_mask(mask,erosion)
        safe_bg=erode_binary_mask(~mask,erosion)
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
        bg_safe=np.zeros(sample_count,dtype=bool)
        visible=np.zeros(sample_count,dtype=bool)
        alpha_fg=np.zeros(sample_count,dtype=bool)
        alpha_bg=np.zeros(sample_count,dtype=bool)
        idx=np.flatnonzero(in_bounds)
        if len(idx):
            x=ix[idx]; y=iy[idx]
            fg_safe[idx]=safe_fg[y,x]
            bg_safe[idx]=safe_bg[y,x]
            visible[idx]=(
                visibility.owner_face_index[y,x]
                == sample_face[idx]
            )
            alpha_fg[idx]=image[y,x,3]>=min_alpha
            alpha_bg[idx]=image[y,x,3]==0

        forward=np.asarray(camera.forward,dtype=np.float64)
        forward=forward/np.linalg.norm(forward)
        cos=np.abs(face_normals@forward)
        face_support[view]=cos
        appearance_support=(
            (fg_safe&alpha_fg)
            |(bg_safe&alpha_bg)
        )
        valid=(
            in_bounds
            &visible
            &appearance_support
            &(cos[sample_face]>=min_cos)
        )
        direct_valid[view]=valid
        if np.any(valid):
            direct_rgba[view,valid]=bilinear_rgba_u8(
                image,xy[valid]
            )

    best_view=np.full(sample_count,-1,dtype=np.int16)
    best_score=np.full(sample_count,-1.0,dtype=np.float64)
    for view in range(8):
        score=face_support[view,sample_face]
        improve=direct_valid[view]&(score>best_score+1.0e-12)
        best_view[improve]=view
        best_score[improve]=score[improve]

    known=best_view>=0
    canonical_rgba=np.zeros((sample_count,4),dtype=np.uint8)
    known_idx=np.flatnonzero(known)
    donors=best_view[known_idx].astype(np.int64)
    canonical_rgba[known_idx]=direct_rgba[donors,known_idx]
    pm=np.zeros((sample_count,4),dtype=np.float64)
    pm[known_idx]=straight_srgb_rgba_u8_to_premultiplied_linear(
        canonical_rgba[known_idx]
    )

    solved_full,stats_full=solve_weighted_surface_dirichlet(
        values=pm,
        known_mask=known,
        sample_component=component,
        positions=np.asarray(positions,dtype=np.float64),
        graph=graph,
    )
    solved_full=np.clip(solved_full,0.0,1.0)
    solved_full[:,:3]=np.minimum(
        solved_full[:,:3],solved_full[:,3:4]
    )
    encoded_full=premultiplied_linear_to_straight_srgb_u8(
        solved_full
    )

    regions=_deterministic_holdout_regions(
        known=known,
        graph=graph,
        component=component,
    )
    holdout=np.zeros(sample_count,dtype=bool)
    for row in regions:
        holdout[np.asarray(row["indices"],dtype=np.int64)]=True
    train_known=known&~holdout
    solved_holdout,stats_holdout=solve_weighted_surface_dirichlet(
        values=pm,
        known_mask=train_known,
        sample_component=component,
        positions=np.asarray(positions,dtype=np.float64),
        graph=graph,
    )
    solved_holdout=np.clip(solved_holdout,0.0,1.0)
    solved_holdout[:,:3]=np.minimum(
        solved_holdout[:,:3],solved_holdout[:,3:4]
    )
    encoded_holdout=premultiplied_linear_to_straight_srgb_u8(
        solved_holdout
    )

    held=np.flatnonzero(holdout)
    aggregate=_metrics(
        encoded_holdout[held],
        canonical_rgba[held],
    )
    per_scale=[]
    for size in sorted({row["requested_size"] for row in regions}):
        ids=np.concatenate([
            np.asarray(row["indices"],dtype=np.int64)
            for row in regions
            if row["requested_size"]==size
        ])
        row_metrics=_metrics(
            encoded_holdout[ids],
            canonical_rgba[ids],
        )
        row_metrics["requested_region_size"]=int(size)
        row_metrics["region_count"]=sum(
            row["requested_size"]==size for row in regions
        )
        per_scale.append(row_metrics)

    mean_limit=float(
        quality_policy["max_structured_holdout_mean_rgba_l1"]
    )
    p95_limit=float(
        quality_policy["max_structured_holdout_p95_rgba_l1"]
    )
    passes=(
        aggregate["mean_rgba_l1"]<=mean_limit
        and aggregate["p95_rgba_l1"]<=p95_limit
    )

    unknown=~known
    completed_alpha=encoded_full[unknown,3].astype(np.float64)
    source_alpha=canonical_rgba[known,3].astype(np.float64)
    return {
        "schema":"RealSaS.CAAVariationalCompletionChallenger.v2",
        "status":"MEASURED",
        "run_id":run_id,
        "control_lattice_resolution":4,
        "face_count":int(face_count),
        "sample_count":int(sample_count),
        "source_anchor_count":int(np.count_nonzero(known)),
        "source_anchor_fraction":float(np.mean(known)),
        "completed_sample_count":int(np.count_nonzero(unknown)),
        "completed_fraction":float(np.mean(unknown)),
        "component_count":len(component_ids),
        "full_completion_solver":stats_full.to_dict(),
        "holdout_solver":stats_holdout.to_dict(),
        "holdout":{
            "mode":"DETERMINISTIC_MULTI_SCALE_CONTIGUOUS_SOURCE_MASK_V1",
            "region_sizes":[16,64,256,1024],
            "regions_per_size":2,
            "region_count":len(regions),
            "regions":[
                {
                    k:v for k,v in row.items()
                    if k!="indices"
                }
                for row in regions
            ],
            "aggregate":aggregate,
            "per_scale":per_scale,
            "frozen_mean_limit":mean_limit,
            "frozen_p95_limit":p95_limit,
            "passes_frozen_structured_holdout_gate":bool(passes),
        },
        "completed_alpha_u8":{
            "minimum":int(np.min(completed_alpha,initial=255)),
            "p01":float(np.quantile(completed_alpha,0.01)) if len(completed_alpha) else 0.0,
            "p05":float(np.quantile(completed_alpha,0.05)) if len(completed_alpha) else 0.0,
            "median":float(np.median(completed_alpha)) if len(completed_alpha) else 0.0,
            "p95":float(np.quantile(completed_alpha,0.95)) if len(completed_alpha) else 0.0,
            "maximum":int(np.max(completed_alpha,initial=0)),
            "zero_alpha_fraction":float(np.mean(completed_alpha==0)) if len(completed_alpha) else 0.0,
        },
        "source_anchor_alpha_u8":{
            "minimum":int(np.min(source_alpha,initial=255)),
            "median":float(np.median(source_alpha)) if len(source_alpha) else 0.0,
            "maximum":int(np.max(source_alpha,initial=0)),
            "zero_alpha_fraction":float(np.mean(source_alpha==0)) if len(source_alpha) else 0.0,
        },
        "source_constraints":{
            "full_solve_premultiplied_linear_exact":bool(
                np.array_equal(solved_full[known],pm[known])
            ),
            "holdout_train_constraints_exact":bool(
                np.array_equal(
                    solved_holdout[train_known],
                    pm[train_known],
                )
            ),
        },
        "authority":{
            "diagnostic_only":True,
            "shipping_policy_changed":False,
            "product_authority_claimed":False,
            "source_foreground_and_safe_transparency_are_hard_constraints":True,
            "learned_pixel_generation_used":False,
            "knight_result_cannot_select_new_thresholds":True,
        },
    }


def main():
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
    print("CAA_VARIATIONAL_CHALLENGER_V2",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
