from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    caa_compile_artifact_from_dict,
    caa_preregistration_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
)
from compiler.realsas_compiler_core.appearance_bake_v2 import (
    straight_rgba_to_premultiplied_float,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    bounded_surface_harmonic_fill,
    surface_sample_neighbors,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    select_other_view_donor_by_support,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    _bounded_edge_holdout_mask,
    _structured_band_mask,
    rgba_l1_premultiplied,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _source_topology_appearance_components,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import _ctx


def summarize(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0, "mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "gt_0_20_fraction": 0.0}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "gt_0_20_fraction": float(np.mean(x > 0.20)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root.resolve(),a.run_id)
    prereg=caa_preregistration_from_dict(stage_output_payload(
        ctx,"20_CAA_BACKEND_PREREGISTERED","RealSaS.CAACompilePreregistrationIR.v2"))
    artifact=caa_compile_artifact_from_dict(stage_output_payload(
        ctx,"21_CAA_COMPILE","RealSaS.CAACompileArtifactIR.v2"))
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))

    arrays=_load_compile_arrays(artifact,required_names={
        "sample_positions","sample_face_index","sample_component_index",
        "face_support_by_view","direct_valid","direct_foreground_donor_valid",
        "direct_rgba","source_xy",
    })
    direct=np.asarray(arrays["direct_valid"],dtype=bool)
    donor_valid=np.asarray(arrays["direct_foreground_donor_valid"],dtype=bool)
    direct_rgba=np.asarray(arrays["direct_rgba"],dtype=np.uint8)
    source_xy=np.asarray(arrays["source_xy"],dtype=np.float32)
    positions=np.asarray(arrays["sample_positions"],dtype=np.float64)
    component=np.asarray(arrays["sample_component_index"],dtype=np.int32)
    face_index=np.asarray(arrays["sample_face_index"],dtype=np.int32)
    face_support=np.asarray(arrays["face_support_by_view"],dtype=np.float64)
    offsets=np.asarray(arrays["face_sample_offsets"],dtype=np.int64)
    resolutions=np.asarray(arrays["face_tile_resolutions"],dtype=np.int32)

    _,_,face_vertex_ids=_source_topology_appearance_components(ctx,candidate)
    graph=surface_sample_neighbors(
        positions=positions,
        face_count=artifact.face_count,
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=face_vertex_ids,
    )
    policy=dict(prereg.completion_quality_policy)

    rows=[]
    global_err=[]
    for target in range(8):
        band=_structured_band_mask(
            direct[target],source_xy[target],view_index=target,
            band_fraction=float(policy["holdout_band_fraction"]))
        holdout=_bounded_edge_holdout_mask(
            candidate_mask=band,
            neighbors=graph,
            max_region_samples=int(policy["max_local_harmonic_region_samples"]),
            max_graph_hops=int(policy["max_local_harmonic_graph_hops"]),
            source_observed_mask=direct[target],
        )
        held=np.flatnonzero(holdout)
        available=direct[target] & ~holdout
        predicted=np.zeros((len(component),4),dtype=np.uint8)
        provenance=np.full(len(component),255,dtype=np.uint8)
        source_view=np.full(len(component),-1,dtype=np.int16)
        has=available.copy()
        predicted[available]=direct_rgba[target,available]
        provenance[available]=CAA_PROVENANCE["DIRECT_SOURCE"]
        source_view[available]=target

        best_view,best_score=select_other_view_donor_by_support(
            target_view_index=target,
            missing=~has,
            direct_valid=direct,
            donor_valid=donor_valid,
            sample_face_index=face_index,
            face_support_by_view=face_support,
        )
        take=(~has)&(best_view>=0)
        idx=np.flatnonzero(take)
        if len(idx):
            donors=best_view[idx].astype(np.int64)
            predicted[idx]=direct_rgba[donors,idx]
            provenance[idx]=CAA_PROVENANCE["OTHER_VIEW_SOURCE"]
            source_view[idx]=donors.astype(np.int16)
            has[idx]=True

        unresolved=holdout & ~has
        if np.any(unresolved):
            bounded_surface_harmonic_fill(
                rgba=predicted,
                provenance=provenance,
                source_view=source_view,
                missing=unresolved.copy(),
                observed_mask=has,
                sample_component=component,
                neighbors=graph,
                max_region_samples=int(policy["max_local_harmonic_region_samples"]),
                max_graph_hops=int(policy["max_local_harmonic_graph_hops"]),
            )

        err=rgba_l1_premultiplied(predicted[held],direct_rgba[target,held])
        global_err.extend(map(float,err))
        held_prov=provenance[held]
        held_sv=source_view[held]

        pm_pred=straight_rgba_to_premultiplied_float(predicted[held])
        pm_true=straight_rgba_to_premultiplied_float(direct_rgba[target,held])
        alpha_abs=np.abs(pm_pred[:,3]-pm_true[:,3])
        rgb_pm_l1=np.mean(np.abs(pm_pred[:,:3]-pm_true[:,:3]),axis=1)

        prov_rows={}
        for name,code in (
            ("OTHER_VIEW_SOURCE",CAA_PROVENANCE["OTHER_VIEW_SOURCE"]),
            ("COMPILED_LOCAL_HARMONIC",CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]),
        ):
            mask=held_prov==code
            prov_rows[name]={
                "error":summarize(err[mask]),
                "alpha_abs":summarize(alpha_abs[mask]),
                "rgb_pm_l1":summarize(rgb_pm_l1[mask]),
            }

        donor_rows=[]
        donor_mask=held_prov==CAA_PROVENANCE["OTHER_VIEW_SOURCE"]
        for donor in range(8):
            if donor==target: continue
            mask=donor_mask & (held_sv==donor)
            if not np.any(mask): continue
            donor_rows.append({
                "donor_view":donor,
                "circular_distance":min(abs(donor-target),8-abs(donor-target)),
                "error":summarize(err[mask]),
                "alpha_abs":summarize(alpha_abs[mask]),
                "rgb_pm_l1":summarize(rgb_pm_l1[mask]),
                "mean_support":float(np.mean(best_score[held][mask])),
            })

        bad=err>0.20
        face_rows=[]
        if np.any(bad):
            bad_faces=face_index[held][bad]
            unique,counts=np.unique(bad_faces,return_counts=True)
            order=np.argsort(counts)[::-1][:12]
            for oi in order:
                fi=int(unique[oi])
                local=(face_index[held]==fi)
                face_rows.append({
                    "face_index":fi,
                    "bad_sample_count":int(counts[oi]),
                    "held_sample_count":int(np.count_nonzero(local)),
                    "p95_error":float(np.quantile(err[local],0.95)),
                    "component_index":int(component[held][np.flatnonzero(local)[0]]),
                    "dominant_source_view":int(np.bincount(
                        np.maximum(held_sv[local],0),minlength=8
                    ).argmax()),
                })

        rows.append({
            "target_view":target,
            "holdout_axis":"x" if target%2==0 else "y",
            "holdout_side":"low" if (target//2)%2==0 else "high",
            "held_count":int(len(held)),
            "overall_error":summarize(err),
            "alpha_abs":summarize(alpha_abs),
            "rgb_pm_l1":summarize(rgb_pm_l1),
            "provenance_breakdown":prov_rows,
            "donor_breakdown":donor_rows,
            "top_bad_faces":face_rows,
        })

    print("STAGE24_HOLDOUT_OWNER_AUDIT="+json.dumps({
        "run_id":a.run_id,
        "policy_p95_limit":float(policy["max_structured_holdout_p95_rgba_l1"]),
        "global_error":summarize(np.asarray(global_err,dtype=np.float64)),
        "views":rows,
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
