from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    caa_compile_artifact_from_dict,
    caa_preregistration_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict
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
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _source_topology_appearance_components,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import _ctx


def summary(x):
    x=np.asarray(x,dtype=np.float64)
    return {
        "count":int(len(x)),
        "mean":float(np.mean(x)) if len(x) else 0.0,
        "p95":float(np.quantile(x,0.95)) if len(x) else 0.0,
        "p99":float(np.quantile(x,0.99)) if len(x) else 0.0,
        "max":float(np.max(x)) if len(x) else 0.0,
        "gt_0_20_fraction":float(np.mean(x>0.20)) if len(x) else 0.0,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
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

    variants={}
    for mode in ("ALL_SUPPORT_DONORS","ADJACENT_ONLY_DONORS","HARMONIC_ONLY"):
        all_error=[]
        per_view=[]
        total_donor=0
        total_harmonic=0
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

            if mode!="HARMONIC_ONLY":
                eligible=donor_valid
                if mode=="ADJACENT_ONLY_DONORS":
                    eligible=np.zeros_like(donor_valid)
                    eligible[(target-1)%8]=donor_valid[(target-1)%8]
                    eligible[(target+1)%8]=donor_valid[(target+1)%8]
                best_view,_=select_other_view_donor_by_support(
                    target_view_index=target,
                    missing=~has,
                    direct_valid=direct,
                    donor_valid=eligible,
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
            all_error.extend(map(float,err))
            hp=provenance[held]
            donor_count=int(np.count_nonzero(hp==CAA_PROVENANCE["OTHER_VIEW_SOURCE"]))
            harmonic_count=int(np.count_nonzero(hp==CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]))
            total_donor+=donor_count
            total_harmonic+=harmonic_count
            per_view.append({
                "view":target,
                "error":summary(err),
                "donor_count":donor_count,
                "harmonic_count":harmonic_count,
            })
        variants[mode]={
            "global":summary(np.asarray(all_error,dtype=np.float64)),
            "donor_count":total_donor,
            "harmonic_count":total_harmonic,
            "per_view":per_view,
        }

    print("STAGE24_DONOR_COUNTERFACTUAL="+json.dumps({
        "run_id":a.run_id,
        "policy_p95_limit":float(policy["max_structured_holdout_p95_rgba_l1"]),
        "variants":variants,
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
