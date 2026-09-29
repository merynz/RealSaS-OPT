from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_holeless_partitioned_dense_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import DEFAULT_MAX_EDGE_RATIO
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix
from tools.audit_knight_probe_conditioned_region_court_v1 import (
    probe_edge_risk, build_probe_partition,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    build_safe_graph, harmonic_complete, motion_metrics,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(
        rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(load_json(
        rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    )).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(
        rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))

    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)
    sids,jids,Wsrc=skin_matrix(surface,skeleton,skin)
    risk=probe_edge_risk(surface,skeleton,cameras,envelope,sids,Wsrc)
    part,unsafe_edges,crossing_edges,closure=build_probe_partition(
        surface,risk,max_edge_ratio=DEFAULT_MAX_EDGE_RATIO)

    carrier=build_component_carrier_policy(
        partition=part,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("COMPONENT_HARMONIC_SEAM_ORACLE",),
            metadata={"automatic":False,"audit_only":True}) for c in part.components),
        metadata={"audit_only":True},
    )
    candidate=build_holeless_partitioned_dense_candidate(
        surface,part,carrier,
        producer_policy_hash=content_sha256({
            "audit":"COMPONENT_HARMONIC_SEAM_ORACLE",
            "partition_components":len(part.components),
        }),
        explicit_face_provenance=explicit,
    )
    validate_canonical_mesh_candidate(
        candidate,surface=surface,partition=part,carrier_policy=carrier)

    rest,Wowner,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64)
    Wowner=np.asarray(Wowner,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)

    unknown=np.asarray([
        str(v.support_binding.mode)=="SEAM_GEOMETRY_INTERPOLATION"
        for v in candidate.vertices
    ],dtype=bool)

    # Candidate faces are component-pure by construction. Build the Laplacian only
    # on candidate edges, so coincident seam copies in different components cannot
    # exchange mechanical information.
    edges=build_safe_graph(rest,faces,np.ones(len(faces),dtype=bool))
    comp_by_vi=[str(v.component_id) for v in candidate.vertices]
    cross_component_graph_edges=sum(
        comp_by_vi[a]!=comp_by_vi[b] for a,b in edges
    )
    if cross_component_graph_edges:
        raise RuntimeError(
            f"COMPONENT_HARMONIC_GRAPH_LEAK::{cross_component_graph_edges}")

    Wharm,solve=harmonic_complete(Wowner,unknown,edges)
    if solve["unresolved_vertex_count"]!=0:
        raise RuntimeError(
            f"COMPONENT_HARMONIC_UNRESOLVED::{solve['unresolved_vertex_count']}")

    anchor_delta=np.abs(Wharm[~unknown]-Wowner[~unknown])
    if anchor_delta.size and float(np.max(anchor_delta))>1e-12:
        raise RuntimeError("COMPONENT_HARMONIC_MUTATED_SOURCE_ANCHOR")
    if np.any(Wharm<-1e-12) or not np.allclose(
        Wharm.sum(axis=1),1.0,atol=1e-8,rtol=0.0):
        raise RuntimeError("COMPONENT_HARMONIC_SIMPLEX_INVALID")

    source_report=load_json(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    owner_motion=motion_metrics(
        rest,Wowner,faces,jids,skeleton,cameras,rr,source_report)
    harm_motion=motion_metrics(
        rest,Wharm,faces,jids,skeleton,cameras,rr,source_report)

    d=np.sum(np.abs(Wharm-Wowner),axis=1)
    generated=d[unknown]
    report={
        "schema":"RealSaS.KnightComponentHarmonicSeamOracle.v1",
        "status":"COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
        "partition":{
            "component_count":len(part.components),
            "unsafe_source_edge_seed_count":len(unsafe_edges),
            "cut_closed_crossing_edge_count":len(crossing_edges),
            "closure":closure,
        },
        "candidate":{
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "generated_vertex_count":int(np.count_nonzero(unknown)),
            "identity_anchor_count":int(np.count_nonzero(~unknown)),
            "cross_component_graph_edge_count":int(cross_component_graph_edges),
        },
        "harmonic_solve":solve,
        "anchor_max_abs_delta":(
            float(np.max(anchor_delta)) if anchor_delta.size else 0.0),
        "generated_weight_l1_delta":{
            "min":float(np.min(generated)),
            "p50":float(np.quantile(generated,.50)),
            "p90":float(np.quantile(generated,.90)),
            "p95":float(np.quantile(generated,.95)),
            "p99":float(np.quantile(generated,.99)),
            "max":float(np.max(generated)),
        },
        "owner_copy_actual_motion":owner_motion,
        "component_harmonic_actual_motion":harm_motion,
        "delta":{
            "max_edge_gt10":int(
                harm_motion["max_edge_gt_10"]-owner_motion["max_edge_gt_10"]),
            "max_edge_gt4":int(
                harm_motion["max_edge_gt_4"]-owner_motion["max_edge_gt_4"]),
            "worst_edge_max":float(
                harm_motion["worst_edge_max"]-owner_motion["worst_edge_max"]),
            "max_edge_p99":float(
                harm_motion["max_edge_p99"]-owner_motion["max_edge_p99"]),
        },
        "face_provenance_replay":prov,
        "claim_boundary":[
            "QualifiedSkinIR source rows are immutable anchors.",
            "Only generated seam/centroid candidate-vertex mechanical weights are solved.",
            "The solve graph contains zero cross-component edges.",
            "Geometry, faces, Stage17 partition, skeleton and actual clips are identical between arms.",
            "This is a causal audit oracle; product promotion requires support-coefficient provenance, not raw-weight invention."
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("COMPONENT_HARMONIC_SEAM_ORACLE="+json.dumps({
        "components":len(part.components),
        "generated":int(np.count_nonzero(unknown)),
        "solve":solve,
        "owner_gt10":owner_motion["max_edge_gt_10"],
        "owner_gt4":owner_motion["max_edge_gt_4"],
        "owner_worst":owner_motion["worst_edge_max"],
        "harm_gt10":harm_motion["max_edge_gt_10"],
        "harm_gt4":harm_motion["max_edge_gt_4"],
        "harm_worst":harm_motion["worst_edge_max"],
        "harm_p99":harm_motion["max_edge_p99"],
    },sort_keys=True))


if __name__=="__main__":
    main()
