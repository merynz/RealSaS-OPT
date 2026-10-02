from __future__ import annotations
import argparse,json
from pathlib import Path

from tools.audit_knight_v9_source_fidelity_v1 import sha,ledger_output_sha
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.geometry_substrate_v2 import geometry_substrate_evidence_from_dict
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    _evaluate_candidate_source_fidelity_v1,
    _source_foreground_masks_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

def summary(rows):
    return {
      "min_recall":min(float(x["silhouette_recall"]) for x in rows),
      "min_precision":min(float(x["silhouette_precision"]) for x in rows),
      "max_hole":max(float(x["largest_coherent_hole_fraction"]) for x in rows),
      "max_interior_uncovered":max(float(x["interior_uncovered_fraction"]) for x in rows),
      "min_component_recall":min(float(x["component_recall"]) for x in rows),
      "max_edge_p95":max(float(x["silhouette_edge_p95_px"]) for x in rows),
      "max_edge_max":max(float(x["silhouette_edge_max_px"]) for x in rows),
      "failed_views":sum(not bool(x["passed"]) for x in rows),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--v9-candidate-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id); rr=ctx["run_root"]
    base=canonical_mesh_candidate_from_dict(json.loads(
        (rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json").read_text()
    ))
    v9=canonical_mesh_candidate_from_dict(json.loads(a.v9_candidate_json.read_text()))

    gpath=rr/"artifacts/13_GEOMETRY_SUBSTRATE_QUALIFIED/geometry_substrate_qualification.json"
    expected=ledger_output_sha(ctx,"13_GEOMETRY_SUBSTRATE_QUALIFIED","RealSaS.GeometrySubstrateQualificationIR.v2")
    actual=sha(gpath)
    if actual!=expected: raise RuntimeError("STAGE13_DRIFT")
    geometry=geometry_substrate_evidence_from_dict(json.loads(gpath.read_text()))
    cameras=qualified_camera_set_from_dict(json.loads(
        (rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json").read_text()
    ))
    obs=qualified_observation_set_from_dict(json.loads(
        (rr/"artifacts/07_OBSERVATION_CONTRACT_QUALIFIED/qualified_observation_set.json").read_text()
    ))
    masks=_source_foreground_masks_v1(ctx,obs)

    bp,brows=_evaluate_candidate_source_fidelity_v1(
        candidate=base,geometry=geometry,cameras=cameras,observation=obs,source_foreground=masks
    )
    vp,vrows=_evaluate_candidate_source_fidelity_v1(
        candidate=v9,geometry=geometry,cameras=cameras,observation=obs,source_foreground=masks
    )
    bs=summary(brows);vs=summary(vrows)
    deltas={
      k:float(vs[k])-float(bs[k])
      for k in ("min_recall","min_precision","max_hole","max_interior_uncovered",
                "min_component_recall","max_edge_p95","max_edge_max")
    }
    per_view=[]
    for br,vr in zip(brows,vrows):
        per_view.append({
          "view_index":int(br["view_index"]),
          "base_passed":bool(br["passed"]),
          "v9_passed":bool(vr["passed"]),
          "recall_delta":float(vr["silhouette_recall"])-float(br["silhouette_recall"]),
          "precision_delta":float(vr["silhouette_precision"])-float(br["silhouette_precision"]),
          "hole_delta":float(vr["largest_coherent_hole_fraction"])-float(br["largest_coherent_hole_fraction"]),
          "interior_uncovered_delta":float(vr["interior_uncovered_fraction"])-float(br["interior_uncovered_fraction"]),
          "component_recall_delta":float(vr["component_recall"])-float(br["component_recall"]),
          "edge_p95_delta":float(vr["silhouette_edge_p95_px"])-float(br["silhouette_edge_p95_px"]),
        })
    report={
      "schema":"RealSaS.KnightV9SourceFidelityAB.v1",
      "status":"MEASURED",
      "base_candidate_lineage_hash":base.candidate_lineage_hash,
      "v9_candidate_lineage_hash":v9.candidate_lineage_hash,
      "base_passed":bool(bp),"v9_passed":bool(vp),
      "base":bs,"v9":vs,"delta_v9_minus_base":deltas,
      "per_view":per_view,
      "finding":{
        "v9_is_material_source_fidelity_regression":bool(
          vs["min_recall"] < bs["min_recall"]-0.002
          or vs["min_precision"] < bs["min_precision"]-0.002
          or vs["max_edge_p95"] > bs["max_edge_p95"]+1.0
        ),
        "both_fail_same_stage13_policy":bool((not bp) and (not vp)),
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_V9_SOURCE_FIDELITY_AB="+json.dumps({
      "base_passed":bp,"v9_passed":vp,
      "base":bs,"v9":vs,"delta":deltas,
      **report["finding"],
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
