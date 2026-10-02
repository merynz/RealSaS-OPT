from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

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

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def ledger_output_sha(ctx,stage_id,schema):
    stage=next((row for row in ctx["ledger"].get("stages") or () if str(row.get("id") or "")==stage_id),None)
    if stage is None:
        raise RuntimeError(f"LEDGER_STAGE_MISSING:{stage_id}")
    rows=[row for row in stage.get("outputs") or () if str(row.get("schema") or "")==schema]
    if len(rows)!=1:
        raise RuntimeError(f"LEDGER_OUTPUT_AMBIGUOUS:{stage_id}:{schema}:{len(rows)}")
    return str(rows[0].get("sha256") or "")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    rr=ctx["run_root"]
    candidate=canonical_mesh_candidate_from_dict(json.loads(a.candidate_json.read_text()))

    gpath=rr/"artifacts/13_GEOMETRY_SUBSTRATE_QUALIFIED/geometry_substrate_qualification.json"
    expected=ledger_output_sha(ctx,"13_GEOMETRY_SUBSTRATE_QUALIFIED","RealSaS.GeometrySubstrateQualificationIR.v2")
    actual=sha(gpath)
    if len(expected)!=64 or actual!=expected:
        raise RuntimeError(f"STAGE13_GEOMETRY_BYTES_DRIFT:{actual}:{expected}")
    geometry=geometry_substrate_evidence_from_dict(json.loads(gpath.read_text()))

    cpath=rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    opath=rr/"artifacts/07_OBSERVATION_CONTRACT_QUALIFIED/qualified_observation_set.json"
    cameras=qualified_camera_set_from_dict(json.loads(cpath.read_text()))
    observation=qualified_observation_set_from_dict(json.loads(opath.read_text()))
    source_foreground=_source_foreground_masks_v1(ctx,observation)

    passed,rows=_evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source_foreground,
    )
    summary=[{
      "view_index":int(x["view_index"]),
      "passed":bool(x["passed"]),
      "silhouette_recall":float(x["silhouette_recall"]),
      "silhouette_precision":float(x["silhouette_precision"]),
      "largest_coherent_hole_fraction":float(x["largest_coherent_hole_fraction"]),
      "interior_uncovered_fraction":float(x["interior_uncovered_fraction"]),
      "component_recall":float(x["component_recall"]),
      "silhouette_edge_mean_px":float(x["silhouette_edge_mean_px"]),
      "silhouette_edge_p95_px":float(x["silhouette_edge_p95_px"]),
      "silhouette_edge_max_px":float(x["silhouette_edge_max_px"]),
    } for x in rows]

    report={
      "schema":"RealSaS.KnightV9SourceFidelityCourt.v1",
      "status":"PASS" if passed else "FAIL_MEASURED",
      "candidate_lineage_hash":candidate.candidate_lineage_hash,
      "vertex_count":len(candidate.vertices),
      "face_count":len(candidate.faces),
      "stage13_sha256":actual,
      "stage13_scientific_every_view_passed":bool(
          (geometry.qualification_report or {}).get("every_view_passed")
      ),
      "policy":dict(geometry.policy or {}),
      "passed":bool(passed),
      "failed_view_count":sum(not bool(x["passed"]) for x in summary),
      "views":summary,
      "aggregate":{
        "min_recall":min(float(x["silhouette_recall"]) for x in summary),
        "min_precision":min(float(x["silhouette_precision"]) for x in summary),
        "max_largest_coherent_hole_fraction":max(float(x["largest_coherent_hole_fraction"]) for x in summary),
        "max_interior_uncovered_fraction":max(float(x["interior_uncovered_fraction"]) for x in summary),
        "min_component_recall":min(float(x["component_recall"]) for x in summary),
        "max_silhouette_edge_p95_px":max(float(x["silhouette_edge_p95_px"]) for x in summary),
        "max_silhouette_edge_max_px":max(float(x["silhouette_edge_max_px"]) for x in summary),
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_V9_SOURCE_FIDELITY="+json.dumps({
      "passed":report["passed"],
      "failed_views":report["failed_view_count"],
      **report["aggregate"],
    },sort_keys=True),flush=True)
    # Scientific failure is intentional evidence; preserve artifact but mark run
    # failed so the gate remains fail-closed.
    if not passed:
        raise SystemExit(2)

if __name__=="__main__":main()
