from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import qualified_camera_set_from_dict
from compiler.realsas_compiler_core.geometry_substrate_v2 import geometry_substrate_evidence_from_dict

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    rr=(a.authority_root/"runs"/a.run_id).resolve()
    ledger=json.loads((rr/"ACTIVE_RUN_V2.json").read_text())
    def row(stage):
        return next(x for x in ledger["stages"] if x["id"]==stage)
    def output(stage,schema):
        hits=[x for x in row(stage).get("outputs",[]) if x.get("schema")==schema]
        if len(hits)!=1: raise RuntimeError(f"OUTPUT_CARDINALITY::{stage}::{schema}::{len(hits)}")
        return hits[0]
    cam_ref=output("05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    cam_path=Path(cam_ref["path"]).resolve()
    stage13_dir=rr/"artifacts"/"13_GEOMETRY_SUBSTRATE_QUALIFIED"
    stage13_files=[]
    geo_candidates=[]
    for p in sorted(stage13_dir.rglob("*.json")):
        try:
            payload=json.loads(p.read_text())
        except Exception as exc:
            stage13_files.append({"path":str(p.resolve()),"parse_error":str(exc)})
            continue
        schema=str(payload.get("schema") or payload.get("schema_version") or "")
        row={"path":str(p.resolve()),"sha256":sha256(p),"schema":schema}
        if schema=="RealSaS.GeometrySubstrateQualificationIR.v2":
            geo_candidates.append((p.resolve(),payload))
            row["camera_set_binding_hash"]=payload.get("camera_set_binding_hash")
            row["substrate_hash"]=payload.get("substrate_hash")
        stage13_files.append(row)
    if len(geo_candidates)!=1:
        raise RuntimeError(f"STAGE13_SCHEMA_SCAN_CARDINALITY::{len(geo_candidates)}::{stage13_files}")
    geo_path,geo_payload=geo_candidates[0]
    cam=qualified_camera_set_from_dict(json.loads(cam_path.read_text()))
    geo=geometry_substrate_evidence_from_dict(geo_payload)
    report={
      "schema":"RealSaS.Stage05Stage13BindingAudit.v1",
      "run_id":a.run_id,
      "stage05":{
        "ledger_status":row("05_CAMERA_CONTRACT_SOLVED").get("status"),
        "path":str(cam_path),
        "ledger_sha256":cam_ref.get("sha256"),
        "actual_sha256":sha256(cam_path),
        "camera_set_hash":cam.camera_set_hash,
        "camera_binding_hashes":list(cam.camera_binding_hashes),
      },
      "stage13":{
        "ledger_status":row("13_GEOMETRY_SUBSTRATE_QUALIFIED").get("status"),
        "path":str(geo_path),
        "ledger_has_schema_output":False,
        "actual_sha256":sha256(geo_path),
        "directory_scan":stage13_files,
        "substrate_hash":geo.substrate_hash,
        "camera_set_binding_hash":geo.camera_set_binding_hash,
        "observation_set_binding_hash":geo.observation_set_binding_hash,
        "zero_surface_binding_hash":geo.zero_surface_binding_hash,
        "normalization_binding_hash":geo.normalization_binding_hash,
        "qualification_report":geo.qualification_report,
        "metadata":geo.metadata,
      },
      "camera_binding_matches":geo.camera_set_binding_hash==cam.camera_set_hash,
      "ledger_bytes_valid":(
          cam_ref.get("sha256")==sha256(cam_path)
          and True
      )
    }
    print("STAGE05_STAGE13_BINDING="+json.dumps(report,sort_keys=True))
    out=Path("/tmp/STAGE05_STAGE13_BINDING_AUDIT.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
