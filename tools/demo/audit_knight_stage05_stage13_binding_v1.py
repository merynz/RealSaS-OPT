from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,
    signed_zero_surface_from_dict,
)
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
    obs_ref=output("07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    norm_ref=output("08_NORMALIZATION_DOMAIN_QUALIFIED","RealSaS.NormalizationDomainIR.v1")
    zero_hits=[x for x in row("12_ZERO_SURFACE_DECODED").get("outputs",[]) if x.get("schema")=="RealSaS.SignedZeroSurfaceSealIR.v1"]
    zero_ref=zero_hits[0] if len(zero_hits)==1 else None
    cam_path=Path(cam_ref["path"]).resolve()
    obs_path=Path(obs_ref["path"]).resolve()
    norm_path=Path(norm_ref["path"]).resolve()
    if zero_ref is not None:
        zero_path=Path(zero_ref["path"]).resolve()
        zero_payload=json.loads(zero_path.read_text())
        zero_source="LEDGER"
    else:
        zc=[]
        for p in sorted((rr/"artifacts"/"12_ZERO_SURFACE_DECODED").rglob("*.json")):
            try:
                payload=json.loads(p.read_text())
            except Exception:
                continue
            schema=str(payload.get("schema") or payload.get("schema_version") or "")
            if schema=="RealSaS.SignedZeroSurfaceSealIR.v1":
                zc.append((p.resolve(),payload))
        if len(zc)!=1:
            raise RuntimeError(f"STAGE12_SCHEMA_SCAN_CARDINALITY::{len(zc)}")
        zero_path,zero_payload=zc[0]
        zero_source="ORPHAN_DIRECTORY_SCAN"
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
        file_row={"path":str(p.resolve()),"sha256":sha256(p),"schema":schema}
        if schema=="RealSaS.GeometrySubstrateQualificationIR.v2":
            geo_candidates.append((p.resolve(),payload))
            file_row["camera_set_binding_hash"]=payload.get("camera_set_binding_hash")
            file_row["substrate_hash"]=payload.get("substrate_hash")
        stage13_files.append(file_row)
    if len(geo_candidates)!=1:
        raise RuntimeError(f"STAGE13_SCHEMA_SCAN_CARDINALITY::{len(geo_candidates)}::{stage13_files}")
    geo_path,geo_payload=geo_candidates[0]
    cam=qualified_camera_set_from_dict(json.loads(cam_path.read_text()))
    obs=qualified_observation_set_from_dict(json.loads(obs_path.read_text()))
    norm=normalization_domain_from_dict(json.loads(norm_path.read_text()))
    zero=signed_zero_surface_from_dict(zero_payload)
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
      "stage07":{
        "ledger_status":row("07_OBSERVATION_CONTRACT_QUALIFIED").get("status"),
        "actual_sha256":sha256(obs_path),
        "ledger_sha256":obs_ref.get("sha256"),
        "observation_set_hash":obs.observation_set_hash,
        "camera_binding_hashes":[v.camera_binding_hash for v in sorted(obs.views,key=lambda x:x.view_index)],
        "camera_bindings_match_stage05":tuple(v.camera_binding_hash for v in sorted(obs.views,key=lambda x:x.view_index))==tuple(cam.camera_binding_hashes),
      },
      "stage08":{
        "ledger_status":row("08_NORMALIZATION_DOMAIN_QUALIFIED").get("status"),
        "actual_sha256":sha256(norm_path),
        "ledger_sha256":norm_ref.get("sha256"),
        "normalization_hash":norm.normalization_hash,
        "camera_set_binding_hash":norm.camera_set_binding_hash,
        "observation_set_binding_hash":norm.observation_set_binding_hash,
        "camera_binding_matches_stage05":norm.camera_set_binding_hash==cam.camera_set_hash,
        "observation_binding_matches_stage07":norm.observation_set_binding_hash==obs.observation_set_hash,
      },
      "stage12":{
        "ledger_status":row("12_ZERO_SURFACE_DECODED").get("status"),
        "actual_sha256":sha256(zero_path),
        "ledger_sha256":(zero_ref.get("sha256") if zero_ref is not None else None),
        "authority_source":zero_source,
        "ledger_has_schema_output":zero_ref is not None,
        "zero_surface_hash":zero.zero_surface_hash,
        "observation_set_binding_hash":zero.observation_set_binding_hash,
        "normalization_binding_hash":zero.normalization_binding_hash,
        "observation_binding_matches_stage07":zero.observation_set_binding_hash==obs.observation_set_hash,
        "normalization_binding_matches_stage08":zero.normalization_binding_hash==norm.normalization_hash,
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
      "upstream_chain_matches":{
        "05_to_07":tuple(v.camera_binding_hash for v in sorted(obs.views,key=lambda x:x.view_index))==tuple(cam.camera_binding_hashes),
        "05_to_08":norm.camera_set_binding_hash==cam.camera_set_hash,
        "07_to_08":norm.observation_set_binding_hash==obs.observation_set_hash,
        "07_to_12":zero.observation_set_binding_hash==obs.observation_set_hash,
        "08_to_12":zero.normalization_binding_hash==norm.normalization_hash,
      },
      "ledger_bytes_valid":all((
          cam_ref.get("sha256")==sha256(cam_path),
          obs_ref.get("sha256")==sha256(obs_path),
          norm_ref.get("sha256")==sha256(norm_path),
          (zero_ref.get("sha256")==sha256(zero_path) if zero_ref is not None else True),
      ))
    }
    print("STAGE05_STAGE13_BINDING="+json.dumps(report,sort_keys=True))
    out=Path("/tmp/STAGE05_STAGE13_BINDING_AUDIT.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
