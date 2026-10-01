from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def walk_strings(x,path=""):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items(): out+=walk_strings(v,f"{path}.{k}" if path else str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x): out+=walk_strings(v,f"{path}[{i}]")
    elif isinstance(x,str):
        out.append((path,x))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    rr=a.authority_root/"runs"/a.run_id
    imp=a.authority_root/"imports"/"knight_frozen_stage08_20260919"/"extracted"
    paths={
      "spec":rr/"demo_stage08_import_spec.json",
      "old05":imp/"05_CAMERA_CONTRACT_SOLVED"/"qualified_camera_set.json",
      "old07":imp/"07_OBSERVATION_CONTRACT_QUALIFIED"/"qualified_observation_set.json",
      "old08":imp/"08_NORMALIZATION_DOMAIN_QUALIFIED"/"normalization_domain.json",
      "s12":rr/"artifacts"/"12_ZERO_SURFACE_DECODED"/"signed_zero_surface_seal.json",
      "s13":rr/"artifacts"/"13_GEOMETRY_SUBSTRATE_QUALIFIED"/"geometry_substrate_qualification.json",
      "s14":rr/"artifacts"/"14_GSA_BUILD"/"rigging_surface_candidate.json",
      "s15":rr/"artifacts"/"15_RIGGING_SURFACE_QUALIFIED"/"qualified_rigging_surface.json",
      "s34":rr/"artifacts"/"34_DEFORMATION_CAPABILITY_ENVELOPE"/"deformation_envelope.json",
    }
    p={k:json.loads(v.read_text()) for k,v in paths.items()}
    h05=p["old05"]["camera_set_hash"]
    h07=p["old07"]["observation_set_hash"]
    h08=p["old08"]["normalization_hash"]
    h12=p["s12"]["zero_surface_hash"]
    checks={
      "12_obs_matches_old07":p["s12"].get("observation_set_binding_hash")==h07,
      "12_norm_matches_old08":p["s12"].get("normalization_binding_hash")==h08,
      "13_camera_matches_old05":p["s13"].get("camera_set_binding_hash")==h05,
      "13_obs_matches_old07":p["s13"].get("observation_set_binding_hash")==h07,
      "13_norm_matches_old08":p["s13"].get("normalization_binding_hash")==h08,
      "13_zero_matches_12":p["s13"].get("zero_surface_binding_hash")==h12,
    }
    target_hashes={"old05":h05,"old07":h07,"old08":h08,"s12":h12}
    string_hits={}
    for key in ("spec","s14","s15","s34"):
        vals=walk_strings(p[key])
        string_hits[key]={
          name:[path for path,val in vals if val==hv]
          for name,hv in target_hashes.items()
          if any(val==hv for _,val in vals)
        }
    report={
      "schema":"RealSaS.FrozenGeometryLineageAudit.v1",
      "paths":{k:str(v.resolve()) for k,v in paths.items()},
      "sha256":{k:sha256(v) for k,v in paths.items()},
      "hashes":{"camera_set_hash":h05,"observation_set_hash":h07,"normalization_hash":h08,"zero_surface_hash":h12},
      "checks":checks,
      "all_12_13_bindings_match":all(checks.values()),
      "import_spec":p["spec"],
      "string_binding_hits":string_hits,
      "stage14_top_keys":sorted(p["s14"].keys()),
      "stage14_metadata":p["s14"].get("metadata"),
      "stage15_top_keys":sorted(p["s15"].keys()),
      "stage15_metadata":p["s15"].get("metadata"),
      "stage34_top_keys":sorted(p["s34"].keys()),
      "stage34_metadata":p["s34"].get("metadata"),
      "old05_camera_binding_hashes":p["old05"].get("camera_binding_hashes"),
    }
    print("FROZEN_GEOMETRY_LINEAGE="+json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
