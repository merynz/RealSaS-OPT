"""Reusable source preparation into verified immutable input packets, not Attempts.

The historical default roster remains FIT8.  A caller may supply a sealed generic roster with the
same ``selected`` row contract (for example Truth139 apparatus output).  Roster identity is bound
into every cache key so packets can never be reused across cohort revisions merely because source
bytes happen to match.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from tools.corpus.iris_observation_admission_v1 import audit_subject, digest

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_ROSTER=ROOT/"canonical/FIT8_DIVERSITY_SELECTION_V1.json"
ALLOWED_ROSTER_SCHEMAS={
    "RealSaS.FIT8DiversitySelection.v1",
    "RealSaS.Truth139ApparatusManifest.v1",
}


def _value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_roster(path):
    path=Path(path).resolve()
    roster=json.loads(path.read_text())
    if roster.get("schema") not in ALLOWED_ROSTER_SCHEMAS:
        raise ValueError("PREPARE_ROSTER_SCHEMA_UNSUPPORTED")
    rows=list(roster.get("selected") or ())
    ids=[str(row.get("subject_key") or "") for row in rows]
    if not rows or any(not sid for sid in ids) or len(set(ids))!=len(ids):
        raise ValueError("PREPARE_ROSTER_SUBJECT_ID_INVALID")
    if roster.get("schema")=="RealSaS.Truth139ApparatusManifest.v1":
        if roster.get("status")!="READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED":
            raise ValueError("PREPARE_TRUTH139_APPARATUS_STATUS")
        if int(roster.get("subject_count",-1))!=len(rows) or roster.get("training_authorized") is not False:
            raise ValueError("PREPARE_TRUTH139_APPARATUS_CARDINALITY_OR_AUTHORITY")
        for row in rows:
            if row.get("import_render_admission")!="PENDING" or row.get("training_authorized") is not False:
                raise ValueError("PREPARE_TRUTH139_SUBJECT_AUTHORITY_DRIFT:"+str(row.get("subject_key")))
    return path,roster,_value_hash(roster)


def prepare(source_root, output_root, blender, probe, roster_path=DEFAULT_ROSTER):
    source_root,output_root=Path(source_root).resolve(),Path(output_root).resolve()
    blender,probe=Path(blender).resolve(),Path(probe).resolve()
    roster_path,roster,roster_sha=load_roster(roster_path)
    policy=json.loads((ROOT/"canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    sources=["tools/observation/blender_controlled_8view_v2.py","tools/observation/blender_controlled_8view_v1.py",
             "tools/observation/source_scene_contract_v2.py","tools/observation/blender_source_parity_export_v1.py",
             "tools/corpus/gltf_bind_scene_v1.py","tools/corpus/iris_observation_admission_v1.py",
             "tools/corpus/prepare_iris_observations_v1.py"]
    binding={"producer_sources":{p:digest(ROOT/p) for p in sources},
             "blender_launcher_sha256":digest(blender),"probe_binary_sha256":digest(probe),"policy":policy,
             "roster_schema":roster["schema"],"roster_identity_sha256":roster_sha}
    artifacts=[]
    for row in roster["selected"]:
        key=row["subject_key"];source=source_root/row["source"]["corpus_relative_path"]
        if digest(source)!=row["source"]["sha256"]:
            raise ValueError("PREPARE_RAW_SOURCE_HASH")
        identity=hashlib.sha256(json.dumps({"binding":binding,"source":row["source"]["sha256"],"subject_key":key},
                                           sort_keys=True,separators=(",",":")).encode()).hexdigest()
        root=output_root/"objects"/identity
        packet=root/"packet"/key
        done=packet/"IRIS_OBSERVATION_ADMISSION.json"
        if done.exists():
            # Never trust an old success marker without checking every bound byte.
            from tools.training.prepare_iris_cohort_v1 import verify_observations
            verify_observations(packet.parent,row)
            print(f"[REUSE] {key} exact_input_packet={identity[:12]}",flush=True)
        else:
            root.mkdir(parents=True,exist_ok=True)
            for role,extra in [
                ("import",["--python",str(ROOT/"tools/observation/blender_source_parity_export_v1.py"),"--",
                            str(source),str(root/"imported"/key/"imported.json")]),
                ("render",["--python",str(ROOT/"tools/observation/blender_controlled_8view_v2.py"),"--",
                            "--source",str(source),"--expected-source-sha256",row["source"]["sha256"],
                            "--subject-id",key,"--out-dir",str(root/"observations"/key)])]:
                target=root/("imported" if role=="import" else "observations")/key
                if target.exists():
                    # Retain failed bytes and create a fresh producer directory.
                    target.rename(target.with_name(key+".failed."+str(time.time_ns())))
                with (root/(role+".log")).open("w") as log:
                    subprocess.run([str(blender),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
            audit_subject(row,source_root,root/"imported",root/"observations",probe,root/"admission",policy)
            packet.mkdir(parents=True,exist_ok=True)
            for path in (root/"observations"/key).iterdir():
                if path.is_file():
                    shutil.copyfile(path,packet/path.name)
            shutil.copyfile(root/"admission"/key/"IRIS_OBSERVATION_ADMISSION.json",done)
        artifacts.append({"subject_id":key,"packet_identity":identity,
                          "proof_sha256":digest(done),"packet_path":str(packet),
                          "source_sha256":row["source"]["sha256"]})
    index={"schema":"RealSaS.PreparedIrisInputPackets.v1","binding":binding,"subjects":artifacts,
           "roster":{"path":str(roster_path),"schema":roster["schema"],"identity_sha256":roster_sha},
           "authority":"PREPARED_EXTERNAL_INPUT_BYTES__NOT_REGISTRY_OR_WORKFLOW_STATE",
           "go_qualification_required":True,"training_authorized":False}
    output_root.mkdir(parents=True,exist_ok=True)
    (output_root/"INPUT_PACKETS.json").write_text(json.dumps(index,indent=2)+"\n")
    return index


def main():
    parser=argparse.ArgumentParser()
    for flag in ("source-root","output-root","blender","probe"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--roster",type=Path,default=DEFAULT_ROSTER)
    args=parser.parse_args()
    prepare(args.source_root,args.output_root,args.blender,args.probe,args.roster)
if __name__=="__main__":
    main()
