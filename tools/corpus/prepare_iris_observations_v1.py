"""Reusable source preparation into verified immutable input packets, not Attempts."""
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

def prepare(source_root, output_root, blender, probe):
    source_root,output_root=Path(source_root).resolve(),Path(output_root).resolve()
    blender,probe=Path(blender).resolve(),Path(probe).resolve()
    roster=json.loads((ROOT/"canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
    policy=json.loads((ROOT/"canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    sources=["tools/observation/blender_controlled_8view_v2.py","tools/observation/blender_controlled_8view_v1.py",
             "tools/observation/source_scene_contract_v2.py","tools/observation/blender_source_parity_export_v1.py",
             "tools/corpus/gltf_bind_scene_v1.py","tools/corpus/iris_observation_admission_v1.py",
             "tools/corpus/prepare_iris_observations_v1.py"]
    binding={"producer_sources":{p:digest(ROOT/p) for p in sources},
             "blender_launcher_sha256":digest(blender),"probe_binary_sha256":digest(probe),"policy":policy}
    artifacts=[]
    for row in roster["selected"]:
        key=row["subject_key"];source=source_root/row["source"]["corpus_relative_path"]
        if digest(source)!=row["source"]["sha256"]:
            raise ValueError("PREPARE_RAW_SOURCE_HASH")
        identity=hashlib.sha256(json.dumps({"binding":binding,"source":row["source"]["sha256"]},
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
           "authority":"PREPARED_EXTERNAL_INPUT_BYTES__NOT_REGISTRY_OR_WORKFLOW_STATE",
           "go_qualification_required":True}
    (output_root/"INPUT_PACKETS.json").write_text(json.dumps(index,indent=2))
    return index

def main():
    parser=argparse.ArgumentParser()
    for flag in ("source-root","output-root","blender","probe"):
        parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source_root,args.output_root,args.blender,args.probe)
if __name__=="__main__":
    main()
