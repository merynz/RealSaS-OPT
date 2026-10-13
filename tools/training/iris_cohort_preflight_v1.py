"""Real source/model/optimizer prefix; honest CPU or on-A100 runtime evidence."""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
import torch
from models.iris.training.cohort_v1 import SharedIrisTrainer, source_identity, file_hash, atomic_json

ROOT=Path(__file__).resolve().parents[2]

def preflight(data_root, policy, checkpoint_root, output_root, device="cuda"):
    data_root=Path(data_root);output_root=Path(output_root)
    manifest=json.loads((data_root/"COHORT_MANIFEST.json").read_text())
    p=copy.deepcopy(policy)
    # Same production architecture, real features/teachers, short isolated policy.
    p.update(steps=16, log_interval=1, evaluation_interval=8,
             checkpoint_interval=4, scope="RUNTIME_PREFLIGHT__NOT_PRODUCTION_FIT")
    if str(device)=="cpu":
        p.update(queries_per_step=512,silhouette_rays_per_step=32,silhouette_depth_samples=32)
    source=source_identity(ROOT)
    trainer=SharedIrisTrainer(data_root,manifest,p,source,checkpoint_root,output_root,device)
    if trainer.step == 16:
        result=trainer.export(False)
        old=output_root/"RUNTIME_PREFLIGHT.json"
        if old.exists():
            receipt=json.loads(old.read_text())
            if receipt["binding_sha256"]==trainer.binding_hash:
                print("[PREFLIGHT] exact completed runtime prefix reused",flush=True)
                return receipt
        raise RuntimeError("PREFLIGHT_COMPLETE_WITHOUT_BOUND_RECEIPT")
    if trainer.step < 8:
        trainer.run(stop_after=8)
    if trainer.step == 8 and trainer.counts != [1]*8:
        raise RuntimeError("PREFLIGHT_NOT_ALL_EIGHT_TRAINED")
    expected={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    del trainer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    resumed=SharedIrisTrainer(data_root,manifest,p,source,checkpoint_root,output_root,device)
    for key,value in expected.items():
        if not torch.equal(value,resumed.model.state_dict()[key].cpu()):
            raise RuntimeError("PREFLIGHT_RESTORE_WEIGHTS_DRIFT")
    # Continue through another complete cycle with the same optimizer and RNG.
    result=resumed.run()
    if resumed.counts != [2]*8 or resumed.step != 16:
        raise RuntimeError("PREFLIGHT_RESUME_PROGRESS_DRIFT")
    receipt={"schema":"RealSaS.SharedIrisRuntimePreflight.v1","status":"PASS",
             "device":str(device),"gpu_name":torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
             "a100_executed":str(device).startswith("cuda") and "A100" in torch.cuda.get_device_name(),
             "production_architecture":p["model_config"],"trainable_parameters":sum(v.numel() for v in resumed.model.parameters()),
             "actual_optimizer_steps":16,"subject_exposures":result["subject_exposures"],
             "weights_exactly_restored":True,"all_eight_evaluated":True,
             "checkpoint_payload_sha256":json.loads((Path(checkpoint_root)/"LATEST.json").read_text())["sha256"],
             "binding_sha256":resumed.binding_hash,"cpu_exact_resume_test":"tests/iris/test_shared_cohort_v1.py",
             "fit8_pass_claimed":False,"go_qualification_claimed":False}
    atomic_json(output_root/"RUNTIME_PREFLIGHT.json",receipt)
    del resumed;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return receipt
