from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

FORBIDDEN_INFERENCE_KEYS = {
    "teacher", "teacher_depth", "teacher_skeleton", "teacher_skin", "truth", "ground_truth", "gt",
    "source_rig", "source_skeleton", "source_weights", "source_skin", "manual_joints", "manual_weights",
    "joint_coordinates", "known_topology", "specimen_id", "asset_id",
}


@dataclass(frozen=True)
class ImageOnlyInferenceRequest:
    images_path: str
    cameras_path: str
    iris_checkpoint: str
    geppetto_checkpoint: str
    arachne_checkpoint: str
    output_dir: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


def _walk(value: Any, path: str = "root"):
    if isinstance(value, dict):
        for k,v in value.items():
            yield path,str(k),v
            yield from _walk(v,f"{path}.{k}")
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): yield from _walk(v,f"{path}[{i}]")


def assert_image_only_request(request: ImageOnlyInferenceRequest | dict) -> None:
    payload=request.to_dict() if isinstance(request,ImageOnlyInferenceRequest) else dict(request)
    keys={str(k).lower() for _,k,_ in _walk(payload)}
    bad=sorted(keys & FORBIDDEN_INFERENCE_KEYS)
    if bad: raise RuntimeError(f"INFERENCE_FIREWALL_FORBIDDEN_KEYS:{bad}")
    required={"images_path","cameras_path","iris_checkpoint","geppetto_checkpoint","arachne_checkpoint","output_dir"}
    missing=sorted(required-set(payload))
    if missing: raise RuntimeError(f"INFERENCE_FIREWALL_MISSING_KEYS:{missing}")
    extra=sorted(set(payload)-required)
    if extra: raise RuntimeError(f"INFERENCE_FIREWALL_UNDECLARED_KEYS:{extra}")
    for key in required:
        if not isinstance(payload[key],str) or not payload[key]: raise RuntimeError(f"INFERENCE_FIREWALL_EMPTY:{key}")


def write_firewall_receipt(request: ImageOnlyInferenceRequest, path: str | Path) -> None:
    assert_image_only_request(request)
    receipt={
        "schema":"RealSaS.ImageOnlyDemoInferenceFirewallReceipt.v1",
        "allowed_inputs":sorted(request.to_dict()),
        "teacher_inputs_present":False,
        "source_rig_inputs_present":False,
        "manual_output_injection":False,
        "generalization_claim":False,
    }
    Path(path).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
