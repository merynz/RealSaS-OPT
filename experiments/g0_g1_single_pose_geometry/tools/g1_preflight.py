from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realsas_iris_sees.g1_io import load_g1_sample, G1_FORBIDDEN_GPU_TARGET_KEYS
from realsas_iris_sees.g1_losses import g1_geometry_loss
from realsas_iris_sees.g1_metrics import g1_geometry_metrics
from realsas_iris_sees.g1_model import G1Config, IRISG1SinglePose, migrate_n1d_state_dict


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            payload = z.read("BEST.pt")
        return (
            torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False),
            hashlib.sha256(payload).hexdigest(),
        )
    return torch.load(path, map_location="cpu", weights_only=False), sha256_file(path)


def cfg_from_checkpoint(ck) -> G1Config:
    c = ck["config"]
    keys = {
        "image_size", "base_dim", "token_dim", "descriptor_dim", "transformer_depth",
        "transformer_heads", "mlp_ratio", "dropout", "input_channels",
    }
    return G1Config(**{k: c[k] for k in keys})


def grad_l1(module: torch.nn.Module) -> float:
    return float(sum(
        p.grad.detach().abs().sum().item()
        for p in module.parameters()
        if p.requires_grad and p.grad is not None
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-a-dir", required=True)
    ap.add_argument("--target-npz", required=True)
    ap.add_argument("--n1d-best", required=True, help="BEST.pt or N1D canonical proof-pack zip")
    ap.add_argument("--expected-family-id", type=int, required=True)
    ap.add_argument("--family-manifest")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    ck_path = Path(args.n1d_best)
    ck, checkpoint_payload_sha = load_checkpoint(ck_path)
    assert ck["schema"] == "RealSaS.IRIS.SEES.N1D.MechanicsCanonical.Checkpoint.v2"

    model = IRISG1SinglePose(cfg_from_checkpoint(ck)).to(device)
    migration = migrate_n1d_state_dict(model, ck["model"])
    assert migration["loaded_fraction"] == 1.0
    assert not migration["missing_destination_keys"]
    assert not migration["shape_mismatches"]
    assert len(migration["archived_only_source_keys"]) == 29
    assert all(k.startswith("differential.") for k in migration["archived_only_source_keys"])

    images, target, family_id = load_g1_sample(
        args.pose_a_dir, args.target_npz, image_size=model.cfg.image_size, device=device
    )
    assert family_id == args.expected_family_id, (family_id, args.expected_family_id)
    assert tuple(images.shape) == (1, 8, 4, model.cfg.image_size, model.cfg.image_size)
    assert not set(target).intersection(G1_FORBIDDEN_GPU_TARGET_KEYS)

    pose_hashes = {
        p.name: sha256_file(p) for p in sorted(Path(args.pose_a_dir).glob("*.png"))
    }
    family_manifest = None
    if args.family_manifest:
        family_manifest = json.loads(Path(args.family_manifest).read_text())
        assert int(family_manifest["family_id"]) == family_id
        assert pose_hashes == family_manifest["poseA_hashes"], (pose_hashes, family_manifest["poseA_hashes"])

    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(images)
    loss, parts = g1_geometry_loss(outputs, target, image_size=256)
    assert torch.isfinite(loss)
    loss.backward()

    grads = {
        "encoder": grad_l1(model.encoder),
        "fusion": grad_l1(model.fusion),
        "decoder": grad_l1(model.decoder),
        "geometry": grad_l1(model.geometry),
        "descriptor": grad_l1(model.descriptor) if model.descriptor is not None else 0.0,
        "camera_residual": grad_l1(model.camera_residual),
    }
    for k in ("encoder", "fusion", "decoder", "geometry"):
        assert grads[k] > 0.0, (k, grads)
    assert grads["descriptor"] == 0.0, grads
    assert grads["camera_residual"] == 0.0, grads
    assert model.descriptor is None or not any(p.requires_grad for p in model.descriptor.parameters())
    assert not any(p.requires_grad for p in model.camera_residual.parameters())

    model.eval()
    with torch.inference_mode():
        outputs = model(images)
        metrics = g1_geometry_metrics(outputs, target, image_size=256)

    result = {
        "schema": "RealSaS.IRIS.G1.ExecutablePreflight.v1",
        "PASS": True,
        "optimizer_steps": 0,
        "family_id": family_id,
        "input_shape": list(images.shape),
        "target_keys_on_device": sorted(target),
        "forbidden_truth_leak": False,
        "checkpoint": {
            "source_schema": ck["schema"],
            "source_epoch": int(ck["epoch"]),
            "source_container_sha256": sha256_file(ck_path),
            "checkpoint_payload_sha256": checkpoint_payload_sha,
            "source_model_tensor_count": len(ck["model"]),
        },
        "migration": migration,
        "gradient_witness_l1": grads,
        "loss_parts": {k: float(v.detach()) for k, v in parts.items()},
        "warmstart_metrics_diagnostic_only": metrics,
        "pose_a_png_sha256": pose_hashes,
        "family_manifest_content_sha256": family_manifest.get("content_sha256") if family_manifest else None,
        "target_npz_sha256": sha256_file(args.target_npz),
        "active_code_sha256": {
            str(q.relative_to(ROOT)): sha256_file(q)
            for q in [
                ROOT / "realsas_iris_sees/g1_model.py",
                ROOT / "realsas_iris_sees/g1_losses.py",
                ROOT / "realsas_iris_sees/g1_metrics.py",
                ROOT / "realsas_iris_sees/g1_sampling.py",
                ROOT / "realsas_iris_sees/g1_targets.py",
                ROOT / "realsas_iris_sees/g1_io.py",
                ROOT / "tools/g1_preflight.py",
            ]
        },
    }
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
