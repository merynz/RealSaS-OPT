from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import os

import torch

from .dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    extract_dinov2_s_patch_map_v2,
    load_exact_dinov2_s_v2,
    tensor_bytes_sha256_v2,
)


@dataclass(frozen=True)
class DINOv2TokenParityResultV2:
    schema: str
    status: str
    authority_hash: str
    feature_contract_hash: str
    observation_contract_hash: str
    reference_tensor_sha256: str
    live_tensor_sha256: str
    shape: tuple[int, ...]
    dtype: str
    bitwise_equal: bool
    max_abs: float
    mean_abs: float
    atol: float
    rtol: float
    optimizer_steps: int
    training_authorized: bool


def _canonical_sha(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def save_dinov2_token_reference_v2(
    path: str | Path,
    patch_map: torch.Tensor,
    *,
    observation_contract_hash: str,
    authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2(),
) -> dict:
    """Seal an exact reference token tensor generated under already-verified authority.

    This function creates evidence; it never authorizes training. Reference creation
    and live parity should be separate executions if the scientific gate requires
    replay across processes/runtimes.
    """
    authority.validate()
    x = patch_map.detach().cpu().float().contiguous()
    if x.ndim != 5 or x.shape[1:] != (8, 384, 37, 37):
        raise ValueError("reference DINO patch map must be [B,8,384,37,37]")
    if not observation_contract_hash:
        raise ValueError("observation contract hash required")
    payload = {
        "schema": "RealSaS.IRIS.DINOv2TokenReference.v2",
        "authority": asdict(authority),
        "authority_hash": authority.authority_hash,
        "feature_contract_hash": authority.feature_contract().contract_hash,
        "observation_contract_hash": str(observation_contract_hash),
        "tensor_sha256": tensor_bytes_sha256_v2(x),
        "shape": tuple(int(v) for v in x.shape),
        "dtype": "float32",
        "patch_map": x,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {k: v for k, v in payload.items() if k != "patch_map"} | {"file_sha256": sha256(path.read_bytes()).hexdigest()}


def load_dinov2_token_reference_v2(
    path: str | Path,
    *,
    expected_observation_contract_hash: str,
    authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2(),
) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "RealSaS.IRIS.DINOv2TokenReference.v2":
        raise ValueError("DINO token reference schema mismatch")
    if payload.get("authority_hash") != authority.authority_hash:
        raise ValueError("DINO token reference authority mismatch")
    if payload.get("feature_contract_hash") != authority.feature_contract().contract_hash:
        raise ValueError("DINO token reference feature contract mismatch")
    if payload.get("observation_contract_hash") != expected_observation_contract_hash:
        raise ValueError("DINO token reference observation mismatch")
    if payload.get("optimizer_steps") != 0 or payload.get("training_authorized") is not False:
        raise ValueError("DINO token reference scientific firewall mismatch")
    x = torch.as_tensor(payload.get("patch_map")).cpu().float().contiguous()
    if tuple(x.shape[1:]) != (8, 384, 37, 37) or payload.get("tensor_sha256") != tensor_bytes_sha256_v2(x):
        raise ValueError("DINO token reference tensor/hash drift")
    return x


def compare_dinov2_tokens_v2(
    live_patch_map: torch.Tensor,
    reference_patch_map: torch.Tensor,
    *,
    authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2(),
    observation_contract_hash: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> DINOv2TokenParityResultV2:
    live = live_patch_map.detach().cpu().float().contiguous()
    ref = reference_patch_map.detach().cpu().float().contiguous()
    if live.shape != ref.shape or tuple(live.shape[1:]) != (8, 384, 37, 37):
        raise ValueError(f"DINO token parity shape mismatch live={tuple(live.shape)} ref={tuple(ref.shape)}")
    if not torch.isfinite(live).all() or not torch.isfinite(ref).all():
        raise ValueError("non-finite DINO tokens")
    diff = (live - ref).abs()
    bitwise = torch.equal(live, ref)
    close = torch.allclose(live, ref, atol=float(atol), rtol=float(rtol))
    status = "PASS_EXACT_TOKEN_PARITY" if close else "FAIL_TOKEN_PARITY"
    return DINOv2TokenParityResultV2(
        schema="RealSaS.IRIS.DINOv2TokenParityResult.v2",
        status=status,
        authority_hash=authority.authority_hash,
        feature_contract_hash=authority.feature_contract().contract_hash,
        observation_contract_hash=str(observation_contract_hash),
        reference_tensor_sha256=tensor_bytes_sha256_v2(ref),
        live_tensor_sha256=tensor_bytes_sha256_v2(live),
        shape=tuple(int(v) for v in live.shape),
        dtype="float32",
        bitwise_equal=bool(bitwise),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        atol=float(atol),
        rtol=float(rtol),
        optimizer_steps=0,
        training_authorized=False,
    )


def write_parity_result_v2(path: str | Path, result: DINOv2TokenParityResultV2) -> dict:
    payload = asdict(result)
    payload["content_sha256"] = _canonical_sha(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact/fail-closed DINOv2-S token parity gate for IRIS V2")
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--weight-path", required=True)
    ap.add_argument("--native-rgb-pt", required=True, help="torch file containing tensor [B,8,3,1024,1024]")
    ap.add_argument("--reference-pt", required=True)
    ap.add_argument("--observation-contract-hash", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    authority = DINOv2FoundationAuthorityV2()
    model, seal = load_exact_dinov2_s_v2(args.source_dir, args.weight_path, device=args.device, authority=authority)
    native = torch.load(args.native_rgb_pt, map_location=args.device, weights_only=True)
    if isinstance(native, dict):
        native = native.get("native_rgb")
    if native is None:
        raise ValueError("native RGB tensor missing")
    live = extract_dinov2_s_patch_map_v2(model, torch.as_tensor(native, device=args.device))
    reference = load_dinov2_token_reference_v2(
        args.reference_pt,
        expected_observation_contract_hash=args.observation_contract_hash,
        authority=authority,
    )
    result = compare_dinov2_tokens_v2(
        live,
        reference,
        authority=authority,
        observation_contract_hash=args.observation_contract_hash,
        atol=args.atol,
        rtol=args.rtol,
    )
    payload = write_parity_result_v2(args.output_json, result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result.status.startswith("PASS"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
