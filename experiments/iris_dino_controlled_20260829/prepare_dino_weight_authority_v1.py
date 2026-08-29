#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

SCHEMA = "RealSaS.DINO.WeightPreprocessAuthority.v1"
DINO_REV = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
TRAIN512_SET_SHA = "1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2"
MASTER_LEDGER_SHA = "475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913"
WITNESS_ASSET = "asset_0004e640bea23f87618cad9e"

CANDIDATES = {
    "S": {"constructor": "dinov2_vits14", "arch_name": "vit_small", "embed_dim": 384, "patch_size": 14},
    "B": {"constructor": "dinov2_vitb14", "arch_name": "vit_base", "embed_dim": 768, "patch_size": 14},
    "L": {"constructor": "dinov2_vitl14", "arch_name": "vit_large", "embed_dim": 1024, "patch_size": 14},
    "g": {"constructor": "dinov2_vitg14", "arch_name": "vit_giant2", "embed_dim": 1536, "patch_size": 14},
}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_sha(obj) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def run_stream(cmd, *, cwd=None, env=None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"COMMAND_FAILED exit={rc}: {' '.join(cmd)}")

def download_resume(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] target={dst}", flush=True)
    print(f"[weights] official_url={url}", flush=True)
    if shutil.which("curl"):
        run_stream([
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
            "--continue-at", "-", "--output", str(dst), url
        ])
    elif shutil.which("wget"):
        run_stream(["wget", "-c", "-O", str(dst), url])
    else:
        raise RuntimeError("CURL_OR_WGET_REQUIRED")
    if not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(f"WEIGHT_DOWNLOAD_EMPTY: {dst}")

def verify_train512_authority(root: Path):
    p = root / "reports" / "post_corpus_audit" / "DINO_TRAIN512_NATIVE1024_AUTHORITY_V2.json"
    if not p.is_file():
        raise RuntimeError(f"TRAIN512_AUTHORITY_MISSING: {p}")
    a = json.loads(p.read_text(encoding="utf-8"))
    checks = {
        "status": a.get("status") == "PASS_512_OF_512",
        "asset_count": a.get("train512_asset_count") == 512,
        "ready_count": a.get("train512_ready_count") == 512,
        "set_sha": a.get("train512_set_sha256") == TRAIN512_SET_SHA,
        "ledger_sha": a.get("selection_ledger_sha256") == MASTER_LEDGER_SHA,
        "optimizer_zero": a.get("scientific_optimizer_steps") == 0,
        "training_not_authorized": a.get("dino_training_authorized_by_this_artifact") is False,
    }
    print("[authority] TRAIN512:", checks, flush=True)
    if not all(checks.values()):
        raise RuntimeError("TRAIN512_AUTHORITY_DRIFT")
    return {
        "path": str(p.relative_to(root)),
        "raw_sha256": sha256_file(p),
        "semantic_checks": checks,
    }

def clone_exact_dino(outdir: Path) -> Path:
    src = outdir / "_dinov2_source_authority"
    if src.exists():
        shutil.rmtree(src)
    run_stream(["git", "clone", "--filter=blob:none", "https://github.com/facebookresearch/dinov2.git", str(src)])
    run_stream(["git", "-C", str(src), "checkout", "--detach", DINO_REV])
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    print("[source] HEAD:", head, flush=True)
    if head != DINO_REV:
        raise RuntimeError(f"DINO_SOURCE_REV_DRIFT expected={DINO_REV} actual={head}")
    return src

def import_source_and_derive_urls(src: Path):
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(src))
    from dinov2.hub import backbones
    from dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name

    derived = {}
    for key, spec in CANDIDATES.items():
        model_base = _make_dinov2_model_name(spec["arch_name"], spec["patch_size"])
        model_full = _make_dinov2_model_name(spec["arch_name"], spec["patch_size"], 0)
        url = f"{_DINOV2_BASE_URL}/{model_base}/{model_full}_pretrain.pth"
        derived[key] = url
    return backbones, derived

def runtime_versions():
    import numpy as np
    import torch
    import torchvision
    import PIL
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "platform": sys.platform,
    }

def preprocess_fixture(root: Path, outdir: Path):
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode

    master_png = root / "master" / "assets" / WITNESS_ASSET / "renders" / "V0" / "cel_clean.png"
    export_png = root / "exports" / "IRIS" / WITNESS_ASSET / "observations" / "V0" / "cel_clean.png"
    if not master_png.is_file():
        raise RuntimeError(f"PREPROCESS_WITNESS_MASTER_MISSING: {master_png}")
    if not export_png.is_file():
        raise RuntimeError(f"PREPROCESS_WITNESS_EXPORT_MISSING: {export_png}")

    master_sha = sha256_file(master_png)
    export_sha = sha256_file(export_png)
    print("[preprocess] witness master SHA:", master_sha, flush=True)
    print("[preprocess] witness export SHA:", export_sha, flush=True)
    if master_sha != export_sha:
        raise RuntimeError("MASTER_EXPORT_OBSERVATION_PIXEL_FILE_DRIFT")

    with Image.open(master_png) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if rgba.shape != (1024, 1024, 4):
        raise RuntimeError(f"WITNESS_RGBA_SHAPE_DRIFT {rgba.shape}")

    rgb = np.ascontiguousarray(rgba[..., :3])
    alpha = np.ascontiguousarray(rgba[..., 3])

    def apply():
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        x = TF.resize(
            x, [518, 518],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        x = (x - mean) / std
        return x.contiguous()

    x1 = apply()
    x2 = apply()
    if not torch.equal(x1, x2):
        raise RuntimeError("PREPROCESS_REPEAT_NOT_BITWISE_DETERMINISTIC")

    arr = np.ascontiguousarray(x1.cpu().numpy().astype("<f4", copy=False))
    fixture = {
        "schema": "RealSaS.DINO.PreprocessFixture.v1",
        "witness_asset": WITNESS_ASSET,
        "view": 0,
        "style": "cel_clean",
        "master_observation_path": str(master_png.relative_to(root)),
        "export_observation_path": str(export_png.relative_to(root)),
        "master_observation_file_sha256": master_sha,
        "export_observation_file_sha256": export_sha,
        "rgba_shape": list(rgba.shape),
        "rgba_uint8_sha256": sha256_bytes(np.ascontiguousarray(rgba).tobytes()),
        "rgb_uint8_sha256": sha256_bytes(rgb.tobytes()),
        "alpha_uint8_sha256": sha256_bytes(alpha.tobytes()),
        "alpha_composited_into_rgb": False,
        "resize_input_hw": [1024, 1024],
        "resize_output_hw": [518, 518],
        "resize_operator": "torchvision.transforms.functional.resize",
        "interpolation": "BICUBIC",
        "antialias": True,
        "tensor_scale": "uint8_RGB/255.0 FP32",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "output_shape": list(arr.shape),
        "output_dtype": "float32_little_endian",
        "preprocessed_tensor_sha256": sha256_bytes(arr.tobytes()),
        "repeat_bitwise_equal": True,
        "torchvision_version": torchvision.__version__,
        "scientific_optimizer_steps": 0,
        "candidate_features_extracted": False,
    }
    fixture["content_sha256"] = canonical_sha(fixture)
    atomic_json(outdir / "DINO_PREPROCESS_FIXTURE_V1.json", fixture)
    print("[preprocess] tensor SHA:", fixture["preprocessed_tensor_sha256"], flush=True)
    print("[preprocess] PASS", flush=True)
    return fixture

def verify_state_dict_meta(backbones, key: str, spec: dict, pth: Path):
    import torch

    ctor = getattr(backbones, spec["constructor"])
    print(f"[compat] {key}: constructing meta model {spec['constructor']}", flush=True)
    with torch.device("meta"):
        model = ctor(pretrained=False)

    if int(getattr(model, "embed_dim")) != spec["embed_dim"]:
        raise RuntimeError(f"{key}_EMBED_DIM_DRIFT")
    patch_size = getattr(model.patch_embed, "patch_size")
    if isinstance(patch_size, tuple):
        if tuple(int(x) for x in patch_size) != (14, 14):
            raise RuntimeError(f"{key}_PATCH_SIZE_DRIFT {patch_size}")
    elif int(patch_size) != 14:
        raise RuntimeError(f"{key}_PATCH_SIZE_DRIFT {patch_size}")
    if int(getattr(model, "num_register_tokens", 0)) != 0:
        raise RuntimeError(f"{key}_REGISTER_TOKEN_DRIFT")

    print(f"[compat] {key}: torch.load(weights_only=True,map_location='meta')", flush=True)
    state = torch.load(str(pth), map_location="meta", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"{key}_STATE_DICT_TYPE {type(state)}")
    state_tensor_keys = [k for k, v in state.items() if torch.is_tensor(v)]
    state_numel = int(sum(int(state[k].numel()) for k in state_tensor_keys))
    model_numel = int(sum(int(p.numel()) for p in model.parameters()))

    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"{key}_STRICT_LOAD_INCOMPATIBLE missing={missing[:5]} unexpected={unexpected[:5]}")

    result = {
        "strict_load_pass": True,
        "state_dict_key_count": len(state),
        "state_tensor_key_count": len(state_tensor_keys),
        "state_numel": state_numel,
        "model_parameter_numel": model_numel,
        "embed_dim": int(model.embed_dim),
        "patch_size": [14, 14],
        "register_tokens": int(getattr(model, "num_register_tokens", 0)),
        "meta_device_validation": True,
    }
    print(f"[compat] {key}: PASS keys={len(state)} model_numel={model_numel}", flush=True)
    del state, model
    gc.collect()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    root = Path(args.corpus_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    weights_dir = outdir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("[DINO-WEIGHT-AUTH] ZERO-STEP WEIGHT + PREPROCESS AUTHORITY", flush=True)
    print("[DINO-WEIGHT-AUTH] feature extraction: FORBIDDEN IN THIS RUN", flush=True)
    print("[DINO-WEIGHT-AUTH] training: FORBIDDEN IN THIS RUN", flush=True)
    print("=" * 96, flush=True)

    authority = verify_train512_authority(root)
    versions = runtime_versions()
    print("[runtime]", json.dumps(versions, indent=2, sort_keys=True), flush=True)

    src = clone_exact_dino(outdir)
    backbones, derived_urls = import_source_and_derive_urls(src)
    print("[source] exact revision PASS", flush=True)

    fixture = preprocess_fixture(root, outdir)

    candidates = {}
    for key in ["S", "B", "L", "g"]:
        spec = dict(CANDIDATES[key])
        url = derived_urls[key]
        filename = url.rsplit("/", 1)[-1]
        pth = weights_dir / filename

        print("=" * 96, flush=True)
        print(f"[candidate {key}] START", flush=True)
        print("=" * 96, flush=True)
        download_resume(url, pth)
        raw_sha = sha256_file(pth)
        size = int(pth.stat().st_size)
        print(f"[candidate {key}] raw_sha256={raw_sha}", flush=True)
        print(f"[candidate {key}] size_bytes={size}", flush=True)

        compat = verify_state_dict_meta(backbones, key, spec, pth)
        candidates[key] = {
            **spec,
            "official_weight_url": url,
            "cached_path": str(pth.relative_to(outdir)),
            "raw_pth_sha256": raw_sha,
            "size_bytes": size,
            "compatibility": compat,
            "candidate_features_extracted": False,
        }

    result = {
        "schema": SCHEMA,
        "status": "PASS_WEIGHT_BYTES_SOURCE_COMPAT_PREPROCESS_SEALED__FEATURE_OUTPUTS_UNOPENED",
        "finished_utc": utc_now(),
        "scientific_optimizer_steps": 0,
        "dino_training_started": False,
        "candidate_features_extracted": False,
        "train512_authority": authority,
        "source_authority": {
            "repository": "facebookresearch/dinov2",
            "revision": DINO_REV,
            "revision_verified": True,
            "base_weight_url_derived_from_exact_source": True,
        },
        "runtime_versions": versions,
        "preprocess_fixture": fixture,
        "candidates": candidates,
        "firewalls": {
            "forward_or_forward_features_called": False,
            "feature_cache_written": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "candidate_specific_preprocess": False,
        },
        "next_gate": "ONLINE_VS_CACHE_TOKEN_PARITY__THEN_SHARED_ARCHITECTURE_AND_SAMPLE_STREAM_SEAL",
    }
    result["content_sha256"] = canonical_sha(result)
    atomic_json(outdir / "DINO_WEIGHT_AUTHORITY_V1.json", result)

    print("=" * 96, flush=True)
    print("[DINO-WEIGHT-AUTH] PASS", flush=True)
    print("[DINO-WEIGHT-AUTH] result:", outdir / "DINO_WEIGHT_AUTHORITY_V1.json", flush=True)
    for key in ["S", "B", "L", "g"]:
        c = candidates[key]
        print(f"[DINO-WEIGHT-AUTH] {key} sha256={c['raw_pth_sha256']} bytes={c['size_bytes']}", flush=True)
    print("[DINO-WEIGHT-AUTH] candidate_features_extracted = false", flush=True)
    print("[DINO-WEIGHT-AUTH] scientific_optimizer_steps = 0", flush=True)
    print("[DINO-WEIGHT-AUTH] training = NOT AUTHORIZED BY THIS RUN", flush=True)
    print("=" * 96, flush=True)

if __name__ == "__main__":
    main()
