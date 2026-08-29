#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

SCHEMA = "RealSaS.DINO.ApparatusFreezeResult.v1"
SEED = 20260825
EXPECTED_MODEL_SHA = "af820b2e739253e716b03399fb1add355412027cf5e5da7409a6eec8b8a0a41f"
EXPECTED_SPEC_SHA = "c66560939e41f38e8f191299a610447f2216049dbfb7c64e61737be28f440b47"
EXPECTED_MEMBERSHIP_SHA = "ae1edcaad47cf4867f11d51305eb57db06acb63ef13fc60ca3263c043ef8f71c"
EXPECTED_SCHEDULE_FILE_SHA = "06aae8a811d8affec2c11e5273c3fb6016f72331a117f32392194043aa4de36c"
EXPECTED_SCHEDULE_DATA_SHA = "2aca93a8840199785cb4392f52e5ef449b26d31cb52a10473e254aa59a4517bd"
EXPECTED_WEIGHT_SEAL_SHA = "b829a176405b4773026704d8e9f5645b35640a1906b7ebae641b59c9798245b1"
EXPECTED_CORRECTION_SHA = "a43560c06e03ca1b3191f0d449be35893eda980bd0721911bec60ddb47892ba2"
EXPECTED_PARAM_COUNT = 5_046_739
EXPECTED_TRAIN512_SET_SHA = "1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2"
EXPECTED_TRAIN_ORDER_HASH = "5e0d84c8a54d61b6c5bc5dcb814e9e89129db8f82531b972781503ad3052dbf7"
EXPECTED_PROXY_ORDER_HASH = "673c35077ee071bd3a5e74029cb8d763ce9bf02acb6d39cc118fa7458a3da548"
EXPECTED_DIAG_ORDER_HASH = "5e2904b384ae923c50e4632e6c9a715e8efc23ebc4f9ee42c04d1b9ceceb4806"
EXPECTED_FIRST_ROWS = [
    [76,246,112,23,391,78,19,292],
    [338,194,437,99,441,172,102,272],
    [129,128,213,388,155,61,68,168],
]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def canonical_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def ordered_id_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("utf-8")).hexdigest()

def tensor_state_sha256(state_dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        a = t.numpy()
        h.update(k.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("ascii") + b"\0")
        h.update(np.asarray(t.shape, dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def import_model(path: Path):
    name = "realsas_dino_shared_depth_learner_v1_authority"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("MODEL_IMPORT_SPEC_FAILED")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def require_file_sha(path: Path, expected: str, label: str):
    if not path.is_file():
        raise RuntimeError(f"{label}_MISSING: {path}")
    got = sha256_file(path)
    print(f"[sha] {label}: {got}", flush=True)
    if got != expected:
        raise RuntimeError(f"{label}_SHA_DRIFT expected={expected} actual={got}")
    return got

def check_train512_authority(root: Path):
    p = root / "reports" / "post_corpus_audit" / "DINO_TRAIN512_NATIVE1024_AUTHORITY_V2.json"
    if not p.is_file():
        raise RuntimeError(f"TRAIN512_AUTHORITY_MISSING: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    checks = {
        "status": d.get("status") == "PASS_512_OF_512",
        "asset_count": d.get("train512_asset_count") == 512,
        "ready_count": d.get("train512_ready_count") == 512,
        "set_sha": d.get("train512_set_sha256") == EXPECTED_TRAIN512_SET_SHA,
        "optimizer_zero": d.get("scientific_optimizer_steps") == 0,
        "not_training_authority": d.get("dino_training_authorized_by_this_artifact") is False,
    }
    print("[train512]", checks, flush=True)
    if not all(checks.values()):
        raise RuntimeError("TRAIN512_AUTHORITY_DRIFT")
    return {"path": str(p.relative_to(root)), "raw_sha256": sha256_file(p), "checks": checks}

def check_weight_authority(root: Path):
    p = root / "reports" / "dino_controlled_preflight" / "DINO_WEIGHT_AUTHORITY_V1" / "DINO_WEIGHT_AUTHORITY_V1.json"
    if not p.is_file():
        raise RuntimeError(f"WEIGHT_AUTHORITY_MISSING: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "PASS_WEIGHT_BYTES_SOURCE_COMPAT_PREPROCESS_SEALED__FEATURE_OUTPUTS_UNOPENED":
        raise RuntimeError("WEIGHT_AUTHORITY_NOT_PASS")
    if d.get("candidate_features_extracted") is not False:
        raise RuntimeError("FEATURE_OUTPUT_FIREWALL_ALREADY_OPEN")
    if d.get("dino_training_started") is not False or d.get("scientific_optimizer_steps") != 0:
        raise RuntimeError("WEIGHT_AUTHORITY_TRAINING_FIREWALL_BREACH")
    return {"path": str(p.relative_to(root)), "raw_sha256": sha256_file(p), "content_sha256": d.get("content_sha256"), "candidate_features_extracted": False}

def audit_one_asset(root: Path, asset_id: str):
    a = root / "master" / "assets" / asset_id
    if not a.is_dir():
        raise RuntimeError(f"{asset_id}:asset_dir_missing")
    for rel in ["primary_geometry.npz", "RENDER_COMPLETE.json"]:
        p = a / rel
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"{asset_id}:{rel}:missing")
    views = []
    for vi in range(8):
        vd = a / "renders" / f"V{vi}"
        cam_p = vd / "camera.json"
        ras_p = vd / "raster_authority.npz"
        cel_p = vd / "cel_clean.png"
        ink_p = vd / "ink_cel.png"
        for p in [cam_p, ras_p, cel_p, ink_p]:
            if not p.is_file() or p.stat().st_size <= 0:
                raise RuntimeError(f"{asset_id}:V{vi}:{p.name}:missing")
        cam = json.loads(cam_p.read_text(encoding="utf-8"))
        if int(round(float(cam.get("yaw_deg", -999)))) != vi * 45:
            raise RuntimeError(f"{asset_id}:V{vi}:yaw_drift")
        if cam.get("image_origin") != "TOP_LEFT" or int(cam.get("vertical_flip_count", -1)) != 1:
            raise RuntimeError(f"{asset_id}:V{vi}:camera_origin_drift")
        with np.load(ras_p, allow_pickle=False) as z:
            if "pixel_linear_index" not in z.files:
                raise RuntimeError(f"{asset_id}:V{vi}:pixel_authority_missing")
            pix = np.asarray(z["pixel_linear_index"], np.int64)
            if len(pix) == 0:
                raise RuntimeError(f"{asset_id}:V{vi}:blank_raster")
            if len(pix) != len(np.unique(pix)):
                raise RuntimeError(f"{asset_id}:V{vi}:duplicate_pixel_authority")
            if np.any(pix < 0) or np.any(pix >= 1024 * 1024):
                raise RuntimeError(f"{asset_id}:V{vi}:pixel_authority_oob")
            if "resolution" in z.files:
                rr = np.asarray(z["resolution"]).reshape(-1).tolist()
                if rr != [1024, 1024]:
                    raise RuntimeError(f"{asset_id}:V{vi}:resolution_drift:{rr}")
        for p in [cel_p, ink_p]:
            with Image.open(p) as im:
                if im.size != (1024, 1024):
                    raise RuntimeError(f"{asset_id}:V{vi}:{p.name}:not_native1024:{im.size}")
        views.append({"view": vi, "authority_rows": int(len(pix))})
    return {"asset": asset_id, "pass": True, "views": views}

def audit_population(root: Path, label: str, ids):
    out = []
    for i, aid in enumerate(ids, 1):
        out.append(audit_one_asset(root, aid))
        if i % 8 == 0 or i == len(ids):
            print(f"[availability] {label}: {i}/{len(ids)}", flush=True)
    return {"label": label, "count": len(ids), "pass_count": len(out), "pass": True, "assets": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--authority-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    root = Path(args.corpus_root)
    auth = Path(args.authority_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_p = auth / "dino_shared_depth_learner_v1.py"
    spec_p = auth / "DINO_SHARED_LEARNER_APPARATUS_SPEC_V1.json"
    memb_p = auth / "DINO_MEMBERSHIP_SAMPLE_STREAM_AUTHORITY_V1.json"
    sched_p = auth / "DINO_SAMPLE_ORDER_INDICES_V1.npy"
    weight_seal_p = auth / "DINO_WEIGHT_AUTHORITY_V1_SEAL.json"
    correction_p = auth / "DINO_ZERO_STEP_ORDERING_CORRECTION_V1.md"

    print("=" * 96, flush=True)
    print("[DINO-APPARATUS] ZERO-STEP FREEZE", flush=True)
    print("[DINO-APPARATUS] candidate DINO feature outputs: FORBIDDEN", flush=True)
    print("[DINO-APPARATUS] optimizer construction: FORBIDDEN", flush=True)
    print("[DINO-APPARATUS] training: FORBIDDEN", flush=True)
    print("=" * 96, flush=True)

    train512 = check_train512_authority(root)
    weight_auth = check_weight_authority(root)

    require_file_sha(model_p, EXPECTED_MODEL_SHA, "MODEL")
    require_file_sha(spec_p, EXPECTED_SPEC_SHA, "APPARATUS_SPEC")
    require_file_sha(memb_p, EXPECTED_MEMBERSHIP_SHA, "MEMBERSHIP")
    require_file_sha(sched_p, EXPECTED_SCHEDULE_FILE_SHA, "SCHEDULE_NPY")
    require_file_sha(weight_seal_p, EXPECTED_WEIGHT_SEAL_SHA, "WEIGHT_SEAL")
    require_file_sha(correction_p, EXPECTED_CORRECTION_SHA, "ORDERING_CORRECTION")

    spec_d = json.loads(spec_p.read_text(encoding="utf-8"))
    memb = json.loads(memb_p.read_text(encoding="utf-8"))
    wseal = json.loads(weight_seal_p.read_text(encoding="utf-8"))
    if spec_d.get("feature_outputs_opened") is not False or spec_d.get("training_authorized") is not False:
        raise RuntimeError("APPARATUS_SPEC_FIREWALL_DRIFT")
    if wseal.get("candidate_features_extracted") is not False or wseal.get("scientific_optimizer_steps") != 0:
        raise RuntimeError("WEIGHT_SEAL_FIREWALL_DRIFT")

    train_ids = list(memb["train_order_512"])
    proxy_ids = list(memb["proxy_32"])
    diag_ids = list(memb["train_diag_32"])
    if len(train_ids) != 512 or len(set(train_ids)) != 512:
        raise RuntimeError("TRAIN_ORDER_NOT_512_UNIQUE")
    if len(proxy_ids) != 32 or len(set(proxy_ids)) != 32:
        raise RuntimeError("PROXY_NOT_32_UNIQUE")
    if len(diag_ids) != 32 or len(set(diag_ids)) != 32:
        raise RuntimeError("DIAG_NOT_32_UNIQUE")
    if proxy_ids and set(proxy_ids) & set(train_ids):
        raise RuntimeError("PROXY_TRAIN_LEAK")
    if diag_ids != train_ids[:32]:
        raise RuntimeError("TRAIN_DIAG32_NOT_NESTED_FIRST32")
    if ordered_id_hash(train_ids) != EXPECTED_TRAIN_ORDER_HASH:
        raise RuntimeError("TRAIN_ORDER_HASH_DRIFT")
    if ordered_id_hash(proxy_ids) != EXPECTED_PROXY_ORDER_HASH:
        raise RuntimeError("PROXY_ORDER_HASH_DRIFT")
    if ordered_id_hash(diag_ids) != EXPECTED_DIAG_ORDER_HASH:
        raise RuntimeError("DIAG_ORDER_HASH_DRIFT")

    schedule = np.load(sched_p, allow_pickle=False)
    if schedule.dtype != np.dtype("<u2") or tuple(schedule.shape) != (32768, 8):
        raise RuntimeError(f"SCHEDULE_CONTRACT_DRIFT dtype={schedule.dtype} shape={schedule.shape}")
    if hashlib.sha256(np.ascontiguousarray(schedule).tobytes()).hexdigest() != EXPECTED_SCHEDULE_DATA_SHA:
        raise RuntimeError("SCHEDULE_DATA_SHA_DRIFT")
    if schedule[:3].tolist() != EXPECTED_FIRST_ROWS:
        raise RuntimeError("SCHEDULE_FIRST_ROWS_DRIFT")
    counts = np.bincount(schedule.reshape(-1), minlength=512)
    if int(counts.min()) != 512 or int(counts.max()) != 512:
        raise RuntimeError(f"SCHEDULE_EXPOSURE_DRIFT min={counts.min()} max={counts.max()}")
    print("[sample-stream] PASS shape=32768x8 exposure=512/family", flush=True)

    mod = import_model(model_p)
    init_hashes = {}
    init_file = out / "DINO_SHARED_INIT_V1.pt"
    first_state = None
    for label in ["S", "B", "L", "g"]:
        torch.manual_seed(SEED)
        model = mod.DINOSharedDepthLearnerV1()
        params = int(sum(p.numel() for p in model.parameters()))
        if params != EXPECTED_PARAM_COUNT:
            raise RuntimeError(f"PARAM_COUNT_DRIFT {params}")
        sh = tensor_state_sha256(model.state_dict())
        init_hashes[label] = sh
        if first_state is None:
            first_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
        gc.collect()
    if len(set(init_hashes.values())) != 1:
        raise RuntimeError(f"CANDIDATE_INIT_STATE_DRIFT {init_hashes}")
    torch.save({"schema": "RealSaS.DINO.SharedInit.v1", "seed": SEED, "model_file_sha256": EXPECTED_MODEL_SHA, "trainable_parameter_count": EXPECTED_PARAM_COUNT, "canonical_tensor_state_sha256": next(iter(init_hashes.values())), "state_dict": first_state}, init_file)
    init_file_sha = sha256_file(init_file)
    print("[init] PASS state_sha=", next(iter(init_hashes.values())), flush=True)
    print("[init] serialized_sha=", init_file_sha, flush=True)

    torch.manual_seed(73)
    q_tests = {}
    for label, native_dim in [("S",384),("B",768),("L",1024),("g",1536)]:
        x = torch.randn(7, native_dim, dtype=torch.float32)
        y = mod.lift_and_normalize_tokens(x, native_dim)
        if tuple(y.shape) != (7, 1536):
            raise RuntimeError(f"{label}_QD_SHAPE_DRIFT")
        tail_exact = True if native_dim == 1536 else bool(torch.count_nonzero(y[:, native_dim:]).item() == 0)
        norm_max_abs = float(torch.max(torch.abs(torch.linalg.vector_norm(y, dim=-1) - 1.0)).item())
        if not tail_exact or norm_max_abs > 2e-6:
            raise RuntimeError(f"{label}_QD_L2_FAIL tail={tail_exact} norm={norm_max_abs}")
        q_tests[label] = {"native_dim": native_dim, "zero_tail_exact": tail_exact, "l2_norm_max_abs": norm_max_abs}
    print("[Q_d] synthetic contract PASS", flush=True)

    proxy_audit = audit_population(root, "FIT_PROXY32", proxy_ids)
    diag_audit = audit_population(root, "TRAIN_DIAG32", diag_ids)

    result = {
        "schema": SCHEMA,
        "status": "PASS_ARCH_INIT_SAMPLE_CACHE_DATA__DTB_CARRIER_PARITY_PENDING",
        "finished_utc": utc_now(),
        "candidate_features_extracted": False,
        "dino_training_started": False,
        "scientific_optimizer_steps": 0,
        "optimizer_constructed": False,
        "dev32_opened": False,
        "train512_authority": train512,
        "weight_authority": weight_auth,
        "authority_input_hashes": {"model": EXPECTED_MODEL_SHA, "apparatus_spec": EXPECTED_SPEC_SHA, "membership": EXPECTED_MEMBERSHIP_SHA, "schedule_npy": EXPECTED_SCHEDULE_FILE_SHA, "schedule_data": EXPECTED_SCHEDULE_DATA_SHA, "weight_seal": EXPECTED_WEIGHT_SEAL_SHA, "ordering_correction": EXPECTED_CORRECTION_SHA},
        "shared_learner": {"trainable_parameter_count": EXPECTED_PARAM_COUNT, "candidate_init_tensor_hashes": init_hashes, "common_init_tensor_sha256": next(iter(init_hashes.values())), "serialized_init_path": str(init_file.relative_to(out)), "serialized_init_sha256": init_file_sha},
        "q_d_synthetic_contract": q_tests,
        "sample_stream": {"shape": [32768, 8], "family_exposure_min": int(counts.min()), "family_exposure_max": int(counts.max()), "first_three_rows": schedule[:3].tolist(), "data_sha256": EXPECTED_SCHEDULE_DATA_SHA},
        "membership": {"train512_count": len(train_ids), "proxy32_count": len(proxy_ids), "train_diag32_count": len(diag_ids), "proxy_train_disjoint": True, "train_diag32_is_train_first32": True, "ordered_hashes": {"train512": EXPECTED_TRAIN_ORDER_HASH, "proxy32": EXPECTED_PROXY_ORDER_HASH, "train_diag32": EXPECTED_DIAG_ORDER_HASH}},
        "native1024_availability": {"fit_proxy32": proxy_audit, "train_diag32": diag_audit},
        "cache_semantics": spec_d["cache"],
        "optimizer_loss_semantics": spec_d["training"],
        "dtb_nd1_binding": spec_d["dtb_nd1_binding"],
        "next_gate": "DTB_ND1_PREDICTED_DEPTH_CARRIER_SEMANTIC_PARITY__NO_DINO_FEATURE_OUTPUTS",
        "training_authorized_by_this_artifact": False,
    }
    result["content_sha256"] = canonical_sha(result)
    atomic_json(out / "DINO_APPARATUS_FREEZE_V1_RESULT.json", result)

    print("=" * 96, flush=True)
    print("[DINO-APPARATUS] COMPLETE: PASS_ARCH_INIT_SAMPLE_CACHE_DATA__DTB_CARRIER_PARITY_PENDING", flush=True)
    print("[DINO-APPARATUS] proxy32 native1024 = 32/32", flush=True)
    print("[DINO-APPARATUS] train_diag32 native1024 = 32/32", flush=True)
    print("[DINO-APPARATUS] candidate_features_extracted = false", flush=True)
    print("[DINO-APPARATUS] scientific_optimizer_steps = 0", flush=True)
    print("[DINO-APPARATUS] training = NOT AUTHORIZED", flush=True)
    print("=" * 96, flush=True)

if __name__ == "__main__":
    main()
