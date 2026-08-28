#!/usr/bin/env python3
"""Frozen one-shot E0 Proxy27 qualification.

This runner does NOT train or retune any consumer. It builds compact D0/D1/D2
packs for the already-frozen 27-member truth-capable Proxy32 intersection using
the exact frozen V1.2 builder, loads the sealed V1.3 Arachne/Geppetto checkpoints,
and evaluates the already-frozen non-inferiority rule.

DEV32 remains closed. Diagnostic outcomes and non-binding expectations cannot
alter PASS/FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from e0_downstream_proxy_v1 import build_compact_pack, load_pack, save_pack, sha256_file
from run_e0_downstream_proxy_calibration_v1 import eval_arachne, eval_geppetto
from e0_downstream_proxy_v1 import make_arachne_model, make_geppetto_model

ARMS = ("D0", "D1", "D2")
CONTRACT_SHA256 = "cf204ad1ef7d8460e402fb6c3db7361d122b3284116aa6e840af032d0fb39a2a"
EXPECTED_INTERSECTION_SHA256 = "fcbdd90d585a3845b8f799fbd1ba6dff526d4714aa581a7f6d301da86d2ecbd9"
EXPECTED_INTERSECTION_IDS_SHA256 = "b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817"
EXPECTED_MARGIN_GIT_BLOB_SHA = "cb723ca9b2c500ad098ac6c4ecc795d3e857e906"
EXPECTED_CALIBRATION_SHA256 = "82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9"
EXPECTED_DIAGNOSTIC_SHA256 = "906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d"
EXPECTED_CHECKPOINTS = {
    "ARACHNE_D0_BEST.pt": "e21e9e7f0a65b1c18a4db0327c0bead708cceea6740b9b9bf9946f0a169b80ce",
    "ARACHNE_D1_BEST.pt": "fedc543d6ac3f1ea067e9ce4fccbaa3c3522a428bc89dc21ee0165f0b9fd088c",
    "ARACHNE_D2_BEST.pt": "72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745",
    "GEPPETTO_D0_BEST.pt": "ff993aeea8f92d2e0ae70d40605e4089f416c7de14f4e682983ec3e0af6d4cde",
    "GEPPETTO_D1_BEST.pt": "f1dec9dd8ea623c36c1de1ac31fd9bcba76a510af460088680ab0f38abf75a6d",
    "GEPPETTO_D2_BEST.pt": "f8c6146fc3ad81146ced01805b9be454ad194b86db3a9ab3d72d7b9eb3747b65",
}
TRANSIENT_ERRNOS = {5, 32, 103, 107, 116}


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_local_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise IOError(5, f"local JSON write mismatch: {path}")


def retry(label, fn, attempts=6):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except OSError as e:
            last = e
            if getattr(e, "errno", None) not in TRANSIENT_ERRNOS or i == attempts:
                raise
            print(f"[retry] {label} {i}/{attempts}: {e!r}", flush=True)
            time.sleep(min(1.5 * i, 5.0))
    raise last


def copy_verified(src: Path, dst: Path, expected_sha: str | None = None) -> str:
    data = retry(f"read {src}", src.read_bytes)
    got = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and got != expected_sha:
        raise RuntimeError(f"SHA mismatch {src}: {got} != {expected_sha}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    if sha256_file(dst) != got:
        raise IOError(5, f"copy verification failed: {dst}")
    return got


def publish_verified(src: Path, dst: Path) -> str:
    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    def _write():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        got = hashlib.sha256(dst.read_bytes()).hexdigest()
        if got != sha:
            raise IOError(5, f"Drive readback SHA mismatch: {dst}: {got} != {sha}")
        return got
    got = retry(f"publish {dst}", _write)
    if got != sha:
        raise RuntimeError(f"publish verification drift: {dst}")
    return sha


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in ids).encode()).hexdigest()


def stage_asset(root: Path, aid: str, stage: Path) -> dict:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    src = root / "master" / "assets" / aid
    if not src.is_dir():
        raise FileNotFoundError(src)
    source = {"geometry_sha256": copy_verified(src / "primary_geometry.npz", stage / "primary_geometry.npz"),
              "raster_sha256": {}}
    for v in range(8):
        source["raster_sha256"][f"V{v}"] = copy_verified(
            src / "renders" / f"V{v}" / "raster_authority.npz",
            stage / "renders" / f"V{v}" / "raster_authority.npz",
        )
    return source


def load_models(checkpoint_root: Path, local_ckpt: Path, device: str, seal: dict):
    if seal.get("checkpoint_source") != "checkpoints_v1_3":
        raise RuntimeError("checkpoint seal is not V1.3 authority")
    if seal.get("result_sha256") != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("calibration result SHA authority drift")
    local_ckpt.mkdir(parents=True, exist_ok=True)
    models = {"arachne": {}, "geppetto": {}}
    for name, expected in EXPECTED_CHECKPOINTS.items():
        p = local_ckpt / name
        copy_verified(checkpoint_root / name, p, expected)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arm = ck["arm"]
        if ck.get("contract_sha256") != CONTRACT_SHA256:
            raise RuntimeError(f"checkpoint contract drift: {name}")
        if name.startswith("ARACHNE_"):
            if ck.get("schema") != "RealSaS.E0.ArachneProxy.v1":
                raise RuntimeError(f"checkpoint schema drift: {name}")
            m = make_arachne_model().to(device)
            models["arachne"][arm] = m
        else:
            if ck.get("schema") != "RealSaS.E0.GeppettoIsolationProxy.v1":
                raise RuntimeError(f"checkpoint schema drift: {name}")
            m = make_geppetto_model().to(device)
            models["geppetto"][arm] = m
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
    return models


def ratio_checks(aggregates: dict, margin: dict) -> tuple[dict, bool]:
    checks = {}
    primary_limit = 1.0 + float(margin["primary_max_relative_degradation"])
    tail_limit = 1.0 + float(margin["geppetto_tail_max_relative_degradation"])
    specs = [
        ("arachne.ce", aggregates["arachne"], "ce", primary_limit),
        ("arachne.influence_disp_mean", aggregates["arachne"], "influence_disp_mean", primary_limit),
        ("geppetto.joint_mean", aggregates["geppetto"], "joint_mean", primary_limit),
        ("geppetto.family_p95", aggregates["geppetto"], "family_p95", tail_limit),
    ]
    for label, src, metric, limit in specs:
        vals = {a: float(src[a][metric]) for a in ARMS}
        for comp, num, den in (("D1/D0", "D1", "D0"), ("D2/D1", "D2", "D1"), ("D2/D0", "D2", "D0")):
            r = vals[num] / vals[den]
            checks[f"{label}:{comp}"] = {"value": r, "max": limit, "pass": bool(r <= limit)}
    p0 = aggregates["geppetto"]["D0"]
    p2 = aggregates["geppetto"]["D2"]
    for metric, key in (("pck_005", "pck_005_D2_minus_D0_min"), ("pck_008", "pck_008_D2_minus_D0_min")):
        delta = float(p2[metric]) - float(p0[metric])
        minimum = float(margin["pck_catastrophe_veto"][key])
        checks[f"geppetto.{metric}:D2-D0"] = {"value": delta, "min": minimum, "pass": bool(delta >= minimum)}
    all_pass = all(v["pass"] for v in checks.values())
    return checks, all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--intersection-json", required=True)
    ap.add_argument("--margin-json", required=True)
    ap.add_argument("--expectation-json", required=True)
    ap.add_argument("--checkpoint-seal", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/content/realsas_e0_proxy27_qualification_v1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.root)
    work = Path(args.work)
    out = Path(args.out)
    local = work / "local"
    packs = local / "packs"
    assets = work / "assets"
    local_ckpt = local / "checkpoints"
    for p in (local, packs, assets, local_ckpt): p.mkdir(parents=True, exist_ok=True)

    result_drive = out / "E0_PROXY27_QUALIFICATION_RESULT_V1.json"
    if result_drive.is_file():
        raise RuntimeError(f"qualification result already exists; refusing second open: {result_drive}")

    intersection_path = Path(args.intersection_json)
    margin_path = Path(args.margin_json)
    expectation_path = Path(args.expectation_json)
    seal_path = Path(args.checkpoint_seal)
    if sha256_file(intersection_path) != EXPECTED_INTERSECTION_SHA256:
        raise RuntimeError("Proxy27 intersection bytes drift")
    intersection = json.loads(intersection_path.read_text())
    margin = json.loads(margin_path.read_text())
    expectation = json.loads(expectation_path.read_text())
    seal = json.loads(seal_path.read_text())
    ids = [x["asset_id"] for x in intersection["truth_capable"]]
    if len(ids) != 27 or intersection.get("truth_capable_count") != 27:
        raise RuntimeError("Proxy27 count drift")
    if ids_sha(ids) != EXPECTED_INTERSECTION_IDS_SHA256:
        raise RuntimeError("Proxy27 ordered-ID SHA drift")
    if margin.get("calibration_result_sha256") != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("margin calibration authority drift")
    if margin.get("margin_changes_after_proxy32_open") != "FORBIDDEN":
        raise RuntimeError("margin firewall drift")
    if expectation.get("binding_to_qualification") is not False:
        raise RuntimeError("expectation record must be explicitly non-binding")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested for frozen qualification but unavailable")
    device = args.device if args.device.startswith("cuda") else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(1862); np.random.seed(1862); torch.manual_seed(1862)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1862)

    print("E0 PROXY27 ONE-SHOT QUALIFICATION")
    print("PROXY27 COUNT", len(ids))
    print("DEV32 CLOSED | NO TRAINING | MARGINS FROZEN | EXPECTATION NON-BINDING")

    records = []
    for i, aid in enumerate(ids, 1):
        print(f"[PACK] {i}/27 {aid}", flush=True)
        stage = assets / aid
        source_hashes = stage_asset(root, aid, stage)
        pack = build_compact_pack(stage, anchor_count=512)
        if str(np.asarray(pack["asset_id"]).item()) != aid:
            raise RuntimeError(f"pack identity drift: {aid}")
        pp = packs / f"{aid}.npz"
        save_pack(pp, pack)
        counts = {arm: int(np.asarray(pack[f"{arm}_skin_valid"]).sum()) for arm in ARMS}
        records.append({
            "asset_id": aid,
            "source_hashes": source_hashes,
            "pack_sha256": sha256_file(pp),
            "arachne_valid_skin_points": counts,
            "original_legal_control_count": int(pack["original_legal_control_count"]),
            "probe_joint_count": int(pack["probe_joint_count"]),
        })
        shutil.rmtree(stage, ignore_errors=True)

    blocked = [r for r in records if min(r["arachne_valid_skin_points"].values()) <= 0]
    if blocked:
        result = {
            "schema": "RealSaS.E0.Proxy27Qualification.v1",
            "status": "BLOCKED_TARGET_AVAILABILITY__NO_DOWNSTREAM_METRICS_EVALUATED",
            "proxy27_downstream_opened": False,
            "dev32_opened": False,
            "population_count": 27,
            "blocked_assets": blocked,
            "pack_records": records,
            "qualification_margins_changed": False,
            "expectation_used_for_decision": False,
        }
        rp = local / "E0_PROXY27_QUALIFICATION_RESULT_V1.json"
        write_local_json(rp, result); publish_verified(rp, result_drive)
        raise RuntimeError("Proxy27 target-availability block; downstream metrics were not evaluated")

    print("TARGET SUPPORT PASS 27/27. OPENING FROZEN PROXY27 DOWNSTREAM ONCE.", flush=True)
    models = load_models(Path(args.checkpoint_root), local_ckpt, device, seal)
    evaluations = {"arachne": {}, "geppetto": {}}
    for arm in ARMS:
        evaluations["arachne"][arm] = eval_arachne(models["arachne"][arm], arm, ids, packs, device)
        evaluations["geppetto"][arm] = eval_geppetto(models["geppetto"][arm], arm, ids, packs, device)
        print("ARACHNE", arm, json.dumps(evaluations["arachne"][arm]["aggregate"], sort_keys=True))
        print("GEPPETTO", arm, json.dumps(evaluations["geppetto"][arm]["aggregate"], sort_keys=True))

    aggregates = {c: {a: evaluations[c][a]["aggregate"] for a in ARMS} for c in ("arachne", "geppetto")}
    checks, passed = ratio_checks(aggregates, margin)
    status = "E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS" if passed else "E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_FAIL"
    result = {
        "schema": "RealSaS.E0.Proxy27Qualification.v1",
        "status": status,
        "scope": "FROZEN_27_MEMBER_TRUTH_CAPABLE_PROXY32_INTERSECTION",
        "population_count": 27,
        "population_ids": ids,
        "population_ids_sha256": EXPECTED_INTERSECTION_IDS_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "intersection_sha256": EXPECTED_INTERSECTION_SHA256,
        "margin_semantic_authority_git_blob_sha": EXPECTED_MARGIN_GIT_BLOB_SHA,
        "margin_runtime_bytes_sha256": sha256_file(margin_path),
        "calibration_result_sha256": EXPECTED_CALIBRATION_SHA256,
        "preproxy_diagnostic_result_sha256": EXPECTED_DIAGNOSTIC_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "proxy27_downstream_opened": True,
        "dev32_opened": False,
        "qualification_margins_changed": False,
        "expectation_used_for_decision": False,
        "expectation_record_sha256": sha256_file(expectation_path),
        "aggregates": aggregates,
        "evaluations": evaluations,
        "gate_checks": checks,
        "all_gate_checks_pass": bool(passed),
        "pack_records": records,
    }
    rp = local / "E0_PROXY27_QUALIFICATION_RESULT_V1.json"
    write_local_json(rp, result)
    result_sha = publish_verified(rp, result_drive)
    seal_out = {
        "schema": "RealSaS.E0.Proxy27QualificationSeal.v1",
        "status": status,
        "result_sha256": result_sha,
        "proxy27_downstream_opened": True,
        "dev32_opened": False,
        "qualification_margins_changed": False,
        "expectation_used_for_decision": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "pack_count": 27,
        "pack_sha256": {r["asset_id"]: r["pack_sha256"] for r in records},
    }
    sp = local / "E0_PROXY27_QUALIFICATION_SEAL_V1.json"
    write_local_json(sp, seal_out)
    seal_sha = publish_verified(sp, out / "E0_PROXY27_QUALIFICATION_SEAL_V1.json")
    print("=== FROZEN QUALIFICATION DECISION ===")
    print(status)
    for k, v in checks.items(): print(k, json.dumps(v, sort_keys=True))
    print("RESULT_SHA256", result_sha)
    print("SEAL_SHA256", seal_sha)


if __name__ == "__main__":
    main()
