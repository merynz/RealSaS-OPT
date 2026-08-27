#!/usr/bin/env python3
"""Operationally hardened preparation of compact D0/D1/D2 downstream proxy packs.

Scientific behavior is identical to run_e0_downstream_proxy_prep_v1.py:
- same frozen split;
- same source-hash checks;
- same exact build_compact_pack(..., anchor_count=512);
- same D0/D1/D2 semantics and MUTUAL_P003 admission.

V1.1 changes only Drive I/O: all JSON and NPZ artifacts are built locally first,
then published to Drive with direct write + readback SHA verification + retry.
No Drive-side tmp.rename/Path.replace operation is used.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from pathlib import Path
import numpy as np

from e0_downstream_proxy_v1 import build_compact_pack, save_pack, sha256_file

TRANSIENT_ERRNOS = {5, 32, 103, 107, 116}
CODE_VERSION = "E0_DOWNSTREAM_PROXY_PREP_V1_1_FUSE_DIRECT_WRITE"


def retry(label, fn, attempts=6):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except OSError as e:
            last = e
            eno = getattr(e, "errno", None)
            if eno not in TRANSIENT_ERRNOS or i == attempts:
                raise
            print(f"[retry] {label} {i}/{attempts}: {e!r}", flush=True)
            time.sleep(min(1.5 * i, 5.0))
    raise last


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_local_json(path: Path, obj):
    """Atomic rename is allowed only on local /content work paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise IOError(5, f"local JSON hash mismatch {path}")


def remote_read_bytes(path: Path) -> bytes:
    return retry(f"read {path}", path.read_bytes)


def remote_sha256(path: Path) -> str:
    return hashlib.sha256(remote_read_bytes(path)).hexdigest()


def publish_bytes_verified(data: bytes, dst: Path) -> str:
    """Drive/FUSE-safe publication: direct write; never tmp.replace on Drive."""
    sha = hashlib.sha256(data).hexdigest()

    def _write_and_verify():
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately direct: Drive FUSE rename/replace is the V1 failure surface.
        dst.write_bytes(data)
        got = hashlib.sha256(dst.read_bytes()).hexdigest()
        if got != sha:
            raise IOError(5, f"post-write hash mismatch {dst}: {got} != {sha}")
        return got

    got = retry(f"publish {dst}", _write_and_verify)
    if got != sha:
        raise RuntimeError(f"publish verification drift {dst}")
    return sha


def publish_file_verified(src: Path, dst: Path) -> str:
    data = src.read_bytes()
    return publish_bytes_verified(data, dst)


def publish_json_verified(obj, local_path: Path, dst: Path) -> str:
    write_local_json(local_path, obj)
    return publish_file_verified(local_path, dst)


def copy_verified(src: Path, dst: Path, expected_sha: str | None = None):
    data = retry(f"read source {src}", src.read_bytes)
    got = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and got != expected_sha:
        raise RuntimeError(f"source hash mismatch {src}: {got} != {expected_sha}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    if sha256_file(dst) != got:
        raise IOError(5, f"local copy hash mismatch {dst}")
    return got


def source_hashes_for(cache_by_id: dict, aid: str):
    r = cache_by_id.get(aid)
    if r is None:
        return None
    return {"geometry_sha256": r["geometry_sha256"], "raster_sha256": dict(r["raster_sha256"])}


def stage_asset(root: Path, aid: str, stage: Path, expected: dict | None):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    src = root / "master" / "assets" / aid
    if not src.is_dir():
        raise FileNotFoundError(src)
    geom_exp = expected["geometry_sha256"] if expected else None
    gh = copy_verified(src / "primary_geometry.npz", stage / "primary_geometry.npz", geom_exp)
    rh = {}
    for v in range(8):
        exp = expected["raster_sha256"][f"V{v}"] if expected else None
        rh[f"V{v}"] = copy_verified(
            src / "renders" / f"V{v}" / "raster_authority.npz",
            stage / "renders" / f"V{v}" / "raster_authority.npz",
            exp,
        )
    return {"geometry_sha256": gh, "raster_sha256": rh}


def try_resume(pp: Path, rp: Path, builder_sha: str):
    try:
        if not pp.is_file() or not rp.is_file():
            return None
        old = json.loads(remote_read_bytes(rp).decode("utf-8"))
        if old.get("builder_source_sha256") != builder_sha:
            return None
        if old.get("pack_sha256") != remote_sha256(pp):
            return None
        return old
    except OSError as e:
        # A transient resume read failure must not silently accept stale data.
        print(f"[resume-read-failed] {pp.name}: {e!r}; rebuild this asset", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/content/e0_downstream_proxy_prep_v1_1")
    ap.add_argument("--builder-source-sha256", required=True)
    a = ap.parse_args()

    root = Path(a.root)
    split = json.loads(Path(a.split_json).read_text())
    cache = json.loads(Path(a.cache_manifest).read_text())
    cache_by_id = {r["asset_id"]: r for r in cache["records"]}

    out = Path(a.out)
    packs = out / "packs"
    reports = out / "reports"
    work = Path(a.work)
    local_artifacts = work / "local_artifacts"
    local_packs = local_artifacts / "packs"
    local_reports = local_artifacts / "reports"
    # Drive dirs may already exist from V1; direct publication uses them.
    packs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    local_packs.mkdir(parents=True, exist_ok=True)
    local_reports.mkdir(parents=True, exist_ok=True)

    ids = list(split["train_ids"]) + list(split["selection_ids"]) + list(split["calibration_ids"])
    if len(ids) != len(set(ids)):
        raise RuntimeError("split duplicate/overlap")
    trainset = set(split["train_ids"]) | set(split["selection_ids"])

    print("PREP V1.1 POPULATION", len(split["train_ids"]), len(split["selection_ids"]), len(split["calibration_ids"]), flush=True)
    print("FIT ONLY | PROXY32 CLOSED | DEV32 CLOSED | DRIVE DIRECT-WRITE VERIFIED", flush=True)

    index = []
    for i, aid in enumerate(ids, 1):
        pp = packs / f"{aid}.npz"
        rp = reports / f"{aid}.json"
        old = try_resume(pp, rp, a.builder_source_sha256)
        if old is not None:
            print(f"[PREP] {i}/{len(ids)} {aid} RESUME", flush=True)
            index.append(old)
            continue

        if aid in trainset and aid not in cache_by_id:
            raise RuntimeError(f"{aid}: frozen train/selection asset missing from cache manifest")
        expected = source_hashes_for(cache_by_id, aid) if aid in trainset else None

        # Builder seeds are namespaced by asset_dir.name; preserve canonical ID.
        stage = work / "assets" / aid
        print(f"[PREP] {i}/{len(ids)} {aid} BUILD", flush=True)
        hashes = stage_asset(root, aid, stage, expected)
        if stage.name != aid:
            raise RuntimeError(f"staged asset identity drift: {stage.name} != {aid}")

        pack = build_compact_pack(stage, anchor_count=512)
        if str(np.asarray(pack["asset_id"]).item()) != aid:
            raise RuntimeError(f"builder asset-id drift for {aid}")

        local_pack = local_packs / f"{aid}.npz"
        save_pack(local_pack, pack)
        local_pack_sha = sha256_file(local_pack)

        report = {
            "schema": "RealSaS.E0.DownstreamCompactPackRecord.v1",
            "asset_id": aid,
            "builder_source_sha256": a.builder_source_sha256,
            "pack_sha256": local_pack_sha,
            "source_hashes": hashes,
            "D0_mean_support": float(np.mean(pack["D0_support_count"])),
            "D1_mean_support": float(np.mean(pack["D1_support_count"])),
            "D2_mean_support": float(np.mean(pack["D2_support_count"])),
            "original_legal_control_count": int(pack["original_legal_control_count"]),
            "probe_joint_count": int(pack["probe_joint_count"]),
            "D0_valid_skin_points": int(pack["D0_skin_valid"].sum()),
            "D1_valid_skin_points": int(pack["D1_skin_valid"].sum()),
            "D2_valid_skin_points": int(pack["D2_skin_valid"].sum()),
            "teacher_identity_consumed_by_D2_admission": False,
            "mutual_cycle_threshold_P": 0.003,
            "prep_operational_version": CODE_VERSION,
        }
        local_report = local_reports / f"{aid}.json"
        write_local_json(local_report, report)

        # Publish pack first, report second. A report is the resume commit record.
        print(f"[PUBLISH] {i}/{len(ids)} {aid} pack", flush=True)
        psha = publish_file_verified(local_pack, pp)
        if psha != local_pack_sha:
            raise RuntimeError(f"published pack SHA drift {aid}")
        print(f"[PUBLISH] {i}/{len(ids)} {aid} report", flush=True)
        publish_file_verified(local_report, rp)
        index.append(report)

        shutil.rmtree(stage, ignore_errors=True)
        local_pack.unlink(missing_ok=True)
        local_report.unlink(missing_ok=True)

    idx = {
        "schema": "RealSaS.E0.DownstreamProxyPrepIndex.v1",
        "status": "FIT_PREP_COMPLETE__PROXY32_DEV32_CLOSED",
        "prep_operational_version": CODE_VERSION,
        "builder_source_sha256": a.builder_source_sha256,
        "split_raw_sha256": sha256_file(a.split_json),
        "record_count": len(index),
        "train_count": len(split["train_ids"]),
        "selection_count": len(split["selection_ids"]),
        "calibration_count": len(split["calibration_ids"]),
        "teacher_identity_consumed_by_D2_admission": False,
        "records": index,
    }
    local_index = local_artifacts / "E0_DOWNSTREAM_PROXY_PREP_INDEX_V1.json"
    write_local_json(local_index, idx)
    ip = out / "E0_DOWNSTREAM_PROXY_PREP_INDEX_V1.json"
    isha = publish_file_verified(local_index, ip)
    print("PREP COMPLETE", len(index), "INDEX_SHA256", isha, flush=True)


if __name__ == "__main__":
    main()
