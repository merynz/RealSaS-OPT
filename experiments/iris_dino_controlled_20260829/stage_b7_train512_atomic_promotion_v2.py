#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image

SCHEMA = "RealSaS.StageB7Train512AtomicPromotion.v2"
BUILD_ID = "REALSAS_STAGE_B7_TRAIN512_ATOMIC_PROMOTION_V2_20260829"
ORIGINAL_BUILD_ID = "REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822"
ROOT_DEFAULT = "/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"
STAGING_REL = Path("reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_REPAIR_STAGING_V2")
B6_RESULT_REL = Path("reports/post_corpus_audit/POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_RESULT_V1.json")
BACKUP_REL = Path("reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_PROMOTION_BACKUP_V2")
RESULT_REL = Path("reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_PROMOTION_RESULT_V2.json")
TRAIN512_AUTH_REL = Path("reports/post_corpus_audit/DINO_TRAIN512_NATIVE1024_AUTHORITY_V2.json")
EXPECTED_STAGE_CONTENT_SHA = "13bee59ea268e232569e041710b52b28ba3900f98e676fc55c36b48ecba85447"
EXPECTED_MASTER_LEDGER_SHA = "475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913"
EXPECTED_TRAIN512_SET_SHA = "1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2"
NATIVE = 1024
REPAIR_ASSETS = (
    "asset_1d6d3b17fe506463b8840c51",
    "asset_4970fb8cc7c69df071970dc0",
    "asset_66c8c63c7a97e830b89d0ba8",
    "asset_674fb6e5571ca86dd0e5c858",
    "asset_681fb76277c10d2307f7509f",
    "asset_88fe0971a994597f3866473b",
    "asset_a873bb17b9a66e6e980845a5",
    "asset_ce3577373d7cc71b7025cc74",
    "asset_f56aff9ecf13f7ddf0afcd55",
)
EXPORT_ALLOW = {
    "IRIS": ["vertices", "faces", "vertex_normals", "canonical_transform", "inverse_canonical_transform"],
    "GEPPETTO": ["vertices", "faces", "vertex_normals", "canonical_transform", "inverse_canonical_transform", "bone_heads", "bone_tails", "parents", "deform_mask"],
    "ARACHNE": ["vertices", "faces", "vertex_normals", "canonical_transform", "inverse_canonical_transform", "bone_heads", "bone_tails", "parents", "deform_mask", "skin"],
}
IMMUTABLE_GLOBALS = (
    "metadata/CANONICAL_VARIANT_SELECTION.json",
    "ledgers/MASTER_VARIANTS.jsonl",
    "ledgers/PROCESSED_VARIANTS.jsonl",
    "exports/IRIS/records.jsonl",
    "exports/GEPPETTO/records.jsonl",
    "exports/ARACHNE/records.jsonl",
)
VIEW_FILES = ("camera.json", "raster_authority.npz", "cel_clean.png", "ink_cel.png", "cel_clean_512.png", "ink_cel_512.png")


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
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def atomic_npz(dst: Path, arr: dict):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    tmp.unlink(missing_ok=True)
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, dst)


def read_jsonl(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def recover_train512(root: Path):
    ledger = root / "ledgers" / "MASTER_VARIANTS.jsonl"
    got = sha256_file(ledger)
    if got != EXPECTED_MASTER_LEDGER_SHA:
        raise RuntimeError(f"MASTER_LEDGER_SHA_DRIFT expected={EXPECTED_MASTER_LEDGER_SHA} actual={got}")
    rows = [r for r in read_jsonl(ledger) if r.get("split") == "FIT"]
    proxy_seed = "PV5_R256_FIT_SCALE_PROXY32_V1"
    train_seed = "PV5_R256_FIT_SCALE_TRAIN_V1"
    proxy_q = {"objaverse_animated_originals": 26, "quaternius_cc0": 3, "kaykit_cc0": 3}
    train_q = {"objaverse_animated_originals": 483, "quaternius_cc0": 24, "kaykit_cc0": 5}
    proxy = []
    for src, q in proxy_q.items():
        pool = [r for r in rows if r["source_registry_id"] == src]
        pool.sort(key=lambda r: hashlib.sha256((proxy_seed + "|" + r["canonical_asset_id"]).encode()).hexdigest())
        proxy.extend(pool[:q])
    proxy_ids = {r["canonical_asset_id"] for r in proxy}
    train = []
    for src, q in train_q.items():
        pool = [r for r in rows if r["source_registry_id"] == src and r["canonical_asset_id"] not in proxy_ids]
        pool.sort(key=lambda r: hashlib.sha256((train_seed + "|" + r["canonical_asset_id"]).encode()).hexdigest())
        train.extend(pool[:q])
    ids = sorted(r["canonical_asset_id"] for r in train)
    set_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(ids) != 512 or set_sha != EXPECTED_TRAIN512_SET_SHA:
        raise RuntimeError(f"TRAIN512_RECOVERY_DRIFT count={len(ids)} sha={set_sha}")
    return ids


def load_selected(root: Path):
    raw = json.loads((root / "metadata" / "CANONICAL_VARIANT_SELECTION.json").read_text(encoding="utf-8"))
    selected = raw.get("selected", raw) if isinstance(raw, dict) else raw
    if isinstance(selected, list):
        return {x["canonical_asset_id"]: x for x in selected}
    return dict(selected)


def decision_from_audit(a):
    return "ADMIT" if a["arachne_capable"] else ("ADMIT_RIG_NO_SKIN" if a["geppetto_capable"] else "ADMIT_GEOMETRY_ONLY")


def patched_admission(old, audit):
    adm = json.loads(json.dumps(old))
    adm["technical"] = {
        "geometry": {"status": "PASS" if audit["geometry_pass"] else "FAIL", "reason": None if audit["geometry_pass"] else ";".join(audit["geometry_reasons"]), "metrics": audit["geometry_metrics"]},
        "rig": {"status": "PASS" if audit["rig_pass"] else "NOT_APPLICABLE", "reason": None if audit["rig_pass"] else ";".join(audit["rig_reasons"]), "metrics": audit["rig_metrics"]},
        "skin": {"status": "PASS" if audit["skin_pass"] else "NOT_APPLICABLE", "reason": None if audit["skin_pass"] else ";".join(audit["skin_reasons"]), "metrics": audit["skin_metrics"]},
        "canonicalization": {"status": "PASS" if audit["canonicalization_pass"] else "FAIL", "reason": None if audit["canonicalization_pass"] else "transform_mismatch", "metrics": {"max_abs_err": audit["canonicalization_max_abs_err"]}},
        "render": {"status": "PASS", "reason": None, "metrics": {"views": 8, "native_resolution": 1024, "repair_build_id": BUILD_ID}},
    }
    caps = {"iris": bool(audit["iris_capable"]), "geppetto": bool(audit["geppetto_capable"]), "arachne": bool(audit["arachne_capable"])}
    adm["capabilities"] = caps
    adm["decision"] = decision_from_audit(audit)
    reasons = list(adm.get("reasons") or [])
    if "DINO_TRAIN512_SELECTIVE_GEOMETRY_RENDER_REPAIR_B7_V2" not in reasons:
        reasons.append("DINO_TRAIN512_SELECTIVE_GEOMETRY_RENDER_REPAIR_B7_V2")
    adm["reasons"] = reasons
    return adm


def validate_render_asset(asset_dir: Path):
    geom = asset_dir / "primary_geometry.npz"
    with np.load(geom, allow_pickle=False) as d:
        V = np.asarray(d["vertices"], np.float64)
        F = np.asarray(d["faces"], np.int64)
    total = 0
    repro = []
    for vi in range(8):
        vd = asset_dir / "renders" / f"V{vi}"
        for fn in VIEW_FILES:
            p = vd / fn
            if not p.is_file() or p.stat().st_size <= 0:
                raise RuntimeError(f"missing render payload {p}")
        cam = json.loads((vd / "camera.json").read_text(encoding="utf-8"))
        if cam.get("image_origin") != "TOP_LEFT" or int(cam.get("vertical_flip_count", -1)) != 1 or float(cam.get("yaw_deg")) != vi * 45:
            raise RuntimeError(f"camera contract fail {asset_dir.name} V{vi}")
        with np.load(vd / "raster_authority.npz", allow_pickle=False) as z:
            pix = np.asarray(z["pixel_linear_index"], np.int64)
            tid = np.asarray(z["triangle_id"], np.int64)
            uv = np.asarray(z["barycentric_uv"], np.float64)
        if len(pix) == 0 or len(pix) != len(np.unique(pix)):
            raise RuntimeError(f"invalid raster rows {asset_dir.name} V{vi}")
        if tid.min() < 0 or tid.max() >= len(F) or not np.isfinite(uv).all():
            raise RuntimeError(f"invalid raster authority {asset_dir.name} V{vi}")
        alpha = np.asarray(Image.open(vd / "cel_clean.png").convert("RGBA"), dtype=np.uint8)[..., 3] > 0
        auth = np.zeros(NATIVE * NATIVE, dtype=bool)
        auth[pix] = True
        auth = auth.reshape(NATIVE, NATIVE)
        if not np.array_equal(alpha, auth):
            raise RuntimeError(f"alpha/authority mismatch {asset_dir.name} V{vi}")
        total += len(pix)
        ys, xs = pix // NATIVE, pix % NATIVE
        w = np.stack([uv[:, 0], uv[:, 1], 1 - uv[:, 0] - uv[:, 1]], 1)
        P = (V[F[tid]] * w[:, :, None]).sum(1)
        right = np.asarray(cam["right"], np.float64)
        up = np.asarray(cam["up"], np.float64)
        he = float(cam["half_extent"])
        px = ((P @ right) / he + 1) * NATIVE / 2 - 0.5
        py = (1 - (P @ up) / he) * NATIVE / 2 - 0.5
        err = np.sqrt((px - xs) ** 2 + (py - ys) ** 2)
        repro.append(float(np.quantile(err, 0.95)))
    mx = max(repro or [0.0])
    if mx > 0.05:
        raise RuntimeError(f"reprojection p95 too high {asset_dir.name}: {mx}")
    return {"foreground_authority_rows_total": int(total), "reprojection_p95_px_max": float(mx)}


def backup_one(src: Path, root: Path, backup_root: Path, absent: list[str]):
    rel = src.relative_to(root)
    dst = backup_root / rel
    if src.exists():
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        absent.append(str(rel))


def restore_one(dst: Path, root: Path, backup_root: Path, absent: set[str]):
    rel = str(dst.relative_to(root))
    src = backup_root / rel
    if rel in absent:
        if dst.exists():
            if dst.is_dir(): shutil.rmtree(dst)
            else: dst.unlink()
        return
    if not src.exists():
        raise RuntimeError(f"backup missing for restore: {rel}")
    if dst.exists():
        if dst.is_dir(): shutil.rmtree(dst)
        else: dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir(): shutil.copytree(src, dst)
    else: shutil.copy2(src, dst)


def audit_train512(root: Path, ids: list[str], repair_map: dict[str, str]):
    bad = []
    deep_repaired = 0
    for i, aid in enumerate(ids, 1):
        a = root / "master" / "assets" / aid
        try:
            if not a.is_dir(): raise RuntimeError("asset_dir_missing")
            geom = a / "primary_geometry.npz"
            marker = a / "RENDER_COMPLETE.json"
            if not geom.is_file() or not marker.is_file(): raise RuntimeError("geometry_or_marker_missing")
            for vi in range(8):
                vd = a / "renders" / f"V{vi}"
                for fn in VIEW_FILES:
                    if not (vd / fn).is_file(): raise RuntimeError(f"V{vi}:{fn}:missing")
                with np.load(vd / "raster_authority.npz", allow_pickle=False) as z:
                    if len(z["pixel_linear_index"]) == 0: raise RuntimeError(f"V{vi}:blank")
            if aid in repair_map:
                if sha256_file(geom) != repair_map[aid]: raise RuntimeError("repaired_geometry_sha")
                validate_render_asset(a)
                deep_repaired += 1
        except Exception as e:
            bad.append({"asset": aid, "error": str(e)})
        if i % 64 == 0 or i == len(ids):
            print(f"[PROMOTE] TRAIN512 availability {i}/{len(ids)} bad={len(bad)}", flush=True)
    return {"count": len(ids), "ready_count": len(ids) - len(bad), "bad": bad, "deep_repaired_count": deep_repaired}


def run(args):
    root = Path(args.root).resolve()
    post = root / "reports" / "post_corpus_audit"
    staging = root / STAGING_REL
    stage_result_path = staging / "STAGE_B7_RESULT_V2.json"
    b6_path = root / B6_RESULT_REL
    result_path = root / RESULT_REL
    backup_root = root / BACKUP_REL
    work = Path("/content/realsas_b7_train512_atomic_promotion_v2")

    print("=" * 88, flush=True)
    print("[PROMOTE] RealSaS Stage-B7 TRAIN512 atomic promotion V2", flush=True)
    print("[PROMOTE] zero optimizer steps; no DINO feature extraction/training", flush=True)
    print("=" * 88, flush=True)

    if result_path.is_file():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS_PROMOTED_9__TRAIN512_NATIVE1024_512_OF_512":
            print("[PROMOTE] already PASS; idempotent exit", flush=True)
            print(json.dumps({k: old.get(k) for k in ["status", "promoted_asset_count", "train512_ready_count", "master_ledger_sha256_after", "scientific_optimizer_steps"]}, indent=2), flush=True)
            return

    if backup_root.exists():
        raise RuntimeError(f"BACKUP_ROOT_ALREADY_EXISTS_WITHOUT_PASS_RESULT: {backup_root}")
    if not stage_result_path.is_file() or not b6_path.is_file():
        raise RuntimeError("required Stage-B7 staging/B6 authority missing")

    stage_result = json.loads(stage_result_path.read_text(encoding="utf-8"))
    if stage_result.get("status") != "PASS_STAGING_ONLY__MASTER_UNCHANGED" or stage_result.get("content_sha256") != EXPECTED_STAGE_CONTENT_SHA:
        raise RuntimeError("STAGE_B7_STAGING_AUTHORITY_DRIFT")
    if stage_result.get("pass_asset_count") != 9 or stage_result.get("staged_asset_count") != 9 or not (stage_result.get("sentinel") or {}).get("pass"):
        raise RuntimeError("STAGE_B7_STAGING_NOT_9_OF_9_SENTINEL_PASS")
    if stage_result.get("master_mutated") is not False or stage_result.get("scientific_optimizer_steps") != 0:
        raise RuntimeError("STAGE_B7_STAGING_FIREWALL_DRIFT")
    print("[PROMOTE] staging authority: sentinel PASS, repair 9/9 PASS", flush=True)

    b6 = json.loads(b6_path.read_text(encoding="utf-8"))
    b6rows = {r["asset"]: r for r in b6["repair_artifacts"]}
    selected = load_selected(root)
    train512 = recover_train512(root)
    print("[PROMOTE] TRAIN512 recovered: 512/512 exact set hash PASS", flush=True)
    if not set(REPAIR_ASSETS).issubset(train512):
        raise RuntimeError("REPAIR9_NOT_SUBSET_OF_TRAIN512")

    immutable_before = {}
    for rel in IMMUTABLE_GLOBALS:
        p = root / rel
        if p.is_file(): immutable_before[rel] = sha256_file(p)
    if immutable_before.get("ledgers/MASTER_VARIANTS.jsonl") != EXPECTED_MASTER_LEDGER_SHA:
        raise RuntimeError("MASTER_VARIANTS_PRE_SHA_DRIFT")

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    prepared = {}
    repair_sha = {}

    for idx, aid in enumerate(REPAIR_ASSETS, 1):
        row = b6rows.get(aid)
        if row is None or aid not in selected:
            raise RuntimeError(f"missing B6/selection authority {aid}")
        if row["old_capabilities"] != row["new_capabilities"]:
            raise RuntimeError(f"CAPABILITY_CHANGE_FORBIDDEN_IN_MINIMAL_PROMOTION {aid}")
        frozen = root / row["normalized_path"]
        source_structure = root / row["source_structure_path"]
        repair_record_path = frozen.parent / "repair_record.json"
        staged_asset = staging / "staged_assets" / aid
        if sha256_file(frozen) != row["normalized_sha256"] or sha256_file(staged_asset / "primary_geometry.npz") != row["normalized_sha256"]:
            raise RuntimeError(f"geometry SHA mismatch {aid}")
        validate_render_asset(staged_asset)
        rr = json.loads(repair_record_path.read_text(encoding="utf-8"))
        audit = rr["new_audit"]
        caps = {"iris": bool(audit["iris_capable"]), "geppetto": bool(audit["geppetto_capable"]), "arachne": bool(audit["arachne_capable"])}
        if caps != row["new_capabilities"] or caps != selected[aid]["capabilities"]:
            raise RuntimeError(f"capability authority mismatch {aid}")

        variant = root / selected[aid]["variant_dir"]
        old_adm = json.loads((variant / "admission.json").read_text(encoding="utf-8"))
        stage = work / aid
        stage_variant = stage / "variant"
        stage_variant.mkdir(parents=True)
        atomic_copy(frozen, stage_variant / "normalized.npz")
        atomic_copy(source_structure, stage_variant / "source_structure.json")
        atomic_json(stage_variant / "technical_audit.json", audit)
        normmeta_path = variant / "normalization_meta.json"
        normmeta = json.loads(normmeta_path.read_text(encoding="utf-8")) if normmeta_path.is_file() else dict(rr.get("normalization") or {})
        normmeta["post_corpus_repair"] = {"schema": SCHEMA, "repair_build_id": BUILD_ID, "frozen_sha256": row["normalized_sha256"], "staging_content_sha256": EXPECTED_STAGE_CONTENT_SHA}
        atomic_json(stage_variant / "normalization_meta.json", normmeta)
        atomic_json(stage_variant / "admission.json", patched_admission(old_adm, audit))

        stage_exports = stage / "exports"
        with np.load(frozen, allow_pickle=False) as d:
            for consumer in ("IRIS", "GEPPETTO", "ARACHNE"):
                if consumer == "ARACHNE" and not caps["arachne"]:
                    continue
                arr = {k: np.asarray(d[k]) for k in EXPORT_ALLOW[consumer] if k in d}
                dst = stage_exports / consumer / ("teacher_geometry.npz" if consumer == "IRIS" else "teacher.npz")
                atomic_npz(dst, arr)

        marker = {
            "build_id": ORIGINAL_BUILD_ID,
            "repair_build_id": BUILD_ID,
            "canonical_asset_id": aid,
            "selected_candidate_id": selected[aid].get("candidate_id"),
            "views": 8,
            "native_resolution": 1024,
            "staging_content_sha256": EXPECTED_STAGE_CONTENT_SHA,
            "scientific_optimizer_steps": 0,
            "utc": utc_now(),
        }
        atomic_json(stage / "RENDER_COMPLETE.json", marker)
        prepared[aid] = {"row": row, "variant": variant, "staged_asset": staged_asset, "stage": stage, "caps": caps}
        repair_sha[aid] = row["normalized_sha256"]
        print(f"[PROMOTE] prepared {idx}/9 {aid}", flush=True)

    # Back up every canonical path BEFORE the first mutation.
    absent = []
    backup_root.mkdir(parents=True, exist_ok=False)
    for aid in REPAIR_ASSETS:
        p = prepared[aid]
        for target in [p["variant"], root / "master" / "assets" / aid, root / "exports" / "IRIS" / aid, root / "exports" / "GEPPETTO" / aid, root / "exports" / "ARACHNE" / aid]:
            backup_one(target, root, backup_root, absent)
    backup_manifest = {
        "schema": SCHEMA + ".BackupManifest",
        "created_utc": utc_now(),
        "repair_assets": list(REPAIR_ASSETS),
        "absent_before": sorted(set(absent)),
        "immutable_global_sha256": immutable_before,
        "staging_content_sha256": EXPECTED_STAGE_CONTENT_SHA,
    }
    atomic_json(backup_root / "BACKUP_MANIFEST.json", backup_manifest)
    print("[PROMOTE] backup complete; opening canonical mutation", flush=True)

    mutated = []
    try:
        for idx, aid in enumerate(REPAIR_ASSETS, 1):
            p = prepared[aid]
            row, variant, staged_asset, stage, caps = p["row"], p["variant"], p["staged_asset"], p["stage"], p["caps"]
            asset_dir = root / "master" / "assets" / aid

            # Variant payload; capability identity is unchanged.
            for fn in ["normalized.npz", "source_structure.json", "technical_audit.json", "normalization_meta.json", "admission.json"]:
                atomic_copy(stage / "variant" / fn, variant / fn)

            # Master payload, marker last.
            (asset_dir / "RENDER_COMPLETE.json").unlink(missing_ok=True)
            atomic_copy(staged_asset / "primary_geometry.npz", asset_dir / "primary_geometry.npz")
            for vi in range(8):
                for fn in VIEW_FILES:
                    atomic_copy(staged_asset / "renders" / f"V{vi}" / fn, asset_dir / "renders" / f"V{vi}" / fn)
            atomic_copy(stage / "RENDER_COMPLETE.json", asset_dir / "RENDER_COMPLETE.json")

            # Consumer payloads; global records remain byte-identical because paths/capabilities do not change.
            iris_dir = root / "exports" / "IRIS" / aid
            gep_dir = root / "exports" / "GEPPETTO" / aid
            atomic_copy(stage / "exports" / "IRIS" / "teacher_geometry.npz", iris_dir / "teacher_geometry.npz")
            atomic_copy(stage / "exports" / "GEPPETTO" / "teacher.npz", gep_dir / "teacher.npz")
            obs = iris_dir / "observations"
            for vi in range(8):
                for fn in VIEW_FILES:
                    atomic_copy(asset_dir / "renders" / f"V{vi}" / fn, obs / f"V{vi}" / fn)
            if caps["arachne"]:
                atomic_copy(stage / "exports" / "ARACHNE" / "teacher.npz", root / "exports" / "ARACHNE" / aid / "teacher.npz")

            # Exact per-asset verification before moving on.
            if sha256_file(variant / "normalized.npz") != row["normalized_sha256"] or sha256_file(asset_dir / "primary_geometry.npz") != row["normalized_sha256"]:
                raise RuntimeError(f"post-copy geometry SHA fail {aid}")
            metrics = validate_render_asset(asset_dir)
            for vi in range(8):
                for fn in VIEW_FILES:
                    if sha256_file(asset_dir / "renders" / f"V{vi}" / fn) != sha256_file(obs / f"V{vi}" / fn):
                        raise RuntimeError(f"IRIS observation parity fail {aid} V{vi} {fn}")
            mutated.append({"asset": aid, "geometry_sha256": row["normalized_sha256"], "render_metrics": metrics})
            print(f"[PROMOTE] promoted {idx}/9 {aid}: PASS", flush=True)

        # Global identity must remain byte-exact.
        immutable_after = {rel: sha256_file(root / rel) for rel in immutable_before}
        if immutable_after != immutable_before:
            drift = {k: {"before": immutable_before[k], "after": immutable_after.get(k)} for k in immutable_before if immutable_before[k] != immutable_after.get(k)}
            raise RuntimeError("IMMUTABLE_GLOBAL_DRIFT " + json.dumps(drift, sort_keys=True))
        print("[PROMOTE] global selection/ledger/records SHA: byte-exact unchanged", flush=True)

        train_audit = audit_train512(root, train512, repair_sha)
        if train_audit["ready_count"] != 512 or train_audit["bad"]:
            raise RuntimeError("TRAIN512_NOT_512_OF_512 " + json.dumps(train_audit["bad"][:10]))

        authority = {
            "schema": "RealSaS.DinoTrain512Native1024Authority.v2",
            "status": "PASS_512_OF_512",
            "train512_asset_count": 512,
            "train512_ready_count": 512,
            "train512_set_sha256": EXPECTED_TRAIN512_SET_SHA,
            "selection_ledger_sha256": EXPECTED_MASTER_LEDGER_SHA,
            "selective_repair_assets": list(REPAIR_ASSETS),
            "selective_repair_count": 9,
            "deep_repaired_count": train_audit["deep_repaired_count"],
            "staging_content_sha256": EXPECTED_STAGE_CONTENT_SHA,
            "scientific_optimizer_steps": 0,
            "dino_training_authorized_by_this_artifact": False,
            "next_gate": "DINO_WEIGHT_SHA_PREPROCESS_ARCHITECTURE_SAMPLE_STREAM_AND_TOKEN_PARITY",
        }
        authority["content_sha256"] = canonical_sha(authority)
        atomic_json(root / TRAIN512_AUTH_REL, authority)

        result = {
            "schema": SCHEMA,
            "build_id": BUILD_ID,
            "status": "PASS_PROMOTED_9__TRAIN512_NATIVE1024_512_OF_512",
            "promoted_asset_count": 9,
            "promoted": mutated,
            "train512_ready_count": 512,
            "train512_set_sha256": EXPECTED_TRAIN512_SET_SHA,
            "master_ledger_sha256_before": immutable_before["ledgers/MASTER_VARIANTS.jsonl"],
            "master_ledger_sha256_after": sha256_file(root / "ledgers" / "MASTER_VARIANTS.jsonl"),
            "immutable_globals_unchanged": True,
            "backup_root": str(BACKUP_REL),
            "authority_path": str(TRAIN512_AUTH_REL),
            "scientific_optimizer_steps": 0,
            "dino_training_started": False,
            "finished_utc": utc_now(),
        }
        result["content_sha256"] = canonical_sha(result)
        atomic_json(result_path, result)
        print("=" * 88, flush=True)
        print("[PROMOTE] PASS_PROMOTED_9__TRAIN512_NATIVE1024_512_OF_512", flush=True)
        print("[PROMOTE] global ledgers/selection/records unchanged: TRUE", flush=True)
        print("[PROMOTE] DINO optimizer steps: 0", flush=True)
        print("[PROMOTE] RESULT:", result_path, flush=True)
        print("[PROMOTE] TRAIN512 AUTHORITY:", root / TRAIN512_AUTH_REL, flush=True)
        print("=" * 88, flush=True)

    except Exception as exc:
        print("[PROMOTE] ERROR; restoring all backed canonical paths:", repr(exc), flush=True)
        absent_set = set(backup_manifest["absent_before"])
        # Restore all 9, not only the current asset, to guarantee all-or-nothing publication.
        for aid in REPAIR_ASSETS:
            p = prepared[aid]
            for target in [p["variant"], root / "master" / "assets" / aid, root / "exports" / "IRIS" / aid, root / "exports" / "GEPPETTO" / aid, root / "exports" / "ARACHNE" / aid]:
                restore_one(target, root, backup_root, absent_set)
        # Global files were not written; still assert byte identity after rollback.
        global_now = {rel: sha256_file(root / rel) for rel in immutable_before}
        rollback_ok = global_now == immutable_before
        fail = {
            "schema": SCHEMA,
            "build_id": BUILD_ID,
            "status": "FAIL_ROLLED_BACK" if rollback_ok else "FAIL_ROLLBACK_GLOBAL_DRIFT",
            "error": repr(exc),
            "rollback_global_identity_pass": rollback_ok,
            "scientific_optimizer_steps": 0,
            "finished_utc": utc_now(),
        }
        atomic_json(result_path, fail)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
