from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

EXPECTED_AUDIT_STATUS = "PASS_MEASUREMENT_COMPLETE__QUALITY_AND_SEMANTIC_FREEZE_NEXT"
EXPECTED_AUDIT_SHA256 = "f71e2fb38b793b9110d4afe34701db1dfa171ff784b026fccf3ce394cfd18b0b"
EXPECTED_STRUCTURAL_MEMBERSHIP_SHA256 = "cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147"
EXPECTED_ASSET_COUNT = 2897
EXPECTED_ARACHNE_COUNT = 2527
EXPECTED_OBJECTIVE_PASS_COUNT = 2874
EXPECTED_OBJECTIVE_ARACHNE_PASS_COUNT = 2509
EXPECTED_OBJECTIVE_PASS_SET_SHA256 = "60096fdf837b9cf277d1e3493a956796faf173d1af8ba7f593009423d77fa78b"
EXPECTED_OBJECTIVE_FAIL_SET_SHA256 = "40f2bbcac14109c3857db2d34e4419c27e8e83bb24e814b059af13e59d6087ad"
EXPECTED_REVIEW_PRIORITY_SET_SHA256 = "4de5424421fb129540d1ed9a99f3c10979de72321766e8b4eaafe5c73a3e3460"

MIN_VIEW_OCCUPANCY = 0.005
REVIEW_MAX_FACE = 0.20
REVIEW_MAX_VISIBLE_TRI = 0.30
REVIEW_MAX_SIGNIFICANT_COMPONENTS = 1
REVIEW_MIN_LARGEST_COMPONENT = 0.90
REVIEW_MAX_CENTROID_OFFSET = 0.25

SEMANTIC_LABELS = {
    "PASS_SINGLE_RIGGABLE_CHARACTER",
    "FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE",
    "REVIEW_AMBIGUOUS",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_sha(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in sorted(ids)).encode()).hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_name(path.name + ".tmp")
    q.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(path)


def _min_view(row: dict, key: str) -> float:
    return min(float(v[key]) for v in row["views"])


def _max_view(row: dict, key: str) -> float:
    return max(float(v[key]) for v in row["views"])


def objective_decision(row: dict) -> tuple[str, list[str], list[str]]:
    hard: list[str] = []
    flags: list[str] = []
    if _min_view(row, "occupancy_fraction") < MIN_VIEW_OCCUPANCY:
        hard.append("TOO_SMALL_NATIVE_SUBSTRATE")
    if int(row["aggregate"]["border_touch_view_count"]) > 0:
        hard.append("BORDER_TOUCH")
    if float(row["mesh"]["max_face_area_fraction"]) >= REVIEW_MAX_FACE:
        flags.append("MESH_FACE_DOMINANCE")
    if _max_view(row, "max_visible_triangle_pixel_fraction") >= REVIEW_MAX_VISIBLE_TRI:
        flags.append("VISIBLE_TRIANGLE_DOMINANCE")
    if _max_view(row, "significant_component_count") > REVIEW_MAX_SIGNIFICANT_COMPONENTS:
        flags.append("MULTI_COMPONENT")
    if _min_view(row, "largest_component_fraction") < REVIEW_MIN_LARGEST_COMPONENT:
        flags.append("FRAGMENTED_SUPPORT")
    if _max_view(row, "centroid_offset_fraction") > REVIEW_MAX_CENTROID_OFFSET:
        flags.append("OFF_CENTER")
    return ("FAIL_OBJECTIVE_RENDER_C0" if hard else "PASS_OBJECTIVE_RENDER_C0", hard, flags)


def load_and_validate_audit(path: Path) -> dict:
    if sha256_file(path) != EXPECTED_AUDIT_SHA256:
        raise RuntimeError("AUDIT_RESULT_SHA_DRIFT")
    o = json.loads(path.read_text(encoding="utf-8"))
    if o.get("status") != EXPECTED_AUDIT_STATUS or o.get("scope_complete") is not True:
        raise RuntimeError("AUDIT_STATUS_DRIFT")
    if int(o.get("asset_count", -1)) != EXPECTED_ASSET_COUNT or int(o.get("arachne_structural_c0_count", -1)) != EXPECTED_ARACHNE_COUNT:
        raise RuntimeError("AUDIT_COUNT_DRIFT")
    if o.get("structural_membership_sha256") != EXPECTED_STRUCTURAL_MEMBERSHIP_SHA256:
        raise RuntimeError("STRUCTURAL_MEMBERSHIP_SHA_DRIFT")
    rows = o.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ASSET_COUNT:
        raise RuntimeError("AUDIT_ROWS_DRIFT")
    if len({r["canonical_asset_id"] for r in rows}) != EXPECTED_ASSET_COUNT:
        raise RuntimeError("DUPLICATE_ASSET_ID")
    return o


def build_objective_manifest(audit: dict) -> dict:
    records = []
    pass_ids, fail_ids, review_ids = [], [], []
    pass_arachne = 0
    for row in audit["rows"]:
        if row.get("status") != "PASS":
            raise RuntimeError(f"UNEXPECTED_AUDIT_ROW_FAIL:{row.get('canonical_asset_id')}")
        status, hard, flags = objective_decision(row)
        aid = row["canonical_asset_id"]
        if hard:
            fail_ids.append(aid)
        else:
            pass_ids.append(aid)
            pass_arachne += int(bool(row["arachne_structural_c0"]))
        if flags:
            review_ids.append(aid)
        records.append({
            "canonical_asset_id": aid,
            "arachne_structural_c0": bool(row["arachne_structural_c0"]),
            "objective_status": status,
            "hard_exclusion_reasons": hard,
            "semantic_review_priority_flags": flags,
        })
    if len(pass_ids) != EXPECTED_OBJECTIVE_PASS_COUNT or pass_arachne != EXPECTED_OBJECTIVE_ARACHNE_PASS_COUNT:
        raise RuntimeError("OBJECTIVE_PASS_COUNT_DRIFT")
    if set_sha(pass_ids) != EXPECTED_OBJECTIVE_PASS_SET_SHA256:
        raise RuntimeError("OBJECTIVE_PASS_SET_SHA_DRIFT")
    if set_sha(fail_ids) != EXPECTED_OBJECTIVE_FAIL_SET_SHA256:
        raise RuntimeError("OBJECTIVE_FAIL_SET_SHA_DRIFT")
    if set_sha(review_ids) != EXPECTED_REVIEW_PRIORITY_SET_SHA256:
        raise RuntimeError("REVIEW_PRIORITY_SET_SHA_DRIFT")
    out = {
        "schema": "RealSaS.GeppettoArachne.ObjectiveRenderC0Manifest.v1",
        "status": "PASS_OBJECTIVE_RENDER_POLICY_APPLIED__IMAGE_AND_SEMANTIC_GATE_NEXT",
        "policy": {
            "min_view_occupancy_fraction": MIN_VIEW_OCCUPANCY,
            "border_touch_forbidden": True,
            "review_only": {
                "max_face_area_fraction_gte": REVIEW_MAX_FACE,
                "max_visible_triangle_pixel_fraction_gte": REVIEW_MAX_VISIBLE_TRI,
                "significant_component_count_gt": REVIEW_MAX_SIGNIFICANT_COMPONENTS,
                "largest_component_fraction_lt": REVIEW_MIN_LARGEST_COMPONENT,
                "centroid_offset_fraction_gt": REVIEW_MAX_CENTROID_OFFSET,
            },
        },
        "counts": {
            "structural_geppetto": EXPECTED_ASSET_COUNT,
            "structural_arachne": EXPECTED_ARACHNE_COUNT,
            "objective_pass_geppetto": len(pass_ids),
            "objective_pass_arachne": pass_arachne,
            "objective_fail": len(fail_ids),
            "semantic_review_priority": len(review_ids),
        },
        "hashes": {
            "audit_result_sha256": EXPECTED_AUDIT_SHA256,
            "structural_membership_sha256": EXPECTED_STRUCTURAL_MEMBERSHIP_SHA256,
            "objective_pass_set_sha256": set_sha(pass_ids),
            "objective_fail_set_sha256": set_sha(fail_ids),
            "semantic_review_priority_set_sha256": set_sha(review_ids),
        },
        "training_authorized": False,
        "records": records,
    }
    out["content_sha256"] = canonical_sha(out)
    return out


def _raster_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
        res = np.asarray(z["resolution"], dtype=np.int64).reshape(-1)
    if len(res) < 2 or tuple(map(int, res[:2])) != (1024, 1024):
        raise ValueError(f"RASTER_RESOLUTION_DRIFT:{path}:{res.tolist()}")
    if len(pix) == 0 or pix.min() < 0 or pix.max() >= 1024 * 1024:
        raise ValueError(f"RASTER_PIXEL_ID_DRIFT:{path}")
    if len(np.unique(pix)) != len(pix):
        raise ValueError(f"RASTER_DUPLICATE_PIXEL:{path}")
    m = np.zeros(1024 * 1024, dtype=bool); m[pix] = True
    return m.reshape(1024, 1024)


def _image_mask(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.size != (1024, 1024):
            raise ValueError(f"NATIVE_IMAGE_CONTRACT_DRIFT:{path}:{im.mode}:{im.size}")
        a = np.asarray(im, dtype=np.uint8)
    alpha = a[:, :, 3]
    if np.any(alpha < 255):
        return alpha >= 128, "ALPHA_THRESHOLD"
    return np.any(a[:, :, :3] != np.asarray([0, 255, 0], dtype=np.uint8), axis=2), "OPAQUE_EXACT_GREEN"


def _measure_one_image_integrity(master_root: str, aid: str) -> tuple[str, dict]:
    asset = Path(master_root) / "master" / "assets" / aid
    views = []
    try:
        for v in range(8):
            vd = asset / "renders" / f"V{v}"
            cam = json.loads((vd / "camera.json").read_text(encoding="utf-8"))
            yaw = float(cam["yaw_deg"])
            if abs(yaw - 45.0 * v) > 1e-4:
                raise ValueError(f"CAMERA_YAW_DRIFT:{aid}:V{v}:{yaw}")
            im, mode = _image_mask(vd / "cel_clean.png")
            ra = _raster_mask(vd / "raster_authority.npz")
            inter = int(np.count_nonzero(im & ra)); union = int(np.count_nonzero(im | ra))
            ni = int(np.count_nonzero(im)); nr = int(np.count_nonzero(ra))
            unsupported = int(np.count_nonzero(im & ~ra)); missing = int(np.count_nonzero(ra & ~im))
            views.append({
                "view_index": v,
                "yaw_deg": yaw,
                "foreground_mode": mode,
                "image_foreground_pixels": ni,
                "raster_foreground_pixels": nr,
                "intersection_pixels": inter,
                "union_pixels": union,
                "iou": float(inter / union) if union else 1.0,
                "image_unsupported_fraction": float(unsupported / max(1, ni)),
                "raster_missing_fraction": float(missing / max(1, nr)),
                "exact_support_equal": bool(np.array_equal(im, ra)),
            })
        return "PASS", {"canonical_asset_id": aid, "status": "PASS_MEASURED", "views": views}
    except Exception as exc:
        return "FAIL", {"canonical_asset_id": aid, "error": f"{type(exc).__name__}: {exc}"}


def measure_image_integrity(master_root: Path, objective_manifest: dict, workers: int = 4) -> dict:
    pass_ids = [r["canonical_asset_id"] for r in objective_manifest["records"] if r["objective_status"] == "PASS_OBJECTIVE_RENDER_C0"]
    records, failures = [], []
    workers = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_measure_one_image_integrity, str(master_root), aid): aid for aid in pass_ids}
        for done, fut in enumerate(as_completed(futures), 1):
            aid = futures[fut]
            try:
                status, payload = fut.result()
            except Exception as exc:
                status, payload = "FAIL", {"canonical_asset_id": aid, "error": f"WORKER_{type(exc).__name__}: {exc}"}
            (records if status == "PASS" else failures).append(payload)
            if done == 1 or done % 25 == 0 or done == len(pass_ids):
                print(f"[image-integrity] {done}/{len(pass_ids)} measured={len(records)} hard_fail={len(failures)}", flush=True)
    records.sort(key=lambda x: x["canonical_asset_id"]); failures.sort(key=lambda x: x["canonical_asset_id"])
    vals_iou = [v["iou"] for r in records for v in r["views"]]
    vals_u = [v["image_unsupported_fraction"] for r in records for v in r["views"]]
    vals_m = [v["raster_missing_fraction"] for r in records for v in r["views"]]
    def dist(x):
        a = np.asarray(x, dtype=np.float64)
        return {"count": int(a.size), "min": float(a.min()), "p01": float(np.percentile(a,1)), "p50": float(np.percentile(a,50)), "p99": float(np.percentile(a,99)), "max": float(a.max())}
    out = {
        "schema": "RealSaS.GeppettoArachne.NativeImageIntegrityMeasurement.v1",
        "status": "PASS_IMAGE_INTEGRITY_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT" if not failures and len(records) == EXPECTED_OBJECTIVE_PASS_COUNT else "FAIL_HARD_IMAGE_AUTHORITY_CONTRACT",
        "objective_pass_expected": EXPECTED_OBJECTIVE_PASS_COUNT,
        "measured_asset_count": len(records),
        "hard_failure_count": len(failures),
        "workers": workers,
        "measurement_only_no_threshold": True,
        "aggregate": {"iou": dist(vals_iou) if vals_iou else None, "image_unsupported_fraction": dist(vals_u) if vals_u else None, "raster_missing_fraction": dist(vals_m) if vals_m else None, "exact_support_equal_view_count": sum(v["exact_support_equal"] for r in records for v in r["views"])},
        "failures": failures,
        "records": records,
        "training_authorized": False,
    }
    out["content_sha256"] = canonical_sha(out)
    return out


def build_contact_sheet(master_root: Path, aid: str, out_path: Path) -> None:
    panel = 512; label_h = 22; cols, rows = 4, 2
    sheet = Image.new("RGB", (cols * panel, rows * (panel + label_h)), (127, 127, 127))
    draw = ImageDraw.Draw(sheet)
    for v in range(8):
        p = master_root / "master" / "assets" / aid / "renders" / f"V{v}" / "cel_clean_512.png"
        with Image.open(p) as im:
            if im.size != (512, 512):
                raise ValueError(f"CONTACT_IMAGE_SIZE_DRIFT:{aid}:V{v}:{im.size}")
            rgba = im.convert("RGBA")
        bg = Image.new("RGBA", (512, 512), (127, 127, 127, 255)); bg.alpha_composite(rgba)
        x = (v % cols) * panel; y = (v // cols) * (panel + label_h)
        sheet.paste(bg.convert("RGB"), (x, y)); draw.text((x + 4, y + panel + 3), f"V{v}", fill=(255,255,255))
    out_path.parent.mkdir(parents=True, exist_ok=True); sheet.save(out_path, optimize=True)


def load_semantic_labels(path: Path, expected_ids: set[str]) -> dict[str, str]:
    o = json.loads(path.read_text(encoding="utf-8"))
    labels = o.get("labels", o)
    if not isinstance(labels, dict):
        raise ValueError("semantic labels must be mapping or {'labels': mapping}")
    unknown = set(labels) - expected_ids
    missing = expected_ids - set(labels)
    bad = {k:v for k,v in labels.items() if v not in SEMANTIC_LABELS}
    if unknown or missing or bad:
        raise RuntimeError(f"SEMANTIC_LABEL_CONTRACT_FAIL:missing={len(missing)} unknown={len(unknown)} bad={len(bad)}")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-result", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--master-root", type=Path)
    ap.add_argument("--measure-image-integrity", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--contact-sheets", action="store_true")
    ap.add_argument("--semantic-labels", type=Path)
    args = ap.parse_args()

    audit = load_and_validate_audit(args.audit_result)
    obj = build_objective_manifest(audit)
    atomic_json(args.out_dir / "OBJECTIVE_RENDER_C0_MANIFEST_V1.json", obj)
    print(f"OBJECTIVE_RENDER_C0=PASS geppetto={obj['counts']['objective_pass_geppetto']} arachne={obj['counts']['objective_pass_arachne']} review_priority={obj['counts']['semantic_review_priority']}", flush=True)

    pass_ids = [r["canonical_asset_id"] for r in obj["records"] if r["objective_status"] == "PASS_OBJECTIVE_RENDER_C0"]
    if args.measure_image_integrity:
        if args.master_root is None: raise SystemExit("--master-root required")
        im = measure_image_integrity(args.master_root, obj, workers=args.workers)
        atomic_json(args.out_dir / "NATIVE_IMAGE_INTEGRITY_MEASUREMENT_V1.json", im)
        print(f"IMAGE_INTEGRITY={im['status']} measured={im['measured_asset_count']} hard_fail={im['hard_failure_count']}", flush=True)
    if args.contact_sheets:
        if args.master_root is None: raise SystemExit("--master-root required")
        for i, aid in enumerate(pass_ids, 1):
            build_contact_sheet(args.master_root, aid, args.out_dir / "semantic_contact_sheets_v1" / f"{aid}.png")
            if i == 1 or i % 25 == 0 or i == len(pass_ids): print(f"[contact-sheet] {i}/{len(pass_ids)}", flush=True)
        atomic_json(args.out_dir / "SEMANTIC_REVIEW_MANIFEST_V1.json", {"schema":"RealSaS.GeppettoArachne.SemanticReviewManifest.v1","asset_count":len(pass_ids),"asset_set_sha256":set_sha(pass_ids),"labels":list(sorted(SEMANTIC_LABELS)),"training_authorized":False})
    if args.semantic_labels:
        # This validates the semantic decision file only. Final membership deliberately remains blocked
        # until a separately frozen image-integrity admission policy/result is provided.
        labels = load_semantic_labels(args.semantic_labels, set(pass_ids))
        c = {k: sum(v == k for v in labels.values()) for k in sorted(SEMANTIC_LABELS)}
        atomic_json(args.out_dir / "SEMANTIC_LABEL_VALIDATION_V1.json", {"schema":"RealSaS.GeppettoArachne.SemanticLabelValidation.v1","status":"PASS_COMPLETE_LABEL_CONTRACT","counts":c,"asset_set_sha256":set_sha(pass_ids),"training_authorized":False})
        print(f"SEMANTIC_LABELS=PASS {c}", flush=True)

if __name__ == "__main__":
    main()
