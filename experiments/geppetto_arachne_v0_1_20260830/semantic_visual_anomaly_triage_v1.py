from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label, maximum_filter

RESOLUTION = 1024
EXPECTED_OBJECTIVE_PASS = 2874
SEED = 20260831
TOP_N = 8
RANDOM_N = 16
LABELS = (
    "PASS_SINGLE_RIGGABLE_CHARACTER",
    "FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE",
    "REVIEW_AMBIGUOUS",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def set_sha(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in sorted(ids)).encode()).hexdigest()


def atomic_json(path: Path, o: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_name(path.name + ".tmp")
    q.write_text(json.dumps(o, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(path)


def image_foreground_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.size != (RESOLUTION, RESOLUTION):
            raise ValueError(f"NATIVE_IMAGE_CONTRACT_DRIFT:{path}:{im.mode}:{im.size}")
        a = np.asarray(im, dtype=np.uint8)
    alpha = a[:, :, 3]
    if np.any(alpha < 255):
        return alpha >= 128
    return np.any(a[:, :, :3] != np.asarray([0, 255, 0], dtype=np.uint8), axis=2)


def component_view_metrics(mask: np.ndarray) -> dict[str, float | int]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        raise ValueError("EMPTY_FOREGROUND")
    structure = np.ones((3, 3), dtype=np.uint8)
    lab, n = label(m, structure=structure)
    counts = np.bincount(lab.reshape(-1))[1:]
    order = np.argsort(counts)[::-1]
    fg = float(m.sum())
    h, w = m.shape
    diag = math.hypot(h, w)
    half_diag = 0.5 * diag
    ys_all, xs_all = np.nonzero(m)
    cx_all = float(xs_all.mean()); cy_all = float(ys_all.mean())
    offcenter = math.hypot(cx_all - (w - 1) * 0.5, cy_all - (h - 1) * 0.5) / half_diag

    comps = []
    for idx in order[: max(2, min(32, len(order)))]:
        lid = int(idx + 1)
        ys, xs = np.nonzero(lab == lid)
        area = float(len(xs))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        bbox_area = float(bw * bh)
        fill = area / bbox_area
        aspect = max(float(bw) / max(1.0, float(bh)), float(bh) / max(1.0, float(bw)))
        bbox_area_fraction = bbox_area / float(h * w)
        rectangle_score = bbox_area_fraction * fill * (1.0 + min(math.log2(max(aspect, 1.0)), 3.0) / 3.0)
        comps.append({
            "area": area,
            "foreground_fraction": area / fg,
            "bbox_area_fraction": bbox_area_fraction,
            "fill_ratio": fill,
            "aspect_ratio": aspect,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "rectangle_score": rectangle_score,
        })

    second_frac = float(comps[1]["foreground_fraction"]) if len(comps) > 1 else 0.0
    multi = 0.0
    if len(comps) > 1:
        sep = math.hypot(
            float(comps[0]["centroid_x"]) - float(comps[1]["centroid_x"]),
            float(comps[0]["centroid_y"]) - float(comps[1]["centroid_y"]),
        ) / diag
        multi = second_frac * sep

    edt = distance_transform_edt(m)
    locmax = m & (edt >= maximum_filter(edt, size=3, mode="constant", cval=0.0) - 1e-12)
    diam = 2.0 * edt[locmax]
    if diam.size == 0:
        raise ValueError("NO_MEDIAL_MAXIMA")

    return {
        "occupancy": fg / float(h * w),
        "component_count": int(n),
        "largest_component_fraction": float(comps[0]["foreground_fraction"]),
        "second_component_fraction": second_frac,
        "dominant_rectangle_score": float(max(float(c["rectangle_score"]) for c in comps)),
        "multi_disconnected_score": float(multi),
        "offcenter_fraction": float(offcenter),
        "medial_p01_px": float(np.percentile(diam, 1)),
        "medial_min_px": float(diam.min()),
    }


def objective_pass_ids(audit: dict) -> tuple[list[str], dict[str, dict]]:
    rows = audit.get("rows", [])
    row_by_id = {r["canonical_asset_id"]: r for r in rows}
    ids = []
    for r in rows:
        min_occ = min(float(v["occupancy_fraction"]) for v in r["views"])
        border = int(r["aggregate"]["border_touch_view_count"])
        if min_occ >= 0.005 and border == 0:
            ids.append(r["canonical_asset_id"])
    ids = sorted(ids)
    if len(ids) != EXPECTED_OBJECTIVE_PASS:
        raise RuntimeError(f"OBJECTIVE_PASS_COUNT_DRIFT:{len(ids)}")
    return ids, row_by_id


def measure_asset(master_root: Path, aid: str, audit_row: dict) -> dict:
    ad = master_root / "master" / "assets" / aid
    views = []
    for v in range(8):
        m = image_foreground_mask(ad / "renders" / f"V{v}" / "cel_clean.png")
        vm = component_view_metrics(m)
        vm["view"] = v
        views.append(vm)

    occ = [float(v["occupancy"]) for v in views]
    min_occ = min(occ); max_occ = max(occ)
    triangle_dom = max(float(v["max_visible_triangle_pixel_fraction"]) for v in audit_row["views"])
    return {
        "canonical_asset_id": aid,
        "views": views,
        "metrics": {
            "dominant_rectangle_score": max(float(v["dominant_rectangle_score"]) for v in views),
            "multi_disconnected_score": max(float(v["multi_disconnected_score"]) for v in views),
            "min_occupancy": min_occ,
            "max_occupancy": max_occ,
            "view_imbalance": max_occ / max(min_occ, 1e-12),
            "max_offcenter_fraction": max(float(v["offcenter_fraction"]) for v in views),
            "min_medial_p01_px": min(float(v["medial_p01_px"]) for v in views),
            "max_visible_triangle_pixel_fraction": triangle_dom,
        },
    }


def rank_select(records: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    def top(metric: str, reverse: bool = True) -> list[str]:
        rr = sorted(records, key=lambda r: (float(r["metrics"][metric]), r["canonical_asset_id"]), reverse=reverse)
        if reverse:
            # reverse=True also reverses asset-id tie order; stabilize explicitly.
            rr = sorted(records, key=lambda r: (-float(r["metrics"][metric]), r["canonical_asset_id"]))
        else:
            rr = sorted(records, key=lambda r: (float(r["metrics"][metric]), r["canonical_asset_id"]))
        return [r["canonical_asset_id"] for r in rr[:TOP_N]]

    buckets = {
        "DOMINANT_RECTANGLE_OR_QUAD": top("dominant_rectangle_score", True),
        "MULTI_DOMINANT_DISCONNECTED": top("multi_disconnected_score", True),
        "TINY_SUBJECT": top("min_occupancy", False),
        "VIEW_IMBALANCE": top("view_imbalance", True),
        "OFF_CENTER_OR_SPLIT_SCENE": top("max_offcenter_fraction", True),
        "EXTREMELY_THIN_SUPPORT": top("min_medial_p01_px", False),
        "RASTER_TRIANGLE_DOMINANCE": top("max_visible_triangle_pixel_fraction", True),
    }
    selected = set(a for xs in buckets.values() for a in xs)
    pool = sorted(r["canonical_asset_id"] for r in records if r["canonical_asset_id"] not in selected)
    rng = random.Random(SEED)
    controls = rng.sample(pool, min(RANDOM_N, len(pool)))
    buckets["BLIND_RANDOM_CONTROL"] = controls
    selected.update(controls)
    ordered = sorted(selected)
    rng.shuffle(ordered)
    return ordered, buckets


def make_contact_sheet(master_root: Path, aid: str, out_path: Path) -> None:
    tile_w, image_h, label_h = 512, 512, 24
    tile_h = image_h + label_h
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), (127, 127, 127))
    draw = ImageDraw.Draw(sheet)
    for v in range(8):
        p = master_root / "master" / "assets" / aid / "renders" / f"V{v}" / "cel_clean_512.png"
        with Image.open(p) as im:
            im = im.convert("RGBA")
            if im.size != (512, 512):
                raise ValueError(f"CONTACT_SHEET_SOURCE_SIZE_DRIFT:{aid}:V{v}:{im.size}")
            bg = Image.new("RGBA", im.size, (127, 127, 127, 255))
            bg.alpha_composite(im)
            tile = bg.convert("RGB")
        col, row = v % 4, v // 4
        x, y = col * tile_w, row * tile_h
        sheet.paste(tile, (x, y + label_h))
        draw.text((x + 6, y + 4), f"V{v}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG", optimize=True)


def run(master_root: Path, audit_result: Path, out_root: Path) -> dict:
    audit = json.loads(audit_result.read_text(encoding="utf-8"))
    ids, row_by_id = objective_pass_ids(audit)
    records = []
    for i, aid in enumerate(ids, 1):
        records.append(measure_asset(master_root, aid, row_by_id[aid]))
        if i == 1 or i % 25 == 0 or i == len(ids):
            print(f"[visual-triage] {i}/{len(ids)}", flush=True)

    ordered, buckets = rank_select(records)
    record_by_id = {r["canonical_asset_id"]: r for r in records}
    reason_by_id: dict[str, list[str]] = {aid: [] for aid in ordered}
    for reason, xs in buckets.items():
        for aid in xs:
            if aid in reason_by_id:
                reason_by_id[aid].append(reason)

    contacts = out_root / "contact_sheets"
    blind = []
    for slot, aid in enumerate(ordered, 1):
        filename = f"slot_{slot:03d}__{aid}.png"
        make_contact_sheet(master_root, aid, contacts / filename)
        blind.append({"slot": slot, "canonical_asset_id": aid, "contact_sheet": f"contact_sheets/{filename}"})

    scheduling = []
    for slot, aid in enumerate(ordered, 1):
        scheduling.append({
            "slot": slot,
            "canonical_asset_id": aid,
            "reasons": sorted(reason_by_id[aid]),
            "metrics": record_by_id[aid]["metrics"],
        })

    out_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "RealSaS.CorpusVisualAnomalyTriage.v1",
        "status": "PASS_DIAGNOSTIC_TRIAGE_COMPLETE__BLIND_REVIEW_REQUIRED",
        "seed": SEED,
        "training_authorized": False,
        "auto_exclusion_authorized": False,
        "population": {
            "objective_pass_count": len(ids),
            "objective_pass_set_sha256": set_sha(ids),
            "audit_result_sha256": sha256_file(audit_result),
        },
        "selection": {
            "top_n_per_priority_bucket": TOP_N,
            "random_control_n": RANDOM_N,
            "selected_unique_count": len(ordered),
            "buckets": buckets,
            "scheduling_records": scheduling,
        },
        "records": records,
    }
    result["content_sha256"] = canonical_sha(result)
    atomic_json(out_root / "CORPUS_VISUAL_ANOMALY_TRIAGE_V1.json", result)
    blind_obj = {
        "schema": "RealSaS.CorpusVisualBlindReviewIndex.v1",
        "seed": SEED,
        "allowed_labels": list(LABELS),
        "training_authorized": False,
        "items": blind,
    }
    blind_obj["content_sha256"] = canonical_sha(blind_obj)
    atomic_json(out_root / "CORPUS_VISUAL_BLIND_REVIEW_INDEX_V1.json", blind_obj)

    with (out_root / "CORPUS_VISUAL_REVIEW_LABELS_TEMPLATE_V1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slot", "canonical_asset_id", "label", "notes"])
        for x in blind:
            w.writerow([x["slot"], x["canonical_asset_id"], "", ""])

    zip_path = out_root / "CORPUS_VISUAL_REVIEW_CONTACT_SHEETS_V1.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=contacts)
    print(json.dumps({
        "status": result["status"],
        "population": result["population"],
        "selected_unique_count": len(ordered),
        "zip": str(zip_path),
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-root", type=Path, required=True)
    ap.add_argument("--audit-result", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    run(args.master_root, args.audit_result, args.out_root)


if __name__ == "__main__":
    main()
