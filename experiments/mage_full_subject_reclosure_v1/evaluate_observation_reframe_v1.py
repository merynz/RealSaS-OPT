from __future__ import annotations

"""Evaluate renderer parity and corrected full-subject observation framing.

Diagnostic-only. Corrected renders are not promoted unless the old-camera parity
reproduces the current source observations and corrected renders clear the frame.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "RealSaS.MageObservationReframeEvaluation.v1"
ALPHA_THRESHOLD = 8


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if arr.shape != (1024, 1024, 4):
        raise RuntimeError(f"MAGE_REFRAME_IMAGE_SHAPE:{path}:{arr.shape}")
    return arr


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _alpha_metrics(ref, got):
    a = ref[..., 3] >= ALPHA_THRESHOLD
    b = got[..., 3] >= ALPHA_THRESHOLD
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(~a & b))
    fn = int(np.count_nonzero(a & ~b))
    return {
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "recall": float(tp / np.count_nonzero(a)) if np.count_nonzero(a) else 1.0,
        "precision": float(tp / np.count_nonzero(b)) if np.count_nonzero(b) else 1.0,
    }


def _rgb_metrics(ref, got):
    support = (ref[..., 3] >= ALPHA_THRESHOLD) & (got[..., 3] >= ALPHA_THRESHOLD)
    if not np.any(support):
        return {"mean_abs_255": None, "p99_abs_255": None}
    d = np.abs(
        ref[..., :3].astype(np.int16) - got[..., :3].astype(np.int16)
    )[support]
    return {
        "mean_abs_255": float(d.mean()),
        "p99_abs_255": float(np.quantile(d, 0.99)),
    }


def _frame(mask):
    b = _bbox(mask)
    if b is None:
        return {"bbox_xyxy": None, "border_contacts": ["EMPTY"], "minimum_margin_px": None}
    x0, y0, x1, y1 = b
    contacts = []
    if x0 == 0:
        contacts.append("LEFT")
    if x1 == 1023:
        contacts.append("RIGHT")
    if y0 == 0:
        contacts.append("TOP")
    if y1 == 1023:
        contacts.append("BOTTOM")
    return {
        "bbox_xyxy": b,
        "border_contacts": contacts,
        "minimum_margin_px": int(min(x0, y0, 1023 - x1, 1023 - y1)),
    }


def _sheet(current, parity, corrected, output):
    rows = []
    for title, paths in (
        ("CURRENT CROPPED AUTHORITY", current),
        ("PARITY RENDER / OLD CAMERA", parity),
        ("CORRECTED FULL-SUBJECT FRAME", corrected),
    ):
        strip = Image.new("RGBA", (8 * 256, 282), (18, 18, 18, 255))
        draw = ImageDraw.Draw(strip)
        draw.text((8, 6), title, fill=(255, 255, 255, 255))
        for i, path in enumerate(paths):
            with Image.open(path) as im:
                thumb = im.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
            strip.alpha_composite(thumb, (i * 256, 26))
        rows.append(strip)
    sheet = Image.new("RGBA", (8 * 256, 3 * 282), (8, 8, 8, 255))
    for i, row in enumerate(rows):
        sheet.alpha_composite(row, (0, i * 282))
    sheet.save(output)


def run(args):
    stage = Path(args.stage)
    render = Path(args.render)
    current_dir = stage / "current_observations"
    parity_dir = render / "parity_current_camera"
    corrected_dir = render / "corrected_full_subject"

    rows = []
    current_paths = []
    parity_paths = []
    corrected_paths = []
    for view in range(8):
        cp = current_dir / f"V{view}.png"
        pp = parity_dir / f"V{view}.png"
        rp = corrected_dir / f"V{view}.png"
        current_paths.append(cp)
        parity_paths.append(pp)
        corrected_paths.append(rp)
        current = _rgba(cp)
        parity = _rgba(pp)
        corrected = _rgba(rp)
        row = {
            "view": view,
            "current_sha256": _sha(cp),
            "parity_alpha": _alpha_metrics(current, parity),
            "parity_rgb": _rgb_metrics(current, parity),
            "current_frame": _frame(current[..., 3] >= ALPHA_THRESHOLD),
            "parity_frame": _frame(parity[..., 3] >= ALPHA_THRESHOLD),
            "corrected_frame": _frame(corrected[..., 3] >= ALPHA_THRESHOLD),
            "corrected_sha256": _sha(rp),
        }
        rows.append(row)
        print("MAGE_REFRAME_V" + str(view) + "=" + json.dumps(row, sort_keys=True))

    parity_min_iou = min(r["parity_alpha"]["iou"] for r in rows)
    corrected_border_contacts = sum(
        len(r["corrected_frame"]["border_contacts"]) for r in rows
    )
    corrected_min_margin = min(
        r["corrected_frame"]["minimum_margin_px"] for r in rows
    )
    parity_status = (
        "PASS_STRONG_RENDERER_PARITY"
        if parity_min_iou >= 0.995
        else "OPEN_RENDERER_PARITY_NEEDS_TUNING"
    )
    framing_status = (
        "PASS_CORRECTED_NO_BORDER_CONTACT"
        if corrected_border_contacts == 0
        else "FAIL_CORRECTED_FRAME_STILL_CLIPPED"
    )

    contact = render / "OBSERVATION_REFRAME_CONTACT_SHEET.png"
    _sheet(current_paths, parity_paths, corrected_paths, contact)
    result = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETED",
        "renderer_parity_status": parity_status,
        "corrected_framing_status": framing_status,
        "parity_min_alpha_iou": parity_min_iou,
        "corrected_border_contact_count": corrected_border_contacts,
        "corrected_minimum_margin_px": corrected_min_margin,
        "views": rows,
        "contact_sheet_path": str(contact),
        "product_authority_promoted": False,
        "teacher_truth_used": False,
        "source_owner_raster_used": False,
    }
    out = render / "OBSERVATION_REFRAME_EVALUATION.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MAGE_REFRAME_EVALUATION=" + str(out))
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--render", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
