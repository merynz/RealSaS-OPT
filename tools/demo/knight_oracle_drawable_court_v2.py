from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
import tools.demo.knight_oracle_drawable_court_v1 as v1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _linear_pm_alpha_composite_layers(
    layer_rgba: dict[int, np.ndarray], order: list[int]
) -> np.ndarray:
    """Composite back-to-front using the same CAA color authority as rasterization.

    Transport inputs are straight-alpha sRGB RGBA8. Composition happens in
    linear-light premultiplied RGBA and converts back only at the output boundary.
    """
    if not order:
        raise RuntimeError("DRAWABLE_ORACLE_EMPTY_ORDER")
    first = np.asarray(layer_rgba[order[0]], dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] != 4:
        raise RuntimeError("DRAWABLE_ORACLE_LAYER_RGBA_SHAPE")
    canvas = np.zeros(first.shape, dtype=np.float64)
    for group in order:
        src_u8 = np.asarray(layer_rgba[group], dtype=np.uint8)
        if src_u8.shape != first.shape:
            raise RuntimeError("DRAWABLE_ORACLE_LAYER_SHAPE_DRIFT")
        src = straight_srgb_rgba_u8_to_premultiplied_linear(src_u8)
        src_a = src[..., 3:4]
        canvas[..., :3] = src[..., :3] + canvas[..., :3] * (1.0 - src_a)
        canvas[..., 3:4] = src_a + canvas[..., 3:4] * (1.0 - src_a)
    return premultiplied_linear_to_straight_srgb_u8(canvas)


def run(*, authority_root: Path, run_id: str, out_dir: Path) -> dict:
    # Scientific design is inherited unchanged from preregistered V1. Only the
    # evaluator compositor is corrected to the canonical CAA color contract.
    v1._alpha_composite_layers = _linear_pm_alpha_composite_layers
    report = v1.run(
        authority_root=authority_root,
        run_id=run_id,
        out_dir=out_dir,
    )
    report = dict(report)
    report["evaluator_revision"] = "KNIGHT_ORACLE_DRAWABLE_COURT_V2_LINEAR_PM"
    report["layer_compositor"] = "CAA_LINEAR_LIGHT_PREMULTIPLIED_OVER"
    report["scientific_design_changed_from_prereg_v1"] = False
    report["baseline_gate_reused_from_sealed_receipt"] = True
    report_path = out_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_ORACLE_DRAWABLE_COURT_V2="
        + json.dumps(
            {
                "status": report.get("status"),
                "evaluator_revision": report["evaluator_revision"],
                "layer_compositor": report["layer_compositor"],
                "report_sha256": _sha256(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        authority_root=a.authority_root,
        run_id=a.run_id,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    main()
