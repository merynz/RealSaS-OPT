#!/usr/bin/env python3
"""Independent synthetic Blender Image.pixels origin witness (no original art).

Writes a 2x2 PNG with four distinct colors through Blender's flat Image.pixels
buffer, reopens it with bpy Image.pixels, and lets an external PIL decoder
independently assert which authored row is at the PNG TOP. No source target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    assert tuple(bpy.app.version) >= (4, 5, 0)
    # First y-buffer row is bottom row:
    # BL red, BR yellow, TL blue, TR green.
    colors_bottom_first = np.asarray([
        [(1., 0., 0., 1.), (1., 1., 0., 1.)],
        [(0., 0., 1., 1.), (0., 1., 0., 1.)],
    ], dtype=np.float32)
    img = bpy.data.images.new("OriginalCatPixelOriginWitness",
                              width=2, height=2, alpha=True)
    img.pixels.foreach_set(colors_bottom_first.ravel())
    dest = args.output.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.filepath_raw = str(dest)
    img.file_format = "PNG"
    img.save()
    assert dest.is_file() and dest.stat().st_size > 0
    reloaded = bpy.data.images.load(str(dest), check_existing=False)
    raw = np.empty(16, dtype=np.float32)
    reloaded.pixels.foreach_get(raw)
    sample = raw.reshape((2, 2, 4))
    assert np.allclose(sample, colors_bottom_first, atol=1e-3, rtol=0)
    print(json.dumps({
        "status": "BLENDER_PIXELS_FLAT_BUFFER_BOTTOM_FIRST_ROUNDTRIP",
        "image_size": [2, 2],
        "blender_version": bpy.app.version_string,
        "raw_first_buffer_pixel_is_red_bottom_left": True,
        "raw_last_buffer_row_is_blue_green_top": True,
        "independent_png_top_row_verification_required": True,
        "no_original_source_bytes": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
