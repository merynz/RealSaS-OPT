from __future__ import annotations

import argparse, json
from pathlib import Path
from time import perf_counter
from PIL import Image

from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import sha256_file
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import run as frozen_preview_run


def gif_from_sheet(sheet_path: Path, out_path: Path, duration_ms: int) -> None:
    sheet = Image.open(sheet_path).convert("RGBA")
    if sheet.width % 4 != 0 or sheet.height % 2 != 0:
        raise RuntimeError(f"UNEXPECTED_FROZEN_SHEET_GEOMETRY:{sheet.size}")
    cell_w = sheet.width // 4
    row_h = sheet.height // 2
    header = 28
    cell_h = row_h - header
    frames = []
    for i in range(4):
        v0 = sheet.crop((i*cell_w, header, (i+1)*cell_w, header+cell_h))
        y2 = row_h + header
        v2 = sheet.crop((i*cell_w, y2, (i+1)*cell_w, y2+cell_h))
        combined = Image.new("RGBA", (cell_w*2, cell_h), (0,0,0,0))
        combined.alpha_composite(v0, (0,0))
        combined.alpha_composite(v2, (cell_w,0))
        target_h = min(640, combined.height)
        target_w = max(1, round(combined.width * target_h / combined.height))
        if (target_w, target_h) != combined.size:
            combined = combined.resize((target_w,target_h), Image.Resampling.LANCZOS)
        frames.append(combined.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(duration_ms), loop=0, disposal=2,
        optimize=False, transparency=0,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()

    out=a.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    preview=out/"frozen_renderer"
    t=perf_counter()
    frozen_preview_run(
        authority_root=a.authority_root.resolve(),
        run_id=a.run_id,
        out_dir=preview,
    )
    render_seconds=perf_counter()-t

    gifs=[]
    for name,duration in (("IDLE",833),("RUN",208),("SLASH",278)):
        sheet=preview/f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        gif=out/f"KNIGHT_{name}_HARMONIC_FIRST_CORRECTED_FROZEN_RENDERER.gif"
        gif_from_sheet(sheet,gif,duration)
        gifs.append({
            "name":name,
            "gif":str(gif),
            "gif_sha256":sha256_file(gif),
            "sheet":str(sheet),
            "sheet_sha256":sha256_file(sheet),
        })

    report={
        "schema":"RealSaS.HarmonicFirstFrozenRendererOnly.v1",
        "status":"PASS",
        "run_id":a.run_id,
        "stage24_qualification_hash":"a6ecf319693a3ad2c37cdba3637e5b0d246ff62f7555b5b635a348805ee0b579",
        "stage24_status":"PASS",
        "frozen_renderer_blob":"7917be02f325ba00b5dcd2b31a786898e45f640f",
        "render_seconds":render_seconds,
        "gifs":gifs,
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("HARMONIC_FIRST_FROZEN_RENDER_PASS="+json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
