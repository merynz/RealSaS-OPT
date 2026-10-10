"""Existing native renderer consumes explicit research layers/order/positions."""
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.visual_oracle_package_v1 import package_frame
from compiler.realsas_compiler_core.visual_oracle_io_v1 import file_ref, write_json, sha
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload


def render_reference(ctx):
    evidence = stage_output_payload(ctx, "47_AUTHORED_VISUAL_ORACLE", "RealSaS.AuthoredVisualOracleEvidence.v1")
    settings = ctx["run_manifest"]["oracle_render"]
    player = Path(settings["player_path"])
    if sha(player.read_bytes()) != settings["player_sha256"]:
        raise ValueError("ORACLE_NATIVE_PLAYER_DRIFT")
    root = ctx["run_root"] / "artifacts" / "48_NATIVE_VISUAL_ORACLE"
    root.mkdir(parents=True, exist_ok=True)
    rows, outputs = [], []
    for variant in settings["variants"]:
        for index, frame in enumerate(evidence["frames"]):
            stem = root / f"{variant}_{index:03d}"
            rss, rgba, owner = stem.with_suffix(".rss"), stem.with_suffix(".rgba"), stem.with_suffix(".owner")
            face_owners = package_frame(evidence, frame, rss, variant)
            result = subprocess.run([str(player), str(rss), "--clip", "oracle", "--view", "V0", "--frame", "0",
                "--out-rgba", str(rgba), "--out-owner", str(owner)], text=True, capture_output=True, check=True, timeout=60)
            if "renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D" not in result.stdout:
                raise ValueError("ORACLE_UNEXPECTED_NATIVE_CONSUMER")
            size = evidence["resolution"]
            pixels = np.fromfile(rgba, dtype=np.uint8).reshape(size, size, 4)
            png = stem.with_suffix(".png"); Image.fromarray(pixels).save(png)
            refs = {"rss": file_ref(rss, "RealSaS.RuntimePackage.v2"), "rgba": file_ref(rgba, "RealSaS.RGBA8Raster.v1"),
                    "owner": file_ref(owner, "RealSaS.NativeFaceOwnerRaster.v1"), "png": file_ref(png, "image/png")}
            outputs.extend(refs.values())
            rows.append({"variant": variant, "frame_index": index, "clip": frame["clip"], "time_ms": frame["time_ms"],
                         "face_owners": face_owners, "files": refs, "native_receipt": result.stdout.strip()})
    payload = {"schema": "RealSaS.NativeVisualOracleConsumption.v1", "evidence_sha256":
        next(o["sha256"] for s in ctx["ledger"]["stages"] if s["id"] == "47_AUTHORED_VISUAL_ORACLE" for o in s["outputs"] if o["schema"] == evidence["schema"]),
        "player": settings, "rows": rows, "full_visual_qualification": False}
    outputs.append(write_json(root / "consumption.json", payload))
    return {"status": "PASS", "outputs": outputs, "diagnostics": {"native_renders": len(rows), "product_qualification": False}}
