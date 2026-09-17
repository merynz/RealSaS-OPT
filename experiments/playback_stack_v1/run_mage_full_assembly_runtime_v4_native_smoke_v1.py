from __future__ import annotations

"""Admit, package, and optionally native-render the complete Mage Runtime-v4 product.

One execution owns the whole product smoke path:
  current full Mage -> render-free admission -> typed render authority ->
  cached Runtime-v4 .rss -> native reference batch render -> 8-view visual GIFs.

The render authority is intentionally NOT a ProductProofBundle. The legacy Runtime-v4
binary field named source_proof_bundle_hash transports the hash, while source_binding
and package manifest explicitly type it as RUNTIME_V4_ADMISSION_CERTIFICATE.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_services.export.runtime_v4_cache import (
    materialize_runtime_v4_archive_cached,
)

import experiments.playback_stack_v1.run_mage_full_assembly_runtime_v4_admission_v1 as admission

SCHEMA = "RealSaS.MageFullAssemblyRuntimeV4NativeSmoke.v1"
AUTHORITY_SCHEMA = "RealSaS.MageRuntimeV4NativeRenderAuthority.v1"
AUTHORITY_KIND = "RUNTIME_V4_ADMISSION_CERTIFICATE"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _promote_certified_clips(projection, certificates):
    certified = {row.clip_id: row for row in certificates}
    clips = []
    for clip in projection.clips:
        cert = certified.get(clip.clip_id)
        if cert is None:
            raise RuntimeError(f"MAGE_V4_NATIVE_CERTIFICATE_MISSING:{clip.clip_id}")
        if not (
            cert.source_backed_active_faces_certified
            and cert.continuity_underlay_certified
            and cert.continuous_embedding_certified
            and cert.body_underlay_always_active_first
        ):
            raise RuntimeError(f"MAGE_V4_NATIVE_CERTIFICATE_NOT_PASS:{clip.clip_id}")
        promoted = replace(clip, runtime_qualified=True)
        validate_runtime_v4_clip(
            projection.contract,
            promoted,
            required_view_ids=admission.VIEW_IDS,
        )
        clips.append(promoted)
    return tuple(clips)


def _stage_textures(admitted, root: Path) -> Path:
    """Materialize exact atlas paths expected by Runtime-v4 without re-encoding."""

    root.mkdir(parents=True, exist_ok=True)
    foreground_dir = Path(admitted["args"].foreground_dir).resolve()
    manifest = admitted["state"]["foreground_manifest"]
    rows = {int(row["view"]): row for row in manifest.get("views") or ()}
    if set(rows) != set(range(8)):
        raise RuntimeError("MAGE_V4_NATIVE_FOREGROUND_VIEW_SET_INCOMPLETE")

    for texture in admitted["projection"].textures:
        view = int(texture.view_id[1:])
        source = foreground_dir / str(rows[view]["atlas_file"])
        target = (root / texture.texture_path).resolve()
        if not source.is_file():
            raise RuntimeError(f"MAGE_V4_NATIVE_TEXTURE_SOURCE_MISSING:{source}")
        if _sha(source) != texture.texture_sha256:
            raise RuntimeError(f"MAGE_V4_NATIVE_TEXTURE_SOURCE_SHA_DRIFT:{texture.view_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and _sha(target) == texture.texture_sha256:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        if _sha(target) != texture.texture_sha256:
            raise RuntimeError(f"MAGE_V4_NATIVE_STAGED_TEXTURE_SHA_DRIFT:{texture.view_id}")
    return root


def _sample_times(duration: float, count: int) -> tuple[float, ...]:
    count = int(count)
    if count < 2:
        raise ValueError("render sample count must be >=2")
    duration = float(duration)
    # Loop visual samples intentionally exclude the exact endpoint because native
    # playback wraps duration -> 0. Continuous certification already covers intervals.
    return tuple(duration * i / count for i in range(count))


def _render_batch(runtime_demo: Path, package: Path, projection, out: Path, sample_count: int):
    if not runtime_demo.is_file():
        raise RuntimeError(f"MAGE_V4_NATIVE_RUNTIME_DEMO_MISSING:{runtime_demo}")
    requests = []
    frame_paths: dict[tuple[str, str], list[Path]] = {}
    for clip in projection.clips:
        times = _sample_times(clip.duration_seconds, sample_count)
        for view_id in admission.VIEW_IDS:
            key = (clip.clip_id, view_id)
            frame_paths[key] = []
            for index, time_seconds in enumerate(times):
                path = out / "frames" / clip.clip_id / view_id / f"frame_{index:03d}.png"
                frame_paths[key].append(path)
                requests.append((clip.clip_id, view_id, time_seconds, path))

    batch_path = out / "RUNTIME_V4_BATCH_RENDER_REQUESTS.tsv"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("w", encoding="utf-8", newline="\n") as handle:
        for clip_id, view_id, time_seconds, path in requests:
            handle.write(f"{clip_id}\t{view_id}\t{time_seconds:.9f}\t{path}\n")

    proc = subprocess.run(
        [str(runtime_demo), str(package), "--batch-file", str(batch_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("MAGE_V4_NATIVE_BATCH_RENDER_FAIL:\n" + proc.stdout)
    if f"batch_rendered={len(requests)}" not in proc.stdout:
        raise RuntimeError("MAGE_V4_NATIVE_BATCH_RENDER_COUNT_DRIFT:\n" + proc.stdout)
    return batch_path, frame_paths, proc.stdout


def _make_contact_gifs(
    projection,
    frame_paths: dict[tuple[str, str], list[Path]],
    out: Path,
    *,
    thumb: int,
):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("MAGE_V4_NATIVE_GIF_REQUIRES_PILLOW") from exc

    thumb = int(thumb)
    if thumb < 64:
        raise ValueError("contact thumbnail size must be >=64")
    outputs = {}
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for clip in projection.clips:
        count = len(frame_paths[(clip.clip_id, admission.VIEW_IDS[0])])
        contact_frames = []
        for frame_index in range(count):
            sheet = Image.new("RGBA", (4 * thumb, 2 * thumb), (0, 0, 0, 0))
            for view_index, view_id in enumerate(admission.VIEW_IDS):
                path = frame_paths[(clip.clip_id, view_id)][frame_index]
                with Image.open(path) as source:
                    image = source.convert("RGBA")
                    image.thumbnail((thumb, thumb), resampling)
                    x = (view_index % 4) * thumb + (thumb - image.width) // 2
                    y = (view_index // 4) * thumb + (thumb - image.height) // 2
                    sheet.alpha_composite(image, (x, y))
            contact_frames.append(sheet)

        gif_path = out / f"{clip.clip_id}_8VIEW_RUNTIME_V4.gif"
        png_path = out / f"{clip.clip_id}_8VIEW_RUNTIME_V4_FRAME0.png"
        contact_frames[0].save(png_path, format="PNG")
        duration_ms = max(
            1,
            int(round(float(clip.duration_seconds) * 1000.0 / max(1, count))),
        )
        contact_frames[0].save(
            gif_path,
            save_all=True,
            append_images=contact_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        outputs[clip.clip_id] = {
            "gif": str(gif_path),
            "gif_sha256": _sha(gif_path),
            "frame0_contact_png": str(png_path),
            "frame0_contact_png_sha256": _sha(png_path),
            "visual_derivative_only": True,
            "native_source_frames_unchanged": True,
        }
    return outputs


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    admitted = admission.run(args)
    admitted["args"] = args
    report = admitted["report"]
    if report.get("status") != "PASS__FULL_MAGE_RUNTIME_V4_RENDER_ADMITTED":
        raise RuntimeError("MAGE_V4_NATIVE_ADMISSION_STATUS_NOT_PASS")

    projection = admitted["projection"]
    product = admitted["product"]
    certificates = admitted["certificates"]
    promoted_clips = _promote_certified_clips(projection, certificates)
    contract_hash = validate_playback_runtime_v4_contract(
        projection.contract,
        required_view_ids=admission.VIEW_IDS,
    )

    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": "PASS__NATIVE_RUNTIME_V4_RENDER_AUTHORIZED",
        "authority_kind": AUTHORITY_KIND,
        "source_admission_schema": report["schema"],
        "source_admission_status": report["status"],
        "source_admission_report_sha256": admitted["report_sha256"],
        "source_product_state_hash": product.product_state_hash,
        "runtime_v4_projection_hash": projection.projection_hash,
        "playback_contract_hash": contract_hash,
        "runtime_qualified_clip_ids": [clip.clip_id for clip in promoted_clips],
        "certificate_hashes": {
            row.clip_id: row.certificate_hash for row in certificates
        },
        "full_product_proof_claimed": False,
        "product_pass_claimed": False,
        "completion_used": False,
        "new_pixels_generated": False,
        "authorization_scope": "NATIVE_RUNTIME_V4_VISUAL_SMOKE_AFTER_RENDER_FREE_ADMISSION",
    }
    authority_path = output_dir / "MAGE_RUNTIME_V4_NATIVE_RENDER_AUTHORITY_V1.json"
    _write_json(authority_path, authority)
    authority_hash = _sha(authority_path)

    texture_root = _stage_textures(
        admitted,
        output_dir / "_runtime_v4_texture_stage",
    )
    package_path = output_dir / "MAGE_FULL_ASSEMBLY_RUNTIME_V4.rss"
    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if str(args.cache_root or "").strip()
        else None
    )
    package = materialize_runtime_v4_archive_cached(
        out_path=package_path,
        texture_root=texture_root,
        contract=projection.contract,
        textures=projection.textures,
        clips=promoted_clips,
        source_product_state_hash=product.product_state_hash,
        source_proof_bundle_hash=authority_hash,
        source_authority_kind=AUTHORITY_KIND,
        required_views=admission.VIEW_IDS,
        cache_root=cache_root,
    )

    native_render = None
    visual_outputs = {}
    if str(args.runtime_demo or "").strip():
        runtime_demo = Path(args.runtime_demo).expanduser().resolve()
        batch_path, frame_paths, stdout = _render_batch(
            runtime_demo,
            package_path,
            projection,
            output_dir / "native_render",
            int(args.render_sample_count),
        )
        native_render = {
            "runtime_demo": str(runtime_demo),
            "batch_file": str(batch_path),
            "batch_file_sha256": _sha(batch_path),
            "request_count": sum(len(rows) for rows in frame_paths.values()),
            "stdout": stdout,
        }
        if not bool(args.no_gifs):
            visual_outputs = _make_contact_gifs(
                projection,
                frame_paths,
                output_dir / "native_render",
                thumb=int(args.contact_thumb),
            )

    result = {
        "schema": SCHEMA,
        "status": (
            "PASS__FULL_MAGE_RUNTIME_V4_NATIVE_RENDER"
            if native_render is not None
            else "PASS__FULL_MAGE_RUNTIME_V4_PACKAGE_ONLY"
        ),
        "source_admission_report_sha256": admitted["report_sha256"],
        "render_authority_sha256": authority_hash,
        "render_authority_kind": AUTHORITY_KIND,
        "source_product_state_hash": product.product_state_hash,
        "runtime_v4_projection_hash": projection.projection_hash,
        "playback_contract_hash": contract_hash,
        "package": package,
        "package_megabytes_decimal": float(package["package_total_bytes"]) / 1_000_000.0,
        "runtime_binary_megabytes_decimal": float(package["runtime_binary_bytes"]) / 1_000_000.0,
        "cache_hit": bool(package.get("cache_hit", False)),
        "native_render": native_render,
        "visual_outputs": visual_outputs,
        "full_product_proof_claimed": False,
        "product_pass_claimed": False,
        "founder_visual_pass_claimed": False,
        "completion_used": False,
        "new_pixels_generated": False,
    }
    result_path = output_dir / "MAGE_FULL_ASSEMBLY_RUNTIME_V4_NATIVE_SMOKE_V1.json"
    _write_json(result_path, result)
    print("MAGE_FULL_ASSEMBLY_RUNTIME_V4_NATIVE_SMOKE_PASS")
    print(json.dumps({
        "result": str(result_path),
        "result_sha256": _sha(result_path),
        "status": result["status"],
        "archive": str(package_path),
        "archive_sha256": package["archive_sha256"],
        "package_megabytes_decimal": result["package_megabytes_decimal"],
        "runtime_binary_megabytes_decimal": result["runtime_binary_megabytes_decimal"],
        "cache_hit": result["cache_hit"],
        "visual_outputs": visual_outputs,
    }, indent=2, sort_keys=True))
    return {
        "result": result,
        "result_path": str(result_path),
        "result_sha256": _sha(result_path),
        "admitted": admitted,
        "promoted_clips": promoted_clips,
        "render_authority": authority,
        "render_authority_sha256": authority_hash,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--p1q-dir", required=True)
    p.add_argument("--foreground-dir", required=True)
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--fit1-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit1-skin", required=True)
    p.add_argument("--expected-p1q-manifest", default="")
    p.add_argument("--expected-foreground-manifest", default="")
    p.add_argument("--expected-assembly-manifest", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-body-source-recall", type=float, default=0.97)
    p.add_argument("--min-body-source-precision", type=float, default=0.995)
    p.add_argument("--min-signed-area2", type=float, default=1.0e-4)
    p.add_argument("--cache-root", default="")
    p.add_argument("--runtime-demo", default="")
    p.add_argument("--render-sample-count", type=int, default=9)
    p.add_argument("--contact-thumb", type=int, default=256)
    p.add_argument("--no-gifs", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
