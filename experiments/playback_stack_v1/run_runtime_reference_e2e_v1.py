from __future__ import annotations

"""Generic RealSaS compiled-puppet -> native runtime -> PNG E2E probe.

This runner is intentionally subject-agnostic. It does not know Mage, FIT1, FITK,
or any model family. A subject is only an opaque provenance label.

The runner exercises the shipped/native presentation path rather than a notebook
renderer. Runtime-v2 archives may be exercised in STRUCTURAL_SMOKE mode, but they
can never produce a depth-qualified/founder visual PASS. Runtime-v3 depth
qualification is required for that claim.

Native reference renders are content-addressed. An unchanged exact runtime package,
renderer binary and (clip, view, time) node is restored from a hash-verified cache;
all restored PNGs still pass the same PNG validation as cold native renders.
"""

import argparse
import json
from pathlib import Path
import zipfile

from compiler.realsas_compiler_services.cache.reference_render import (
    NativeReferenceRenderCache,
    REFERENCE_RENDER_CACHE_PRODUCER,
)


REPORT_SCHEMA = "RealSaS.PlaybackReferenceE2EProbe.v1"


def read_runtime_manifest(archive: Path) -> dict:
    if archive.is_dir():
        p = archive / "manifest.json"
        if not p.is_file():
            raise RuntimeError("E2E_RUNTIME_MANIFEST_MISSING")
        return json.loads(p.read_text(encoding="utf-8"))
    if not archive.is_file():
        raise RuntimeError(f"E2E_RUNTIME_PACKAGE_MISSING:{archive}")
    with zipfile.ZipFile(archive, "r") as z:
        try:
            return json.loads(z.read("manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise RuntimeError("E2E_RUNTIME_MANIFEST_MISSING") from exc


def classify_runtime(manifest: dict) -> dict:
    schema = str(manifest.get("schema_version") or "")
    # Runtime v2 is explicitly no-depth. Future v3 packages must opt in with
    # explicit rendering capabilities rather than inheriting an assumption.
    caps = dict(manifest.get("render_capabilities") or {})
    depth_test = caps.get("depth_test") is True
    posed_depth = caps.get("posed_vertex_depth") is True
    full_surface = caps.get("full_surface_geometry_authority") is True
    depth_qualified = bool(depth_test and posed_depth and full_surface)
    return {
        "schema_version": schema,
        "depth_test": depth_test,
        "posed_vertex_depth": posed_depth,
        "full_surface_geometry_authority": full_surface,
        "depth_qualified": depth_qualified,
    }


def parse_csv(text: str) -> tuple[str, ...]:
    rows = tuple(x.strip() for x in str(text).split(",") if x.strip())
    if not rows:
        raise ValueError("empty csv list")
    return rows


def parse_times(text: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in parse_csv(text))
    if any(x < 0.0 for x in out):
        raise ValueError("times must be nonnegative")
    return out


def render_frame(
    *,
    runtime_demo: Path,
    archive: Path,
    clip: str,
    view: str,
    time: float,
    out_png: Path,
    render_cache: NativeReferenceRenderCache | None = None,
    cache_root: Path | None = None,
) -> dict:
    cache = render_cache or NativeReferenceRenderCache(
        runtime_package=archive,
        runtime_demo=runtime_demo,
        cache_root=cache_root,
    )
    return cache.render(
        clip=clip,
        view=view,
        time_seconds=time,
        out_png=out_png,
    )


def run(args: argparse.Namespace) -> dict:
    archive = Path(args.rss).resolve()
    runtime_demo = Path(args.runtime_demo).resolve()
    out = Path(args.out).resolve()
    if not runtime_demo.is_file():
        raise RuntimeError(f"E2E_RUNTIME_DEMO_MISSING:{runtime_demo}")

    manifest = read_runtime_manifest(archive)
    runtime = classify_runtime(manifest)
    if args.require_depth_v3 and not runtime["depth_qualified"]:
        raise RuntimeError(
            "E2E_DEPTH_QUALIFIED_RUNTIME_REQUIRED:"
            "runtime-v2/no-depth output may be used only as STRUCTURAL_SMOKE"
        )

    views = parse_csv(args.views)
    if tuple(views) != tuple(f"V{i}" for i in range(8)) and args.require_exact_8_views:
        raise RuntimeError(f"E2E_REQUIRES_EXACT_V0_TO_V7:{views}")
    times = parse_times(args.times)

    cache_root_arg = getattr(args, "cache_root", None)
    render_cache = NativeReferenceRenderCache(
        runtime_package=archive,
        runtime_demo=runtime_demo,
        cache_root=(Path(cache_root_arg).resolve() if cache_root_arg else None),
    )

    frames = []
    for view in views:
        for index, t in enumerate(times):
            target = out / "frames" / str(args.clip) / view / f"F{index:04d}_T{t:.6f}.png"
            frames.append(render_frame(
                runtime_demo=runtime_demo,
                archive=archive,
                clip=args.clip,
                view=view,
                time=t,
                out_png=target,
                render_cache=render_cache,
            ))

    claim = "RUNTIME_DEPTH_QUALIFIED_E2E" if runtime["depth_qualified"] else "STRUCTURAL_SMOKE_ONLY"
    cache_hits = sum(1 for frame in frames if frame.get("render_cache_hit") is True)
    cache_misses = len(frames) - cache_hits
    package_identity = render_cache.package_identity
    renderer_identity = render_cache.renderer_identity
    report = {
        "schema": REPORT_SCHEMA,
        "subject_id": str(args.subject_id),
        "subject_is_fixture_not_pipeline_policy": True,
        "goal": "AUTOMATIC_8_DIRECTION_SPINE_CLASS_PUPPET_PLAYBACK",
        "runtime": runtime,
        "claim_class": claim,
        "founder_visual_pass_claimed": False,
        "appearance_product_pass_claimed": False,
        "clip_id": str(args.clip),
        "views": list(views),
        "times": list(times),
        "runtime_package": str(archive),
        # Preserve the historical file-only field while also exposing a stable
        # content identity for unpacked runtime directories.
        "runtime_package_sha256": package_identity["sha256"] if package_identity["kind"] == "file" else None,
        "runtime_package_content_sha256": package_identity["sha256"],
        "native_reference_renderer": str(runtime_demo),
        "native_reference_renderer_sha256": renderer_identity["sha256"],
        "reference_render_cache": {
            "producer_fingerprint": REFERENCE_RENDER_CACHE_PRODUCER,
            "cache_root": str(render_cache.cache.root),
            "node_count": len(frames),
            "hit_count": cache_hits,
            "miss_count": cache_misses,
            "all_nodes_hit": bool(frames) and cache_misses == 0,
            "cache_is_acceleration_not_proof_authority": True,
        },
        "frames": frames,
    }
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "PLAYBACK_REFERENCE_E2E_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS__" + claim,
        "report": str(report_path),
        "frame_count": len(frames),
        "render_cache_hits": cache_hits,
        "render_cache_misses": cache_misses,
    }, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rss", required=True, help="Compiled .rss/.realsas archive or unpacked package")
    ap.add_argument("--runtime-demo", required=True, help="Native C++ realsas_runtime_demo executable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", default=None, help="Persistent content-addressed cache root; defaults to REALSAS_CACHE_ROOT or ~/.cache/realsas")
    ap.add_argument("--subject-id", default="UNNAMED_TEST_SUBJECT")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--views", default=",".join(f"V{i}" for i in range(8)))
    ap.add_argument("--times", default="0.0")
    ap.add_argument("--require-exact-8-views", action="store_true", default=True)
    ap.add_argument("--require-depth-v3", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
