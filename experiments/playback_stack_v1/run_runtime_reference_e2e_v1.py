from __future__ import annotations

"""Generic RealSaS compiled-puppet -> native runtime -> PNG E2E probe.

This runner is intentionally subject-agnostic. It does not know Mage, FIT1, FITK,
or any model family. A subject is only an opaque provenance label.

The runner exercises the shipped/native presentation path rather than a notebook
renderer. Runtime-v2 archives may be exercised in STRUCTURAL_SMOKE mode, but they
can never produce a depth-qualified/founder visual PASS. Runtime-v3 depth
qualification is required for that claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


REPORT_SCHEMA = "RealSaS.PlaybackReferenceE2EProbe.v1"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def render_frame(*, runtime_demo: Path, archive: Path, clip: str, view: str, time: float, out_png: Path) -> dict:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runtime_demo), str(archive),
        "--clip", str(clip),
        "--view", str(view),
        "--time", f"{float(time):.9f}",
        "--out", str(out_png),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "E2E_NATIVE_REFERENCE_RENDER_FAILED:"
            f"view={view}:time={time}:rc={proc.returncode}\n{proc.stdout}"
        )
    if not out_png.is_file():
        raise RuntimeError(f"E2E_REFERENCE_RENDER_DID_NOT_CREATE_PNG:{out_png}")
    raw = out_png.read_bytes()
    if not raw.startswith(PNG_MAGIC):
        raise RuntimeError(f"E2E_REFERENCE_RENDER_OUTPUT_NOT_PNG:{out_png}")
    return {
        "view": view,
        "time_seconds": float(time),
        "path": str(out_png),
        "sha256": sha256_file(out_png),
        "stdout": proc.stdout.strip(),
    }


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
            ))

    claim = "RUNTIME_DEPTH_QUALIFIED_E2E" if runtime["depth_qualified"] else "STRUCTURAL_SMOKE_ONLY"
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
        "runtime_package_sha256": sha256_file(archive) if archive.is_file() else None,
        "native_reference_renderer": str(runtime_demo),
        "native_reference_renderer_sha256": sha256_file(runtime_demo),
        "frames": frames,
    }
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "PLAYBACK_REFERENCE_E2E_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS__" + claim,
        "report": str(report_path),
        "frame_count": len(frames),
    }, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rss", required=True, help="Compiled .rss/.realsas archive or unpacked package")
    ap.add_argument("--runtime-demo", required=True, help="Native C++ realsas_runtime_demo executable")
    ap.add_argument("--out", required=True)
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
