from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

EXPECTED_UFBX_REVISION = "5955c5c0b042ac2dc6f32955ab206aaec0620cfc"
SCHEMA_VERSION = "RealSaS.UFBXStaticCensusProbeBuildReceipt.v1"
STATUS = "VERIFIED_PINNED_UFBX_BUILD"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _run(argv: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"UFBX_BUILD_COMMAND_FAILED:{completed.returncode}:{' '.join(argv)}:{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def build_probe_with_receipt_v1(
    *,
    ufbx_repo: Path,
    probe_source: Path,
    output_binary: Path,
    compiler: str = "cc",
) -> dict:
    ufbx_repo = ufbx_repo.resolve()
    probe_source = probe_source.resolve()
    output_binary = output_binary.resolve()
    if not (ufbx_repo / ".git").exists():
        raise ValueError("UFBX_BUILD_REPO_NOT_GIT_CHECKOUT")
    if not probe_source.is_file():
        raise ValueError("UFBX_BUILD_PROBE_SOURCE_MISSING")
    ufbx_c = ufbx_repo / "ufbx.c"
    ufbx_h = ufbx_repo / "ufbx.h"
    if not ufbx_c.is_file() or not ufbx_h.is_file():
        raise ValueError("UFBX_BUILD_SOURCE_FILES_MISSING")

    head = _run(["git", "rev-parse", "HEAD"], cwd=ufbx_repo)
    if head != EXPECTED_UFBX_REVISION:
        raise ValueError(f"UFBX_BUILD_REVISION_MISMATCH:{head}")
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ufbx_repo)
    if dirty:
        raise ValueError("UFBX_BUILD_WORKTREE_DIRTY")

    compiler_path_raw = shutil.which(compiler)
    if compiler_path_raw is None:
        raise ValueError("UFBX_BUILD_COMPILER_NOT_FOUND")
    compiler_path = Path(compiler_path_raw).resolve()
    compiler_version = _run([str(compiler_path), "--version"]).splitlines()[0]
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(compiler_path),
        "-O2",
        "-std=c99",
        "-I",
        str(ufbx_repo),
        str(probe_source),
        str(ufbx_c),
        "-lm",
        "-o",
        str(output_binary),
    ]
    _run(argv)
    if not output_binary.is_file() or output_binary.stat().st_size <= 0:
        raise RuntimeError("UFBX_BUILD_BINARY_MISSING")

    receipt = {
        "schema": SCHEMA_VERSION,
        "status": STATUS,
        "expected_ufbx_revision": EXPECTED_UFBX_REVISION,
        "observed_ufbx_revision": head,
        "ufbx_worktree_clean": True,
        "ufbx_c_sha256": _sha256_file(ufbx_c),
        "ufbx_h_sha256": _sha256_file(ufbx_h),
        "probe_source_sha256": _sha256_file(probe_source),
        "compiler_path": str(compiler_path),
        "compiler_sha256": _sha256_file(compiler_path),
        "compiler_version": compiler_version,
        "compile_argv": argv,
        "binary_sha256": _sha256_file(output_binary),
        "binary_size": int(output_binary.stat().st_size),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_build_receipt_v1(receipt: dict, *, expected_probe_source_sha256: str, binary_path: Path) -> None:
    if receipt.get("schema") != SCHEMA_VERSION or receipt.get("status") != STATUS:
        raise ValueError("UFBX_BUILD_RECEIPT_SCHEMA_OR_STATUS_INVALID")
    if receipt.get("expected_ufbx_revision") != EXPECTED_UFBX_REVISION:
        raise ValueError("UFBX_BUILD_RECEIPT_EXPECTED_REVISION_INVALID")
    if receipt.get("observed_ufbx_revision") != EXPECTED_UFBX_REVISION:
        raise ValueError("UFBX_BUILD_RECEIPT_OBSERVED_REVISION_INVALID")
    if receipt.get("ufbx_worktree_clean") is not True:
        raise ValueError("UFBX_BUILD_RECEIPT_WORKTREE_NOT_CLEAN")
    if receipt.get("probe_source_sha256") != expected_probe_source_sha256:
        raise ValueError("UFBX_BUILD_RECEIPT_PROBE_SOURCE_SHA_MISMATCH")
    if receipt.get("receipt_sha256") != _canonical_sha256(receipt):
        raise ValueError("UFBX_BUILD_RECEIPT_SELF_HASH_MISMATCH")
    binary_path = binary_path.resolve()
    if not binary_path.is_file():
        raise ValueError("UFBX_BUILD_RECEIPT_BINARY_MISSING")
    if receipt.get("binary_sha256") != _sha256_file(binary_path):
        raise ValueError("UFBX_BUILD_RECEIPT_BINARY_SHA_MISMATCH")
    if int(receipt.get("binary_size", -1)) != binary_path.stat().st_size:
        raise ValueError("UFBX_BUILD_RECEIPT_BINARY_SIZE_MISMATCH")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ufbx-repo", required=True, type=Path)
    parser.add_argument("--probe-source", type=Path, default=Path(__file__).with_name("ufbx_static_census_probe_v3.c"))
    parser.add_argument("--output-binary", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--compiler", default="cc")
    args = parser.parse_args()
    receipt = build_probe_with_receipt_v1(
        ufbx_repo=args.ufbx_repo,
        probe_source=args.probe_source,
        output_binary=args.output_binary,
        compiler=args.compiler,
    )
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


__all__ = [
    "EXPECTED_UFBX_REVISION",
    "SCHEMA_VERSION",
    "STATUS",
    "build_probe_with_receipt_v1",
    "validate_build_receipt_v1",
]

if __name__ == "__main__":
    main()
