#!/usr/bin/env python3
"""Create/reuse a per-runner Python environment without serializing the shared CI queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

SCHEMA = "RealSaS.PlatformCIBootstrap.v2"


def _requirement_file_state(paths: list[Path] | tuple[Path, ...]) -> list[dict]:
    rows: list[dict] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"REQUIREMENTS_FILE_MISSING:{path}")
        data = path.read_bytes()
        rows.append({
            "name": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })
    return rows


def fingerprint(requirements: list[str], requirement_files: list[Path] | tuple[Path, ...] = ()) -> str:
    payload = {
        "python": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        "requirements": requirements,
        "requirement_files": _requirement_file_state(requirement_files),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ensure_venv(
    cache_root: Path,
    requirements: list[str],
    *,
    requirement_files: list[Path] | tuple[Path, ...] = (),
    pip_cache: Path | None = None,
) -> dict:
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    requirement_files = tuple(Path(path).expanduser().resolve() for path in requirement_files)
    requirement_file_state = _requirement_file_state(requirement_files)
    digest = fingerprint(requirements, requirement_files)
    venv_root = cache_root / "venvs"
    venv_root.mkdir(parents=True, exist_ok=True)
    target = venv_root / digest
    lock_path = venv_root / f"{digest}.lock"
    started = time.monotonic()
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        stamp = target / ".realsas-env.json"
        python = target / "bin" / "python"
        if python.is_file() and stamp.is_file():
            try:
                state = json.loads(stamp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
            if state.get("fingerprint") == digest:
                os.utime(stamp, None)
                return {
                    "schema": SCHEMA,
                    "status": "CACHE_HIT",
                    "fingerprint": digest,
                    "venv": str(target),
                    "bin": str(target / "bin"),
                    "wall_seconds": round(time.monotonic() - started, 6),
                }

        tmp_parent = cache_root / ".tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"venv-{digest[:12]}-", dir=tmp_parent))
        backup = target.with_name(target.name + ".old")
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(tmp)
            env = os.environ.copy()
            if pip_cache is not None:
                pip_cache = pip_cache.expanduser().resolve()
                pip_cache.mkdir(parents=True, exist_ok=True)
                env["PIP_CACHE_DIR"] = str(pip_cache)
            command = [
                str(tmp / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--progress-bar",
                "off",
            ]
            for path in requirement_files:
                command.extend(["-r", str(path)])
            command.extend(requirements)
            if requirement_files or requirements:
                subprocess.run(command, check=True, env=env)
            (tmp / ".realsas-env.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "fingerprint": digest,
                        "requirements": requirements,
                        "requirement_files": requirement_file_state,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                target.rename(backup)
            tmp.rename(target)
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return {
            "schema": SCHEMA,
            "status": "CREATED",
            "fingerprint": digest,
            "venv": str(target),
            "bin": str(target / "bin"),
            "wall_seconds": round(time.monotonic() - started, 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pip-cache", type=Path)
    parser.add_argument("--require", action="append", dest="requirements", default=[])
    parser.add_argument("--requirements-file", action="append", type=Path, dest="requirement_files", default=[])
    parser.add_argument("--github-path", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = ensure_venv(
        args.cache_root,
        args.requirements,
        requirement_files=args.requirement_files,
        pip_cache=args.pip_cache,
    )
    if args.github_path:
        with args.github_path.open("a", encoding="utf-8") as handle:
            handle.write(report["bin"] + "\n")
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
