from __future__ import annotations

"""Fail-closed bridge to the exact historical RealSaS v0.5 downstream.

This module does not reimplement FK/LBS/ARAP/XPBD.  It verifies the full
historical source ZIP, safely extracts it, then runs a clean subprocess whose
PYTHONPATH points at the historical compiler tree.  The worker consumes only a
current-product projection JSON and returns evaluated 2D mesh frames.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys
import tempfile
import zipfile

V05_SOURCE_SHA256 = "03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130"
V05_SOURCE_BASENAME = "RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip"
CRITICAL_MEMBER_SHA256 = {
    "compiler/realsas_deformation/production_arap.py": "9b1d45ddca06bc26fdb830181b3a454e422eab399c04b5e203731fa045734cee",
    "compiler/realsas_deformation/secondary_xpbd.py": "a21982e7a9724e65ca89d375a20f79aae38967c1a663cb068d2dcd26d5ae7539",
    "compiler/realsas_deformation/contact_sdf.py": "4db53c35355cb949e0c4613c9e8575351cd93da5a2b15fde0e0b601fdf6327a4",
    "compiler/realsas_reference_runtime/deformation_evaluator.py": "16cca7f09f81959c9ecbb3365d9c736e1f7de6196007629cf05ea78e0b841bd4",
}


@dataclass(frozen=True)
class HistoricalRuntimeResult:
    output: dict[str, Any]
    source_zip_sha256: str
    worker_stdout: str
    worker_stderr: str


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_v05_source_zip(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"V05_SOURCE_ZIP_NOT_FOUND:{p}")
    actual = _sha_file(p)
    if actual != V05_SOURCE_SHA256:
        raise RuntimeError(f"V05_SOURCE_ZIP_SHA_MISMATCH:{actual}!={V05_SOURCE_SHA256}")
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        for suffix, expected in CRITICAL_MEMBER_SHA256.items():
            hits = [name for name in names if name.replace("\\", "/").endswith(suffix)]
            if len(hits) != 1:
                raise RuntimeError(f"V05_CRITICAL_MEMBER_RESOLUTION_FAIL:{suffix}:hits={len(hits)}")
            actual_member = sha256(zf.read(hits[0])).hexdigest()
            if actual_member != expected:
                raise RuntimeError(f"V05_CRITICAL_MEMBER_SHA_MISMATCH:{suffix}:{actual_member}!={expected}")
    return p


def _safe_extract(zip_path: Path, target: Path) -> Path:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            dest = (target / info.filename).resolve()
            if target != dest and target not in dest.parents:
                raise RuntimeError(f"V05_UNSAFE_ZIP_MEMBER:{info.filename}")
        zf.extractall(target)
    candidates = [p for p in target.rglob("compiler") if (p / "realsas_reference_runtime" / "deformation_evaluator.py").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"V05_COMPILER_ROOT_RESOLUTION_FAIL:hits={len(candidates)}")
    return candidates[0]


def run_historical_reference_runtime(
    request: dict[str, Any],
    *,
    source_zip: str | Path,
    worker_script: str | Path,
    python_executable: str | None = None,
    timeout_seconds: int = 180,
) -> HistoricalRuntimeResult:
    source = verify_v05_source_zip(source_zip)
    worker = Path(worker_script).resolve()
    if not worker.is_file():
        raise FileNotFoundError(f"HISTORICAL_RUNTIME_WORKER_NOT_FOUND:{worker}")
    with tempfile.TemporaryDirectory(prefix="realsas_v05_runtime_") as td:
        root = Path(td)
        compiler_root = _safe_extract(source, root / "src")
        request_path = root / "request.json"
        output_path = root / "output.json"
        request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(compiler_root)
        cmd = [python_executable or sys.executable, str(worker), str(request_path), str(output_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_seconds)
        if proc.returncode != 0:
            raise RuntimeError(
                "HISTORICAL_RUNTIME_WORKER_FAIL:"
                f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_path.is_file():
            raise RuntimeError("HISTORICAL_RUNTIME_WORKER_NO_OUTPUT")
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if output.get("status") != "PASS":
            raise RuntimeError(f"HISTORICAL_RUNTIME_OUTPUT_FAIL:{output}")
        return HistoricalRuntimeResult(output, V05_SOURCE_SHA256, proc.stdout, proc.stderr)
