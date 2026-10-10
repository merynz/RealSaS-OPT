"""Exact-byte, resumable render witnesses; never mints product authority."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkout_identity(root: Path, expected_sha: str) -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if sha != expected_sha:
        raise RuntimeError("RENDER_CHECKOUT_SHA_MISMATCH")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError("RENDER_CHECKOUT_DIRTY__SOURCE_OVERLAY_FORBIDDEN")
    return sha


class RenderCheckpoint:
    def __init__(self, root: Path, contract: dict):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "CHECKPOINT.json"
        self.contract = contract
        self.sha = identity(contract)
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            if (self.value.get("contract_sha256") != self.sha
                    or self.value.get("contract") != contract):
                raise RuntimeError("RENDER_RESUME_CONTRACT_DRIFT")
        else:
            self.value = {"schema": "RealSaS.RenderCheckpoint.v1", "contract": contract,
                          "contract_sha256": self.sha, "frames": {}}
            atomic_json(self.path, self.value)

    def read(self, key: str) -> tuple[Path, dict] | None:
        row = self.value["frames"].get(key)
        if row is None:
            return None
        path = (self.root / row["file"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or digest(path) != row["sha256"]:
            raise RuntimeError("RENDER_CHECKPOINT_FRAME_BYTES_DRIFT:" + key)
        return path, row["measurement"]

    def commit(self, key: str, path: Path, measurement: dict) -> None:
        relative = path.resolve().relative_to(self.root)
        if self.read(key) is not None:
            raise RuntimeError("RENDER_CHECKPOINT_FRAME_ALREADY_COMMITTED:" + key)
        self.value["frames"][key] = {"file": str(relative), "sha256": digest(path),
                                     "measurement": measurement}
        atomic_json(self.path, self.value)
