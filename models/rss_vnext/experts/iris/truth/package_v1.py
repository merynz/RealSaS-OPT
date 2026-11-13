"""Immutable local staging packages; Platform CAS import is a separate step."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .contracts_v1 import validate_manifest


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def array_bytes(value):
    a = np.asarray(value)
    if a.dtype.kind not in "biuf" or a.dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError("numeric/bool arrays only; no pickle/object/string payloads")
    a = np.array(a, dtype=a.dtype.newbyteorder("<"), order="C", copy=True)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, a, version=(1, 0), allow_pickle=False)
    return a, stream.getvalue()


def write_package(destination, envelope, arrays):
    """Publish a fresh derived directory, manifest last. Never replace a path.

    A crash before manifest publication leaves an unsealed directory; callers
    must not treat that directory as a package or retry by overwriting it.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(canonical_json(envelope))
    if value.get("payloads") != {}:
        raise ValueError("writer computes payload descriptors; caller must leave them empty")
    blobs = {}
    value["payloads"] = {}
    for name, data in sorted(arrays.items()):
        a, raw = array_bytes(data)
        value["payloads"][name] = {"path": "arrays/" + name + ".npy",
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                    "dtype": a.dtype.str, "shape": list(a.shape)}
        blobs[name] = raw
    validate_manifest(value)
    raw_manifest = canonical_json(value)
    staging = Path(tempfile.mkdtemp(prefix=".truth-staging-", dir=destination.parent))
    try:
        (staging / "arrays").mkdir()
        for name, raw in blobs.items():
            (staging / "arrays" / (name + ".npy")).write_bytes(raw)
        (staging / "manifest.json").write_bytes(raw_manifest)
        # Exclusive reservation prevents a second writer replacing even an empty
        # destination. Commit files inside the reserved directory, manifest last.
        destination.mkdir()
        try:
            os.rename(staging / "arrays", destination / "arrays")
            os.rename(staging / "manifest.json", destination / "manifest.json")
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return hashlib.sha256(raw_manifest).hexdigest()


def verify_package(directory, *, expected_manifest_sha256):
    root = Path(directory)
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("manifest identity mismatch")
    value = validate_manifest(json.loads(raw))
    arrays = {}
    for name, desc in value["payloads"].items():
        path = root / desc["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("payload escapes package")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != desc["sha256"]:
            raise ValueError("payload identity mismatch: " + name)
        a = np.load(io.BytesIO(data), allow_pickle=False)
        if a.dtype.str != desc["dtype"] or list(a.shape) != desc["shape"]:
            raise ValueError("payload shape/dtype mismatch")
        arrays[name] = a
    return value, arrays
