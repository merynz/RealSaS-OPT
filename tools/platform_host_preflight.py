#!/usr/bin/env python3
"""Bounded, read-only evidence inventory; never imports or qualifies a result."""
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def inventory_report(root, inventory, *, deadline_seconds=20, max_files=100000):
    root = Path(root).resolve()
    if inventory.get("schema") != "RealSaS.ExternalInputInventory.v1":
        raise ValueError("INPUT_INVENTORY_SCHEMA_INVALID")
    required = inventory["required_files"]
    names = {row["name"] for row in required}
    if len(names) != len(required) or any(Path(name).name != name for name in names):
        raise ValueError("INPUT_INVENTORY_NAMES_INVALID")
    found = {name: [] for name in names}
    started = time.monotonic()
    count = 0
    complete = root.is_dir()
    if complete:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in sorted(dirs) if not (Path(directory) / name).is_symlink()]
            if count + len(files) > max_files or time.monotonic() - started > deadline_seconds:
                complete = False
                break
            count += len(files)
            for name in sorted(names.intersection(files)):
                path = Path(directory) / name
                if path.is_symlink() or len(found[name]) >= 20:
                    continue
                expected = next(row for row in required if row["name"] == name)
                size = path.stat().st_size
                digest = ""
                # Do not read arbitrary oversized files which merely share a name.
                if size == expected["size_bytes"]:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                found[name].append({"path": str(path), "size_bytes": size, "sha256": digest,
                                    "exact": size == expected["size_bytes"] and digest == expected["sha256"]})
    rows = [{**row, "matches": found[row["name"]],
             "exact_available": any(match["exact"] for match in found[row["name"]])}
            for row in required]
    return {"schema": "RealSaS.PlatformHostPreflight.v1", "authority_root": str(root),
            "scan_complete": complete, "scanned_files": count,
            "wall_seconds": round(time.monotonic() - started, 6), "files": rows,
            "input_bytes_available": complete and all(row["exact_available"] for row in rows),
            "scientific_pass_claimed": False, "qualification_minted": False,
            "remaining_bindings": inventory.get("remaining_bindings", [])}


def endpoint_available(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inventory_report(args.authority_root, json.loads(args.inventory.read_text()))
    report["default_local_endpoints"] = {"operator_api_8080": endpoint_available(8080),
                                         "temporal_7233": endpoint_available(7233)}
    report["endpoint_scope"] = "DEFAULT_LOOPBACK_PORTS_ONLY__NOT_SERVICE_IDENTITY_OR_REMOTE_CONFIGURATION_PROOF"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
