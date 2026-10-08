#!/usr/bin/env python3
"""Foreground service entrypoints; systemd, not an Actions job, owns children."""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("postgres", "migrate"))
    args = parser.parse_args()
    root = Path(os.environ["REALSAS_DEPLOY_ROOT"])
    from pgserver._commands import POSTGRES_BIN_PATH
    if args.service == "postgres":
        # A dedicated cluster, private Unix socket, no TCP database listener.
        binary = POSTGRES_BIN_PATH / "postgres"
        os.execv(str(binary), [str(binary), "-D", str(root / "data/postgres"),
                              "-k", str(root / "data/socket"),
                              "-c", "listen_addresses=", "-c", "port=55432"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = subprocess.run([str(POSTGRES_BIN_PATH / "pg_isready"),
                                 "-h", str(root / "data/socket"), "-p", "55432"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=5, check=False)
        if result.returncode == 0:
            os.execv(str(root / "current/bin/realsas-migrate"),
                     [str(root / "current/bin/realsas-migrate"), "up"])
        time.sleep(1)
    raise RuntimeError("DEDICATED_POSTGRES_NOT_READY")


if __name__ == "__main__":
    main()
