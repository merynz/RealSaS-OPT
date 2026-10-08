#!/usr/bin/env python3
"""Install canonical main as persistent loopback-only developer user services.

No sudo, background-job orphaning, database reset or implicit service takeover.
If the user manager is unavailable, stop before installing anything and report
the operator action. Temporal's SQLite dev server is NOT production deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_VERSION = "1.9.1"
TEMPORAL_SHA256 = "09a0326a51db84d02735e53542b9ebd8c4758daf47482a9ab0abce15844e60d5"
TEMPORAL_URL = ("https://github.com/temporalio/cli/releases/download/v1.9.1/"
                "temporal_cli_1.9.1_linux_amd64.tar.gz")
UNITS = ("postgres", "temporal", "migrate", "api", "control", "engine")


def command(args, *, cwd=None, capture=False, timeout=300, env=None):
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, text=True,
                          capture_output=capture, timeout=timeout, env=env)


def all_services_active(stdout):
    statuses = stdout.strip().splitlines()
    # systemctl is-active exits zero if ANY supplied unit is active, not all.
    return len(statuses) == len(UNITS) and all(value == "active" for value in statuses)


def python_library_dir(python):
    # setup-python's shared build needs this even outside the Actions job.
    base = command([python, "-c", "import sys; print(sys.base_prefix)"], capture=True).stdout.strip()
    directory = Path(base) / "lib"
    if not directory.is_dir():
        raise RuntimeError("PYTHON_BASE_LIBRARY_DIRECTORY_MISSING")
    return str(directory)


def runtime_environment(root, authority, library_dir):
    return {"REALSAS_DEPLOY_ROOT": str(root), "REALSAS_REPO_ROOT": str(root / "current/source"),
            "REALSAS_AUTHORITY_ROOT": str(authority), "REALSAS_ARTIFACT_ROOT": str(root / "artifacts"),
            "REALSAS_DATABASE_URL": f"postgresql:///postgres?host={root}/data/socket&port=55432",
            "REALSAS_TEMPORAL_ADDRESS": "127.0.0.1:7233", "REALSAS_TEMPORAL_NAMESPACE": "default",
            "LD_LIBRARY_PATH": library_dir, "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def python_launcher(python, library_dir):
    return ("#!/bin/sh\n" + f"LD_LIBRARY_PATH={shlex.quote(library_dir)}\n"
            + "export LD_LIBRARY_PATH\n" + f"exec {shlex.quote(str(python))} \"$@\"\n")


def startup_diagnostics():
    # Only our managed units; never print the service environment/credentials.
    subprocess.run(["systemctl", "--user", "show", *[f"realsas-platform-{name}.service" for name in UNITS],
                    "--property=Id,ActiveState,SubState,ExecMainCode,ExecMainStatus"], check=False, timeout=10)
    subprocess.run(["journalctl", "--user", "-u", "realsas-platform-postgres.service",
                    "-u", "realsas-platform-migrate.service", "-u", "realsas-platform-engine.service",
                    "-u", "realsas-platform-control.service", "-n", "30", "--no-pager"], check=False, timeout=10)


def safe_root(value):
    path = Path(value)
    if not path.is_absolute() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise RuntimeError("DEPLOY_ROOT_MUST_BE_AN_ABSOLUTE_SIMPLE_PATH")
    if path != path.resolve() or path in (Path("/"), Path.home(), ROOT) or len(path.parts) < 4:
        raise RuntimeError("DEPLOY_ROOT_UNSAFE_OR_SYMLINKED")
    return path


def socket_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def manager_environment():
    # A configured user manager is the legitimate durable execution owner.
    # Never change RUNNER_TRACKING_ID or spawn detached runner-owned processes.
    runtime = Path(f"/run/user/{os.getuid()}")
    if (runtime / "bus").is_socket() and runtime.stat().st_uid == os.getuid():
        os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))
        os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")
    result = subprocess.run(["systemctl", "--user", "show-environment"],
                            capture_output=True, text=True, timeout=10, check=False)
    if result.returncode:
        raise RuntimeError("USER_SYSTEMD_UNAVAILABLE: operator must enable a user manager; "
                           "on WSL run sudo loginctl enable-linger $(id -un), then retry")


def preflight(root, expected_sha):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("LINUX_AMD64_REQUIRED")
    if os.getuid() == 0:
        raise RuntimeError("NONROOT_OPERATOR_REQUIRED")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise RuntimeError("EXACT_MAIN_SHA_REQUIRED")
    head = command(["git", "rev-parse", "HEAD"], cwd=ROOT, capture=True).stdout.strip()
    remote = command(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT,
                     capture=True).stdout.split()[0]
    if head != expected_sha or remote != expected_sha:
        raise RuntimeError("CANONICAL_MAIN_CHANGED_OR_CHECKOUT_MISMATCH")
    if command(["git", "status", "--porcelain"], cwd=ROOT, capture=True).stdout.strip():
        raise RuntimeError("CLEAN_CANONICAL_CHECKOUT_REQUIRED")
    manager_environment()
    marker = root / "deployment.json"
    if root.exists() and not marker.exists():
        raise RuntimeError("UNMANAGED_DEPLOY_ROOT: refusing to overwrite existing data")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("schema") != "RealSaS.DeveloperDeployment.v1" or previous.get("root") != str(root):
            raise RuntimeError("DEPLOY_MARKER_IDENTITY_DRIFT")
        # Do not stop/restart services while an Attempt is running.
        socket_dir = root / "data/socket"
        if (socket_dir / ".s.PGSQL.55432").exists():
            # Use the deployed interpreter to locate its pinned psql, not PATH.
            script = "from pgserver._commands import POSTGRES_BIN_PATH; print(POSTGRES_BIN_PATH/'psql')"
            psql = command([root / "current/bin/realsas-python", "-c", script], capture=True).stdout.strip()
            active = command([psql, "-h", socket_dir, "-p", "55432", "-d", "postgres", "-Atc",
                              "SELECT count(*) FROM attempts WHERE final_state='OPEN'"], capture=True).stdout.strip()
            if active != "0":
                raise RuntimeError("OPEN_ATTEMPTS_REQUIRE_OPERATOR_DRAIN_BEFORE_UPGRADE")
    unit_dir = Path.home() / ".config/systemd/user"
    for name in (*UNITS, "target"):
        target = unit_dir / ("realsas-platform.target" if name == "target" else f"realsas-platform-{name}.service")
        if target.exists() and "# Managed by RealSaS developer deployment" not in target.read_text():
            raise RuntimeError(f"UNMANAGED_UNIT:{target.name}")
    for name, port in (("api", 8080), ("temporal", 7233)):
        if not socket_open(port):
            continue
        if not marker.exists() or not previous.get("current_release"):
            raise RuntimeError("PLATFORM_PORT_CONFLICT: no service takeover permitted")
        result = subprocess.run(["systemctl", "--user", "is-active", f"realsas-platform-{name}.service"],
                                capture_output=True, timeout=10, check=False)
        if result.returncode:
            raise RuntimeError("PLATFORM_PORT_CONFLICT: managed service does not own the active port")


def service_units(root):
    current = root / "current"
    header = "# Managed by RealSaS developer deployment\n"
    common = (f"WorkingDirectory={current}/source\nEnvironmentFile={root}/platform.env\n"
              "UMask=0077\nTimeoutStopSec=60\n")
    launcher = f"{current}/venv/bin/python {current}/source/tools/platform_service.py"
    starts = {
        "postgres": launcher + " postgres",
        "temporal": f"{current}/bin/temporal server start-dev --ip 127.0.0.1 --headless --db-filename {root}/data/temporal.sqlite",
        "migrate": launcher + " migrate",
        "api": f"{current}/bin/realsas-api",
        "control": f"{current}/bin/realsas-control-worker",
        "engine": f"{current}/venv/bin/python -m compiler.realsas_compiler_services.platform_worker.worker",
    }
    units = {}
    for name, start in starts.items():
        dependencies = ""
        if name == "migrate":
            dependencies = "Requires=realsas-platform-postgres.service\nAfter=realsas-platform-postgres.service\n"
        elif name in ("api", "control", "engine"):
            dependencies = "Requires=realsas-platform-migrate.service\nAfter=realsas-platform-migrate.service\n"
            if name != "api":
                dependencies += "Requires=realsas-platform-temporal.service\nAfter=realsas-platform-temporal.service\n"
        behavior = ("Type=oneshot\nRemainAfterExit=yes\n" if name == "migrate" else
                    "Type=simple\nRestart=on-failure\nRestartSec=3\n")
        units[f"realsas-platform-{name}.service"] = (header + "[Unit]\n"
            f"Description=RealSaS developer {name}\nPartOf=realsas-platform.target\n"
            "StartLimitIntervalSec=0\n" + dependencies + "\n[Service]\n" + behavior + common + f"ExecStart={start}\n")
    units["realsas-platform.target"] = (header + "[Unit]\nDescription=RealSaS local developer platform\n"
        "Wants=" + " ".join(f"realsas-platform-{name}.service" for name in UNITS) + "\n\n[Install]\nWantedBy=default.target\n")
    return units


def extract_temporal(archive, destination):
    if hashlib.sha256(archive).hexdigest() != TEMPORAL_SHA256:
        raise RuntimeError("TEMPORAL_ARCHIVE_HASH_MISMATCH")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as data:
        entries = [entry for entry in data.getmembers() if entry.name == "temporal" and entry.isfile()]
        if len(entries) != 1:
            raise RuntimeError("TEMPORAL_BINARY_MISSING_OR_AMBIGUOUS")
        destination.write_bytes(data.extractfile(entries[0]).read())
        destination.chmod(0o755)


def install(root, expected_sha, authority):
    preflight(root, expected_sha)
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    root.chmod(0o700)
    marker = root / "deployment.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous is None:
        marker.write_text(json.dumps({"schema": "RealSaS.DeveloperDeployment.v1", "root": str(root), "state": "PREPARING"}) + "\n")
    release = root / "releases" / f"{expected_sha}-{uuid.uuid4().hex[:8]}"
    release.mkdir(parents=True, mode=0o700)
    source, binaries, venv = release / "source", release / "bin", release / "venv"
    source.mkdir()
    binaries.mkdir()
    # Immutable source snapshot; source branches never become execution state.
    archive = subprocess.run(["git", "archive", expected_sha], cwd=ROOT, check=True,
                             capture_output=True, timeout=60).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as data:
        data.extractall(source, filter="data")
    command([sys.executable, "-m", "venv", venv])
    python = venv / "bin/python"
    command([python, "-m", "pip", "install", "--no-input", "--progress-bar", "off",
             "-r", source / "requirements/platform-deployment.txt"], timeout=600)
    for name in ("realsas-api", "realsas-control-worker", "realsas-migrate", "realsasctl"):
        command(["go", "build", "-o", binaries / name, f"./cmd/{name}"], cwd=source / "platform", timeout=300)
    # A pinned, checksum-verified official binary, never curl|sh/latest.
    with urllib.request.urlopen(TEMPORAL_URL, timeout=60) as response:
        temporal = response.read(64 * 1024 * 1024 + 1)
    extract_temporal(temporal, binaries / "temporal")
    for name in ("data", "data/socket", "artifacts", "receipts"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    pgdata = root / "data/postgres"
    if not (pgdata / "PG_VERSION").exists():
        if pgdata.exists() and any(pgdata.iterdir()):
            raise RuntimeError("PARTIAL_PGDATA_REQUIRES_OPERATOR_INSPECTION")
        command([python, "-c", "import sys; from pgserver import initdb; from pathlib import Path; "
                 "initdb(['-A','trust','--encoding=UTF8','--no-locale'], pgdata=Path(sys.argv[1]))", pgdata])
    # Verify imports WITHOUT any inherited Actions environment before switching.
    library_dir = python_library_dir(python)
    launcher = binaries / "realsas-python"
    launcher.write_text(python_launcher(python, library_dir))
    launcher.chmod(0o755)
    env = runtime_environment(root, authority, library_dir)
    command([launcher, "-c", "from compiler.realsas_compiler_services.platform_worker import worker; "
             "print('Engine imports verified in isolated service environment')"], cwd=source, env=env)
    units = service_units(root)
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, text in units.items():
        # Validate actual built executables before the current symlink exists.
        (release / name).write_text(text.replace(str(root / "current"), str(release)))
    command(["systemd-analyze", "--user", "verify", *[release / name for name in units]], timeout=30)
    # A long dependency build must not switch the host to a superseded main.
    if command(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, capture=True).stdout.split()[0] != expected_sha:
        raise RuntimeError("CANONICAL_MAIN_ADVANCED_DURING_PACKAGE_BUILD: live services unchanged")
    if previous and previous.get("current_release"):
        command(["systemctl", "--user", "stop", "realsas-platform.target"], timeout=90)
    temporary_link = root / f".current-{uuid.uuid4().hex}"
    temporary_link.symlink_to(release)
    os.replace(temporary_link, root / "current")
    (root / "platform.env").write_text("".join(f"{key}={value}\n" for key, value in env.items()))
    (root / "platform.env").chmod(0o600)
    for name, text in units.items():
        (unit_dir / name).write_text(text)
    receipt = {"schema": "RealSaS.DeveloperDeployment.v1", "root": str(root),
               "code_sha": expected_sha, "current_release": str(release), "previous_release": (previous or {}).get("current_release"),
               "temporal_version": TEMPORAL_VERSION, "temporal_archive_sha256": TEMPORAL_SHA256,
               "execution_mode": "LOCAL_DEVELOPER", "production_ready": False,
               "authority_root": str(authority), "state": "STARTING"}
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    command(["systemctl", "--user", "daemon-reload"])
    command(["systemctl", "--user", "enable", "--now", "realsas-platform.target"], timeout=120)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        active = subprocess.run(["systemctl", "--user", "is-active", *[f"realsas-platform-{name}.service" for name in UNITS]],
                                text=True, capture_output=True, check=False, timeout=10)
        if all_services_active(active.stdout) and socket_open(8080) and socket_open(7233):
            receipt["state"] = "SERVICES_ACTIVE"
            marker.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, sort_keys=True))
            return
        time.sleep(1)
    startup_diagnostics()
    raise RuntimeError("PLATFORM_SERVICES_NOT_READY: startup diagnostics printed; no data was reset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = safe_root(args.root)
    authority = safe_root(args.authority_root)
    if args.preflight_only:
        preflight(root, args.expected_main_sha)
        print("DEPLOY_PREFLIGHT_OK: no host writes")
    else:
        install(root, args.expected_main_sha, authority)


if __name__ == "__main__":
    main()
