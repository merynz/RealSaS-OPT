#!/usr/bin/env python3
"""Isolated ephemeral CI databases on the persistent local RealSaS PostgreSQL service.

The persistent server is transport only. Each CI run gets unique databases; the
canonical developer database is never migrated/reset by CI. Stale CI databases
are garbage-collected only when old and connection-idle.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from urllib.parse import quote

DEFAULT_ADMIN_URL = "postgresql:///postgres?host=/home/monster/realsas_platform/data/socket&port=55432"
PREFIX = "realsas_ci_"
NAME_RE = re.compile(r"^realsas_ci_(\d{10})_[0-9]+_[0-9]+_(?:main|migrations)$")


def psql_binary() -> str:
    from pgserver._commands import POSTGRES_BIN_PATH
    return str(POSTGRES_BIN_PATH / "psql")


def run_psql(admin_url: str, sql: str, *, capture: bool = True) -> str:
    result = subprocess.run(
        [
            psql_binary(),
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            admin_url,
            "-Atc",
            sql,
        ],
        check=True,
        text=True,
        capture_output=capture,
        timeout=30,
    )
    return result.stdout.strip() if capture else ""


def validate_name(name: str) -> str:
    if not NAME_RE.fullmatch(name) or len(name) > 63:
        raise ValueError("CI_DATABASE_NAME_INVALID")
    return name


def database_url(name: str, *, socket_dir: str, port: int) -> str:
    validate_name(name)
    return f"postgresql:///{name}?host={quote(socket_dir, safe='/')}&port={int(port)}"


def create_database(admin_url: str, name: str) -> None:
    validate_name(name)
    exists = run_psql(admin_url, f"SELECT 1 FROM pg_database WHERE datname='{name}'")
    if exists:
        raise RuntimeError(f"CI_DATABASE_ALREADY_EXISTS:{name}")
    run_psql(admin_url, f'CREATE DATABASE "{name}"', capture=False)
    run_psql(admin_url, f'COMMENT ON DATABASE "{name}" IS \'RealSaS ephemeral CI database\'', capture=False)


def drop_database(admin_url: str, name: str, *, force: bool = False) -> None:
    validate_name(name)
    active = run_psql(admin_url, f"SELECT count(*) FROM pg_stat_activity WHERE datname='{name}'")
    if active and int(active) > 0 and not force:
        raise RuntimeError(f"CI_DATABASE_HAS_ACTIVE_CONNECTIONS:{name}:{active}")
    suffix = " WITH (FORCE)" if force else ""
    run_psql(admin_url, f'DROP DATABASE IF EXISTS "{name}"{suffix}', capture=False)


def gc_databases(admin_url: str, *, max_age_seconds: int, now: int | None = None) -> dict:
    now = int(time.time()) if now is None else int(now)
    raw = run_psql(admin_url, f"SELECT datname FROM pg_database WHERE datname LIKE '{PREFIX}%'")
    names = [line.strip() for line in raw.splitlines() if line.strip()]
    deleted: list[str] = []
    retained_active: list[str] = []
    ignored_invalid: list[str] = []
    for name in sorted(names):
        match = NAME_RE.fullmatch(name)
        if not match:
            ignored_invalid.append(name)
            continue
        created = int(match.group(1))
        if now - created <= max_age_seconds:
            continue
        active = int(run_psql(admin_url, f"SELECT count(*) FROM pg_stat_activity WHERE datname='{name}'") or "0")
        if active:
            retained_active.append(name)
            continue
        drop_database(admin_url, name)
        deleted.append(name)
    return {
        "schema": "RealSaS.PlatformCIDatabaseGC.v1",
        "max_age_seconds": max_age_seconds,
        "deleted": deleted,
        "retained_active": retained_active,
        "ignored_invalid": ignored_invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-url", default=os.environ.get("REALSAS_CI_POSTGRES_ADMIN_URL", DEFAULT_ADMIN_URL))
    parser.add_argument("--socket-dir", default="/home/monster/realsas_platform/data/socket")
    parser.add_argument("--port", type=int, default=55432)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    drop = sub.add_parser("drop")
    drop.add_argument("--name", required=True)
    drop.add_argument("--force", action="store_true")
    gc = sub.add_parser("gc")
    gc.add_argument("--max-age-hours", type=float, default=24)
    args = parser.parse_args()

    if args.command == "create":
        create_database(args.admin_url, args.name)
        print(database_url(args.name, socket_dir=args.socket_dir, port=args.port))
    elif args.command == "drop":
        drop_database(args.admin_url, args.name, force=args.force)
        print(json.dumps({"status": "DROPPED", "name": args.name}, sort_keys=True))
    else:
        print(json.dumps(gc_databases(args.admin_url, max_age_seconds=int(args.max_age_hours * 3600)), sort_keys=True))


if __name__ == "__main__":
    main()
