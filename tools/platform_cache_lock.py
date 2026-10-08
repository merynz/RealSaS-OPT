"""Persistent tool-cache lifetime lock; all local cache users hold a shared lease."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import wraps
from pathlib import Path
import subprocess


@contextmanager
def cache_lease(root: Path, *, exclusive: bool = False, nonblocking: bool = False):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".use.lock").open("a+b") as handle:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            mode |= fcntl.LOCK_NB
        fcntl.flock(handle.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def shared_cache_use(func):
    @wraps(func)
    def wrapped(root, *args, **kwargs):
        with cache_lease(root):
            return func(root, *args, **kwargs)
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a cache-consuming command is required")
    with cache_lease(args.root):
        raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
