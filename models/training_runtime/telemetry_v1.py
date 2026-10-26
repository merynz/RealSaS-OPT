"""Independent phase heartbeat plus immediate, append-only worker events."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time
import traceback


class WorkerTelemetry:
    def __init__(self, path, *, heartbeat_seconds=30, stream=None):
        import sys
        if not 0 < heartbeat_seconds <= 60:
            raise ValueError("WORKER_HEARTBEAT_MUST_BE_WITHIN_60_SECONDS")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = stream or sys.stdout
        self.interval = heartbeat_seconds
        self.started = time.monotonic()
        self.state = {"phase": "BOOTSTRAP", "global_step": 0}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.emit("BEGIN")
        self.thread.start()
        return self

    def emit(self, event, **fields):
        with self.lock:
            row = {"event": event, "elapsed_seconds": round(time.monotonic()-self.started, 3), **self.state, **fields}
            text = json.dumps(row, sort_keys=True, allow_nan=False)
            with self.path.open("a") as stream:
                stream.write(text+"\n")
                stream.flush()
            print(text, file=self.stream, flush=True)

    def update(self, *, phase=None, **fields):
        with self.lock:
            self.state.update(fields)
            if phase is not None:
                self.state["phase"] = phase

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            self.emit("HEARTBEAT")

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if exc is not None:
            self.emit("CRASH", exception_type=exc_type.__name__, message=str(exc),
                      traceback="".join(traceback.format_exception(exc_type, exc, tb)))
        else:
            self.emit("WORKER_EXIT")
        # Worker exit is operational only, never a scientific PASS.
        return False
