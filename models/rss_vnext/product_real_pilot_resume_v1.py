from __future__ import annotations

"""Crash-safe, fail-closed D/E pilot resume identity.

The underlying D and E production trainers own PyTorch checkpoint state
(optimizer moments, model, RNG and source registries). This sidecar binds
that state to the *user-selected pilot phase plan*, the exact incoming
C/D/E handoff, the optimizer owner ABI, the cohort and the sampler cursor.

A checkpoint write followed by an interrupted sidecar write is deliberately
NOT resumable until a coherent, byte-matched pair exists. Neither this module
nor its tests grant real GPU or source-teacher admission.
"""

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile

SCHEMA = "RealSaS.RSSvNext.PilotDEPhaseResume.v1"
MAX_RECEIPT_BYTES = 32 * 1024


def _sha(v: object) -> bool:
    return type(v) is str and len(v) == 64 and all(
        ch in "0123456789abcdef" for ch in v
    )


def pilot_phase_plan_sha256_v1(plan: object) -> str:
    if not is_dataclass(plan) or isinstance(plan, type):
        raise TypeError("RSS_VNEXT_PILOT_RESUME_PLAN_DATACLASS_REQUIRED")
    validate = getattr(plan, "validate", None)
    if not callable(validate):
        raise TypeError("RSS_VNEXT_PILOT_RESUME_PLAN_VALIDATOR_REQUIRED")
    validate()
    return sha256(json.dumps(
        asdict(plan), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf8")).hexdigest()


@dataclass(frozen=True)
class PilotDEPhaseResumeIdentityV1:
    stage: str
    cohort_sha256: str
    plan_sha256: str
    incoming_model_state_sha256: str
    optimizer_binding_sha256: str
    objective_policy_sha256: str

    def validate(self) -> None:
        from .product_real_pilot_execution_v1 import STAGES
        if (
            self.stage not in STAGES[3:]
            or any(not _sha(value) for value in (
                self.cohort_sha256, self.plan_sha256,
                self.incoming_model_state_sha256,
                self.optimizer_binding_sha256,
                self.objective_policy_sha256,
            ))
        ):
            raise ValueError("RSS_VNEXT_PILOT_RESUME_IDENTITY_INVALID")


def _regular(path: str | Path) -> Path:
    file = Path(path).expanduser()
    if file.is_symlink() or not file.is_file():
        raise ValueError("RSS_VNEXT_PILOT_RESUME_FILE_MISSING_OR_SYMLINK")
    return file


def _file_sha(path: Path) -> str:
    digest = sha256()
    with _regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".pilot-resume.json")


def write_pilot_de_resume_v1(
    checkpoint_path: str | Path,
    *, identity: PilotDEPhaseResumeIdentityV1,
    step: int, sampler_cursor: int,
) -> Path:
    identity.validate()
    if (
        type(step) is not int or step < 0
        or type(sampler_cursor) is not int or sampler_cursor < 0
    ):
        raise ValueError("RSS_VNEXT_PILOT_RESUME_CURSOR_INVALID")
    checkpoint = _regular(checkpoint_path)
    payload = {
        "schema": SCHEMA,
        "identity": asdict(identity),
        "checkpoint_sha256": _file_sha(checkpoint),
        "step": step,
        "sampler_cursor": sampler_cursor,
    }
    target = _sidecar(checkpoint)
    if target.is_symlink():
        raise ValueError("RSS_VNEXT_PILOT_RESUME_SIDECAR_SYMLINK")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=target.parent,
        prefix="rss-pilot-de-", suffix=".json", delete=False,
    ) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def read_pilot_de_resume_v1(
    checkpoint_path: str | Path,
    *, identity: PilotDEPhaseResumeIdentityV1,
    max_steps: int,
) -> tuple[int, int]:
    identity.validate()
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("RSS_VNEXT_PILOT_RESUME_BUDGET_INVALID")
    checkpoint = _regular(checkpoint_path)
    sidecar = _regular(_sidecar(checkpoint))
    if sidecar.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("RSS_VNEXT_PILOT_RESUME_RECEIPT_TOO_LARGE")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("RSS_VNEXT_PILOT_RESUME_RECEIPT_UNREADABLE") from exc
    if (
        type(payload) is not dict
        or payload.get("schema") != SCHEMA
        or payload.get("identity") != asdict(identity)
        or not _sha(payload.get("checkpoint_sha256"))
        or payload["checkpoint_sha256"] != _file_sha(checkpoint)
        or type(payload.get("step")) is not int
        or not 0 <= payload["step"] <= max_steps
        or type(payload.get("sampler_cursor")) is not int
        or payload["sampler_cursor"] < 0
    ):
        raise ValueError("RSS_VNEXT_PILOT_RESUME_PROVENANCE_OR_BYTES_DRIFT")
    return payload["step"], payload["sampler_cursor"]


__all__ = [
    "SCHEMA", "PilotDEPhaseResumeIdentityV1",
    "pilot_phase_plan_sha256_v1", "write_pilot_de_resume_v1",
    "read_pilot_de_resume_v1",
]
