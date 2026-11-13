from __future__ import annotations

"""Editable control-state overlay in the existing RealSaS .rss transport.

This stores an editor intervention without duplicating or altering its source
mesh, texture, motion, camera, or graphics entries. Reopening verifies the
unchanged base-package bytes, exact canonical parents, typed control/slot
domains, and deterministic Mechanics+LBS edit consequences.

This is a Python runtime roundtrip witness, NOT native C++ renderer parity,
source-contact truth, ModelRelease admission, or ProductPASS.
"""

from collections import OrderedDict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .controllable_puppet_runtime_v1 import (
    ControllablePuppetBundleV1, ControllablePuppetSessionV1,
)
from .hashing import content_sha256
from .mechanics_control_solver_v1 import ControlTransformIRV1
from .runtime_package_v2 import _entry_bytes, read_rss_v2, write_rss_v2
from .types import QualificationError

EDITOR_ENTRY = "editor_state_v1.json"
SCHEMA = "RealSaS.PuppetEditorRSSReopen.v1"
MAX_EDITOR_STATE_BYTES = 256 * 1024
REQUIRED_BASE_ENTRIES = frozenset({"manifest.txt", "mesh.bin", "cameras.bin"})


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualificationError("RSS_EDITOR_STATE_NOT_CANONICAL_JSON") from exc


def _identity(bundle: ControllablePuppetBundleV1) -> dict[str, str | None]:
    return {
        "mesh_lineage_hash": bundle.mesh.mesh_lineage_hash,
        "rig_lineage_hash": bundle.rig.rig_lineage_hash,
        "skin_lineage_hash": bundle.skin.skin_lineage_hash,
        "control_program_hash": bundle.rig.control_program_hash,
        "product_revision_id": bundle.product_revision_id,
        "product_qualification_receipt": bundle.product_qualification_receipt,
        "runtime_graphics_hash": (
            bundle.graphics.runtime_graphics_hash if bundle.graphics else None
        ),
    }


def _base_entries(entries: OrderedDict[str, bytes]) -> OrderedDict[str, bytes]:
    result = OrderedDict(
        (key, val) for key, val in entries.items() if key != EDITOR_ENTRY
    )
    if not REQUIRED_BASE_ENTRIES.issubset(result):
        raise QualificationError("RSS_EDITOR_BASE_PACKAGE_ENTRIES_MISSING")
    if not result["mesh.bin"] or not result["cameras.bin"]:
        raise QualificationError("RSS_EDITOR_BASE_PACKAGE_EMPTY_GEOMETRY")
    try:
        manifest = result["manifest.txt"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("RSS_EDITOR_BASE_MANIFEST_ENCODING_INVALID") from exc
    if "schema=RealSaS.RuntimePackage.v2\n" not in (manifest + "\n"):
        raise QualificationError("RSS_EDITOR_BASE_MANIFEST_SCHEMA_DRIFT")
    return result


def _edit_state(session: ControllablePuppetSessionV1, base_sha: str) -> dict:
    controls = []
    for control_id, row in sorted(session.control_states.items()):
        if type(row) is not ControlTransformIRV1 or row.canonical_control_id != control_id:
            raise QualificationError("RSS_EDITOR_CONTROL_STATE_DOMAIN_DRIFT")
        controls.append({
            "canonical_control_id": control_id,
            "translation": list(row.translation),
            "rotation_quat_wxyz": list(row.rotation_quat_wxyz),
            "scale": list(row.scale),
        })
    state = {
        "schema": SCHEMA,
        "source_package_sha256": base_sha,
        "canonical_parents": _identity(session.bundle),
        "view_index": session.view_index,
        "control_states": controls,
        "active_attachments": dict(sorted(session.active_attachments.items())),
        "claim": "EDITABLE_CANONICAL_CONSUMER_NOT_PRODUCT_PASS",
    }
    # Compare real pose and deformed vertex consequences, not only edit inputs.
    # Explicitly normalize control insertion order before computing the witness.
    restored = _restore_state(session.bundle, state, check_consequence=False)
    original = session.frame()
    actual = restored.frame()
    if (
        original.pose.joints != actual.pose.joints
        or original.deformed.positions != actual.deformed.positions
        or original.composition != actual.composition
    ):
        raise QualificationError("RSS_EDITOR_EDIT_ORDER_NONDETERMINISTIC")
    state["edit_consequence"] = {
        "pose_sha256": content_sha256(actual.pose.to_dict()),
        "deformed_mesh_sha256": content_sha256(actual.deformed.to_dict()),
        "view_id": actual.view_id,
        "visible_attachment_ids": sorted(m.attachment_id for m in actual.meshes),
    }
    state["state_sha256"] = sha256(_json_bytes(state)).hexdigest()
    return state


def _typed_numbers(value: Any, n: int, *, positive: bool = False) -> tuple[float, ...]:
    if (
        type(value) is not list or len(value) != n
        or any(type(v) not in (int, float) for v in value)
    ):
        raise QualificationError("RSS_EDITOR_CHANNEL_TYPED_DOMAIN_INVALID")
    values = tuple(float(v) for v in value)
    if any(not math.isfinite(v) or (positive and v <= 0) for v in values):
        raise QualificationError("RSS_EDITOR_CHANNEL_FINITE_DOMAIN_INVALID")
    return values


def _restore_state(
    bundle: ControllablePuppetBundleV1,
    state: Mapping[str, Any],
    *, check_consequence: bool,
) -> ControllablePuppetSessionV1:
    if (
        type(state) is not dict
        or state.get("schema") != SCHEMA
        or state.get("claim") != "EDITABLE_CANONICAL_CONSUMER_NOT_PRODUCT_PASS"
        or state.get("canonical_parents") != _identity(bundle)
        or type(state.get("view_index")) is not int
        or type(state.get("control_states")) is not list
        or type(state.get("active_attachments")) is not dict
    ):
        raise QualificationError("RSS_EDITOR_SCHEMA_OR_CANONICAL_PARENT_DRIFT")
    if len(state["control_states"]) > len(bundle.rig.controls):
        raise QualificationError("RSS_EDITOR_CONTROL_SET_TOO_LARGE")
    session = ControllablePuppetSessionV1(bundle)
    session.set_view_index(state["view_index"])
    if set(state["active_attachments"]) != set(session.active_attachments):
        raise QualificationError("RSS_EDITOR_SLOT_SET_DRIFT")
    for slot_id, attachment_id in sorted(state["active_attachments"].items()):
        if attachment_id is not None and type(attachment_id) is not str:
            raise QualificationError("RSS_EDITOR_ATTACHMENT_TYPED_DOMAIN_INVALID")
        session.set_attachment(slot_id, attachment_id)
    previous = ""
    for row in state["control_states"]:
        if type(row) is not dict or set(row) != {
            "canonical_control_id", "translation", "rotation_quat_wxyz", "scale",
        }:
            raise QualificationError("RSS_EDITOR_CONTROL_ROW_INVALID")
        control_id = row.get("canonical_control_id")
        if type(control_id) is not str or control_id <= previous:
            raise QualificationError("RSS_EDITOR_CONTROL_ORDER_OR_ID_DRIFT")
        previous = control_id
        session.set_control(ControlTransformIRV1(
            canonical_control_id=control_id,
            translation=_typed_numbers(row["translation"], 3),
            rotation_quat_wxyz=_typed_numbers(row["rotation_quat_wxyz"], 4),
            scale=_typed_numbers(row["scale"], 3, positive=True),
        ))
    if check_consequence:
        result = session.frame()
        witness = state.get("edit_consequence")
        if type(witness) is not dict or witness != {
            "pose_sha256": content_sha256(result.pose.to_dict()),
            "deformed_mesh_sha256": content_sha256(result.deformed.to_dict()),
            "view_id": result.view_id,
            "visible_attachment_ids": sorted(m.attachment_id for m in result.meshes),
        }:
            raise QualificationError("RSS_EDITOR_REOPEN_EDIT_CONSEQUENCE_DRIFT")
    return session


def write_edited_rss_v1(
    session: ControllablePuppetSessionV1,
    *, source_path: Path, destination_path: Path,
) -> dict[str, str | int | bool]:
    """Persist edits as an overlay without modifying any existing .rss entry."""
    source_path, destination_path = Path(source_path), Path(destination_path)
    if (
        source_path.suffix.lower() != ".rss"
        or destination_path.suffix.lower() != ".rss"
        or source_path.resolve() == destination_path.resolve()
    ):
        raise QualificationError("RSS_EDITOR_SOURCE_DESTINATION_INVALID")
    base = _base_entries(read_rss_v2(source_path))
    fingerprint = sha256(_entry_bytes(base)).hexdigest()
    state = _edit_state(session, fingerprint)
    raw = _json_bytes(state)
    if len(raw) > MAX_EDITOR_STATE_BYTES:
        raise QualificationError("RSS_EDITOR_STATE_SIZE_LIMIT")
    entries = OrderedDict(base)
    entries[EDITOR_ENTRY] = raw
    receipt = write_rss_v2(destination_path, entries)
    return {
        "schema": "RealSaS.PuppetEditorRSSWriteReceipt.v1",
        "archive_sha256": str(receipt["archive_sha256"]),
        "source_package_sha256": fingerprint,
        "edit_state_sha256": state["state_sha256"],
        "native_renderer_parity_claimed": False,
        "product_revision_minted": False,
    }


def reopen_edited_rss_v1(
    path: Path, bundle: ControllablePuppetBundleV1,
) -> ControllablePuppetSessionV1:
    path = Path(path)
    if path.suffix.lower() != ".rss":
        raise QualificationError("RSS_EDITOR_PACKAGE_SUFFIX_INVALID")
    entries = read_rss_v2(path)
    body = entries.get(EDITOR_ENTRY)
    if type(body) is not bytes or len(body) > MAX_EDITOR_STATE_BYTES:
        raise QualificationError("RSS_EDITOR_STATE_ENTRY_MISSING_OR_TOO_LARGE")
    try:
        state = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError("RSS_EDITOR_STATE_JSON_INVALID") from exc
    if type(state) is not dict:
        raise QualificationError("RSS_EDITOR_STATE_JSON_OBJECT_REQUIRED")
    base = _base_entries(entries)
    if state.get("source_package_sha256") != sha256(_entry_bytes(base)).hexdigest():
        raise QualificationError("RSS_EDITOR_BASE_PACKAGE_CHANGED")
    claimed = state.get("state_sha256")
    without_hash = dict(state)
    without_hash.pop("state_sha256", None)
    if (
        type(claimed) is not str
        or claimed != sha256(_json_bytes(without_hash)).hexdigest()
        or body != _json_bytes(state)
    ):
        raise QualificationError("RSS_EDITOR_STATE_HASH_OR_CANONICAL_BYTES_DRIFT")
    return _restore_state(bundle, state, check_consequence=True)


__all__ = [
    "EDITOR_ENTRY", "write_edited_rss_v1", "reopen_edited_rss_v1",
]
