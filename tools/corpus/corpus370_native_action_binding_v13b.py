"""Exact original-importer action to Blender Action.name lookup (no inference).

A source teacher can name a native FBX take as "Rig|Take" while Blender's
Action.name is distinct. Captured bindings are keyed by the *resolved* Action
identity, never by the source receipt alias. This is teacher-side only.
"""
from __future__ import annotations

def bound_rows_for_source_action(source_action_name, resolved_actions, native_bindings):
    if not isinstance(source_action_name, str) or not source_action_name:
        raise ValueError("V13B_EMPTY_IMPORTED_SOURCE_ACTION")
    action=resolved_actions.get(source_action_name)
    if action is None:
        raise ValueError("V13B_UNRESOLVED_ORIGINAL_SOURCE_ACTION")
    canonical=getattr(action, "name", None)
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("V13B_INVALID_RESOLVED_BLENDER_ACTION_NAME")
    if canonical not in native_bindings:
        raise ValueError("V13B_RESOLVED_ORIGINAL_BLENDER_BINDING_NOT_CAPTURED")
    rows=native_bindings[canonical]
    if not isinstance(rows, list):
        raise ValueError("V13B_MALFORMED_IMPORTER_NATIVE_BINDINGS")
    return rows
