#!/usr/bin/env python3
"""Independent fail-closed audit of deep rig and lookdev diagnostic receipts.

It validates observation integrity, not rig/skin/source-render truth authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMAS = {
    "control": "RealSaS.Corpus4008ImportedControlProfile.v1",
    "lookdev": "RealSaS.Corpus4008LookdevStudy.v1",
}
SHA = re.compile(r"^[a-f0-9]{64}$")
MODES = {
    "SOURCE_MATERIAL_DIAGNOSTIC",
    "STYLIZED_FLAT_DIAGNOSTIC",
    "STYLIZED_OUTLINE_DIAGNOSTIC",
}


def require(value, message):
    if not value:
        raise ValueError("DEEP_DIAGNOSTIC_" + message)


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_control(doc):
    require(doc.get("schema") == SCHEMAS["control"], "CONTROL_SCHEMA")
    require(doc.get("status") == "PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
            "CONTROL_STATUS")
    require(doc.get("training_authorized") is False
            and doc.get("model_family_authorized") is False
            and doc.get("semantic_truth_authorized") is False,
            "CONTROL_FAMILY_AUTHORITY")
    c = doc.get("imported_control", {})
    require(c.get("training_authorized") is False, "CONTROL_TRAINING")
    for name in ("actual_ik_functionality", "ik_absence_in_authored_source",
                 "controller_bone_semantics", "control_group_semantics",
                 "fk_ik_switch_functionality", "retargetability",
                 "source_anatomical_up", "source_anatomical_front"):
        require(c.get(name) == "UNKNOWN", "CONTROL_UNKNOWN_" + name)
    bones = [x for arm in c.get("armatures", []) for x in arm.get("bones", [])]
    constraints = [cc for b in bones for cc in b.get("constraints", [])]
    constraints += [cc for obj in c.get("object_constraints", [])
                    for cc in obj.get("constraints", [])]
    require(len(bones) == c.get("imported_pose_bone_count"),
            "BONE_COUNT")
    require(len(c.get("armatures", [])) == c.get("imported_armature_count"),
            "ARMATURE_COUNT")
    require(len(constraints) == c.get("imported_constraint_count"),
            "CONSTRAINT_COUNT")
    require(sum(x.get("type") == "IK" for x in constraints) ==
            c.get("imported_ik_constraint_count"), "IK_COUNT")
    require(sum(not b["use_deform"] for b in bones) ==
            c.get("imported_nondeform_bone_count"), "NONDEFORM_COUNT")
    require(len(c.get("drivers", [])) == c.get("imported_driver_count"),
            "DRIVER_COUNT")
    require(len(c.get("nla_strips", [])) == c.get("imported_nla_strip_count"),
            "NLA_COUNT")
    require(sum(len(x.get("curves", [])) for x in c.get("actions", [])) ==
            c.get("imported_action_curve_count"), "CURVE_COUNT")
    require(len(c.get("actions", [])) == c.get("imported_action_count"),
            "ACTION_COUNT")
    for b in bones:
        require(b.get("is_control_bone_courted") is False and
                b.get("ik_solver_functional_courted") is False,
                "CONTROLLER_COURT_FORGERY")
    return {
        "armatures": c["imported_armature_count"],
        "bones": len(bones),
        "imported_ik_constraints": c["imported_ik_constraint_count"],
        "imported_action_curves": c["imported_action_curve_count"],
        "imported_driver_channels": c["imported_driver_count"],
    }


def verify_lookdev(doc, root):
    require(doc.get("schema") == SCHEMAS["lookdev"], "LOOKDEV_SCHEMA")
    require(doc.get("status") == "LOOKDEV_DIAGNOSTIC_ONLY",
            "LOOKDEV_STATUS")
    require(doc.get("training_authorized") is False and
            doc.get("stylization_training_authorized") is False and
            doc.get("source_rgba_truth_authorized") is False,
            "LOOKDEV_AUTHORITY")
    require(doc.get("source_material_parity_courted") is False and
            doc.get("texture_fidelity_courted") is False, "LOOKDEV_PARITY")
    require(doc.get("source_anatomical_up") == "UNKNOWN" and
            doc.get("source_anatomical_front") == "UNKNOWN", "LOOKDEV_AXIS")
    outputs = doc.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 3,
            "LOOKDEV_OUTPUT_COUNT")
    require({x.get("mode") for x in outputs} == MODES, "LOOKDEV_MODES")
    for output in outputs:
        require(output.get("render_approved_as_source_rgb") is False and
                output.get("render_approved_as_model_target") is False,
                "LOOKDEV_RENDER_AUTHORITY")
        filename = output.get("path_private")
        require(isinstance(filename, str) and filename.endswith(".png")
                and Path(filename).name == filename, "LOOKDEV_PATH")
        path = root / filename
        require(path.is_file() and path.stat().st_size > 0,
                "LOOKDEV_RENDER_MISSING")
        require(output.get("byte_count") == path.stat().st_size,
                "LOOKDEV_BYTE_COUNT")
        require(output.get("sha256") == _hash(path), "LOOKDEV_SHA")
    return {"render_studies": len(outputs)}


def verify_pair(control, lookdev, root):
    require(isinstance(control, dict) and isinstance(lookdev, dict),
            "INPUT_TYPE")
    ids = ("asset_id", "candidate_id")
    require(all(control.get(k) and control[k] == lookdev.get(k) for k in ids),
            "CROSS_ASSET_ID")
    expected = control.get("original_sha256")
    require(isinstance(expected, str) and SHA.fullmatch(expected) and
            expected == lookdev.get("source_sha256"), "CROSS_SOURCE_SHA")
    c = verify_control(control)
    l = verify_lookdev(lookdev, root)
    return {
        "schema": "RealSaS.Corpus4008DeepDiagnosticVerifiedReceipt.v1",
        "source_sha256": expected, "asset_id": control["asset_id"],
        "candidate_id": control["candidate_id"],
        "counts": dict(c, **l),
        "control_truth_qualified": False,
        "source_rgba_truth_qualified": False,
        "training_authorized": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--lookdev", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    control = json.loads(args.control.read_text())
    lookdev = json.loads(args.lookdev.read_text())
    report = verify_pair(control, lookdev, args.lookdev.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
