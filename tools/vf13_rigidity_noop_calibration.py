from __future__ import annotations

"""VF-13 subject-free rigidity no-op calibration/validation.

The shipping tolerance is preregistered in the frozen presentation policy. This
apparatus does not tune from a subject and does not search a threshold. It validates
that exact one-hot rigid groups sit below the frozen numerical tolerance while
multiple independent LBS deformation controls sit strictly above it.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.product_state_v2 import (
    _face_group_rigidity_noop_probe,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "canonical" / "PRESENTATION_PARTITION_POLICY_V1_20260921.json"
OUT = Path(os.environ.get("REALSAS_VF13_OUT", "vf13_out"))
OUT.mkdir(parents=True, exist_ok=True)


def _mesh(*, scale: float = 1.0, offset=(0.0, 0.0, 0.0)):
    s = float(scale)
    o = np.asarray(offset, dtype=np.float64)
    base = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    xyz = base * s + o[None, :]
    return SimpleNamespace(
        vertices=tuple(
            SimpleNamespace(
                canonical_mesh_vertex_id=f"v{i}",
                component_id="c0",
                P=tuple(map(float, xyz[i])),
            )
            for i in range(4)
        ),
        faces=(("v0", "v1", "v2"), ("v1", "v3", "v2")),
        mesh_lineage_hash="m" * 64,
    )


def _skin(rows):
    return SimpleNamespace(
        rows=tuple(
            SimpleNamespace(
                canonical_mesh_vertex_id=f"v{i}",
                influences=tuple(rows[i]),
            )
            for i in range(4)
        ),
        mesh_skin_lineage_hash="w" * 64,
    )


def _measure(name: str, mesh, skin, policy: dict) -> dict:
    result = _face_group_rigidity_noop_probe(
        (0, 1),
        mesh,
        skin,
        relative_edge_tolerance=float(
            policy["rigidity_noop_relative_edge_tolerance"]
        ),
        probe_rotation_degrees=float(
            policy["rigidity_noop_probe_rotation_degrees"]
        ),
    )
    return {
        "name": name,
        **result,
    }


def main() -> int:
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = dict(doc["mechanical_binding_policy"])
    if bool(policy["rigidity_noop_required"]) is not True:
        raise RuntimeError("VF13_RIGIDITY_NOOP_POLICY_NOT_REQUIRED")
    if bool(policy["legacy_weight_thresholds_final_authority"]):
        raise RuntimeError("VF13_LEGACY_WEIGHT_AUTHORITY_MUST_BE_FALSE")

    rigid_rows = {
        i: (("j1", 1.0),)
        for i in range(4)
    }
    hard_rigid = [
        _measure(
            "ONE_HOT_UNIT_SCALE",
            _mesh(scale=1.0),
            _skin(rigid_rows),
            policy,
        ),
        _measure(
            "ONE_HOT_MICRO_SCALE_TRANSLATED",
            _mesh(scale=1.0e-6, offset=(2.3, -1.7, 0.9)),
            _skin(rigid_rows),
            policy,
        ),
        _measure(
            "ONE_HOT_LARGE_SCALE_TRANSLATED",
            _mesh(scale=1.0e6, offset=(-3.0e6, 2.0e6, 7.0e5)),
            _skin(rigid_rows),
            policy,
        ),
    ]

    smooth_gradient = _skin(
        {
            0: (("j0", 0.85), ("j1", 0.15)),
            1: (("j0", 0.65), ("j1", 0.35)),
            2: (("j0", 0.35), ("j1", 0.65)),
            3: (("j0", 0.15), ("j1", 0.85)),
        }
    )
    uniform_soft = _skin(
        {
            i: (("j1", 0.999), ("j0", 0.001))
            for i in range(4)
        }
    )
    legacy_false_positive = _skin(
        {
            0: (("j1", 1.0),),
            1: (("j1", 0.999), ("j0", 0.001)),
            2: (("j1", 1.0),),
            3: (("j1", 1.0),),
        }
    )
    split_owner = _skin(
        {
            0: (("j0", 1.0),),
            1: (("j0", 1.0),),
            2: (("j1", 1.0),),
            3: (("j1", 1.0),),
        }
    )

    deformable = [
        _measure(
            "SMOOTH_MULTIJOINT_GRADIENT",
            _mesh(scale=1.0),
            smooth_gradient,
            policy,
        ),
        _measure(
            "UNIFORM_0999_0001_SOFT_BLEND",
            _mesh(scale=1.0),
            uniform_soft,
            policy,
        ),
        _measure(
            "LEGACY_0999_BOUNDARY_FALSE_POSITIVE",
            _mesh(scale=1.0),
            legacy_false_positive,
            policy,
        ),
        _measure(
            "SPLIT_ONE_HOT_OWNERS",
            _mesh(scale=1.0),
            split_owner,
            policy,
        ),
    ]

    tolerance = float(policy["rigidity_noop_relative_edge_tolerance"])
    max_rigid = max(float(row["max_relative_edge_error"]) for row in hard_rigid)
    min_deform = min(float(row["max_relative_edge_error"]) for row in deformable)

    if not all(bool(row["passed"]) for row in hard_rigid):
        raise RuntimeError("VF13_HARD_RIGID_CONTROL_FAILED")
    if not all(not bool(row["passed"]) for row in deformable):
        raise RuntimeError("VF13_DEFORMABLE_CONTROL_FALSE_RIGID")
    if not max_rigid < tolerance < min_deform:
        raise RuntimeError("VF13_FROZEN_TOLERANCE_NOT_SEPARATING_CONTROLS")

    owner_min = float(policy["min_rigid_owner_weight"])
    other_max = float(policy["max_rigid_other_mass"])
    # This control is intentionally inside the historical threshold predicate.
    legacy_rows = legacy_false_positive.rows
    ranked = [
        sorted(
            ((str(j), float(w)) for j, w in row.influences),
            key=lambda item: (-item[1], item[0]),
        )
        for row in legacy_rows
    ]
    legacy_min_owner = min(row[0][1] for row in ranked)
    legacy_max_other = max(
        sum(w for j, w in row if j != row[0][0])
        for row in ranked
    )
    legacy_predicate_would_say_rigid = (
        legacy_min_owner >= owner_min
        and legacy_max_other <= other_max
        and len({row[0][0] for row in ranked}) == 1
    )
    if not legacy_predicate_would_say_rigid:
        raise RuntimeError("VF13_LEGACY_FALSE_POSITIVE_CONTROL_INVALID")

    payload = {
        "schema": "RealSaS.VF13RigidityNoopCalibration.v1",
        "status": "PASS_SUBJECT_FREE_FROZEN_TOLERANCE_VALIDATED",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "category_labels_used": False,
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_status": doc["status"],
        "frozen_policy": {
            "min_rigid_owner_weight": owner_min,
            "max_rigid_other_mass": other_max,
            "rigidity_noop_relative_edge_tolerance": tolerance,
            "rigidity_noop_probe_rotation_degrees": float(
                policy["rigidity_noop_probe_rotation_degrees"]
            ),
            "legacy_weight_thresholds_final_authority": False,
        },
        "hard_rigid_controls": hard_rigid,
        "deformable_controls": deformable,
        "separation": {
            "maximum_hard_rigid_relative_edge_error": max_rigid,
            "frozen_tolerance": tolerance,
            "minimum_deformable_relative_edge_error": min_deform,
            "strictly_separated": True,
            "threshold_search_used": False,
            "selection_semantics": (
                "PREREGISTERED_NUMERICAL_TOLERANCE_VALIDATED_IN_OPEN_GAP"
            ),
        },
        "legacy_false_positive_control": {
            "historical_weight_predicate_would_classify_rigid": True,
            "minimum_owner_weight": legacy_min_owner,
            "maximum_other_mass": legacy_max_other,
            "noop_probe_passed": next(
                row["passed"]
                for row in deformable
                if row["name"] == "LEGACY_0999_BOUNDARY_FALSE_POSITIVE"
            ),
        },
        "claim_boundary": (
            "This validates mechanical rigid/no-op classification only. It does not "
            "infer object category, detachability, runtime presentation keyability, "
            "or artist semantic intent."
        ),
    }
    path = OUT / "VF13_RIGIDITY_NOOP_CALIBRATION.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
