from __future__ import annotations

"""Typed executable constraint vocabulary shared by AXIS, Compiler and Mechanics.

An operator is admitted only when the runtime has enough canonical information to
execute it deterministically. Unsupported production-rig concepts (for example a
PATH constraint without a canonical path/curve object) are deliberately absent
rather than represented by a name with no semantics.
"""

from dataclasses import dataclass
from typing import Mapping, Any
import math

from .types import QualificationError


@dataclass(frozen=True)
class RigConstraintOperatorSchemaV1:
    operator_id: str
    min_control_inputs: int
    max_control_inputs: int
    min_joint_outputs: int
    max_joint_outputs: int
    chain_requirement: str  # REQUIRED | OPTIONAL | FORBIDDEN
    parameter_names: tuple[str, ...]


RIG_CONSTRAINT_OPERATOR_SCHEMAS_V1: tuple[RigConstraintOperatorSchemaV1, ...] = (
    RigConstraintOperatorSchemaV1(
        "IK", 1, 2, 0, 256, "REQUIRED",
        ("p0", "p1", "p2", "p3"),  # weight, pole_weight, max_stretch, softness
    ),
    RigConstraintOperatorSchemaV1(
        "AIM", 1, 1, 1, 256, "FORBIDDEN",
        ("p0", "p1", "p2", "p3"),  # weight, roll, reserved, reserved
    ),
    RigConstraintOperatorSchemaV1(
        "FOLLOW", 1, 1, 1, 256, "OPTIONAL",
        ("p0", "p1", "p2", "p3"),  # translation/rotation blend controls
    ),
    RigConstraintOperatorSchemaV1(
        "COPY_TRANSFORM", 1, 1, 1, 256, "FORBIDDEN",
        ("p0", "p1", "p2", "p3"),
    ),
    RigConstraintOperatorSchemaV1(
        "STRETCH", 1, 1, 0, 256, "REQUIRED",
        ("p0", "p1", "p2", "p3"),  # weight, max_ratio, reserved, reserved
    ),
)

RIG_CONSTRAINT_OPERATOR_IDS_V1 = tuple(x.operator_id for x in RIG_CONSTRAINT_OPERATOR_SCHEMAS_V1)
_SCHEMA_BY_ID = {x.operator_id: x for x in RIG_CONSTRAINT_OPERATOR_SCHEMAS_V1}


def rig_constraint_operator_schema_v1(operator_id: str) -> RigConstraintOperatorSchemaV1:
    try:
        return _SCHEMA_BY_ID[str(operator_id)]
    except KeyError as exc:
        raise QualificationError(f"RIG_CONSTRAINT_OPERATOR_UNSUPPORTED:{operator_id}") from exc


def validate_rig_constraint_operator_v1(
    *,
    operator_id: str,
    input_control_count: int,
    output_joint_count: int,
    has_chain: bool,
    parameters: Mapping[str, Any],
) -> None:
    schema = rig_constraint_operator_schema_v1(operator_id)
    if not schema.min_control_inputs <= int(input_control_count) <= schema.max_control_inputs:
        raise QualificationError(f"RIG_CONSTRAINT_CONTROL_ARITY:{schema.operator_id}")
    if not schema.min_joint_outputs <= int(output_joint_count) <= schema.max_joint_outputs:
        raise QualificationError(f"RIG_CONSTRAINT_OUTPUT_ARITY:{schema.operator_id}")
    if schema.chain_requirement == "REQUIRED" and not has_chain:
        raise QualificationError(f"RIG_CONSTRAINT_CHAIN_REQUIRED:{schema.operator_id}")
    if schema.chain_requirement == "FORBIDDEN" and has_chain:
        raise QualificationError(f"RIG_CONSTRAINT_CHAIN_FORBIDDEN:{schema.operator_id}")
    keys = set(map(str, parameters))
    allowed = set(schema.parameter_names)
    if not keys.issubset(allowed):
        raise QualificationError(f"RIG_CONSTRAINT_PARAMETER_UNSUPPORTED:{schema.operator_id}")
    for key, value in parameters.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise QualificationError(f"RIG_CONSTRAINT_PARAMETER_NONNUMERIC:{schema.operator_id}:{key}") from exc
        if not math.isfinite(scalar):
            raise QualificationError(f"RIG_CONSTRAINT_PARAMETER_NONFINITE:{schema.operator_id}:{key}")


__all__ = [
    "RIG_CONSTRAINT_OPERATOR_IDS_V1",
    "RIG_CONSTRAINT_OPERATOR_SCHEMAS_V1",
    "RigConstraintOperatorSchemaV1",
    "rig_constraint_operator_schema_v1",
    "validate_rig_constraint_operator_v1",
]
