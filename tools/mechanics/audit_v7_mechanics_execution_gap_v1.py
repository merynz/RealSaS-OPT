from __future__ import annotations

"""Read-only V7 producer vs Mechanics consumer compatibility audit.

The V7 research model (PR #352, 2026-11-06) explicitly reuses the
AXIS V6 RigProgram event vocabulary. We examine ACTUAL V6 declared
operator and DOF language and ACTUAL Mechanics V1 executable schema
without importing torch or changing any canonical/producer file.

Does not test V7 weight learning, cross-PR merge safety or actual
source-to-canonical correctness. This is a necessary vocabulary-gap
receipt, not a product release gate or solver proof.
"""

import argparse
import ast
import json
from pathlib import Path


SCHEMA="RealSaS.V7MechanicsExecutionGapResearchAudit.v1"


def _assignment(tree: ast.AST, name: str) -> ast.expr:
    for item in getattr(tree, "body", ()):
        if (isinstance(item,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in item.targets)):
            return item.value
        if (isinstance(item,ast.AnnAssign)
                and isinstance(item.target,ast.Name)
                and item.target.id==name and item.value is not None):
            return item.value
    raise ValueError("V7_GAP_AUDIT_SYMBOL_MISSING:"+name)


def _literal_tuple(tree: ast.AST, name: str) -> tuple[str,...]:
    value=ast.literal_eval(_assignment(tree,name))
    if not isinstance(value,tuple) or not value or not all(
        isinstance(x,str) for x in value
    ):
        raise ValueError("V7_GAP_AUDIT_VOCAB_INVALID:"+name)
    if len(set(value))!=len(value):
        raise ValueError("V7_GAP_AUDIT_VOCAB_DUPLICATE:"+name)
    return value


def _constraint_schema_ops(tree: ast.AST) -> tuple[str,...]:
    node=_assignment(tree,"RIG_CONSTRAINT_OPERATOR_SCHEMAS_V1")
    if not isinstance(node,ast.Tuple):
        raise ValueError("V7_GAP_AUDIT_RUNTIME_OP_SCHEMA_UNRECOGNIZED")
    ops=[]
    for call in node.elts:
        if not isinstance(call,ast.Call) or not call.args:
            raise ValueError("V7_GAP_AUDIT_RUNTIME_OP_SCHEMA_UNRECOGNIZED")
        op=ast.literal_eval(call.args[0])
        if not isinstance(op,str):
            raise ValueError("V7_GAP_AUDIT_RUNTIME_OP_SCHEMA_UNRECOGNIZED")
        ops.append(op)
    if len(ops)!=len(set(ops)):
        raise ValueError("V7_GAP_AUDIT_RUNTIME_OP_DUPLICATE")
    return tuple(ops)


def _multigoal_rotations(tree: ast.AST) -> tuple[str,...]:
    matches=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="parameter_axes"
                        for t in node.targets)
                and isinstance(node.value,ast.Dict)):
            matches.append(tuple(ast.literal_eval(key) for key in node.value.keys))
    if len(matches)!=1 or not matches[0] or not all(
        isinstance(x,str) for x in matches[0]
    ):
        raise ValueError("V7_GAP_AUDIT_MULTIGOAL_DOF_SCHEMA_UNRECOGNIZED")
    return matches[0]


def _event_fields(tree: ast.AST) -> tuple[str,...]:
    classes=[
        node for node in getattr(tree,"body",())
        if isinstance(node,ast.ClassDef) and node.name=="RigProgramEventV6"
    ]
    if len(classes)!=1:
        raise ValueError("V7_GAP_AUDIT_RIG_EVENT_SCHEMA_MISSING")
    return tuple(
        stmt.target.id for stmt in classes[0].body
        if isinstance(stmt,ast.AnnAssign)
        and isinstance(stmt.target,ast.Name)
    )


def audit_v7_mechanics_execution_gap_v1(root: Path) -> dict:
    paths={
        "producer":"models/axis/rig_program_model_v6.py",
        "runtime":"compiler/realsas_compiler_core/rig_constraint_schema_v1.py",
        "multigoal":"compiler/realsas_compiler_core/mechanics_multigoal_solver_v1.py",
    }
    trees={}
    for label,filename in paths.items():
        candidate=root/filename
        if not candidate.is_file():
            raise ValueError("V7_GAP_AUDIT_SOURCE_FILE_MISSING:"+label)
        trees[label]=ast.parse(candidate.read_text(encoding="utf-8"))

    producer_ops=_literal_tuple(trees["producer"],"CONSTRAINT_OPERATORS_V6")
    producer_dofs=_literal_tuple(trees["producer"],"DOF_LABELS_V6")
    producer_controls=_literal_tuple(trees["producer"],"CONTROL_KINDS_V6")
    runtime_ops=_constraint_schema_ops(trees["runtime"])
    multi_dofs=_multigoal_rotations(trees["multigoal"])
    event_fields=_event_fields(trees["producer"])
    missing_ops=tuple(x for x in producer_ops if x not in runtime_ops)
    missing_dofs=tuple(x for x in producer_dofs if x not in multi_dofs)
    # The event has generic FRAME_6D, but not a distinct ANCHOR frame
    # payload. This observation is about the typed V6 event schema; the
    # V7 generator's use of the event is separately version-referenced.
    declared_anchor_frame_payload=any(name in event_fields for name in (
        "anchor_local_basis","anchor_basis_6d","anchor_orientation",
        "anchor_rotation",
    ))
    distinct_relative_bridge_payload=any(name in event_fields for name in (
        "attachment_bridge","relative_attachment_frame",
        "relative_grip_transform",
    ))
    gaps=[]
    if missing_ops:
        gaps.append("PRODUCER_CONSTRAINT_VOCAB_NOT_EXECUTABLE_IN_CANONICAL_MECHANICS")
    if missing_dofs:
        gaps.append("PRODUCER_DOF_VOCAB_NOT_EXECUTABLE_IN_MULTIGOAL_IK")
    if not declared_anchor_frame_payload:
        gaps.append("NO_SEPARATE_TYPED_ANCHOR_ORIENTATION_EVENT_PAYLOAD")
    if not distinct_relative_bridge_payload:
        gaps.append("NO_SEPARATE_TYPED_RELATIVE_6D_ATTACHMENT_BRIDGE_EVENT_PAYLOAD")
    return {
        "schema":SCHEMA,
        "status":(
            "GAP__NO_V7_MECHANICS_PRODUCT_READINESS"
            if gaps else "STATIC_VOCAB_COVERAGE_ONLY__NO_MODEL_OR_PRODUCT_PROOF"
        ),
        "producer_contract":"AXIS_V7_PR352_REUSES_V6_RIG_PROGRAM_VOCAB",
        "source_paths":paths,
        "observations":{
            "producer_constraint_operator_ids":producer_ops,
            "canonical_runtime_constraint_operator_ids":runtime_ops,
            "producer_only_operator_ids":missing_ops,
            "producer_dof_labels":producer_dofs,
            "multigoal_analytic_rotational_dof_labels":multi_dofs,
            "producer_dofs_not_executable_in_analytic_multigoal":missing_dofs,
            "producer_control_kinds":producer_controls,
            "producer_rig_event_payload_fields":event_fields,
            "explicit_typed_anchor_frame_payload_present":declared_anchor_frame_payload,
            "explicit_typed_relative_bridge_payload_present":distinct_relative_bridge_payload,
        },
        "gaps":gaps,
        "authority":{
            "source_teacher_truth_proven":False,
            "v7_trained_model_available":False,
            "training_admission_minted":False,
            "rig_program_execution_product_ready":False,
            "unseen_generalization_proven":False,
            "graphics_authority_claimed":False,
            "default_product_compiler_mutated":False,
            "scope":"STATIC_CROSS_OWNER_VOCAB_AUDIT_ONLY",
        },
        "next_qualified_gate":"Full typed operator/DOF/attachment frame/relative bridge capability compatibility, qualified teacher and codegen, independent source intervention and learning execution",
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    result=audit_v7_mechanics_execution_gap_v1(args.repo_root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
