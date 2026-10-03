from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_rig_tokens_v1 import (
    build_canonical_rig_tokens_v1,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualificationError,
)


def _skeleton(ids=("root","a","b"), *, shuffled=False, renamed=False):
    rid, aid, bid = ids
    rows = [
        QualifiedJoint(rid,(0.0,0.0,0.0),None,(),"src-root"),
        QualifiedJoint(aid,(0.0,1.0,0.0),rid,(),"src-a"),
        QualifiedJoint(bid,(1.0,2.0,0.0),aid,(),"src-b"),
    ]
    if shuffled:
        rows = [rows[2], rows[0], rows[1]]
    return QualifiedSkeletonIR(tuple(rows),rid,{"status":"PASS"},"lineage")


def _semantic_payload(tokens):
    return [
        (
            t.sequence_index,
            t.parent_sequence_index,
            t.is_root,
            tuple(round(x,8) for x in t.position_normalized),
            tuple(round(x,8) for x in t.parent_delta_normalized),
            round(t.parent_distance_normalized,8),
            t.child_count,
        )
        for t in tokens.tokens
    ]


def test_canonical_rig_tokens_are_input_order_invariant():
    a = build_canonical_rig_tokens_v1(_skeleton())
    b = build_canonical_rig_tokens_v1(_skeleton(shuffled=True))
    assert a.token_set_hash == b.token_set_hash
    assert _semantic_payload(a) == _semantic_payload(b)


def test_canonical_rig_tokens_are_joint_id_and_source_id_invariant():
    a = build_canonical_rig_tokens_v1(_skeleton())
    b = build_canonical_rig_tokens_v1(_skeleton(ids=("X","Y","Z")))
    assert a.token_set_hash == b.token_set_hash
    assert _semantic_payload(a) == _semantic_payload(b)


def test_canonical_rig_tokens_change_when_parent_structure_changes():
    base = _skeleton()
    rows = list(base.joints)
    rows[2] = replace(rows[2], parent_canonical_id="root")
    changed = QualifiedSkeletonIR(tuple(rows),"root",{"status":"PASS"},"lineage2")
    a = build_canonical_rig_tokens_v1(base)
    b = build_canonical_rig_tokens_v1(changed)
    assert a.token_set_hash != b.token_set_hash


def test_canonical_rig_tokens_do_not_serialize_lineage_or_source_ids_into_hash():
    base = _skeleton()
    rows = tuple(replace(j, source_proposal_id="teacher-name-leak") for j in base.joints)
    changed = QualifiedSkeletonIR(rows,base.root_id,{"status":"DIFFERENT"},"different-lineage")
    assert build_canonical_rig_tokens_v1(base).token_set_hash == build_canonical_rig_tokens_v1(changed).token_set_hash


def test_canonical_rig_tokens_fail_closed_on_disconnected_or_bad_root():
    s = _skeleton()
    rows = list(s.joints)
    rows[2] = replace(rows[2], parent_canonical_id="missing")
    bad = QualifiedSkeletonIR(tuple(rows),"root",{"status":"PASS"},"x")
    with pytest.raises(QualificationError):
        build_canonical_rig_tokens_v1(bad)

    with pytest.raises(QualificationError):
        build_canonical_rig_tokens_v1(
            QualifiedSkeletonIR(s.joints,"missing",{"status":"PASS"},"x")
        )
