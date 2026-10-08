import numpy as np
import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_motion_safety_v1 import (
    POLICY,
    apply_repair_domains,
    compile_motion_repair,
    select_repair_domains,
)


def _square():
    rest = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    domains = np.zeros(4, dtype=np.int32)
    return rest, faces, domains


def _area_ratio(rest, posed, faces):
    rb = np.stack((rest[faces[:, 1]] - rest[faces[:, 0]],
                   rest[faces[:, 2]] - rest[faces[:, 0]]), axis=2)
    pb = np.stack((posed[faces[:, 1]] - posed[faces[:, 0]],
                   posed[faces[:, 2]] - posed[faces[:, 0]]), axis=2)
    return np.linalg.det(pb @ np.linalg.inv(rb))


def test_safe_body_domain_is_not_repaired():
    rest, faces, domains = _square()
    owners = np.zeros(4, dtype=np.int32)
    frames = np.stack((rest, rest + [2.0, -3.0]))
    selected, diagnostics = select_repair_domains(
        rest_positions=rest, visual_faces=faces, domain_id=domains,
        vertex_attachment_owner=owners, frames_xy=frames)
    assert selected.tolist() == []
    assert diagnostics[0]["selected"] is False


def test_unsafe_body_domain_is_projected_to_proper_unit_scale_motion():
    rest, faces, domains = _square()
    owners = np.zeros(4, dtype=np.int32)
    safe = rest + [4.0, 2.0]
    collapsed = rest * 0.01 + [7.0, -1.0]
    fields = np.concatenate((np.stack((safe, collapsed)), np.ones((2, 4, 1))), axis=2)
    result = compile_motion_repair(
        rest_positions=rest, visual_faces=faces, domain_id=domains,
        vertex_attachment_owner=owners, clip_fields={"clip": fields})
    assert result["repair_domain_ids"].tolist() == [0]
    repaired = result["repaired_fields"]["clip"][:, :, :2]
    assert np.all(_area_ratio(rest, repaired[0], faces) > POLICY["minimum_signed_area_ratio"])
    assert np.all(_area_ratio(rest, repaired[1], faces) > POLICY["minimum_signed_area_ratio"])
    for frame in repaired:
        before = np.linalg.norm(rest[faces[:, 1]] - rest[faces[:, 0]], axis=1)
        after = np.linalg.norm(frame[faces[:, 1]] - frame[faces[:, 0]], axis=1)
        assert np.allclose(after, before, atol=1e-10, rtol=0)


def test_repair_selection_is_witness_union_and_applies_to_all_frames():
    rest, faces, domains = _square()
    owners = np.zeros(4, dtype=np.int32)
    safe_but_scaled = rest * 0.9 + [1.0, 1.0]
    collapsed = rest * 0.01 + [2.0, 2.0]
    fields = np.concatenate((
        np.stack((safe_but_scaled, collapsed)), np.zeros((2, 4, 1))), axis=2)
    result = compile_motion_repair(
        rest_positions=rest, visual_faces=faces, domain_id=domains,
        vertex_attachment_owner=owners, clip_fields={"clip": fields})
    repaired = result["repaired_fields"]["clip"][:, :, :2]
    assert result["repair_domain_ids"].tolist() == [0]
    assert not np.allclose(repaired[0], safe_but_scaled)
    assert np.allclose(
        np.linalg.norm(repaired[0, 1] - repaired[0, 0]),
        np.linalg.norm(rest[1] - rest[0]), atol=1e-10, rtol=0)


def test_attachment_domain_is_never_claimed_by_body_repair():
    rest, faces, domains = _square()
    owners = np.ones(4, dtype=np.int32)
    collapsed = np.stack((rest * 0.001, rest * 0.002))
    selected, diagnostics = select_repair_domains(
        rest_positions=rest, visual_faces=faces, domain_id=domains,
        vertex_attachment_owner=owners, frames_xy=collapsed)
    assert selected.tolist() == []
    assert diagnostics[0]["reason"] == "TARGET_ATTACHMENT_EXCLUDED"


def test_apply_repair_rejects_attachment_scope():
    rest, _, domains = _square()
    with pytest.raises(QualificationError, match="REPAIR_SCOPE_INVALID"):
        apply_repair_domains(
            rest_positions=rest, frame_xy=rest.copy(), domain_id=domains,
            vertex_attachment_owner=np.ones(4, dtype=np.int32),
            repair_domain_ids=np.asarray([0], dtype=np.int32))
