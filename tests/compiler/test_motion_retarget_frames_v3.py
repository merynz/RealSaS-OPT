import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_rows
from compiler.realsas_compiler_core.motion_retarget_frames_v3 import compile_retargeted_keys
from compiler.realsas_compiler_core.types import QualificationError


class RetargetFrameTests(unittest.TestCase):
    def source(self):
        rows = [dict(source_joint_id="root", parent_source_joint_id=None, rest_position=[0, 0, 0]),
                dict(source_joint_id="shoulder", parent_source_joint_id="root", rest_position=[0, 0, 1]),
                dict(source_joint_id="hand", parent_source_joint_id="shoulder", rest_position=[1, 0, 1])]
        frames = derive_joint_frames_from_rows(rows, joint_id_key="source_joint_id", parent_id_key="parent_source_joint_id", position_key="rest_position")
        tracks = []
        for sid in frames:
            F = np.array(frames[sid].rotation_matrix)
            keys = []
            for time in (0., 1.):
                # Only the omitted source shoulder moves: a 90-degree Z turn.
                R = Rotation.from_euler("z", 90 * time if sid == "shoulder" else 0, degrees=True).as_matrix()
                keys.append(dict(time_seconds=time, local_rotation_quat_xyzw=Rotation.from_matrix(F.T @ R @ F).as_quat().tolist(), local_translation_xyz=[0., 0., 0.], local_scale_xyz=[1., 1., 1.]))
            tracks.append(dict(source_joint_id=sid, keyframes=keys))
        return dict(source_skeleton=rows, tracks=tracks, metadata=dict(source_body_scale=2.))

    def test_skipped_shoulder_motion_survives_different_target_axes(self):
        # Target hand local axes deliberately disagree with source hand axes.
        F = Rotation.from_euler("y", 43, degrees=True).as_matrix()
        frames = {"r": SimpleNamespace(parent_id=None, rest_position=(0, 0, 0), rotation_matrix=np.eye(3)),
                  "h": SimpleNamespace(parent_id="r", rest_position=(2, 0, 2), rotation_matrix=F)}
        out = compile_retargeted_keys(self.source(), target_frames=frames,
            target_to_source={"r": "root", "h": "hand"}, object_rotation=np.eye(3), target_body_scale=4.)
        local = Rotation.from_quat(out["h"][-1]["local_rotation_quat_xyzw"]).as_matrix()
        expected = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_allclose(F @ local @ F.T, expected, atol=1e-12)
        # Source hand rotates about (0,0,1), not its own head. Folding ancestry
        # must retain that orbital translation as well as its rotation.
        np.testing.assert_allclose(F @ np.array(out["h"][-1]["local_translation_xyz"]), [-2, 2, 0], atol=1e-12)

    def test_object_basis_rotation_is_applied_before_target_local_rebase(self):
        frames = {"r": SimpleNamespace(parent_id=None, rest_position=(0, 0, 0), rotation_matrix=np.eye(3)),
                  "h": SimpleNamespace(parent_id="r", rest_position=(0, 2, 2), rotation_matrix=np.eye(3))}
        C = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        out = compile_retargeted_keys(self.source(), target_frames=frames,
            target_to_source={"r": "root", "h": "hand"}, object_rotation=C, target_body_scale=4.)
        actual = Rotation.from_quat(out["h"][-1]["local_rotation_quat_xyzw"]).as_matrix()
        np.testing.assert_allclose(actual, Rotation.from_euler("y", -90, degrees=True).as_matrix(), atol=1e-12)

    def test_reflection_is_rejected(self):
        with self.assertRaisesRegex(QualificationError, "OBJECT_BASIS_INVALID"):
            compile_retargeted_keys(self.source(), target_frames={}, target_to_source={},
                object_rotation=np.diag([-1, 1, 1]), target_body_scale=1.)


if __name__ == "__main__":
    unittest.main()
