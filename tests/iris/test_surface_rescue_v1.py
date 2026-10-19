from __future__ import annotations

import unittest

import torch

from models.iris.vnext.surface_rescue_v1 import FrozenHSurfaceRescueConfigV1, FrozenHSurfaceRescueV1, iris_cell_step_v1


class FrozenHSurfaceRescueTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(7)
        return FrozenHSurfaceRescueV1(FrozenHSurfaceRescueConfigV1(local_feature_dim=12, point_width=32, point_residual_blocks=2, cube_residual_width=48))

    def _inputs(self):
        b, n = 2, 16
        h = torch.randn(b, n, 12)
        xyz = torch.rand(b, n, 3) * 1.8 - 0.9
        field = torch.randn(b, n) * 0.2
        delta = torch.tanh(torch.randn(b, n, 3)) * 0.025
        return h, xyz, field, delta

    def test_zero_step_native_r24_reproduces_v8_surface_inputs_exactly(self):
        model = self._model()
        h, xyz, field, delta = self._inputs()
        out = model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=24)
        self.assertTrue(torch.equal(out["field"], field))
        self.assertTrue(torch.allclose(out["deformed_xyz_iris"], xyz + delta, atol=2e-7, rtol=0.0))
        cubes = torch.tensor([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]])
        inherited = torch.randn(2,2,21)
        cube = model.cube_parameters(point_features=out["point_features"], cubes=cubes, inherited_v8_raw_parameters=inherited)
        self.assertTrue(torch.equal(cube["raw_parameters"], inherited))
        self.assertTrue(torch.count_nonzero(cube["cube_parameter_residual"]) == 0)

    def test_inherited_deformation_is_cell_relative_across_resolution(self):
        model = self._model()
        h, xyz, field, delta = self._inputs()
        native = model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=24)
        fine = model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=48)
        self.assertTrue(torch.allclose(native["delta_cells"], fine["delta_cells"], atol=1e-6, rtol=0.0))
        native_abs = native["deformed_xyz_iris"] - xyz
        fine_abs = fine["deformed_xyz_iris"] - xyz
        self.assertTrue(torch.allclose(fine_abs, native_abs * 0.5, atol=2e-7, rtol=0.0))
        self.assertEqual(iris_cell_step_v1(24), 2.0/24.0)

    def test_total_deformation_remains_inside_preregistered_cell_bound(self):
        model = self._model()
        with torch.no_grad():
            model.delta_residual_logits.bias.fill_(100.0)
        h, xyz, field, delta = self._inputs()
        out = model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=48)
        self.assertLessEqual(float(out["delta_cells"].abs().max()), 0.4500001)

    def test_grad_carrying_v8_evidence_is_rejected(self):
        model = self._model()
        h, xyz, field, delta = self._inputs()
        h.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "FROZEN_V8_INPUT_REQUIRES_GRAD:H"):
            model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=24)

    def test_inherited_v8_delta_must_fit_declared_cell_semantics(self):
        model = self._model()
        h, xyz, field, delta = self._inputs()
        delta[:] = 0.04
        with self.assertRaisesRegex(ValueError, "INHERITED_DELTA_EXCEEDS_CELL_BOUND"):
            model.point_decode(h=h, xyz_iris=xyz, raw_v8_field=field, v8_extraction_delta=delta, resolution=24)


if __name__ == "__main__":
    unittest.main()
