import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.iris.truth.contracts_v1 import FAMILIES, SourceLocusRefV1, validate_manifest
from models.iris.truth.measurements_v1 import (
    affine_relative_frames, interval_velocity, oriented_pair_gap,
    response_gram, triangle_metric_strain,
)
from models.iris.truth.package_v1 import verify_package, write_package
from models.iris.truth.ray_compositing_v1 import measure_ray, removal_rgb_delta


def envelope():
    result = {"schema": "RealSaS.TruthPackageManifest.v1", "subject_key": "synthetic-test-only",
              "producer_source_commit": "1" * 40, "authority": "UNQUALIFIED_DERIVED_MEASUREMENTS",
              "coordinate_contract": {"frame": "SOURCE_NORMALIZED", "length_unit": "SOURCE_UNIT",
                                      "seconds_per_time_unit": 1, "normalization_sha256": "a" * 64},
              "payloads": {}, "families": []}
    for k in ("source_sha256", "topology_sha256", "teacher_scene_sha256",
              "intervention_spec_sha256", "observation_bundle_sha256"):
        result[k] = "b" * 64
    for name in sorted(FAMILIES):
        result["families"].append({"family": name, "measurement_status": "UNAVAILABLE",
                                   "observability": "UNKNOWN", "witness_sha256": None,
                                   "reason": "synthetic test does not supply this family",
                                   "validity_array": None, "payload_arrays": []})
    return result


class GeometryTests(unittest.TestCase):
    def test_relative_frames_cancel_common_affine_gauge(self):
        a, b, root = np.eye(4), np.eye(4), np.eye(4)
        a[:3, :3] = [[-1, .2, 0], [0, 2, .1], [0, 0, 1]]
        a[:3, 3], b[:3, 3] = [2, 3, 1], [6, -4, 2]
        root[:3, :3] = [[0, -2, 0], [1, .5, 0], [0, 0, 3]]
        root[:3, 3] = [100, -20, 5]
        relative, valid = affine_relative_frames(a, b)
        changed, changed_valid = affine_relative_frames(root @ a, root @ b)
        self.assertTrue(valid and changed_valid)
        np.testing.assert_allclose(changed, relative, atol=1e-12)

    def test_zero_scale_abstains_instead_of_fabricating_inverse(self):
        a = np.eye(4)
        a[0, 0] = 0
        result, valid = affine_relative_frames(a, np.eye(4))
        self.assertFalse(valid)
        self.assertTrue(np.isnan(result).all())

    def test_response_law_duplicate_and_basis_invariance(self):
        r = np.array([[[1., 2., 0.], [3., 0., 1.]], [[-1., 0., 0.], [2., 4., 0.]]])
        base, _ = response_gram(r, [.25, .75], np.ones((2, 2), dtype=bool))
        duplicate, _ = response_gram(r[:, [0, 0, 1]], [.125, .125, .75], np.ones((2, 3), bool))
        np.testing.assert_allclose(base, duplicate)
        # For an isotropic action law, orthogonal transport of the control basis.
        q = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        isotropic, _ = response_gram(r, [.5, .5], np.ones((2, 2), bool))
        rotated, _ = response_gram(np.einsum("icd,ck->ikd", r, q), [.5, .5], np.ones((2, 2), bool))
        np.testing.assert_allclose(isotropic, rotated)
        scaled, _ = response_gram(2 * r, [.25, .75], np.ones((2, 2), bool))
        np.testing.assert_allclose(scaled, 4 * base)

    def test_missing_response_does_not_renormalize_to_false_similarity(self):
        r = np.ones((2, 2, 3))
        r[0, 1] = np.nan
        gram, mask = response_gram(r, [.5, .5], np.array([[True, False], [True, True]]))
        self.assertFalse(mask[0, 1])
        self.assertTrue(np.isnan(gram[0, 1]))

    def test_rigid_rotation_has_no_surface_strain_but_stretch_does(self):
        x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        rot = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        strain, area, valid = triangle_metric_strain(x, x @ rot + [2, 5, 7], degeneracy_tolerance=1e-12)
        self.assertTrue(valid)
        np.testing.assert_allclose(strain, 0, atol=1e-12)
        self.assertAlmostEqual(float(area), 1)
        strain, area, _ = triangle_metric_strain(x, x * [2, 1, 1], degeneracy_tolerance=1e-12)
        np.testing.assert_allclose(np.linalg.eigvalsh(strain), [0, 1.5])
        self.assertAlmostEqual(float(area), 2)

    def test_degenerate_teacher_triangle_has_no_deformation_authority(self):
        x = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        strain, _, valid = triangle_metric_strain(x, x, degeneracy_tolerance=1e-12)
        self.assertFalse(valid)
        self.assertTrue(np.isnan(strain).all())

    def test_gap_and_velocity_keep_sign_units_and_actual_time(self):
        gap, speed = oriented_pair_gap([0, 0, 0], [0, 0, .02], [0, 0, 1], [0, 0, 0], [0, 0, -.3])
        self.assertAlmostEqual(float(gap), .02)
        self.assertAlmostEqual(float(speed), -.3)
        vel, mask = interval_velocity([[[0, 0, 0], [2, 0, 0], [5, 0, 0]]], [0, .5, 2], np.ones((1, 3), bool))
        np.testing.assert_allclose(vel[0, :, 0], [4, 2])
        self.assertTrue(mask.all())


class RayTests(unittest.TestCase):
    def measure(self, z, alpha, rgb, **kwargs):
        return measure_ray(z, alpha, rgb, complete=True, tie_epsilon=1e-9, **kwargs)

    def test_transparent_stack_and_counterfactual_recompose(self):
        result = self.measure([1, 2], [.5, .5], [[1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(result.transmittance, [1, .5])
        np.testing.assert_allclose(result.contribution, [.5, .25])
        np.testing.assert_allclose(result.premultiplied_rgb, [.5, 0, .25])
        self.assertAlmostEqual(result.alpha, .75)
        delta, valid = removal_rgb_delta([1, 2], [.5, .5], [[1, 0, 0], [0, 0, 1]], 0,
                                         complete=True, tie_epsilon=1e-9)
        self.assertTrue(valid)
        np.testing.assert_allclose(delta, [.5, 0, -.25])

    def test_same_colour_occluder_zero_delta_does_not_erase_occlusion(self):
        ray = self.measure([1, 2], [1, 1], [[1, 0, 0], [1, 0, 0]])
        delta, valid = removal_rgb_delta([1, 2], [1, 1], [[1, 0, 0], [1, 0, 0]], 0,
                                         complete=True, tie_epsilon=1e-9)
        self.assertTrue(valid)
        np.testing.assert_array_equal(delta, [0, 0, 0])
        self.assertEqual(ray.transmittance[1], 0)

    def test_tie_permutation_keeps_only_order_independent_alpha(self):
        for colours in ([[1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0]]):
            result = self.measure([1, 1], [.5, .5], colours)
            self.assertFalse(result.per_hit_valid or result.composite_valid)
            self.assertTrue(result.alpha_valid)
            self.assertAlmostEqual(result.alpha, .75)

    def test_incomplete_unknown_and_additive_cannot_claim_normal_visibility(self):
        incomplete = measure_ray([1], [1], [[1, 0, 0]], complete=False, tie_epsilon=0)
        unknown = self.measure([1], [np.nan], [[1, 0, 0]])
        additive = self.measure([1], [1], [[1, 0, 0]], blend_mode="ADDITIVE")
        for result in (incomplete, unknown, additive):
            self.assertFalse(result.per_hit_valid or result.composite_valid or result.alpha_valid)

    def test_empty_complete_ray_is_transparent(self):
        result = self.measure([], [], np.empty((0, 3)))
        self.assertTrue(result.composite_valid)
        self.assertEqual(result.alpha, 0)
        np.testing.assert_array_equal(result.premultiplied_rgb, [0, 0, 0])

    def test_unknown_colour_can_retain_geometric_transmittance(self):
        result = self.measure([1], [.5], [[np.nan, 0, 0]])
        self.assertTrue(result.per_hit_valid and result.alpha_valid)
        self.assertFalse(result.composite_valid)


class PackageTests(unittest.TestCase):
    def test_source_locus_is_topology_bound_and_rejects_invalid_barycentric(self):
        SourceLocusRefV1("a" * 64, "b" * 64, 0, 1, (.2, .3, .5))
        for bary in ((1, 1, -1), (.2, .2, .2), (float("nan"), 0, 1)):
            with self.assertRaises(ValueError):
                SourceLocusRefV1("a" * 64, "b" * 64, 0, 1, bary)

    def test_reproducible_arrays_manifest_and_tamper_detection(self):
        e = envelope()
        family = next(x for x in e["families"] if x["family"] == "SURFACE_TRAJECTORY")
        family.update(measurement_status="MEASURED", observability="PRIVILEGED_ONLY",
                      validity_array="valid", payload_arrays=["position"])
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "one", Path(tmp) / "two"
            arrays = {"valid": np.ones((1, 2), bool), "position": np.zeros((1, 2, 3))}
            h1, h2 = (write_package(p, e, arrays) for p in (one, two))
            self.assertEqual(h1, h2)
            verify_package(one, expected_manifest_sha256=h1)
            with self.assertRaises(FileExistsError):
                write_package(one, e, arrays)
            with (one / "arrays/position.npy").open("ab") as out:
                out.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "payload identity"):
                verify_package(one, expected_manifest_sha256=h1)

    def test_conditioned_truth_requires_witness_and_labels_cannot_replace_measurements(self):
        e = envelope()
        f = e["families"][0]
        f["observability"] = "OBSERVATION_CONDITIONED"
        with self.assertRaisesRegex(ValueError, "witness"):
            validate_manifest(e)
        e = envelope()
        e["families"][0]["family"] = "ATTACHMENT"
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_manifest(e)

    def test_path_escape_and_fake_qualification_rejected(self):
        e = envelope()
        e["payloads"] = {"bad": {"path": "../raw.npy", "sha256": "a" * 64, "dtype": "<f8", "shape": [1]}}
        with self.assertRaisesRegex(ValueError, "escape"):
            validate_manifest(e)
        e = envelope()
        e["authority"] = "QUALIFIED"
        with self.assertRaisesRegex(ValueError, "authority"):
            validate_manifest(e)

    def test_numeric_payload_preserves_scalar_and_endian_identity(self):
        from models.iris.truth.package_v1 import array_bytes
        a, raw = array_bytes(np.array(3.5, dtype=">f8"))
        self.assertEqual(a.shape, ())
        little, other = array_bytes(np.array(3.5, dtype="<f8"))
        self.assertEqual(little.shape, ())
        self.assertEqual(raw, other)

    def test_object_payload_and_nonboolean_validity_rejected(self):
        from models.iris.truth.package_v1 import array_bytes
        with self.assertRaises(ValueError):
            array_bytes(np.array([{"owner": "hidden-bone-id"}], dtype=object))
        with self.assertRaisesRegex(ValueError, "mask/weight"):
            response_gram(np.ones((2, 2, 3)), [.5, .5], np.ones((2, 2), int))


if __name__ == "__main__":
    unittest.main()
