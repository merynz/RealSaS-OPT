# RealSaS — Animation-Grade Mesh Conditioning Calibration V1 — 2026-09-18

Status: FROZEN_BEFORE_SELF_HOSTED_CALIBRATION_RESULT
Scope: G3 numeric lower-bound calibration only; subject-independent.
Forbidden inputs: Knight meshes/results, Mage mesh percentile distributions, teacher/source mesh topology.

## Question

What rest-triangle conditioning floor is sufficient for the current float32 product geometry/deformation measurement path to remain stable under a small, dimensionless posed-vertex perturbation budget?

This study does not prove artist-quality topology and does not close QualifiedMesh. It calibrates the numerical/conditioning lower bound used by G3. G5 visible coverage and G1 geometric-refinement bounds remain separate policies.

## Exact synthetic family

Rest triangles are isosceles with minimum angles:

0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60 degrees.

They are evaluated at local scales:

1e-3, 1, 1e3.

The posed material maps are frozen as identity, rigid 3D Y rotation by 37 degrees, anisotropic principal stretches 2 and 0.5, unit shear, and compression with principal stretches 1 and 0.25.

A fixed deterministic three-vertex perturbation pattern is applied to posed XYZ at amplitudes:

0, 1e-5, 1e-4, 1e-3 times local_scale.

No random seed or subject data is used.

## Metric

For every rest/posed triangle the exact rest material plane is mapped to posed 3D and the 3x2 deformation gradient is measured by singular values. This makes the metric invariant to global rigid 3D rotation.

Against the analytical transform singular values, measure relative error in minimum principal stretch, area ratio, and deformation condition number. The per-case error is the maximum of those three relative errors.

## Frozen selection rule

An angle passes only if the worst error over all frozen scales, transforms and perturbation amplitudes is <= 5%.

The animation-grade numerical minimum-angle candidate is the smallest swept angle that passes.

The paired aspect candidate uses longest_edge / minimum_altitude. Its integer ceiling is frozen as ceil(aspect_of_the_selected_isosceles_triangle).

A separately preregistered visual/raster stability study may make the final product threshold stricter, but Knight or Mage outcome inspection may not relax it.

## Nonclaims

- No Knight or Mage product PASS.
- No G5 coverage threshold.
- No G1 dense-refinement threshold.
- No universal motion envelope.
- No claim that an individual 3D triangle has a meaningful rest-normal signed inversion under arbitrary rigid rotation. 3D collapse and anisotropy are measured by singular values; screen-space visibility/foldover remains a downstream visual/motion proof concern.
