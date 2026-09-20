# Complete Appearance Authority V2 — Subject-Free Numerical Policy

**Date:** 2026-09-20  
**Status:** FROZEN before any Knight V2 witness execution.

This policy gives appearance the same preregistration discipline as geometry and mechanics. It is not tuned from Knight or Mage output.

## Analytic atlas constraint

CAA V1 uses one unique barycentric tile per canonical face. With tile resolution 8 and two texels of mandatory bleed on every side, tile stride is 12. A 2048 texture admits 170 tiles per dimension, therefore 28,900 faces maximum without violating the product texture cap.

If Stage18 produces more faces, appearance may not silently increase texture size. Mesh/atlas representation must be repaired upstream under a new qualified policy.

## Source lock

Direct source lock requires exact camera/observation correspondence, canonical first-hit visibility, a one-pixel boundary-safe region, and abs(normal dot camera-forward) >= cos(80 degrees). The angular criterion measures surface-to-raster correspondence support, not color quality.

Safe source background with alpha zero is observed transparent art. It is not UNKNOWN.

## Totality and quality are separate

Total defined fraction and direct-source mutation protection are exact 1.0 requirements.

Completion quality is independently measured by structured anisotropic connected-band holdouts and provenance-boundary color/gradient discontinuity. The thresholds are fault-detection guardrails frozen before the witness; they are not claims that values just below them are perceptually optimal.

## Rest artwork is a product gate

Stage25 must qualify the final rendered source-foreground RGBA, not only locked texels, and must separately qualify source-alpha recall, precision, coherent holes, and interior uncovered support. Geometry-visible transparent pixels are diagnostic because transparent CAA is valid authored appearance.

## Dynamic honesty budget

At Stage45, undefined visible provenance is forbidden. Native/reference mismatch is forbidden. Compiled-unobserved appearance is allowed only within the frozen 20% screen-space exposure budget.

This budget does not turn inferred art into source truth; provenance remains explicit in the sealed asset.
