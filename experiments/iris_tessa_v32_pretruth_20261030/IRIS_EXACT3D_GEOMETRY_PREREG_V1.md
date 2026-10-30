# IRIS Exact-3D Geometry Target Court V1

Date: 2026-10-30
Status: PREREGISTRATION_SOURCE__PLATFORM_EXECUTION_REQUIRED

## Question
Does replacing legacy F*-only geometry authority with exact qualified 3D TSDF/sign/normal truth improve watertight geometry and sign placement without degrading source consistency?

## Fixed architecture
- Same observation encoder and Dense3D shared carrier Z.
- Same query capacity unless an arm explicitly changes only the geometry output/objective.
- Same source views/cameras and deterministic extraction/courts.
- F* remains available to every arm as measured source-consistency evidence.

## Arms
A0 LEGACY_FSTAR_ONLY: current signed_field/F* lineage.
A1 EXACT3D_TSDF_SIGN: exact watertight 3D TSDF + inside/outside supervision; F* auxiliary only.
A2 EXACT3D_TSDF_SIGN_NORMAL: A1 + oriented surface-normal supervision in the qualified near-surface band.
A3 EXACT3D_TSDF_SIGN_NORMAL_NO_FSTAR_AUX: ablation to measure whether source auxiliary is still necessary.

## Primary metrics
- exact 3D sign disagreement, stratified by near-surface/interior/exterior;
- zero-surface displacement to truth mesh;
- watertightness / boundary-edge count;
- connected component and hole morphology;
- surface normal angular error;
- source-view silhouette/coverage courts;
- unseen/hidden-region geometry error where truth is observable only from 3D authority.

## Interpretation
A candidate may replace F*-only authority only if it improves exact 3D sign/zero-surface truth and does not violate source consistency or watertightness gates. Low scalar field MAE alone cannot select an arm.

## Non-claims
No training split, ModelRelease promotion, product authority or generalization claim is created by this preregistration source file. Canonical execution requires a Platform Experiment/Attempt, EngineRelease on reviewed main, and Artifact Registry/CAS evidence.
