# Oracle-Z Representation Court V1 — preregistration

Status: SEALED BEFORE OUTCOME
Authority: research-only; no product/FIT8/Truth139 promotion.

## Question

Before Truth Builder training, test whether the current **triplane + local query** representation is a plausible carrier for the local evidence families we care about, independent of DINO/camera/scene-encoder quality.

This court intentionally bypasses the image encoder and directly optimizes an oracle scene carrier `Z`. It therefore tests representation/query capacity and inductive bias, not scene inference.

## Arms

All learned arms use the same query points, targets, optimizer, steps and seeds.

1. `TRIPLANE_Q`
   - three 2-D planes;
   - bilinear XY/XZ/YZ sampling with `align_corners=False` semantics;
   - concatenated samples -> shared nonlinear query MLP;
   - no explicit xyz input.

2. `TRIPLANE_Q_XYZ`
   - identical carrier;
   - explicit xyz appended only at the query MLP;
   - diagnostic for whether forcing Q to reconstruct coordinates from Z is a material burden.

3. `MULTIRES_TRIPLANE_Q`
   - two triplane scales under approximately the same carrier-scalar budget as Arm 1;
   - diagnostic for local/high-frequency surface evidence.

4. `DENSE3D_Q`
   - trilinear 3-D carrier under the same carrier-scalar budget as Arm 1;
   - adversarial representational control, not a proposed production architecture.

The probe implementation reports exact carrier and shared-network parameter counts. A larger arm is not selected merely because it wins.

## Truth families

Each synthetic scene has random parameters unknown to the shared query network. Therefore xyz alone cannot solve the task; scene information must flow through Z.

### A. SURFACE_STRUCTURED
Union of randomly placed anisotropic ellipsoids. Targets at each 3-D locus:
- signed geometry field;
- nearest-surface authored appearance scalar;
- mechanics/rigidity scalar;
- causal-support scalar tied to component-local intervention family.

### B. THIN_SURFACE
Same construction with one or more deliberately thin anisotropic components and narrow support transitions. Boundary-heavy samples are oversampled in evaluation.

### C. VOLUMETRIC_3WAY_CONTROL
Smooth random 3-D Fourier/hash-like field with genuine xyz interaction and no surface prior. This is deliberately unfavorable to triplanes and exists to prove that the court can expose a representation mismatch.

## Controls

- Same random scene definitions across arms.
- Shared query network across scenes inside one arm; per-scene carrier only.
- Held-out continuous query points are never optimized directly.
- Random-target firewall must stay near chance / high error.
- The volumetric control is not allowed to veto triplanes for RealSaS surface evidence by itself.

## Interpretation gates

For each surface family, report held-out normalized RMSE and boundary subset RMSE. `TRIPLANE_Q` is considered **surface-capacity supported at tiny scale** only if:
- mean normalized RMSE <= 0.06 on SURFACE_STRUCTURED;
- mean normalized RMSE <= 0.10 on THIN_SURFACE;
- no seed exceeds 0.15 on either surface family.

`TRIPLANE_Q_XYZ` is preferred only if it improves worst-seed surface error by >=20% relative while not materially degrading another surface family. Otherwise explicit xyz is not justified by this court.

`MULTIRES_TRIPLANE_Q` is preferred only if it improves THIN_SURFACE worst-seed error by >=20% while SURFACE_STRUCTURED remains within +0.01 absolute normalized RMSE and carrier budget is not larger.

`DENSE3D_Q` is a diagnostic control. A dense-grid win on VOLUMETRIC_3WAY_CONTROL alone is expected and does not select dense 3-D production state.

If current triplane fails a surface gate, the result is **not** immediately a production FAIL because this is a deliberately small CPU proxy. Failure escalates to an exact/reference-scale court (GPU may then be required). Passing is sufficient evidence that the representation mechanism itself is not an obvious blocker for that family.

## Non-claims

This court does not establish:
- learnability from images;
- causal truth correctness;
- joint IRIS/TESSA success;
- FIT8 genericity;
- production resolution;
- superiority of triplane over every representation.
