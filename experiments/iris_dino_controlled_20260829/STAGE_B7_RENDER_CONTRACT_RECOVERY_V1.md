# RealSaS — Stage-B7 Native Render Contract Recovery V1

**Date:** 2026-08-29  
**Status:** `RECOVERED_FROM_PERSISTED_PRODUCTION_ARTIFACTS__ZERO_DINO_OPTIMIZER_STEPS`  
**Scope:** selective publication/rerender for the exact nine Stage-B6 repair-pending members of the recovered historical train-512 population.

## Why this recovery exists

The exact historical 512-family DINO comparison population contains nine assets whose Stage-B6 corrected geometry is frozen but whose canonical A-pass images/raster authority still correspond to the pre-repair geometry. Reusing those stale images with repaired truth is forbidden, and replacing/dropping the families would change the preregistered population.

Therefore the old production render contract was recovered from persisted master artifacts before writing the selective Stage-B7 publisher.

## Recovered camera contract

Eight yaw views:

`yaw = 0,45,90,135,180,225,270,315 degrees`

For yaw `theta`:

- `forward = [sin(theta), cos(theta), 0]`
- `right   = [cos(theta),-sin(theta), 0]`
- `up      = [0,0,1]`
- `screen_up = up`
- image origin = `TOP_LEFT`
- image y direction = `DOWN`
- NDC y direction = `UP`
- vertical flip count = `1`

A single asset-level half extent is used for all eight views:

`half_extent = 1.08 * max_view(max(abs(P·right_view)), max(abs(P·up)))`

This formula reproduces persisted production cameras. Example `asset_00aa1b666ba193851a498194`:

- reconstructed: `0.5570941582437483`
- persisted/report authority: approximately `0.5570941257476807`

The forward depth bounds for each view are exactly geometry-derived:

- `z_min = min(P·forward)`
- `z_max = max(P·forward)`

Pixel-center mapping recovered from persisted raster authority:

- `x_pixel = (x_ndc + 1) * 1024 / 2 - 0.5`
- `y_pixel = (1 - y_ndc) * 1024 / 2 - 0.5`

On a persisted real V0 raster this maps barycentric reconstructed surface points back to their stored pixel centers at near floating-point precision.

## Raster authority contract

The existing controlled-IRIS package explicitly describes the authority as `nvdiffrast barycentric_uv`.

Per visible native-1024 pixel store:

- `pixel_linear_index`
- `triangle_id`
- `barycentric_uv`
- `resolution = [1024,1024]`
- `origin = TOP_LEFT`

Persisted `barycentric_uv = [lambda_0, lambda_1]`; `lambda_2 = 1-lambda_0-lambda_1`.

A zero-step independent software rasterization check on a real persisted V0 witness recovered:

- exact visible support size: historical `172531`, independent renderer `172532`;
- support XOR: `61` pixels;
- common-pixel triangle-id agreement: approximately `99.7484%`.

The residual is consistent with GPU rasterizer edge/tie convention, so Stage-B7 must use nvdiffrast rather than declaring the independent CPU rasterizer canonical.

## Cel-clean shading contract

Normal interpolation uses the persisted raster barycentrics and canonical vertex normals.

Normalize:

`L = normalize([0.35,-0.45,0.82])`

For interpolated normalized surface normal `N`:

`q = dot(N,L)`

Three bands:

- dark if `q < 0.35`
- mid if `0.35 <= q < 0.72`
- light if `q >= 0.72`

On the real `asset_f56aff9ecf13f7ddf0afcd55 / V0` persisted native render this rule reproduces **172531 / 172531 foreground color-band assignments exactly**.

The asset palette itself is preserved from that asset's previous production render rather than re-derived. The three nontransparent `cel_clean` colors are sorted dark -> mid -> light by luminance and reused for the corrected geometry. This preserves asset appearance while changing only the geometry/raster publication.

Example old palette for `asset_f56aff...`:

- dark `[94,95,39]`
- mid `[141,143,58]`
- light `[197,198,81]`

A second persisted asset uses a different palette, confirming that palette is asset-specific and should be preserved from the asset's prior publication.

## Ink-cel contract

`ink_cel` uses exactly the same alpha/support and cel band colors except for a one-native-pixel interior silhouette outline.

Recovered outline mask:

`outline = foreground AND NOT binary_erosion_3x3(foreground, iterations=1)`

The 3x3 erosion uses all eight neighbors.

On the same real V0 witness:

- changed pixels between cel and ink: `5374`
- recovered 3x3 interior boundary pixels: `5374`
- mask XOR: `0`

Outline RGB is integer floor of 10% of the underlying cel RGB:

- `[94,95,39] -> [9,9,3]`
- `[141,143,58] -> [14,14,5]`
- `[197,198,81] -> [19,19,8]`

Alpha is unchanged.

## 512 derivatives

The native RGBA image is authoritative. `*_512.png` is generated from native `1024x1024` RGBA with the corpus LANCZOS transform.

A prior byte-domain corpus audit demonstrated native `cel_clean.png -> cel_clean_512.png` reproduction with:

- RGB MAE `0`
- alpha MAE `0`
- support IoU `1.0`

No separate 512 renderer is allowed.

## Publication discipline

The selective Stage-B7 runner must:

1. use only the already-frozen Stage-B6 `normalized.npz` for the nine exact assets;
2. preserve each asset's previous cel palette;
3. use nvdiffrast for native 1024 rasterization;
4. derive camera metadata from corrected geometry by the recovered formulas;
5. generate cel/ink native images from the recovered shading/outline rules;
6. derive 512 images only by LANCZOS;
7. write to a staging tree first;
8. validate all eight views and geometry/raster/image lineage;
9. publish atomically only after the full nine-asset staging audit passes;
10. never replace/drop a train family.

DINO feature extraction and optimizer steps remain forbidden until the resulting exact train population passes 512/512 native-authority preflight.
