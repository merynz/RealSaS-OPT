# IRIS Single-Pose V2 — Resolution & Coordinate Contract

Status: `FROZEN_FOR_CODE_PREFLIGHT__TRAINING_NOT_AUTHORIZED`
Date: 2026-08-24

Prior experiments mixed 1024 authority, 512 derivatives, 256 model input, 128 matcher candidates and normalized-grid tolerances. V2 forbids implicit resolution semantics.

## Resolution names

- `R_authority`: native raster/correspondence authority; primary 1024.
- `R_input`: actual IRIS image; controls 256/512, primary 1024.
- `R_coarse = R_input/8`: Z_coarse lattice.
- `R_fine = R_input/2`: P/N/U_geo/Z_fine lattice.

Primary: `1024 -> coarse 128 -> fine 512`. Matched 512 control: `1024 authority -> input 512 -> coarse 64 -> fine 256`. Matched 256 control: `1024 authority -> input 256 -> coarse 32 -> fine 128`.

## Coordinate authority

Image origin TOP_LEFT, x right, y down, coordinates refer to pixel centers. Neural sampling uses `grid_sample(..., align_corners=False)`.

Pixel center p at resolution R:

`g = 2*(p+0.5)/R - 1`

Inverse:

`p = (g+1)*R/2 - 0.5`

`coords.py` is the only executable conversion authority.

## Coarse containment

Coarse Recall@K uses exact containing-cell identity, not a free normalized tolerance. Truth is mapped to one target coarse cell under the align_corners=False partition; a candidate hits only if it names that cell.

## Fine localization

Fine/end-to-end error is always reported in native authority pixels, with mean/median/p90/p95 and PCK <=2/4/8/16 native px. Lower-resolution controls naturally pay more authority pixels per field-cell error; that is the information-loss measurement.

## Row corridor

Known-camera vertical geometry is expressed in coarse-row cells. Default controlled safety envelope is query coarse row +/-1 target coarse cell; exact-row results must also be reportable.

## Fine local window

Fine search radius is expressed in fine cells. Default audit candidate is +/-4 fine cells, step 1. Since R_fine=R_input/2, one fine cell corresponds to two input pixels.

## Object-space P/N

P is canonical/object-frame geometry and is never image-resolution scaled. Full selected-corpus P min/max/p99/p100 must be audited before any output-range activation is accepted. N is a unit vector; evaluation reports angular degrees, not only 1-cos loss.

## Forbidden evaluator practices

- normalized tolerance without native-pixel interpretation;
- counting styles as independent physical assets;
- mean-only promotion where G0 requires tails;
- global fine-descriptor ranking while calling it local refinement;
- changing resolution and architecture in one comparison without a matched control;
- treating teacher query selection as autonomous source-track extraction.
