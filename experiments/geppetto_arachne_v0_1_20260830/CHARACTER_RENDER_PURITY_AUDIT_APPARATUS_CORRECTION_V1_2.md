# Character/Render Purity Audit — Apparatus Correction V1.2

Status: **PREREGISTERED_APPARATUS_CORRECTION__NO_QUALITY_RESULT_OPENED**  
Date: 2026-08-31

## Trigger

The first real-path smoke asset failed before any corpus-quality measurement was opened:

`asset_00027381105293114a49dc90: raster resolution mismatch: (1024, 1024)`

The V1.1 apparatus had incorrectly reused the derived `cel_clean_512.png` size as the expected size of canonical `raster_authority.npz`.

## Canonical authority

The RealSaS native frontend/master raster authority is **1024×1024 per view**. The `cel_clean_512.png` artifact is a separate derived 512×512 image representation.

## Frozen correction

V1.2 changes only apparatus resolution binding:

- `raster_authority.npz`: require exactly **1024×1024**;
- `cel_clean_512.png`: continue existence-only verification in this objective gate;
- no PNG pixel decoding;
- all occupancy, bbox, frame-margin, connected-component, triangle-dominance and skin-visibility metrics remain defined as normalized fractions and are otherwise unchanged;
- no quality threshold is selected;
- no semantic character classifier is applied;
- no Geppetto/Arachne output is read;
- training remains unauthorized.

## Regression test

The V1.2 fixture intentionally separates the authorities:

- native raster: 1024×1024;
- derived cel image: 512×512.

A 512×512 raster must fail closed. This directly guards the bug that triggered the correction.

Because the failure occurred in the four-asset real-path smoke before the full measurement loop, this is an apparatus correction, not a post-hoc quality-threshold change.
