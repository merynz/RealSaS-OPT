# IRIS Single-Pose V2 — Active Audit Implementation

Status: `PREFLIGHTED_SOURCE__TRAINING_FORBIDDEN_UNTIL_AUDIT_CLOSES`

This directory replaces `experiments/iris_controlled_v1/` as the only candidate active IRIS implementation path. V1 remains historical/diagnostic provenance and must not be launched during the architecture audit.

## Frozen mission

`ONE neutral pose x 8 ordered RGBA views -> observable geometry/persistence evidence -> deterministic SurfaceBuilder -> RiggingSurface`

IRIS V2 does not predict hidden authored owners, skeleton topology, parents, skin weights or mandatory GFDR.

## Neural roles

```text
input R x R x 8
 -> shared encoder f2/f4/f8/f16
 -> full-f16 within-view axial context
 -> fixed-size pooled cross-view context (known yaw)
 -> upsample/fuse context into full local f16
 -> decoder
      P / N / U_geo @ R/2
      Z_coarse       @ R/8   [global high-recall role]
      Z_fine         @ R/2   [local refinement role only]
```

There is no fixed `max_w`, no fixed 128 matcher lattice, and no global `Z_fine` authority.

## Deterministic matcher roles

1. build target candidate basins from alpha support on the actual Z_coarse lattice;
2. apply known-camera row corridor in coarse cells;
3. retain high-recall basin union from Z_coarse and predicted P;
4. order coarse basins using only coarse/P evidence;
5. use Z_fine only inside local patches around admitted basins;
6. preserve coarse-basin order across basins;
7. emit top-k hypotheses plus evidence; no calibrated singleton is claimed yet;
8. reciprocal/cycle are evidence/qualification channels, not hidden identity.

## Training roles

- P/N/U_geo: dense geometry supervision.
- Z_coarse: observation-level multi-positive correspondence, pairwise bidirectional matching, hard-negative margin; no view pooling before supervision.
- Z_fine: local offset classification only around a truth-containing local basin.
- P cross-view consistency: common-frame geometric coherence.

## Files

- `ARCHITECTURE_CONTRACT_V2.md`
- `RESOLUTION_COORDINATE_CONTRACT_V2.md`
- `CORPUS_INTERFACE_V2.md`
- `EVALUATION_CONTRACT_V2.md`
- `coords.py`, `model.py`, `losses.py`, `matcher.py`, `metrics.py`, `preflight.py`

## Authorization

No trainer or Colab notebook belongs in this directory until `audit/IRIS_ARCHITECTURE_AUDIT_20260824.md` closes with every mini-gate mismatch resolved.
